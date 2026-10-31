// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "focal_sparse_conv.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>

#include "deterministic_sparse_conv.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace openpcdet::ext {

namespace {

// Option C: the split importance sigmoid uses the shared deterministic
// det_sigmoid (extensions/deterministic_sparse_conv.h) instead of glibc
// std::exp, so the host-side split/threshold stage is bit-identical no matter
// which backend produced the conv_imp importance map (and would stay identical
// if split ever moves on-device, where expf/~2ULP vs glibc ~0.5ULP diverge).
inline float sigmoidf(float x) { return det::det_sigmoid(x); }

// ---------------------------------------------------------------------------
// Indice-map cache: FocalSparseConv runs several SubM convolutions over the
// exact same active voxel set (conv2.1/conv2.2, conv_imp vs the preceding
// SubM layer), so rebuilding the rulebook every call is pure waste. The cache
// is a tiny FIFO keyed by a 64-bit content hash of (coordinates, shape,
// params); on a hit the full coordinate buffer is compared before reuse, so a
// hash collision can never yield a wrong map.
// ---------------------------------------------------------------------------

struct map_cache_entry {
    uint64_t hash = 0;
    std::vector<int32_t> coordinates;
    std::vector<int32_t> spatial_shape;
    sparse_conv_params params;
    sparse_indice_map map;
};

struct map_cache {
    std::vector<map_cache_entry> entries;  // FIFO, bounded
};

map_cache &cached_maps() {
    static map_cache cache;
    return cache;
}

uint64_t hash_bytes(const void *data, size_t size, uint64_t seed) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    uint64_t h = seed;
    for (size_t i = 0; i < size; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t params_hash(const sparse_conv_params &params) {
    uint64_t h = 1469598103934665603ull;
    for (const auto &values :
         {params.kernel_size, params.stride, params.padding, params.dilation}) {
        h = hash_bytes(values.data(), values.size() * sizeof(int32_t), h);
    }
    h = hash_bytes(&params.submanifold, sizeof(bool), h);
    h = hash_bytes(&params.transposed, sizeof(bool), h);
    return h;
}

bool params_equal(const sparse_conv_params &a, const sparse_conv_params &b) {
    return a.kernel_size == b.kernel_size && a.stride == b.stride &&
           a.padding == b.padding && a.dilation == b.dilation &&
           a.submanifold == b.submanifold && a.transposed == b.transposed;
}

// Linearized spconv coordinate (batch omitted): z * y_max * x_max + y * x_max +
// x.
inline int64_t linearized_coord(const int32_t *coord,
                                int64_t y_max,
                                int64_t x_max) {
    // Mixed-radix key mirrored from PyTorch check_repeat:
    //   z * (y.max * x.max) + y * x.max + x.
    // y_max / x_max are the max over the coordinate set, so this
    // (intentionally) mirrors Py exactly, including any theoretical collisions.
    return static_cast<int64_t>(coord[1]) * y_max * x_max +
           static_cast<int64_t>(coord[2]) * x_max +
           static_cast<int64_t>(coord[3]);
}

}  // namespace

bool build_sparse_indice_map_cached(const sparse_tensor &input,
                                    const sparse_conv_params &params,
                                    sparse_indice_map &map,
                                    std::string &error) {
    uint64_t hash = params_hash(params);
    hash = hash_bytes(input.spatial_shape.data(),
                      input.spatial_shape.size() * sizeof(int32_t), hash);
    hash = hash_bytes(input.coordinates.data(),
                      input.coordinates.size() * sizeof(int32_t), hash);

    map_cache &cache = cached_maps();
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->hash != hash || it->coordinates != input.coordinates ||
            it->spatial_shape != input.spatial_shape ||
            !params_equal(it->params, params))
            continue;
        map = it->map;
        if (it != cache.entries.begin())
            std::rotate(cache.entries.begin(), it, it + 1);
        return true;
    }

    if (!build_sparse_indice_map(input, params, map, error)) return false;

    map_cache_entry entry;
    entry.hash = hash;
    entry.coordinates = input.coordinates;
    entry.spatial_shape = input.spatial_shape;
    entry.params = params;
    entry.map = map;
    constexpr size_t max_entries = 4;
    if (cache.entries.size() >= max_entries) cache.entries.pop_back();
    cache.entries.insert(cache.entries.begin(), std::move(entry));
    return true;
}

std::vector<float> streamed_sparse_conv_weight(
        const std::vector<float> &weights,
        int kernel_volume,
        int in_channels,
        int out_channels) {
    // The GGUF stores a sparse conv weight as a dense [K*Cin, Cout] matrix
    // (ne[0]=K*Cin, ne[1]=Cout). Row-major indexing data[(k*Cin+c)*Cout+o] is
    // exactly the [K, Cin, Cout] layout ext::sparse_convolution expects, so the
    // buffer is passed through unchanged.
    (void)kernel_volume;
    (void)in_channels;
    (void)out_channels;
    return weights;
}

std::vector<std::array<int32_t, 3>> focal_kernel_offsets(int kernel_size) {
    const int half = kernel_size / 2;
    std::vector<std::array<int32_t, 3>> offsets;
    offsets.reserve(static_cast<size_t>(kernel_size) * kernel_size *
                    kernel_size);
    for (int z = -half; z <= half; ++z)
        for (int y = -half; y <= half; ++y)
            for (int x = -half; x <= half; ++x) {
                if (z == 0 && y == 0 && x == 0) continue;  // remove center
                offsets.push_back({z, y, x});
            }
    return offsets;
}

void focal_check_repeat(const std::vector<float> &features,
                        const std::vector<int32_t> &coordinates,
                        int channels,
                        bool flip_first,
                        const std::vector<float> *features_add,
                        std::vector<float> &out_features,
                        std::vector<int32_t> &out_coordinates,
                        std::vector<float> *out_add) {
    const int64_t n = static_cast<int64_t>(coordinates.size()) / 4;
    out_features.clear();
    out_coordinates.clear();
    if (out_add) out_add->clear();
    if (n == 0) return;

    int64_t y_max = 0, x_max = 0;
    for (int64_t i = 0; i < n; ++i) {
        y_max = std::max<int64_t>(y_max, coordinates[i * 4 + 2]);
        x_max = std::max<int64_t>(x_max, coordinates[i * 4 + 3]);
    }
    if (y_max < 0) y_max = 0;
    if (x_max < 0) x_max = 0;

    // Pack (linearized key, row) into one 64-bit word and radix-sort it. The
    // key is z * y_max * x_max + y * x_max + x, which stays below 2^32 for
    // every real point-cloud grid, so key << 32 | row is collision-free; LSD
    // radix sort is stable, giving the exact same row order as the previous
    // std::stable_sort over keys. This avoids the order[] indirection, the
    // intermediate sorted copies and the n log n comparison work.
    std::vector<uint64_t> packed(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const int64_t key = linearized_coord(&coordinates[i * 4], y_max, x_max);
        packed[static_cast<size_t>(i)] =
                (static_cast<uint64_t>(key) << 32) | static_cast<uint32_t>(i);
    }
    radix_sort_u64(packed);

    // unique_consecutive over the sorted keys. With flip_first the permuted
    // run is reversed, so the coordinate of the merged row is the FIRST row of
    // the run instead of the last (torch scatter_ overwrites with the last
    // written index), and the output groups come out in DESCENDING key order
    // (unique_consecutive runs over the flipped sequence). Accumulation order
    // matches PyTorch exactly.
    std::vector<int64_t> group_first;
    group_first.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t key = packed[static_cast<size_t>(i)] >> 32;
        if (i == 0 || key != packed[static_cast<size_t>(i - 1)] >> 32)
            group_first.push_back(i);
    }
    const int64_t uniq = static_cast<int64_t>(group_first.size());

    out_coordinates.assign(static_cast<size_t>(uniq) * 4, 0);
    out_features.assign(static_cast<size_t>(uniq) * channels, 0.0f);
    if (out_add) out_add->assign(static_cast<size_t>(uniq), 0.0f);
    auto row_of = [&](int64_t index) {
        return static_cast<int64_t>(packed[static_cast<size_t>(index)] &
                                    0xFFFFFFFFull);
    };
// Each unique group writes disjoint output rows, so the merge can run in
// parallel without changing any accumulation order inside a group.
#pragma omp parallel for num_threads(std::min(8, omp_get_max_threads()))
    for (int64_t u = 0; u < uniq; ++u) {
        // PyTorch flips the sorted sequence before unique_consecutive, so the
        // output order is the reversed (descending-key) group order.
        const int64_t gu = flip_first ? (uniq - 1 - u) : u;
        const int64_t begin = group_first[static_cast<size_t>(gu)];
        const int64_t end =
                (gu + 1 < uniq) ? group_first[static_cast<size_t>(gu + 1)] : n;
        const int64_t first = row_of(begin);
        const int64_t last = row_of(end - 1);
        const int64_t coord_row = flip_first ? first : last;
        std::copy_n(&coordinates[coord_row * 4], 4, &out_coordinates[u * 4]);
        float *feat = &out_features[static_cast<size_t>(u) * channels];
        if (flip_first) {
            for (int64_t r = end - 1; r >= begin; --r) {
                const int64_t src = row_of(r);
                for (int c = 0; c < channels; ++c)
                    feat[c] +=
                            features[static_cast<size_t>(src) * channels + c];
            }
            if (out_add) {
                float sum = 0.0f;
                for (int64_t r = end - 1; r >= begin; --r)
                    sum += (*features_add)[static_cast<size_t>(row_of(r))];
                (*out_add)[static_cast<size_t>(u)] =
                        sum / static_cast<float>(end - begin);
            }
        } else {
            for (int64_t r = begin; r < end; ++r) {
                const int64_t src = row_of(r);
                for (int c = 0; c < channels; ++c)
                    feat[c] +=
                            features[static_cast<size_t>(src) * channels + c];
            }
            if (out_add) {
                float sum = 0.0f;
                for (int64_t r = begin; r < end; ++r)
                    sum += (*features_add)[static_cast<size_t>(row_of(r))];
                (*out_add)[static_cast<size_t>(u)] =
                        sum / static_cast<float>(end - begin);
            }
        }
    }
}

bool focal_split_voxels(
        const std::vector<float> &features,
        const std::vector<int32_t> &coordinates,
        int channels,
        const std::vector<float> &imps,
        int kernel_volume,
        int kernel_size,
        const std::vector<std::array<int32_t, 3>> &kernel_offsets,
        const std::vector<int32_t> &spatial_shape,
        bool mask_multi,
        bool topk,
        float threshold,
        std::vector<float> &fore_features,
        std::vector<int32_t> &fore_coordinates,
        std::vector<float> &back_features,
        std::vector<int32_t> &back_coordinates,
        std::vector<float> &mask_kernel_fore,
        std::string &error) {
    const int64_t n = static_cast<int64_t>(coordinates.size()) / 4;
    const int off_count =
            kernel_volume - 1;  // kernel neighbors excluding center
    if (off_count != static_cast<int>(kernel_offsets.size())) {
        error = "focal kernel offset count mismatch";
        return false;
    }
    if (static_cast<int>(imps.size()) != n * kernel_volume) {
        error = "focal importance map shape mismatch";
        return false;
    }

    // mask_voxel / mask_kernel (sigmoid).
    std::vector<float> mask_voxel(n), mask_kernel(n * off_count);
#pragma omp parallel for num_threads(std::min(8, omp_get_max_threads()))
    for (int64_t i = 0; i < n; ++i) {
        mask_voxel[i] = sigmoidf(imps[i * kernel_volume + (kernel_volume - 1)]);
        for (int k = 0; k < off_count; ++k)
            mask_kernel[i * off_count + k] =
                    sigmoidf(imps[i * kernel_volume + k]);
    }

    // Foreground/background partition.
    std::vector<int64_t> idx_fore, idx_back;
    if (topk) {
        // Pack (descending mask bits, row) and radix-sort. mask_voxel is a
        // sigmoid output in (0, 1), so its IEEE-754 bit pattern is monotonic
        // with the value; inverting the bits turns the ascending radix sort
        // into a descending mask sort that is stable within equal masks (row
        // ascending) — exactly the previous std::stable_sort semantics.
        std::vector<uint64_t> packed(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            uint32_t bits;
            std::memcpy(&bits, &mask_voxel[static_cast<size_t>(i)],
                        sizeof(bits));
            packed[static_cast<size_t>(i)] =
                    (static_cast<uint64_t>(0xFFFFFFFFu - bits) << 32) |
                    static_cast<uint32_t>(i);
        }
        radix_sort_u64(packed);
        const int64_t cut = static_cast<int64_t>(n * threshold);
        idx_fore.resize(static_cast<size_t>(cut));
        idx_back.resize(static_cast<size_t>(n - cut));
        for (int64_t i = 0; i < cut; ++i)
            idx_fore[static_cast<size_t>(i)] = static_cast<int64_t>(
                    packed[static_cast<size_t>(i)] & 0xFFFFFFFFull);
        for (int64_t i = cut; i < n; ++i)
            idx_back[static_cast<size_t>(i - cut)] = static_cast<int64_t>(
                    packed[static_cast<size_t>(i)] & 0xFFFFFFFFull);
    } else {
        for (int64_t i = 0; i < n; ++i) {
            if (mask_voxel[i] > threshold)
                idx_fore.push_back(i);
            else
                idx_back.push_back(i);
        }
    }

    const int64_t fcount = static_cast<int64_t>(idx_fore.size());
    // features_fore = features_ori[idx_fore] (optionally multiplied by
    // mask_voxel).
    fore_features.assign(static_cast<size_t>(fcount) * channels, 0.0f);
    fore_coordinates.assign(static_cast<size_t>(fcount) * 4, 0);
#pragma omp parallel for num_threads(std::min(8, omp_get_max_threads()))
    for (int64_t i = 0; i < fcount; ++i) {
        const int64_t src = idx_fore[i];
        const float mult = mask_multi ? mask_voxel[src] : 1.0f;
        for (int c = 0; c < channels; ++c)
            fore_features[i * channels + c] =
                    features[src * channels + c] * mult;
        std::copy_n(&coordinates[src * 4], 4, &fore_coordinates[i * 4]);
    }

    // Expand foreground voxels to their kernel neighbors. Two passes so the
    // selection can run in parallel: count per row, prefix-sum the offsets,
    // then fill the output at disjoint offsets.
    std::vector<int64_t> sel_offsets(static_cast<size_t>(fcount) + 1, 0);
#pragma omp parallel for num_threads(std::min(8, omp_get_max_threads()))
    for (int64_t i = 0; i < fcount; ++i) {
        const int64_t src = idx_fore[i];
        const int32_t *base = &coordinates[src * 4];
        int64_t count = 0;
        for (int k = 0; k < off_count; ++k) {
            if (mask_kernel[src * off_count + k] < threshold) continue;
            const int32_t z = base[1] + kernel_offsets[k][0];
            const int32_t y = base[2] + kernel_offsets[k][1];
            const int32_t x = base[3] + kernel_offsets[k][2];
            if (z > 0 && y > 0 && x > 0 && z < spatial_shape[0] &&
                y < spatial_shape[1] && x < spatial_shape[2])
                ++count;
        }
        sel_offsets[static_cast<size_t>(i)] = count;
    }
    // Exclusive prefix sum: sel_offsets[i] becomes the start offset of row i
    // (sum of counts before i) and sel_offsets[fcount] the total.
    int64_t running = 0;
    for (int64_t i = 0; i < fcount; ++i) {
        const int64_t count = sel_offsets[static_cast<size_t>(i)];
        sel_offsets[static_cast<size_t>(i)] = running;
        running += count;
    }
    sel_offsets[static_cast<size_t>(fcount)] = running;
    const int64_t mcount = running;
    std::vector<std::array<int32_t, 3>> sel_coords(static_cast<size_t>(mcount));
    std::vector<float> sel_mask(static_cast<size_t>(mcount));
#pragma omp parallel for num_threads(std::min(8, omp_get_max_threads()))
    for (int64_t i = 0; i < fcount; ++i) {
        const int64_t src = idx_fore[i];
        const int32_t *base = &coordinates[src * 4];
        int64_t out = sel_offsets[static_cast<size_t>(i)];
        for (int k = 0; k < off_count; ++k) {
            if (mask_kernel[src * off_count + k] < threshold) continue;
            const int32_t z = base[1] + kernel_offsets[k][0];
            const int32_t y = base[2] + kernel_offsets[k][1];
            const int32_t x = base[3] + kernel_offsets[k][2];
            if (z > 0 && y > 0 && x > 0 && z < spatial_shape[0] &&
                y < spatial_shape[1] && x < spatial_shape[2]) {
                if (out >= mcount) {
                    // Unreachable: offsets are the exclusive prefix sum of the
                    // same per-row counts used here.
                    std::fprintf(stderr,
                                 "focal split expansion out of range: %lld >= "
                                 "%lld\n",
                                 (long long)out, (long long)mcount);
                    std::abort();
                }
                sel_coords[static_cast<size_t>(out)] = {z, y, x};
                sel_mask[static_cast<size_t>(out)] =
                        mask_kernel[src * off_count + k];
                ++out;
            }
        }
    }
    // features_fore_cat = [fore; zeros(M)], coords_cat = [fore; b+sel], mask =
    // [ones; sel].
    std::vector<float> cat_feat(static_cast<size_t>(fcount + mcount) * channels,
                                0.0f);
    std::vector<int32_t> cat_coord(static_cast<size_t>(fcount + mcount) * 4, 0);
    std::vector<float> cat_mask(static_cast<size_t>(fcount + mcount), 1.0f);
    std::copy(fore_features.begin(), fore_features.end(), cat_feat.begin());
    std::copy(fore_coordinates.begin(), fore_coordinates.end(),
              cat_coord.begin());
#pragma omp parallel for num_threads(std::min(8, omp_get_max_threads()))
    for (int64_t i = 0; i < mcount; ++i) {
        const int32_t b =
                fore_coordinates[0];  // batch id is constant for this element
        cat_coord[(fcount + i) * 4 + 0] = b;
        cat_coord[(fcount + i) * 4 + 1] = sel_coords[static_cast<size_t>(i)][0];
        cat_coord[(fcount + i) * 4 + 2] = sel_coords[static_cast<size_t>(i)][1];
        cat_coord[(fcount + i) * 4 + 3] = sel_coords[static_cast<size_t>(i)][2];
        cat_mask[fcount + i] = sel_mask[static_cast<size_t>(i)];
    }

    focal_check_repeat(cat_feat, cat_coord, channels, true, &cat_mask,
                       fore_features, fore_coordinates, &mask_kernel_fore);

    // Background.
    const int64_t bcount = static_cast<int64_t>(idx_back.size());
    back_features.assign(static_cast<size_t>(bcount) * channels, 0.0f);
    back_coordinates.assign(static_cast<size_t>(bcount) * 4, 0);
#pragma omp parallel for num_threads(std::min(8, omp_get_max_threads()))
    for (int64_t i = 0; i < bcount; ++i) {
        const int64_t src = idx_back[i];
        std::copy_n(&features[src * channels], channels,
                    &back_features[i * channels]);
        std::copy_n(&coordinates[src * 4], 4, &back_coordinates[i * 4]);
    }
    return true;
}

bool focal_combine_out(const std::vector<float> &fore_features,
                       const std::vector<int32_t> &fore_coordinates,
                       const std::vector<float> &back_features,
                       const std::vector<int32_t> &back_coordinates,
                       int channels,
                       int batch_size,
                       std::vector<float> &features,
                       std::vector<int32_t> &coordinates,
                       std::string &error) {
    const int64_t nf = static_cast<int64_t>(fore_coordinates.size()) / 4;
    const int64_t nb = static_cast<int64_t>(back_coordinates.size()) / 4;
    std::vector<int32_t> all_coord(static_cast<size_t>(nf + nb) * 4, 0);
    std::vector<float> all_feat(static_cast<size_t>(nf + nb) * channels, 0.0f);
    std::copy(fore_coordinates.begin(), fore_coordinates.end(),
              all_coord.begin());
    std::copy(back_coordinates.begin(), back_coordinates.end(),
              all_coord.begin() + static_cast<int64_t>(nf) * 4);
    std::copy(fore_features.begin(), fore_features.end(), all_feat.begin());
    std::copy(back_features.begin(), back_features.end(),
              all_feat.begin() + static_cast<int64_t>(nf) * channels);

    features.clear();
    coordinates.clear();
    if (batch_size == 1) {
        // Fast path: no batch filtering needed, deduplicate in one pass.
        focal_check_repeat(all_feat, all_coord, channels, false, nullptr,
                           features, coordinates, nullptr);
        return true;
    }
    for (int b = 0; b < batch_size; ++b) {
        std::vector<int32_t> b_coord;
        std::vector<float> b_feat;
        for (int64_t i = 0; i < nf + nb; ++i) {
            if (all_coord[i * 4] == b) {
                std::copy_n(&all_coord[i * 4], 4, std::back_inserter(b_coord));
                std::copy_n(&all_feat[i * channels], channels,
                            std::back_inserter(b_feat));
            }
        }
        std::vector<int32_t> o_coord;
        std::vector<float> o_feat;
        focal_check_repeat(b_feat, b_coord, channels, false, nullptr, o_feat,
                           o_coord, nullptr);
        coordinates.insert(coordinates.end(), o_coord.begin(), o_coord.end());
        features.insert(features.end(), o_feat.begin(), o_feat.end());
    }
    return true;
}

}  // namespace openpcdet::ext
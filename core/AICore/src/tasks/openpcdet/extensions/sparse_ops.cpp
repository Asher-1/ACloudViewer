// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "sparse_ops.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Memory-bandwidth-bound OpenMP regions (hash probes, radix counting) scale
// poorly beyond ~8 threads on this class of NUMA machines: random access to a
// shared table saturates the L2/L3 interconnect and 32 threads measured ~6x
// SLOWER than 8. Cap the parallelism explicitly instead of trusting the
// environment default.
static inline int map_threads() {
#ifdef _OPENMP
    return std::min(8, omp_get_max_threads());
#else
    return 1;
#endif
}

namespace openpcdet::ext {

namespace {

// Scatter block budget for the compact lowering (see
// sparse_set_scatter_max_blocks).
int g_scatter_max_blocks = 27;

struct coord4 {
    int32_t b = 0, z = 0, y = 0, x = 0;
    bool operator==(const coord4 &other) const {
        return b == other.b && z == other.z && y == other.y && x == other.x;
    }
    bool operator<(const coord4 &other) const {
        return std::array<int32_t, 4>{b, z, y, x} <
               std::array<int32_t, 4>{other.b, other.z, other.y, other.x};
    }
};

coord4 read_coord(const sparse_tensor &tensor, int64_t row) {
    const int32_t *p = tensor.coordinates.data() + row * (tensor.ndim + 1);
    if (tensor.ndim == 3) return {p[0], p[1], p[2], p[3]};
    return {p[0], 0, p[1], p[2]};
}

void append_coord(sparse_tensor &tensor, const coord4 &c) {
    tensor.coordinates.push_back(c.b);
    if (tensor.ndim == 3) tensor.coordinates.push_back(c.z);
    tensor.coordinates.push_back(c.y);
    tensor.coordinates.push_back(c.x);
}

std::array<int32_t, 3> spatial3(const sparse_tensor &tensor) {
    if (tensor.ndim == 3) {
        return {tensor.spatial_shape[0], tensor.spatial_shape[1],
                tensor.spatial_shape[2]};
    }
    return {1, tensor.spatial_shape[0], tensor.spatial_shape[1]};
}

std::array<int32_t, 3> param3(const std::vector<int32_t> &values,
                              int ndim,
                              int32_t leading_2d) {
    if (ndim == 3) return {values[0], values[1], values[2]};
    return {leading_2d, values[0], values[1]};
}

bool inside(const coord4 &c, const std::array<int32_t, 3> &shape) {
    return c.b >= 0 && c.z >= 0 && c.y >= 0 && c.x >= 0 && c.z < shape[0] &&
           c.y < shape[1] && c.x < shape[2];
}

int64_t product(const std::vector<int32_t> &values) {
    int64_t result = 1;
    for (int32_t value : values) result *= value;
    return result;
}

std::vector<std::array<int32_t, 3>> kernel_offsets(
        const std::array<int32_t, 3> &kernel) {
    std::vector<std::array<int32_t, 3>> result;
    result.reserve(static_cast<size_t>(kernel[0]) * kernel[1] * kernel[2]);
    for (int z = 0; z < kernel[0]; z++)
        for (int y = 0; y < kernel[1]; y++)
            for (int x = 0; x < kernel[2]; x++) result.push_back({z, y, x});
    return result;
}

// ---------------------------------------------------------------------------
// Linearized-coordinate lookup table.
//
// spconv-style rulebook generation: instead of hashing a 4-tuple through a
// node-based std::unordered_map (one heap allocation + several cache misses
// per probe), every coordinate is compressed into a single mixed-radix int64
// key (batch-major, monotonic with [b, z, y, x] lexicographic order) and
// stored in a flat open-addressing table. Lookups are ~10x cheaper because
// the whole table stays in L2 and a probe is one 8-byte compare.
// ---------------------------------------------------------------------------

// Mixed-radix key: ((b * Z + z) * Y + y) * X + x + 1. The +1 keeps 0 as the
// "empty slot" sentinel while remaining monotonic with [b, z, y, x].
inline int64_t linearized_key(const coord4 &c,
                              const std::array<int32_t, 3> &shape) {
    return ((static_cast<int64_t>(c.b) * shape[0] + c.z) * shape[1] + c.y) *
                   shape[2] +
           c.x + 1;
}

class coord_table {
public:
    // Reserve for `count` entries (load factor <= 0.5).
    void build(int64_t count) {
        const int64_t min_capacity = 8;
        int64_t capacity = 1;
        while (capacity < std::max<int64_t>(min_capacity, count * 2))
            capacity <<= 1;
        slots_.assign(static_cast<size_t>(capacity), slot{0, -1});
        mask_ = static_cast<uint32_t>(capacity) - 1;
        size_ = 0;
    }

    // Inserts key -> value. Returns false when the key already exists.
    // Empty slots are marked by value == -1 (keys may legitimately be 0:
    // the linearized coordinate of voxel (0,0,0)).
    bool insert(int64_t key, int32_t value) {
        size_t idx = hash_slot(key);
        for (;;) {
            if (slots_[idx].value < 0) {
                slots_[idx] = slot{key, value};
                ++size_;
                return true;
            }
            if (slots_[idx].key == key) return false;
            idx = (idx + 1) & mask_;
        }
    }

    // Returns the stored value or -1 when the key is absent.
    int32_t find(int64_t key) const {
        size_t idx = hash_slot(key);
        for (;;) {
            const slot found = slots_[idx];
            if (found.value < 0) return -1;
            if (found.key == key) return found.value;
            idx = (idx + 1) & mask_;
        }
    }

    size_t size() const { return size_; }

private:
    // Key and value packed into one 16-byte slot: a linear probe then touches
    // a single cache line instead of one line per array (keys_ + values_).
    struct slot {
        int64_t key;
        int32_t value;
        int32_t padding;
    };

    size_t hash_slot(int64_t key) const {
        // The linearized keys are dense, low-entropy integers (clustered in
        // [0, Z*Y*X]), so the low bits are already well distributed once the
        // table is a power of two sized for load <= 0.5. A plain mask avoids
        // the multiply and, more importantly, maps neighbouring keys to
        // neighbouring slots, so probe sequences hit the same cache lines.
        return static_cast<size_t>(static_cast<uint64_t>(key) & mask_);
    }

    std::vector<slot> slots_;
    uint32_t mask_ = 0;
    size_t size_ = 0;
};

}  // namespace

// LSD radix sort over non-negative uint64 keys (16-bit digits, 4 passes).
// Replaces std::sort for the output-candidate key arrays: keys generated by a
// stride-2 kernel are 8x the unique output count, and the counting sort pays
// linear cost per pass instead of n log n compares. Stable (LSD order).
void radix_sort_u64(std::vector<uint64_t> &values) {
    const size_t n = values.size();
    if (n < 2) return;
    std::vector<uint64_t> buffer(n);
#ifdef _OPENMP
    // Parallel LSD radix sort with 8-bit digits: per-thread histograms are
    // 1 KB, so they stay L1-resident (the 16-bit variant's 256 KB tables
    // thrashed L2 and measured slower than the serial path).
    constexpr int passes = 8;  // 64-bit / 8-bit digits
    constexpr size_t buckets = 1u << 8;
    const int max_threads = map_threads();
    const int n_threads =
            std::min<int>(max_threads, std::max<int>(1, (int)(n / 8192)));
    if (n_threads <= 1) {
        for (int digit = 0; digit < passes; ++digit) {
            const unsigned shift = static_cast<unsigned>(digit * 8);
            std::array<uint32_t, buckets> counts{};
            for (uint64_t value : values) ++counts[(value >> shift) & 0xFF];
            uint32_t offset = 0;
            for (size_t b = 0; b < buckets; ++b) {
                const uint32_t count = counts[b];
                counts[b] = offset;
                offset += count;
            }
            for (uint64_t value : values)
                buffer[counts[(value >> shift) & 0xFF]++] = value;
            values.swap(buffer);
        }
        return;
    }
    std::vector<std::array<uint32_t, buckets>> thread_counts(
            static_cast<size_t>(n_threads));
    std::vector<std::array<uint32_t, buckets>> cursors(
            static_cast<size_t>(n_threads));
    const int64_t chunk = (int64_t)((n + n_threads - 1) / n_threads);
    for (int digit = 0; digit < passes; ++digit) {
        const unsigned shift = static_cast<unsigned>(digit * 8);
        for (auto &counts : thread_counts) counts.fill(0);
#pragma omp parallel for num_threads(n_threads) schedule(static)
        for (int64_t t = 0; t < n_threads; ++t) {
            const int64_t begin = t * chunk;
            const int64_t end = std::min<int64_t>(begin + chunk, (int64_t)n);
            auto &counts = thread_counts[static_cast<size_t>(t)];
            for (int64_t i = begin; i < end; ++i)
                ++counts[(values[static_cast<size_t>(i)] >> shift) & 0xFF];
        }
        // Global bucket offsets: prefix sum over the thread histograms.
        std::array<uint32_t, buckets> offsets{};
        for (size_t b = 0; b < buckets; ++b) {
            uint32_t total = 0;
            for (int t = 0; t < n_threads; ++t)
                total += thread_counts[static_cast<size_t>(t)][b];
            offsets[b] = total;
        }
        uint32_t running = 0;
        for (size_t b = 0; b < buckets; ++b) {
            const uint32_t count = offsets[b];
            offsets[b] = running;
            running += count;
        }
        // Per-thread scatter cursors: global offsets + prefix of thread counts.
        for (size_t b = 0; b < buckets; ++b) {
            uint32_t acc = offsets[b];
            for (int t = 0; t < n_threads; ++t) {
                const uint32_t count = thread_counts[static_cast<size_t>(t)][b];
                cursors[static_cast<size_t>(t)][b] = acc;
                acc += count;
            }
        }
#pragma omp parallel for num_threads(n_threads) schedule(static)
        for (int64_t t = 0; t < n_threads; ++t) {
            const int64_t begin = t * chunk;
            const int64_t end = std::min<int64_t>(begin + chunk, (int64_t)n);
            auto &cursor = cursors[static_cast<size_t>(t)];
            for (int64_t i = begin; i < end; ++i) {
                const uint64_t value = values[static_cast<size_t>(i)];
                buffer[cursor[(value >> shift) & 0xFF]++] = value;
            }
        }
        values.swap(buffer);
    }
#else
    constexpr int passes = 8;  // 64-bit / 8-bit digits
    constexpr size_t buckets = 1u << 8;
    for (int digit = 0; digit < passes; ++digit) {
        const unsigned shift = static_cast<unsigned>(digit * 8);
        std::array<uint32_t, buckets> counts{};
        for (uint64_t value : values) ++counts[(value >> shift) & 0xFF];
        uint32_t offset = 0;
        for (size_t b = 0; b < buckets; ++b) {
            const uint32_t count = counts[b];
            counts[b] = offset;
            offset += count;
        }
        for (uint64_t value : values)
            buffer[counts[(value >> shift) & 0xFF]++] = value;
        values.swap(buffer);
    }
#endif
}

int64_t sparse_tensor::active_count() const {
    return ndim == 2 || ndim == 3
                   ? static_cast<int64_t>(coordinates.size()) / (ndim + 1)
                   : 0;
}

int64_t sparse_indice_map::output_active_count() const {
    if (device_output_active_count > 0) return device_output_active_count;
    return ndim == 2 || ndim == 3
                   ? static_cast<int64_t>(coordinates.size()) / (ndim + 1)
                   : 0;
}

// Derive the CSR-style compact rulebook from the dense `indices` layout:
// count valid neighbors per kernel offset, prefix-sum the group offsets, then
// fill compact_indices (input rows, grouped by kernel) and scatter_rows
// (compact column per (output row, kernel), -1 when invalid). The scatter
// column is stable within a kernel group because the traversal below visits
// output rows in ascending order.
//
// Both passes parallelize over output rows / kernel offsets: the count pass
// writes per-thread histograms, the fill pass gives each kernel offset its
// own output-range cursor, so no atomic or lock is needed.
void build_compact_rulebook(sparse_indice_map &map) {
    const size_t k_volume = static_cast<size_t>(map.kernel_volume);
    const size_t n_out = static_cast<size_t>(map.output_active_count());
    std::vector<int32_t> valid_per_k(k_volume, 0);
#ifdef _OPENMP
    const int n_threads = map_threads();
    if (n_out > 4096 && n_threads > 1) {
        // Per-thread histograms (k_volume ints each), merged afterwards; the
        // shared-vector reduction form is not portable across compilers.
        std::vector<std::vector<int32_t>> thread_counts(
                static_cast<size_t>(n_threads),
                std::vector<int32_t>(k_volume, 0));
        const int64_t chunk = (int64_t)((n_out + n_threads - 1) / n_threads);
#pragma omp parallel for num_threads(n_threads) schedule(static)
        for (int64_t t = 0; t < n_threads; ++t) {
            auto &counts = thread_counts[static_cast<size_t>(t)];
            const int64_t begin = t * chunk;
            const int64_t end =
                    std::min<int64_t>(begin + chunk, (int64_t)n_out);
            for (int64_t out_row = begin; out_row < end; ++out_row) {
                const int32_t *row = map.indices.data() +
                                     static_cast<size_t>(out_row) * k_volume;
                for (size_t ki = 0; ki < k_volume; ++ki)
                    if (row[ki] >= 0) ++counts[ki];
            }
        }
        for (size_t ki = 0; ki < k_volume; ++ki)
            for (const auto &counts : thread_counts)
                valid_per_k[ki] += counts[ki];
    } else
#endif
    {
        for (int64_t out_row_i = 0; out_row_i < static_cast<int64_t>(n_out);
             ++out_row_i) {
            const int32_t *row = map.indices.data() +
                                 static_cast<size_t>(out_row_i) * k_volume;
            for (size_t ki = 0; ki < k_volume; ++ki)
                if (row[ki] >= 0) ++valid_per_k[ki];
        }
    }
    map.compact_offsets.assign(k_volume + 1, 0);
    int32_t running = 0;
    for (size_t ki = 0; ki < k_volume; ++ki) {
        map.compact_offsets[ki] = running;
        running += valid_per_k[ki];
    }
    map.compact_offsets[k_volume] = running;
    map.compact_indices.assign(static_cast<size_t>(running), 0);
    map.scatter_rows.assign(n_out * k_volume, -1);
#ifdef _OPENMP
#pragma omp parallel for num_threads(map_threads()) \
        schedule(dynamic) if (k_volume > 1)
#endif
    for (int64_t ki = 0; ki < static_cast<int64_t>(k_volume); ++ki) {
        int32_t pos = 0;
        for (size_t out_row = 0; out_row < n_out; ++out_row) {
            const int32_t in_row =
                    map.indices[out_row * k_volume + static_cast<size_t>(ki)];
            if (in_row < 0) continue;
            map.compact_indices[static_cast<size_t>(map.compact_offsets[ki] +
                                                    pos)] = in_row;
            map.scatter_rows[out_row * k_volume + static_cast<size_t>(ki)] =
                    map.compact_offsets[ki] + pos;
            ++pos;
        }
    }
}

bool build_sparse_indice_map(const sparse_tensor &input,
                             const sparse_conv_params &params,
                             sparse_indice_map &map,
                             std::string &error) {
    if (input.ndim != 2 && input.ndim != 3) {
        error = "sparse tensor ndim must be 2 or 3";
        return false;
    }
    if (static_cast<int>(input.spatial_shape.size()) != input.ndim ||
        input.channels <= 0 ||
        input.coordinates.size() % static_cast<size_t>(input.ndim + 1) != 0) {
        error = "bad sparse coordinate metadata";
        return false;
    }
    const auto input_shape = spatial3(input);
    const int ndim = input.ndim;
    auto check_param = [&](const std::vector<int32_t> &values,
                           const char *name) {
        if (static_cast<int>(values.size()) != ndim ||
            std::any_of(values.begin(), values.end(),
                        [](int32_t x) { return x <= 0; })) {
            error = std::string("bad sparse convolution ") + name;
            return false;
        }
        return true;
    };
    if (!check_param(params.kernel_size, "kernel_size") ||
        !check_param(params.stride, "stride") ||
        !check_param(params.dilation, "dilation") ||
        static_cast<int>(params.padding.size()) != ndim ||
        std::any_of(params.padding.begin(), params.padding.end(),
                    [](int32_t x) { return x < 0; })) {
        if (error.empty()) error = "bad sparse convolution padding";
        return false;
    }
    if (params.submanifold &&
        (params.transposed ||
         std::any_of(params.stride.begin(), params.stride.end(),
                     [](int32_t x) { return x != 1; }))) {
        error = "submanifold convolution requires stride 1 and non-transposed "
                "mode";
        return false;
    }

    const auto in_shape = spatial3(input);
    const auto kernel = param3(params.kernel_size, ndim, 1);
    const auto stride = param3(params.stride, ndim, 1);
    const auto padding = param3(params.padding, ndim, 0);
    const auto dilation = param3(params.dilation, ndim, 1);
    std::array<int32_t, 3> out_shape{};
    for (int axis = 0; axis < 3; ++axis) {
        if (params.submanifold)
            out_shape[axis] = in_shape[axis];
        else if (params.transposed) {
            out_shape[axis] = (in_shape[axis] - 1) * stride[axis] -
                              2 * padding[axis] +
                              dilation[axis] * (kernel[axis] - 1) + 1;
        } else {
            out_shape[axis] = (in_shape[axis] + 2 * padding[axis] -
                               dilation[axis] * (kernel[axis] - 1) - 1) /
                                      stride[axis] +
                              1;
        }
        if (out_shape[axis] <= 0) {
            error = "sparse convolution output shape is empty";
            return false;
        }
    }

    // Flat linearized-key table over the input rows. The insert also serves as
    // the duplicate check, so the former second unordered_map pass is gone.
    coord_table input_table;
    input_table.build(input.active_count());
    for (int64_t row = 0; row < input.active_count(); ++row) {
        const coord4 coordinate = read_coord(input, row);
        if (!inside(coordinate, input_shape)) {
            error = "sparse coordinate outside spatial shape";
            return false;
        }
        if (!input_table.insert(linearized_key(coordinate, input_shape),
                                static_cast<int>(row))) {
            error = "duplicate sparse coordinate";
            return false;
        }
    }

    const auto offsets = kernel_offsets(kernel);
    std::vector<coord4> output_coords;
    if (params.submanifold) {
        // SubM keeps the input active set in input row order.
        output_coords.reserve(input.active_count());
        for (int64_t row = 0; row < input.active_count(); ++row)
            output_coords.push_back(read_coord(input, row));
    } else {
        // Candidate output coordinates: dedupe through the flat open-addressing
        // table (insert is O(1) amortized, cache-friendly), then sort only the
        // UNIQUE keys. The previous 4-pass radix sort over all candidates
        // (input_active * kernel_volume keys, e.g. 470k for a 17.5k stride-2
        // layer) is replaced by one linear pass plus a std::sort over the
        // output_active keys only (e.g. 29k), which is ~10x fewer elements.
        // The unique-key sort keeps the output in coordinate order, so the
        // engine's row layout (and compare_taps row alignment) is unchanged.
        std::vector<int64_t> candidates;
        candidates.reserve(static_cast<size_t>(input.active_count()) *
                           offsets.size());
        for (int64_t row = 0; row < input.active_count(); ++row) {
            const coord4 in = read_coord(input, row);
            for (const auto &k : offsets) {
                coord4 out{in.b, 0, 0, 0};
                const int32_t iv[3] = {in.z, in.y, in.x};
                int32_t *ov[3] = {&out.z, &out.y, &out.x};
                bool valid = true;
                for (int axis = 0; axis < 3; ++axis) {
                    if (params.transposed) {
                        *ov[axis] = iv[axis] * stride[axis] - padding[axis] +
                                    k[axis] * dilation[axis];
                    } else {
                        const int32_t numerator = iv[axis] + padding[axis] -
                                                  k[axis] * dilation[axis];
                        if (numerator % stride[axis] != 0) {
                            valid = false;
                            break;
                        }
                        *ov[axis] = numerator / stride[axis];
                    }
                }
                if (valid && inside(out, out_shape))
                    candidates.push_back(linearized_key(out, out_shape));
            }
        }
        coord_table candidate_table;
        candidate_table.build(static_cast<int64_t>(candidates.size()));
        std::vector<int64_t> unique_keys;
        unique_keys.reserve(candidates.size());
        for (int64_t key : candidates)
            if (candidate_table.insert(key, 0)) unique_keys.push_back(key);
        std::vector<int64_t>().swap(candidates);
        std::sort(unique_keys.begin(), unique_keys.end());
        output_coords.reserve(unique_keys.size());
        for (int64_t key : unique_keys) {
            int64_t n = key - 1;
            coord4 out;
            out.x = static_cast<int32_t>(n % out_shape[2]);
            n /= out_shape[2];
            out.y = static_cast<int32_t>(n % out_shape[1]);
            n /= out_shape[1];
            out.z = static_cast<int32_t>(n % out_shape[0]);
            out.b = static_cast<int32_t>(n / out_shape[0]);
            output_coords.push_back(out);
        }
    }

    map = {};
    map.ndim = ndim;
    map.input_spatial_shape = input.spatial_shape;
    map.input_coordinates = input.coordinates;
    map.kernel_volume = static_cast<int32_t>(offsets.size());
    map.spatial_shape =
            ndim == 3 ? std::vector<int32_t>{out_shape[0], out_shape[1],
                                             out_shape[2]}
                      : std::vector<int32_t>{out_shape[1], out_shape[2]};
    map.coordinates.reserve(output_coords.size() * (ndim + 1));
    map.indices.assign(output_coords.size() * offsets.size(), -1);
    sparse_tensor coord_sink;
    coord_sink.ndim = ndim;
    for (const coord4 &c : output_coords) append_coord(coord_sink, c);
    map.coordinates = std::move(coord_sink.coordinates);

    // Reverse neighbor resolution through the flat table: kernel_volume x
    // output_active flat probes, each one compare in L2 instead of a hashed
    // 4-tuple probe through a node-based map. Rows write disjoint index
    // ranges, so the probe loop parallelizes.
    // Mixed-radix axis weights for the SubM key-delta shortcut below.
    const int64_t w_yx = static_cast<int64_t>(in_shape[1]) * in_shape[2];
    const int64_t w_x = in_shape[2];
#pragma omp parallel for num_threads(map_threads())
    for (int64_t out_row_i = 0;
         out_row_i < static_cast<int64_t>(output_coords.size()); ++out_row_i) {
        const size_t out_row = static_cast<size_t>(out_row_i);
        const coord4 out = output_coords[out_row];
        // SubM (stride 1): key(in) = key(out) + sum(axis delta * axis weight)
        // because the inside() bounds check guarantees no digit carry. This
        // removes one mixed-radix linearization per (row, kernel) pair.
        const int64_t base_key =
                params.submanifold ? linearized_key(out, in_shape) : 0;
        for (size_t ki = 0; ki < offsets.size(); ++ki) {
            const auto &k = offsets[ki];
            coord4 in{out.b, 0, 0, 0};
            const int32_t ov[3] = {out.z, out.y, out.x};
            int32_t *iv[3] = {&in.z, &in.y, &in.x};
            bool valid = true;
            for (int axis = 0; axis < 3; ++axis) {
                if (params.transposed) {
                    const int32_t numerator =
                            ov[axis] + padding[axis] - k[axis] * dilation[axis];
                    if (numerator % stride[axis] != 0) {
                        valid = false;
                        break;
                    }
                    *iv[axis] = numerator / stride[axis];
                } else {
                    *iv[axis] = ov[axis] * stride[axis] - padding[axis] +
                                k[axis] * dilation[axis];
                }
            }
            if (!valid || !inside(in, in_shape)) continue;
            const int64_t key =
                    params.submanifold
                            ? base_key +
                                      static_cast<int64_t>(k[0] - padding[0]) *
                                              dilation[0] * w_yx +
                                      static_cast<int64_t>(k[1] - padding[1]) *
                                              dilation[1] * w_x +
                                      static_cast<int64_t>(k[2] - padding[2]) *
                                              dilation[2]
                            : linearized_key(in, in_shape);
            const int32_t found = input_table.find(key);
            if (found >= 0) map.indices[out_row * offsets.size() + ki] = found;
        }
    }
    build_compact_rulebook(map);
    return true;
}

bool build_sparse_inverse_indice_map(const sparse_indice_map &forward,
                                     sparse_indice_map &inverse,
                                     std::string &error) {
    inverse = {};
    const int coordinate_width = forward.ndim + 1;
    if ((forward.ndim != 2 && forward.ndim != 3) ||
        forward.kernel_volume <= 0 || forward.reuse_of >= 0 ||
        forward.spatial_shape.size() != static_cast<size_t>(forward.ndim) ||
        forward.input_spatial_shape.size() !=
                static_cast<size_t>(forward.ndim) ||
        forward.coordinates.size() % coordinate_width != 0 ||
        forward.input_coordinates.size() % coordinate_width != 0 ||
        forward.indices.size() !=
                static_cast<size_t>(forward.output_active_count()) *
                        forward.kernel_volume) {
        error = "bad forward sparse map for inverse convolution";
        return false;
    }
    const int64_t restored_count = static_cast<int64_t>(
            forward.input_coordinates.size() / coordinate_width);
    inverse.ndim = forward.ndim;
    inverse.spatial_shape = forward.input_spatial_shape;
    inverse.coordinates = forward.input_coordinates;
    inverse.input_spatial_shape = forward.spatial_shape;
    inverse.input_coordinates = forward.coordinates;
    inverse.kernel_volume = forward.kernel_volume;
    inverse.indices.assign(
            static_cast<size_t>(restored_count) * inverse.kernel_volume, -1);
    for (int64_t coarse = 0; coarse < forward.output_active_count(); ++coarse) {
        for (int kernel = 0; kernel < forward.kernel_volume; ++kernel) {
            const int32_t fine =
                    forward.indices[coarse * forward.kernel_volume + kernel];
            if (fine < 0) continue;
            if (fine >= restored_count) {
                error = "forward sparse map contains an invalid input row";
                inverse = {};
                return false;
            }
            int32_t &source = inverse.indices[static_cast<size_t>(fine) *
                                                      inverse.kernel_volume +
                                              kernel];
            if (source >= 0 && source != coarse) {
                error = "forward sparse map is ambiguous for inverse "
                        "convolution";
                inverse = {};
                return false;
            }
            source = static_cast<int32_t>(coarse);
        }
    }
    build_compact_rulebook(inverse);
    return true;
}

void remap_dense_indices(const sparse_indice_map &map,
                         int32_t zero_row,
                         std::vector<int32_t> &out) {
    out.resize(map.indices.size());
    for (size_t i = 0; i < map.indices.size(); ++i) {
        const int32_t v = map.indices[i];
        out[i] = v < 0 ? zero_row : v;
    }
}

ggml_tensor *build_sparse_convolution(ggml_context *ctx,
                                      ggml_tensor *features,
                                      ggml_tensor *dense_indices,
                                      ggml_tensor *compact_indices,
                                      ggml_tensor *scatter_rows,
                                      const sparse_indice_map &map,
                                      ggml_tensor *weights,
                                      ggml_tensor *bias,
                                      bool use_compact) {
    GGML_ASSERT(ctx && features && weights);
    GGML_ASSERT(features->ne[2] == 1 && features->ne[3] == 1);
    GGML_ASSERT(weights->ne[0] == features->ne[0] * map.kernel_volume);
    GGML_ASSERT(map.compact_offsets.size() ==
                static_cast<size_t>(map.kernel_volume) + 1);

    const int64_t in_channels = features->ne[0];
    const int64_t out_channels = weights->ne[1];

    // Quantized weights cannot be sliced per kernel offset: Q8_0 rows are
    // padded to 32-element blocks, so the linear k*in_channels stride no
    // longer matches the block-aligned row layout (and ggml_view_2d aborts
    // on the out-of-range offset). Fall back to the dense GEMM, which
    // consumes the whole weight tensor through mul_mat's native quantized
    // path.
    if (ggml_is_quantized(weights->type)) {
        use_compact = false;
    }
    if (use_compact) {
        GGML_ASSERT(compact_indices && scatter_rows);
        GGML_ASSERT(compact_indices->type == GGML_TYPE_I32);
        GGML_ASSERT(scatter_rows->type == GGML_TYPE_I32);
        // Gather only the valid neighbors: [in_channels, sum(valid_k)].
        ggml_tensor *gathered =
                ggml_sparse_gather(ctx, features, compact_indices);
        ggml_tensor *gathered_2d =
                ggml_reshape_2d(ctx, gathered, in_channels,
                                ggml_nelements(gathered) / in_channels);
        // One GEMM per non-empty kernel offset; each becomes one scatter
        // source (no concat at all). The multi-block SPARSE_SCATTER_ADD
        // accumulates up to 27 blocks; the [base, end) ranges ride in
        // op_params: [0] = n_a, [1..n_a] = compact base, [1+n_a..1+2n_a] =
        // exclusive end.
        const int64_t kv = map.kernel_volume;
        const size_t weight_stride =
                static_cast<size_t>(in_channels) * ggml_element_size(weights);
        std::vector<ggml_tensor *> blocks;
        std::vector<int32_t> bases;
        int64_t n_valid = 0;
        for (int64_t k = 0; k < kv; ++k)
            if (map.compact_offsets[k + 1] > map.compact_offsets[k]) ++n_valid;
        GGML_ASSERT(n_valid > 0);
        GGML_ASSERT(n_valid <= 27);
        blocks.reserve(static_cast<size_t>(n_valid));
        bases.reserve(static_cast<size_t>(n_valid));
        for (int64_t k = 0; k < kv; ++k) {
            const int64_t begin = map.compact_offsets[k];
            const int64_t count = map.compact_offsets[k + 1] - begin;
            if (count <= 0) continue;
            ggml_tensor *gk = ggml_view_2d(ctx, gathered_2d, in_channels, count,
                                           gathered_2d->nb[1],
                                           begin * gathered_2d->nb[1]);
            ggml_tensor *wk =
                    ggml_view_2d(ctx, weights, in_channels, out_channels,
                                 weights->nb[1], k * weight_stride);
            ggml_tensor *block =
                    ggml_mul_mat(ctx, wk, gk);  // [out_channels, valid_k]
            // Strict fp32 accumulation for the detector accuracy gate (same
            // policy as the dense lowering; the kernel-offset summation order
            // still matches the CPU compact path exactly).
            ggml_mul_mat_set_prec(block, GGML_PREC_F32);
            blocks.push_back(block);
            bases.push_back(static_cast<int32_t>(begin));
        }
        ggml_tensor *result = ggml_sparse_scatter_add_n(
                ctx, blocks.data(), (int)blocks.size(), scatter_rows, bias);
        for (size_t i = 0; i < blocks.size(); ++i) {
            result->op_params[1 + i] = bases[i];
            result->op_params[1 + blocks.size() + i] =
                    bases[i] + static_cast<int32_t>(ggml_nelements(blocks[i]) /
                                                    out_channels);
        }
        return result;
    }

    GGML_ASSERT(dense_indices && dense_indices->type == GGML_TYPE_I32);
    // The dense index tensor may be capacity-sized (cached CUDA graphs keep
    // the largest observed output count); only divisibility is required.
    GGML_ASSERT(dense_indices->ne[0] % map.kernel_volume == 0 &&
                dense_indices->ne[0] >=
                        map.kernel_volume * map.output_active_count());

    const int64_t n_output = dense_indices->ne[0] / map.kernel_volume;

    // Stock-op lowering of the upstream dense GEMM path. get_rows emits one
    // contiguous [in_channels] chunk per rulebook entry, so the reshape
    // produces the [in_channels * kernel_volume, n_output] operand whose row
    // index is k*in_channels + c — byte-compatible with the fused sparse
    // weights. The kernel-offset (K) reduction is part of the GEMM, so no
    // scatter op is required.
    ggml_tensor *gathered = ggml_get_rows(ctx, features, dense_indices);
    ggml_tensor *dense_input = ggml_reshape_2d(
            ctx, gathered, in_channels * map.kernel_volume, n_output);
    ggml_tensor *result = ggml_mul_mat(ctx, weights, dense_input);
    // cuBLAS defaults to TF32 for plain f32 GEMM on CUDA, which the detector
    // accuracy gate cannot absorb on the sparse backbone (same policy as the
    // lingbot/lightglue f32 matmuls).
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    if (bias) {
        GGML_ASSERT(bias->ne[0] == weights->ne[1] &&
                    ggml_nelements(bias) == weights->ne[1]);
        result = ggml_add(ctx, result, bias);
    }
    return result;
}

bool sparse_tensor::validate(std::string &error) const {
    if (ndim != 2 && ndim != 3) {
        error = "sparse tensor ndim must be 2 or 3";
        return false;
    }
    if (static_cast<int>(spatial_shape.size()) != ndim) {
        error = "bad sparse spatial shape";
        return false;
    }
    if (channels <= 0) {
        error = "sparse tensor channels must be positive";
        return false;
    }
    if (coordinates.size() % static_cast<size_t>(ndim + 1) != 0) {
        error = "sparse coordinate array has incomplete row";
        return false;
    }
    if (features.size() != static_cast<size_t>(active_count()) * channels) {
        error = "sparse feature shape mismatch";
        return false;
    }
    const auto shape = spatial3(*this);
    coord_table seen;
    seen.build(active_count());
    for (int64_t row = 0; row < active_count(); row++) {
        const coord4 c = read_coord(*this, row);
        if (!inside(c, shape)) {
            error = "sparse coordinate outside spatial shape";
            return false;
        }
        if (!seen.insert(linearized_key(c, shape), static_cast<int>(row))) {
            error = "duplicate sparse coordinate";
            return false;
        }
    }
    return true;
}

}  // namespace openpcdet::ext

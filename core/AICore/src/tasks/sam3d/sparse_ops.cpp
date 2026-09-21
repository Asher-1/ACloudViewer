// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Host-side tables for the SLat sparse stages - see sparse_ops.hpp.
// These mirror torch's SparseDownsample (//2 coords, unique-sorted codes,
// scatter mean), SparseUpsample (gather by the downsample inverse index) and
// spconv's SubMConv3d (submanifold: output set == input set, missing
// neighbours contribute zero).
#include "sparse_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <tuple>

#include "mesh_ape_reference.hpp"

namespace sam3d {

namespace {

using Key = std::tuple<int32_t, int32_t, int32_t>;

}  // namespace

bool SlatTables::build(const int32_t* coords,
                       int64_t n_fine,
                       int channels,
                       bool dump_tables) {
    nf = n_fine;
    if (nf <= 0) return false;

    // ---- downsample: coarse key = (b, x/2, y/2, z/2) --------------------
    // torch orders coarse rows by code.unique() with a mixed-radix code over
    // the coarse grid maxima -> lexicographic (b, x, y, z) order. Row ids are
    // the SORTED position (torch's return_inverse), not insertion order.
    std::map<std::tuple<int32_t, Key>, int32_t>
            coarse_index;  // sorted = code order
    fine_to_coarse.assign((size_t)nf, 0);
    {
        // pass 1: collect the unique coarse keys (any temp id)
        std::map<std::tuple<int32_t, Key>, int32_t> tmp;
        for (int64_t i = 0; i < nf; i++) {
            Key k{coords[i * 4 + 1] / 2, coords[i * 4 + 2] / 2,
                  coords[i * 4 + 3] / 2};
            tmp.emplace(std::make_tuple(coords[i * 4 + 0], k), 0);
        }
        // pass 2: assign sorted row ids
        int32_t id = 0;
        for (auto& [k, v] : tmp) v = id++;
        coarse_index = std::move(tmp);
        nc = (int64_t)id;
        std::vector<int32_t> counts((size_t)nc, 0);
        for (int64_t i = 0; i < nf; i++) {
            Key k{coords[i * 4 + 1] / 2, coords[i * 4 + 2] / 2,
                  coords[i * 4 + 3] / 2};
            const int32_t c =
                    coarse_index.at(std::make_tuple(coords[i * 4 + 0], k));
            fine_to_coarse[(size_t)i] = c;
            counts[(size_t)c]++;
        }

        coarse_count.resize((size_t)nc);
        // children table (c*8+j): gather rows for the mean-pool; a coarse
        // cell's 2x2x2 block is often partial, missing slots -> sentinel (nf =
        // zero row)
        std::vector<int32_t> cnts((size_t)nc, 0);
        coarse_children.assign((size_t)(8 * nc), (int32_t)nf);
        for (int64_t i = 0; i < nf; i++) {
            const int32_t c = fine_to_coarse[(size_t)i];
            const int slot = cnts[(size_t)c]++;
            if (slot < 8) coarse_children[(size_t)(c * 8 + slot)] = (int32_t)i;
        }
        for (int64_t c = 0; c < nc; c++) {
            const int32_t k = cnts[(size_t)c];
            // NOTE: torch scatter_reduce(reduce="mean") averages over the
            // written values PLUS the initial zero row, i.e. sum/(count+1) -
            // this quirk is part of the reference behaviour and must be
            // reproduced. The divisor itself is stored (not a reciprocal)
            // because the reference divides in F32 opmath instead of
            // multiplying a rounded inverse.
            coarse_count[(size_t)c] = (float)(k + 1);
            // pack children compactly (keep the used slots first)
            int w = 0;
            for (int j = 0; j < 8; j++) {
                const int32_t v = coarse_children[(size_t)(c * 8 + j)];
                if (v != (int32_t)nf)
                    coarse_children[(size_t)(c * 8 + w++)] = v;
            }
            for (int j = w; j < 8; j++)
                coarse_children[(size_t)(c * 8 + j)] = (int32_t)nf;
        }
    }

    if (dump_tables) {
        fprintf(stderr, "[tbl] first 12 coarse keys:\n");
        int printed = 0;
        for (const auto& [bk, row] : coarse_index) {
            if (printed++ >= 40) break;
            fprintf(stderr, "[tbl] row %d: b=%d xyz=(%d,%d,%d)\n", row,
                    std::get<0>(bk), std::get<0>(std::get<1>(bk)),
                    std::get<1>(std::get<1>(bk)), std::get<2>(std::get<1>(bk)));
        }
    }
    // ---- submanifold conv neighbour tables ------------------------------
    // hash the point sets for O(1) membership
    std::map<std::tuple<int32_t, Key>, int32_t> fine_index;
    for (int64_t i = 0; i < nf; i++)
        fine_index.emplace(
                std::make_tuple(coords[i * 4 + 0],
                                Key{coords[i * 4 + 1], coords[i * 4 + 2],
                                    coords[i * 4 + 3]}),
                (int32_t)i);

    auto build_conv =
            [&](const std::map<std::tuple<int32_t, Key>, int32_t>& idx,
                int64_t n, std::vector<int32_t>& table) {
                table.assign((size_t)(27 * n),
                             (int32_t)n);  // sentinel = n (zero row)
                for (const auto& [bk, row] : idx) {
                    const auto& [b, k] = bk;
                    for (int o = 0; o < 27; o++) {
                        const int32_t dx = o / 9 - 1, dy = (o / 3) % 3 - 1,
                                      dz = o % 3 - 1;
                        auto it = idx.find({b, Key{std::get<0>(k) + dx,
                                                   std::get<1>(k) + dy,
                                                   std::get<2>(k) + dz}});
                        if (it != idx.end())
                            table[(size_t)(row * 27 + o)] = it->second;
                    }
                }
            };
    build_conv(fine_index, nf, conv_fine);
    build_conv(coarse_index, nc, conv_coarse);

    auto build_spconv = [](const std::vector<int32_t>& token_major, int64_t n,
                           std::vector<int32_t>& offset_major,
                           std::vector<int32_t>& masks,
                           std::vector<int32_t>& argsort) {
        offset_major.assign((size_t)(27 * n), -1);
        masks.assign((size_t)n, 0);
        argsort.resize((size_t)n);
        for (int64_t token = 0; token < n; token++) {
            argsort[(size_t)token] = (int32_t)token;
            for (int offset = 0; offset < 27; offset++) {
                const int32_t row = token_major[(size_t)(token * 27 + offset)];
                if (row >= 0 && row < n) {
                    offset_major[(size_t)(offset * n + token)] = row;
                    masks[(size_t)token] |= (int32_t)(1u << offset);
                }
            }
        }

        // spconv's implicit-GEMM path stable-sorts the per-output mask and
        // passes the matching source-row permutation to the Cumm kernel.
        // The kernel uses both buffers together; an unsorted mask paired with
        // identity indices selects unrelated sparse rows and causes error to
        // accumulate across diffusion blocks.
        std::stable_sort(argsort.begin(), argsort.end(),
                         [&](int32_t lhs, int32_t rhs) {
                             return masks[(size_t)lhs] < masks[(size_t)rhs];
                         });
        std::vector<int32_t> sorted_masks((size_t)n);
        for (int64_t sorted = 0; sorted < n; ++sorted) {
            sorted_masks[(size_t)sorted] =
                    masks[(size_t)argsort[(size_t)sorted]];
        }
        masks = std::move(sorted_masks);
    };
    build_spconv(conv_fine, nf, conv_fine_spconv, conv_fine_mask,
                 conv_fine_argsort);
    build_spconv(conv_coarse, nc, conv_coarse_spconv, conv_coarse_mask,
                 conv_coarse_argsort);

    // ---- AbsolutePositionEmbedder for the coarse grid -------------------
    // per-axis: outer(coord, freqs) -> [sin | cos] (2*freq_dim per axis),
    // axes concatenated, zero-padded to `channels`.
    const int freq_dim = channels / 3 / 2;
    ape.assign((size_t)channels * nc, 0.0f);
    // coarse coords in row order
    std::vector<int32_t> cc((size_t)nc * 3);
    for (const auto& [bk, row] : coarse_index) {
        const Key& k = std::get<1>(bk);
        cc[(size_t)row * 3 + 0] = std::get<0>(k);
        cc[(size_t)row * 3 + 1] = std::get<1>(k);
        cc[(size_t)row * 3 + 2] = std::get<2>(k);
    }
    std::vector<float> freqs((size_t)freq_dim);
    for (int i = 0; i < freq_dim; i++)
        freqs[(size_t)i] = 1.0f / powf(10000.0f, (float)i / (float)freq_dim);
    for (int64_t n = 0; n < nc; n++) {
        for (int ax = 0; ax < 3; ax++) {
            const float v = (float)cc[(size_t)n * 3 + ax];
            float* sin_seg = &ape[(size_t)n * channels + ax * 2 * freq_dim];
            float* cos_seg = sin_seg + freq_dim;
            for (int i = 0; i < freq_dim; i++) {
                const float ph = v * freqs[(size_t)i];
                sin_seg[i] = sinf(ph);
                cos_seg[i] = cosf(ph);
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// GS decoder swin-window tables (see sparse_ops.hpp). Mirrors torch
// calc_window_partition: window id over shifted coords with mixed-radix
// OFFSET, tokens grouped by window id (order inside a window is irrelevant -
// attention is permutation invariant), then compacted into same-length
// buckets for batched attention.
// ---------------------------------------------------------------------------
bool GsTables::build(const int32_t* coords, int64_t n_fine) {
    if (n_fine <= 0) return false;
    const int W = 8;  // window_size from slat_decoder_gs.yaml

    for (int si = 0; si < 2; si++) {
        const int32_t sh = 4 * si;  // swin shift: (4,4,4) for odd blocks
        GsWindowTables& t = shifts[si];
        t.nf = n_fine;

        // window key per token: (b, x/8, y/8, z/8) of the shifted coords.
        // torch computes the NUM_WINDOWS grid from the shifted maxima; the
        // mixed-radix id ordering equals lexicographic (b, wx, wy, wz).
        std::map<std::tuple<int32_t, int32_t, int32_t, int32_t>, int32_t>
                win_id;
        const int64_t nf = n_fine;
        std::vector<int32_t> wid((size_t)nf);
        for (int64_t i = 0; i < nf; i++) {
            std::tuple<int32_t, int32_t, int32_t, int32_t> k{
                    coords[i * 4 + 0], (coords[i * 4 + 1] + sh) / W,
                    (coords[i * 4 + 2] + sh) / W, (coords[i * 4 + 3] + sh) / W};
            auto it = win_id.find(k);
            if (it == win_id.end())
                it = win_id.emplace(k, (int32_t)win_id.size()).first;
            wid[(size_t)i] = it->second;
        }

        // rows sorted by window id (stable: keep table order inside a window)
        std::vector<int32_t> order((size_t)nf);
        for (int64_t i = 0; i < nf; i++) order[(size_t)i] = (int32_t)i;
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return wid[(size_t)a] < wid[(size_t)b];
        });

        // window segments in sorted order
        std::vector<std::pair<int32_t, int32_t>> segs;  // (row_start, len)
        for (int64_t i = 0; i < nf;) {
            int64_t j = i;
            while (j < nf && wid[(size_t)order[(size_t)j]] ==
                                     wid[(size_t)order[(size_t)i]])
                j++;
            segs.emplace_back((int32_t)i, (int32_t)(j - i));
            i = j;
        }

        // bucket by segment length (map iterates len ascending; torch makes
        // no layout promise across windows, only the partition matters)
        std::map<int32_t, std::vector<int32_t>> by_len;  // len -> seg indices
        for (int32_t s = 0; s < (int32_t)segs.size(); s++)
            by_len[segs[(size_t)s].second].push_back(s);

        t.bkt_idx.assign((size_t)nf, 0);
        t.bwd_idx.assign((size_t)nf, 0);
        t.buckets.clear();
        int32_t row = 0;
        for (auto& [len, seg_list] : by_len) {
            const int32_t off = row;
            for (int32_t s : seg_list) {
                const auto& [start, slen] = segs[(size_t)s];
                for (int32_t j = 0; j < slen; j++) {
                    const int32_t orig = order[(size_t)(start + j)];
                    t.bkt_idx[(size_t)(row)] = orig;  // orig -> bucket pos
                    t.bwd_idx[(size_t)orig] =
                            (int32_t)row;  // bucket pos -> orig
                    row++;
                }
            }
            t.buckets.push_back({len, off, (int32_t)seg_list.size()});
        }
        if (row != (int32_t)nf) return false;
    }

    // ---- AbsolutePositionEmbedder for the fine grid (768 ch) ------------
    {
        const int channels = 768;
        const int freq_dim = channels / 3 / 2;
        const int64_t nf = n_fine;
        ape.assign((size_t)channels * nf, 0.0f);
        static_assert(mesh_ape_reference::kFrequencyCount == freq_dim,
                      "official mesh APE frequency count changed");
        bool has_official_coordinate_range = true;
        for (int64_t n = 0; n < nf && has_official_coordinate_range; ++n) {
            for (int ax = 0; ax < 3; ++ax) {
                const int32_t coordinate = coords[n * 4 + 1 + ax];
                has_official_coordinate_range =
                        coordinate >= 0 &&
                        coordinate < mesh_ape_reference::kCoordinateCount;
                if (!has_official_coordinate_range) break;
            }
        }
        std::vector<float> freqs;
        if (!has_official_coordinate_range) {
            freqs.resize((size_t)freq_dim);
            for (int i = 0; i < freq_dim; i++) {
                freqs[(size_t)i] =
                        1.0f / powf(10000.0f, (float)i / (float)freq_dim);
            }
        }
        for (int64_t n = 0; n < nf; n++) {
            for (int ax = 0; ax < 3; ax++) {
                float* sin_seg = &ape[(size_t)n * channels + ax * 2 * freq_dim];
                float* cos_seg = sin_seg + freq_dim;
                if (has_official_coordinate_range) {
                    const int32_t coordinate = coords[n * 4 + 1 + ax];
                    std::memcpy(sin_seg,
                                mesh_ape_reference::kSinCosBits[coordinate][0],
                                sizeof(float) * freq_dim);
                    std::memcpy(cos_seg,
                                mesh_ape_reference::kSinCosBits[coordinate][1],
                                sizeof(float) * freq_dim);
                } else {
                    const float value =
                            static_cast<float>(coords[n * 4 + 1 + ax]);
                    for (int i = 0; i < freq_dim; i++) {
                        const float phase = value * freqs[(size_t)i];
                        sin_seg[i] = sinf(phase);
                        cos_seg[i] = cosf(phase);
                    }
                }
            }
        }
    }
    return true;
}

namespace {

using CoordKey = std::tuple<int32_t, int32_t, int32_t, int32_t>;

bool build_mesh_conv_table(MeshConvLevel& level) {
    if (level.n <= 0 ||
        level.coords.size() != static_cast<size_t>(level.n) * 4) {
        return false;
    }

    std::map<CoordKey, int32_t> index;
    for (int64_t row = 0; row < level.n; ++row) {
        const int32_t* coord = &level.coords[static_cast<size_t>(row) * 4];
        if (coord[1] < 0 || coord[2] < 0 || coord[3] < 0 ||
            coord[1] >= level.resolution || coord[2] >= level.resolution ||
            coord[3] >= level.resolution) {
            return false;
        }
        if (!index.emplace(CoordKey{coord[0], coord[1], coord[2], coord[3]},
                           static_cast<int32_t>(row))
                     .second) {
            return false;
        }
    }

    level.conv.assign(static_cast<size_t>(level.n) * 27,
                      static_cast<int32_t>(level.n));
    for (int64_t row = 0; row < level.n; ++row) {
        const int32_t* coord = &level.coords[static_cast<size_t>(row) * 4];
        for (int offset = 0; offset < 27; ++offset) {
            const int32_t dx = offset / 9 - 1;
            const int32_t dy = (offset / 3) % 3 - 1;
            const int32_t dz = offset % 3 - 1;
            const auto it = index.find(CoordKey{coord[0], coord[1] + dx,
                                                coord[2] + dy, coord[3] + dz});
            if (it != index.end()) {
                level.conv[static_cast<size_t>(row) * 27 + offset] = it->second;
            }
        }
    }
    return true;
}

bool subdivide_mesh_level(const MeshConvLevel& input, MeshConvLevel& output) {
    if (input.n <= 0 || input.n > INT32_MAX / 8 ||
        input.coords.size() != static_cast<size_t>(input.n) * 4) {
        return false;
    }
    output.n = input.n * 8;
    output.resolution = input.resolution * 2;
    output.coords.resize(static_cast<size_t>(output.n) * 4);
    for (int64_t parent = 0; parent < input.n; ++parent) {
        const int32_t* source = &input.coords[static_cast<size_t>(parent) * 4];
        for (int child = 0; child < 8; ++child) {
            // torch.nonzero over a [2, 2, 2] tensor: x, y, z lexicographic.
            const int32_t dx = child / 4;
            const int32_t dy = (child / 2) % 2;
            const int32_t dz = child % 2;
            int32_t* target =
                    &output.coords[static_cast<size_t>(parent * 8 + child) * 4];
            target[0] = source[0];
            target[1] = source[1] * 2 + dx;
            target[2] = source[2] * 2 + dy;
            target[3] = source[3] * 2 + dz;
        }
    }
    return build_mesh_conv_table(output);
}

}  // namespace

bool MeshTables::build(const int32_t* coords,
                       int64_t n_fine,
                       int base_resolution) {
    if (coords == nullptr || n_fine <= 0 || base_resolution <= 0) {
        return false;
    }
    if (!swin.build(coords, n_fine)) {
        return false;
    }

    levels[0].n = n_fine;
    levels[0].resolution = base_resolution;
    levels[0].coords.assign(coords, coords + static_cast<size_t>(n_fine) * 4);
    if (!build_mesh_conv_table(levels[0])) {
        return false;
    }
    return subdivide_mesh_level(levels[0], levels[1]) &&
           subdivide_mesh_level(levels[1], levels[2]);
}

}  // namespace sam3d

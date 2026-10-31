// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "anchor_head.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>
#include <vector>

namespace openpcdet {

namespace {

// Parallel-for over [0,n): splits into contiguous chunks, one per HW thread.
template <class F>
void parallel_for(int64_t n, F&& body) {
    if (n <= 0) return;
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    nt = (unsigned)std::min<int64_t>(nt, n);
    if (nt <= 1) {
        body(int64_t{0}, n);
        return;
    }
    const int64_t chunk = (n + nt - 1) / nt;
    std::vector<std::thread> ths;
    ths.reserve(nt);
    for (int64_t a = 0; a < n; a += chunk)
        ths.emplace_back(
                [&body, a, b = std::min(n, a + chunk)] { body(a, b); });
    for (auto& t : ths) t.join();
}

// 1x1 conv with bias: y = W^T x + b, where W is [1, 1, IC, OC] and x is
// [W, H, IC, N]. Output is [W, H, OC, N]. For 1x1, this is just a matmul
// over the IC dim, equivalent to:
//   reshape x to [W*H, IC, N] (or [W*H*IC, N]), W to [IC, OC]
//   mm = mul_mat(W_reshape, x)  -> [OC, W*H, N]
//   reshape to [W, H, OC, N]
//   add bias broadcast over OC
ggml_tensor* conv1x1_with_bias(ggml_context* ctx,
                               ggml_tensor* conv_w,  // [1, 1, IC, OC]
                               ggml_tensor* bias,    // [OC]
                               ggml_tensor* x) {     // [W, H, IC, N]
    const bool packed = ggml_n_dims(conv_w) == 2;
    const int64_t OC = packed ? conv_w->ne[1] : conv_w->ne[3];
    const int64_t IC = packed ? conv_w->ne[0] : conv_w->ne[2];
    const int64_t W = x->ne[0], H = x->ne[1], N = x->ne[3];
    ggml_tensor* wflat = packed ? conv_w : ggml_reshape_2d(ctx, conv_w, IC, OC);
    ggml_tensor* x_chw = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
    ggml_tensor* xflat = ggml_reshape_2d(ctx, x_chw, IC, W * H * N);
    ggml_tensor* mm = ggml_mul_mat(ctx, wflat, xflat);  // [OC, W*H*N]
    ggml_tensor* out_cwhn = ggml_reshape_4d(ctx, mm, OC, W, H, N);
    ggml_tensor* out = ggml_cont(ctx, ggml_permute(ctx, out_cwhn, 2, 0, 1, 3));
    if (bias) {
        ggml_tensor* b = ggml_reshape_4d(ctx, bias, 1, 1, OC, 1);
        out = ggml_add(ctx, out, b);
    }
    return out;
}

}  // namespace

bool load_anchor_head_weights(model_file& file,
                              const hparams& hp,
                              anchor_head_weights& w,
                              std::string& err) {
    w.conv_cls_w = file.require("head.conv_cls.weight");
    w.conv_cls_b = file.require("head.conv_cls.bias");
    w.conv_box_w = file.require("head.conv_box.weight");
    w.conv_box_b = file.require("head.conv_box.bias");
    if (!file.error.empty()) {
        err = file.error;
        return false;
    }
    if (hp.use_direction_classifier) {
        w.conv_dir_cls_w = file.require("head.conv_dir_cls.weight");
        w.conv_dir_cls_b = file.require("head.conv_dir_cls.bias");
        if (!file.error.empty()) {
            err = file.error;
            return false;
        }
    }
    auto check = [&](ggml_tensor* tensor, ggml_tensor* bias, int out_channels,
                     const char* name) {
        const int in_channels =
                hp.bb_num_upsample_filters.empty()
                        ? hp.bb_num_filters.back()
                        : std::accumulate(hp.bb_num_upsample_filters.begin(),
                                          hp.bb_num_upsample_filters.end(), 0);
        const bool weight_ok = ggml_n_dims(tensor) == 2
                                       ? tensor->ne[0] == in_channels &&
                                                 tensor->ne[1] == out_channels
                                       : tensor->ne[0] == 1 &&
                                                 tensor->ne[1] == 1 &&
                                                 tensor->ne[2] == in_channels &&
                                                 tensor->ne[3] == out_channels;
        if (!weight_ok || !bias || ggml_nelements(bias) != out_channels) {
            err = std::string("head weight shape mismatch: ") + name;
            return false;
        }
        return true;
    };
    const int anchors = hp.num_anchors_per_location();
    if (!check(w.conv_cls_w, w.conv_cls_b, anchors * hp.num_class,
               "conv_cls") ||
        !check(w.conv_box_w, w.conv_box_b, anchors * hp.code_size,
               "conv_box") ||
        (hp.use_direction_classifier &&
         !check(w.conv_dir_cls_w, w.conv_dir_cls_b, anchors * hp.num_dir_bins,
                "conv_dir_cls"))) {
        return false;
    }
    return true;
}

head_outputs build_anchor_head_graph(ggml_context* ctx,
                                     const anchor_head_weights& w,
                                     const hparams& hp,
                                     ggml_tensor* spatial_2d) {
    head_outputs out;
    out.cls = conv1x1_with_bias(ctx, w.conv_cls_w, w.conv_cls_b, spatial_2d);
    out.box = conv1x1_with_bias(ctx, w.conv_box_w, w.conv_box_b, spatial_2d);
    if (hp.use_direction_classifier && w.conv_dir_cls_w) {
        out.dir = conv1x1_with_bias(ctx, w.conv_dir_cls_w, w.conv_dir_cls_b,
                                    spatial_2d);
    }
    // The PyTorch head does .permute(0, 2, 3, 1) on the [N, C, H, W] output
    // to get [N, H, W, C]. ggml's native layout is already [W, H, C, N] (i.e.
    // WHC with N outermost) — equivalent to NHWC after a cont() if needed.
    // We leave the tensors in ggml's native layout; the CPU-side decode loop
    // reads them with the proper strides.
    return out;
}

anchor_grid generate_anchors(const hparams& hp,
                             int feature_map_w,
                             int feature_map_h) {
    anchor_grid g;
    const int num_sets = (int)hp.anchor_sizes.size();
    if (num_sets == 0 || (int)hp.anchor_rotations.size() != num_sets ||
        (int)hp.anchor_bottom_heights.size() != num_sets ||
        (int)hp.feature_map_strides.size() != num_sets ||
        (int)hp.anchor_align_center.size() != num_sets) {
        return g;
    }
    g.feature_map_w = feature_map_w;
    g.feature_map_h = feature_map_h;

    const float x0 = hp.point_cloud_range[0], x1 = hp.point_cloud_range[3];
    const float y0 = hp.point_cloud_range[1], y1 = hp.point_cloud_range[4];

    int A_per_loc = 0;
    for (int s = 0; s < num_sets; s++) {
        if (hp.feature_map_strides[s] <= 0 ||
            hp.grid_size[0] / hp.feature_map_strides[s] != feature_map_w ||
            hp.grid_size[1] / hp.feature_map_strides[s] != feature_map_h) {
            return anchor_grid{};
        }
        A_per_loc += (int)(hp.anchor_sizes[s].size() *
                           hp.anchor_rotations[s].size() *
                           hp.anchor_bottom_heights[s].size());
    }
    const int A_total = feature_map_w * feature_map_h * A_per_loc;
    g.total = A_total;
    g.anchors.resize((size_t)A_total * 7);

    // torch.cat(self.anchors, dim=-3) produces per-location ordering:
    // [set][height][size][rotation], with x changing faster than y globally.
    int idx = 0;
    for (int yi = 0; yi < feature_map_h; yi++) {
        for (int xi = 0; xi < feature_map_w; xi++) {
            for (int s = 0; s < num_sets; s++) {
                const auto& sizes = hp.anchor_sizes[s];
                const auto& rots = hp.anchor_rotations[s];
                const auto& hts = hp.anchor_bottom_heights[s];
                const bool align_center = hp.anchor_align_center[s] != 0;
                if (!align_center &&
                    (feature_map_w <= 1 || feature_map_h <= 1)) {
                    return anchor_grid{};
                }
                const float xs = (x1 - x0) / (align_center ? feature_map_w
                                                           : feature_map_w - 1);
                const float ys = (y1 - y0) / (align_center ? feature_map_h
                                                           : feature_map_h - 1);
                const float xc = x0 + (align_center ? xs / 2 : 0.0f) + xi * xs;
                const float yc = y0 + (align_center ? ys / 2 : 0.0f) + yi * ys;
                for (size_t hi = 0; hi < hts.size(); hi++) {
                    for (size_t si = 0; si < sizes.size(); si++) {
                        const float dx = sizes[si][0], dy = sizes[si][1],
                                    dz = sizes[si][2];
                        for (size_t ri = 0; ri < rots.size(); ri++) {
                            float* a = &g.anchors[(size_t)idx * 7];
                            a[0] = xc;
                            a[1] = yc;
                            a[2] = hts[hi] + dz / 2;
                            a[3] = dx;
                            a[4] = dy;
                            a[5] = dz;
                            a[6] = rots[ri];
                            idx++;
                        }
                    }
                }
            }
        }
    }
    return g;
}

anchor_grid generate_multihead_anchors(const hparams& hp,
                                       const std::vector<int32_t>& set_indices,
                                       int feature_map_w,
                                       int feature_map_h) {
    anchor_grid g;
    g.feature_map_w = feature_map_w;
    g.feature_map_h = feature_map_h;
    const float x0 = hp.point_cloud_range[0], x1 = hp.point_cloud_range[3];
    const float y0 = hp.point_cloud_range[1], y1 = hp.point_cloud_range[4];
    int total = 0;
    for (int32_t set : set_indices) {
        if (set < 0 || set >= (int32_t)hp.anchor_sizes.size() ||
            hp.feature_map_strides[set] <= 0 ||
            hp.grid_size[0] / hp.feature_map_strides[set] != feature_map_w ||
            hp.grid_size[1] / hp.feature_map_strides[set] != feature_map_h)
            return {};
        total += feature_map_w * feature_map_h * hp.anchor_sizes[set].size() *
                 hp.anchor_rotations[set].size() *
                 hp.anchor_bottom_heights[set].size();
    }
    g.total = total;
    g.anchors.resize((size_t)total * 7);
    int index = 0;
    for (int32_t set : set_indices) {
        const bool align = hp.anchor_align_center[set] != 0;
        if (!align && (feature_map_w <= 1 || feature_map_h <= 1)) return {};
        const float xs =
                (x1 - x0) / (align ? feature_map_w : feature_map_w - 1);
        const float ys =
                (y1 - y0) / (align ? feature_map_h : feature_map_h - 1);
        for (const auto& size : hp.anchor_sizes[set]) {
            for (float rotation : hp.anchor_rotations[set]) {
                for (float bottom : hp.anchor_bottom_heights[set]) {
                    for (int y = 0; y < feature_map_h; y++) {
                        for (int x = 0; x < feature_map_w; x++) {
                            float* anchor =
                                    g.anchors.data() + (size_t)index++ * 7;
                            anchor[0] = x0 + (align ? xs / 2 : 0) + x * xs;
                            anchor[1] = y0 + (align ? ys / 2 : 0) + y * ys;
                            anchor[2] = bottom + size[2] / 2;
                            anchor[3] = size[0];
                            anchor[4] = size[1];
                            anchor[5] = size[2];
                            anchor[6] = rotation;
                        }
                    }
                }
            }
        }
    }
    return g;
}

std::vector<float> whc_to_hwc(const float* src, int W, int H, int C) {
    std::vector<float> out((size_t)W * H * C);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < C; c++) {
                out[((size_t)y * W + x) * C + c] =
                        src[(size_t)x + (size_t)W * (y + (size_t)H * c)];
            }
        }
    }
    return out;
}

std::vector<float> decode_boxes(const hparams& hp,
                                const float* box_enc,
                                int32_t N,
                                int32_t A,
                                const float* anchors) {
    // ResidualCoder.decode_torch (no sincos for PointPillar):
    //   xg = xt * diag + xa       diag = sqrt(dxa^2 + dya^2)
    //   yg = yt * diag + ya
    //   zg = zt * dza + za
    //   dxg = exp(dxt) * dxa
    //   dyg = exp(dyt) * dya
    //   dzg = exp(dzt) * dza
    //   rg  = rt + ra
    std::vector<float> out((size_t)N * A * 7);
    const int64_t total_a = (int64_t)N * A;
    parallel_for(total_a, [&](int64_t a0, int64_t a1) {
        for (int64_t a = a0; a < a1; a++) {
            const float* enc = box_enc + ((size_t)a) * hp.code_size;
            const float* an = anchors + (size_t)a * 7;
            float* o = out.data() + ((size_t)a) * 7;
            const float xa = an[0], ya = an[1], za = an[2];
            const float dxa = std::max(an[3], 1e-5f);
            const float dya = std::max(an[4], 1e-5f);
            const float dza = std::max(an[5], 1e-5f);
            const float ra = an[6];
            const float diag = std::sqrt(dxa * dxa + dya * dya);
            o[0] = enc[0] * diag + xa;
            o[1] = enc[1] * diag + ya;
            o[2] = enc[2] * dza + za;
            o[3] = std::exp(enc[3]) * dxa;
            o[4] = std::exp(enc[4]) * dya;
            o[5] = std::exp(enc[5]) * dza;
            if (hp.encode_angle_by_sincos) {
                // Not used for PointPillar; included for completeness.
                const float rg_cos = enc[6] + std::cos(ra);
                const float rg_sin = enc[7] + std::sin(ra);
                o[6] = std::atan2(rg_sin, rg_cos);
            } else {
                o[6] = enc[6] + ra;
            }
        }
    });
    return out;
}

void apply_direction_classifier(const hparams& hp,
                                const float* dir_cls,
                                int32_t N,
                                int32_t A,
                                std::vector<float>& boxes) {
    // For each (n, a): pick the bin with max logit, add period * bin_label to
    // the heading, after wrapping into [-period/2 + dir_offset, ...).
    // Matches AnchorHeadTemplate.generate_predicted_boxes:
    //   period = 2*pi / num_dir_bins
    //   dir_rot = limit_period(box[..., 6] - dir_offset, dir_limit_offset,
    //   period) box[..., 6] = dir_rot + dir_offset + period * dir_label
    const float period = 2.0f * (float)M_PI / hp.num_dir_bins;
    const float dir_off = hp.dir_offset;
    const float dir_lim_off = hp.dir_limit_offset;
    const int64_t total_a = (int64_t)N * A;
    parallel_for(total_a, [&](int64_t a0, int64_t a1) {
        for (int64_t a = a0; a < a1; a++) {
            const float* d = dir_cls + ((size_t)a) * hp.num_dir_bins;
            int bin = 0;
            float best = d[0];
            for (int b = 1; b < hp.num_dir_bins; b++) {
                if (d[b] > best) {
                    best = d[b];
                    bin = b;
                }
            }
            float* box = boxes.data() + ((size_t)a) * 7;
            // limit_period(x, offset, period) = x - floor(x / period + offset)
            // * period
            float x = box[6] - dir_off;
            x = x - std::floor(x / period + dir_lim_off) * period;
            box[6] = x + dir_off + period * (float)bin;
        }
    });
}

}  // namespace openpcdet

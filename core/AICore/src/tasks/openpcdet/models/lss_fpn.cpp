// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "lss_fpn.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace openpcdet {

namespace {

// 2× bilinear upsampling on the CPU.
// src: [W, H, C, B] in GGML column-major (W fastest).
// Returns [2*W, 2*H, C, B].
std::vector<float> bilinear_upsample_2x(
        const float* src, int W, int H, int C, int B) {
    const int oW = 2 * W, oH = 2 * H;
    std::vector<float> out((size_t)oW * oH * C * B);

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < oH; oh++) {
                for (int ow = 0; ow < oW; ow++) {
                    // align_corners=False 2× upsample:
                    //   in_coord = (out_coord + 0.5) * 0.5 - 0.5
                    float fy = (oh + 0.5f) * 0.5f - 0.5f;
                    float fx = (ow + 0.5f) * 0.5f - 0.5f;

                    int y0 = (int)std::floor(fy);
                    int x0 = (int)std::floor(fx);
                    int y1 = y0 + 1;
                    int x1 = x0 + 1;
                    float wy = fy - y0;
                    float wx = fx - x0;

                    y0 = std::max(0, std::min(y0, H - 1));
                    y1 = std::max(0, std::min(y1, H - 1));
                    x0 = std::max(0, std::min(x0, W - 1));
                    x1 = std::max(0, std::min(x1, W - 1));

                    // GGML layout: index = w + W*(h + H*(c + C*b))
                    size_t base = (size_t)W * H * C * b + (size_t)W * H * c;
                    float v00 = src[base + y0 * W + x0];
                    float v01 = src[base + y0 * W + x1];
                    float v10 = src[base + y1 * W + x0];
                    float v11 = src[base + y1 * W + x1];

                    float val = (1 - wy) * ((1 - wx) * v00 + wx * v01) +
                                wy * ((1 - wx) * v10 + wx * v11);

                    size_t oidx = (size_t)b * oW * oH * C +
                                  (size_t)c * oW * oH + (size_t)oh * oW + ow;
                    out[oidx] = val;
                }
            }
        }
    }
    return out;
}

}  // namespace

// ---- Weight loading ----

bool load_lss_fpn_weights(model_file& file,
                          const hparams& hp,
                          lss_fpn_weights& w,
                          std::string& error) {
    const auto& in_channels = hp.image_neck_in_channels;
    const int out_channels = hp.image_neck_out_channels;
    const int num_ins = (int)in_channels.size();
    const int start_level = 0;
    const int backbone_end_level = num_ins - 1;  // END_LEVEL == -1

    const int num_levels = backbone_end_level - start_level;
    w.levels.resize(num_levels);

    for (int i = 0; i < num_levels; i++) {
        // Lateral conv input channels:
        //   in_channels[i+start_level] + (in_channels[i+start_level+1] if i ==
        //   num_levels-1 else out_channels)
        int lat_in;
        if (i == num_levels - 1) {
            lat_in = in_channels[i + start_level] +
                     in_channels[i + start_level + 1];
        } else {
            lat_in = in_channels[i + start_level] + out_channels;
        }

        char buf[256];
        // lateral_convs.{i}: 1×1 conv, BN folded
        snprintf(buf, sizeof(buf), "neck.lateral_convs.%d.conv.weight", i);
        w.levels[i].lateral.conv_w = file.require(buf);
        snprintf(buf, sizeof(buf), "neck.lateral_convs.%d.bn.eval_scale", i);
        w.levels[i].lateral.bn_scale = file.require(buf);
        snprintf(buf, sizeof(buf), "neck.lateral_convs.%d.bn.eval_bias", i);
        w.levels[i].lateral.bn_bias = file.require(buf);

        // fpn_convs.{i}: 3×3 conv, BN folded
        snprintf(buf, sizeof(buf), "neck.fpn_convs.%d.conv.weight", i);
        w.levels[i].fpn_conv.conv_w = file.require(buf);
        snprintf(buf, sizeof(buf), "neck.fpn_convs.%d.bn.eval_scale", i);
        w.levels[i].fpn_conv.bn_scale = file.require(buf);
        snprintf(buf, sizeof(buf), "neck.fpn_convs.%d.bn.eval_bias", i);
        w.levels[i].fpn_conv.bn_bias = file.require(buf);

        if (!file.error.empty()) {
            error = file.error;
            return false;
        }
    }
    return true;
}

// ---- Forward ----

bool run_lss_fpn(engine_backend& backend,
                 const hparams& hp,
                 const lss_fpn_weights& w,
                 const std::vector<std::vector<float>>& swin_features,
                 const std::vector<int>& widths,
                 const std::vector<int>& heights,
                 const std::vector<int>& channels,
                 int batch_size,
                 lss_fpn_output& result,
                 const tap_sink& sink,
                 std::string& error) {
    const int out_channels = hp.image_neck_out_channels;
    const int num_swin = (int)swin_features.size();
    const int backbone_end_level = num_swin - 1;
    const int num_levels = backbone_end_level;  // outputs

    if (num_levels <= 0) {
        error = "FPN: not enough Swin inputs";
        return false;
    }

    // Build laterals: convert Swin features from BCHW [B,C,H,W] (as
    // produced by swin_transformer.cpp) to GGML [W,H,C,B] column-major,
    // which is what bilinear_upsample_2x, concat, and GGML conv expect.
    std::vector<std::vector<float>> laterals(num_swin);
    std::vector<int> lat_W(num_swin), lat_H(num_swin), lat_C(num_swin);
    for (int i = 0; i < num_swin; i++) {
        int W = widths[i], H = heights[i], C = channels[i];
        size_t HW = (size_t)H * W;
        size_t n = (size_t)C * HW * batch_size;
        laterals[i].resize(n);
        for (int b = 0; b < batch_size; b++)
            for (int c = 0; c < C; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++) {
                        // src: BCHW → b*C*HW + c*HW + h*W + w
                        size_t src = (size_t)b * C * HW + (size_t)c * HW +
                                     (size_t)h * W + w;
                        // dst: GGML [W,H,C,B] → w + W*(h + H*(c + C*b))
                        size_t dst = (size_t)w + (size_t)W * h +
                                     (size_t)W * H * c + (size_t)W * H * C * b;
                        laterals[i][dst] = swin_features[i][src];
                    }
        lat_W[i] = W;
        lat_H[i] = H;
        lat_C[i] = C;
    }

    // Top-down path: from deepest to shallowest
    for (int i = num_levels - 1; i >= 0; i--) {
        // Upsample laterals[i+1] to match laterals[i] spatial size
        int upW = lat_W[i], upH = lat_H[i];
        int srcW = lat_W[i + 1], srcH = lat_H[i + 1];
        int srcC = lat_C[i + 1];  // = out_channels (or in_channels for deepest)

        std::vector<float> up_data;
        if (srcW != upW || srcH != upH) {
            up_data = bilinear_upsample_2x(laterals[i + 1].data(), srcW, srcH,
                                           srcC, batch_size);
        } else {
            up_data = laterals[i + 1];
        }

        // Concat: [laterals[i], upsampled] along channel dim
        // GGML layout: [W, H, C, B] → concat channels → [W, H, C1+C2, B]
        int curC = lat_C[i];
        int concatC = curC + srcC;
        size_t spatial = (size_t)upW * upH;
        std::vector<float> concat_data(spatial * concatC * batch_size);

        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < upH; h++) {
                for (int w = 0; w < upW; w++) {
                    size_t out_base = ((size_t)b * concatC + 0) * spatial +
                                      (size_t)h * upW + w;
                    size_t src1_base = ((size_t)b * curC + 0) * spatial +
                                       (size_t)h * lat_W[i] + w;
                    size_t src2_base = ((size_t)b * srcC + 0) * spatial +
                                       (size_t)h * upW + w;
                    // Copy lateral channels
                    for (int c = 0; c < curC; c++) {
                        concat_data[out_base + (size_t)c * spatial] =
                                laterals[i][src1_base + (size_t)c * spatial];
                    }
                    // Copy upsampled channels
                    for (int c = 0; c < srcC; c++) {
                        concat_data[out_base + (size_t)(curC + c) * spatial] =
                                up_data[src2_base + (size_t)c * spatial];
                    }
                }
            }
        }

        // Build conv graph: lateral (1×1) → fpn_conv (3×3)
        const size_t graph_nodes = 256;
        ggml_init_params gp = {
                ggml_tensor_overhead() * graph_nodes +
                        ggml_graph_overhead_custom(graph_nodes, false),
                nullptr,
                /*no_alloc=*/true,
        };
        ggml_context* ctx = ggml_init(gp);
        if (!ctx) {
            error = "FPN graph ctx init failed";
            return false;
        }

        ggml_tensor* input_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, upW, upH,
                                                  concatC, batch_size);
        ggml_set_name(input_t, "fpn_input");
        ggml_set_input(input_t);

        // Lateral conv (1×1, BN, ReLU)
        ggml_tensor* lat =
                build_conv_bn_relu(ctx, w.levels[i].lateral, input_t, 1, 0);
        // FPN conv (3×3, pad=1, BN, ReLU)
        ggml_tensor* out_t =
                build_conv_bn_relu(ctx, w.levels[i].fpn_conv, lat, 1, 1);
        ggml_set_name(out_t, "fpn_out");
        ggml_set_output(out_t);

        ggml_cgraph* gf = ggml_new_graph_custom(ctx, graph_nodes, false);
        ggml_build_forward_expand(gf, out_t);

        if (!ggml_gallocr_alloc_graph(backend.galloc, gf)) {
            error = "FPN graph alloc failed";
            ggml_free(ctx);
            return false;
        }

        ggml_backend_tensor_set(input_t, concat_data.data(), 0,
                                concat_data.size() * sizeof(float));

        if (ggml_backend_graph_compute(backend.be, gf) != GGML_STATUS_SUCCESS) {
            error = "FPN graph compute failed";
            ggml_free(ctx);
            return false;
        }

        // Readback — keep in GGML [W,H,C,B] format for the next
        // top-down iteration (bilinear_upsample_2x, concat, GGML conv
        // all expect GGML layout).
        int oW = upW, oH = upH;
        size_t n = (size_t)oW * oH * out_channels * batch_size;
        laterals[i].resize(n);
        ggml_backend_tensor_get(out_t, laterals[i].data(), 0,
                                n * sizeof(float));
        lat_W[i] = oW;
        lat_H[i] = oH;
        lat_C[i] = out_channels;

        ggml_free(ctx);
    }

    // Build output: first num_levels entries of laterals.
    // Convert from GGML [W,H,C,B] to BCHW [B,C,H,W] for downstream
    // consumers (DepthLSS concat, conv_fuser, tap comparison).
    result.features.resize(num_levels);
    result.widths.resize(num_levels);
    result.heights.resize(num_levels);
    result.channels.resize(num_levels);
    for (int i = 0; i < num_levels; i++) {
        int W = lat_W[i], H = lat_H[i], C = lat_C[i];
        size_t HW = (size_t)H * W;
        size_t n = (size_t)W * H * C * batch_size;
        result.features[i].resize(n);
        for (int b = 0; b < batch_size; b++)
            for (int c = 0; c < C; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++) {
                        // src: GGML [W,H,C,B] → w + W*(h + H*(c + C*b))
                        size_t src = (size_t)w + (size_t)W * h +
                                     (size_t)W * H * c + (size_t)W * H * C * b;
                        // dst: BCHW → b*C*HW + c*HW + h*W + w
                        size_t dst = (size_t)b * C * HW + (size_t)c * HW +
                                     (size_t)h * W + w;
                        result.features[i][dst] = laterals[i][src];
                    }
        result.widths[i] = W;
        result.heights[i] = H;
        result.channels[i] = C;

        // Dump FPN output tap as [B*H*W, C] (row-major HWC per batch).
        // Data is now BCHW.
        if (sink) {
            int64_t rows = (int64_t)batch_size * H * W;
            std::vector<float> tap((size_t)rows * C);
            for (int b = 0; b < batch_size; b++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        for (int c = 0; c < C; c++) {
                            size_t src = (size_t)b * C * HW + (size_t)c * HW +
                                         (size_t)h * W + w;
                            size_t dst2 =
                                    ((size_t)b * HW + (size_t)h * W + w) * C +
                                    c;
                            tap[dst2] = result.features[i][src];
                        }
            char nm[64];
            snprintf(nm, sizeof(nm), "bevfusion.fpn.%d", i);
            sink(nm, tap.data(), rows, C);
        }
    }
    return true;
}

}  // namespace openpcdet

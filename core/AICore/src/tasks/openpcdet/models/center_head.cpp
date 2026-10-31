// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "center_head.h"

#include <ggml.h>

#include <numeric>

namespace openpcdet {

namespace {

bool check_conv(ggml_tensor* weight,
                ggml_tensor* bias,
                int input_channels,
                int output_channels,
                const std::string& name,
                std::string& error) {
    // Quantized (q8_0) kernels are stored as a packed matrix [IC*K*K, OC]
    // (the converter flattens 4-D PyTorch [OC,IC,KH,KW] weights before
    // quantization); f32/f16 kernels keep the 4-D [KW,KH,IC,OC] layout. ggml
    // trims trailing 1 dims, so OC==1 packed kernels arrive 1-D [IC*K*K]. All
    // layouts are handled by conv_2d_any; mirror bev_backbone's shape check.
    const bool ok =
            ggml_n_dims(weight) == 1
                    ? weight->ne[0] == (int64_t)input_channels * 3 * 3 &&
                              output_channels == 1
            : ggml_n_dims(weight) == 2
                    ? weight->ne[0] == (int64_t)input_channels * 3 * 3 &&
                              weight->ne[1] == output_channels
                    : weight->ne[0] == 3 && weight->ne[1] == 3 &&
                              weight->ne[2] == input_channels &&
                              weight->ne[3] == output_channels;
    if (!ok || !bias || ggml_nelements(bias) != output_channels) {
        error = "CenterHead tensor shape mismatch: " + name;
        return false;
    }
    return true;
}

bool load_sequence(model_file& file,
                   const std::string& prefix,
                   int channels,
                   int middle_count,
                   int output_channels,
                   center_sequence_weights& sequence,
                   std::string& error) {
    sequence.middle.resize(middle_count);
    for (int middle = 0; middle < middle_count; middle++) {
        const std::string base =
                prefix + ".middle." + std::to_string(middle) + ".";
        auto& block = sequence.middle[middle];
        block.conv_w = file.require((base + "conv.weight").c_str());
        block.bn_scale = file.require((base + "bn_fold.scale").c_str());
        block.bn_bias = file.require((base + "bn_fold.bias").c_str());
        if (!file.error.empty()) {
            error = file.error;
            return false;
        }
        if (!check_conv(block.conv_w, block.bn_bias, channels, channels, base,
                        error) ||
            ggml_nelements(block.bn_scale) != channels)
            return false;
    }
    sequence.output_weight = file.require((prefix + ".out.weight").c_str());
    sequence.output_bias = file.require((prefix + ".out.bias").c_str());
    if (!file.error.empty()) {
        error = file.error;
        return false;
    }
    return check_conv(sequence.output_weight, sequence.output_bias, channels,
                      output_channels, prefix + ".out", error);
}

ggml_tensor* build_conv_bias_only(ggml_context* ctx,
                                  const center_sequence_weights& sequence,
                                  ggml_tensor* input) {
    return build_conv_bias(ctx, sequence.output_weight, sequence.output_bias,
                           input, 1, 1);
}

}  // namespace

bool load_center_head_weights(model_file& file,
                              const hparams& hp,
                              center_head_weights& weights,
                              std::string& error) {
    if (hp.head_kind != dense_head_kind::center || hp.center_num_heads <= 0) {
        error = "invalid CenterHead metadata";
        return false;
    }
    const int backbone_channels =
            hp.bb_num_upsample_filters.empty()
                    ? hp.bb_num_filters.back()
                    : std::accumulate(hp.bb_num_upsample_filters.begin(),
                                      hp.bb_num_upsample_filters.end(), 0);
    weights.shared.conv_w = file.require("head.center.shared.conv.weight");
    weights.shared.bn_scale = file.require("head.center.shared.bn_fold.scale");
    weights.shared.bn_bias = file.require("head.center.shared.bn_fold.bias");
    if (!file.error.empty()) {
        error = file.error;
        return false;
    }
    if (!check_conv(weights.shared.conv_w, weights.shared.bn_bias,
                    backbone_channels, hp.center_shared_channels,
                    "head.center.shared", error) ||
        ggml_nelements(weights.shared.bn_scale) != hp.center_shared_channels)
        return false;

    weights.heads.resize(hp.center_num_heads);
    for (int head = 0; head < hp.center_num_heads; head++) {
        const std::string prefix = "head.center." + std::to_string(head) + ".";
        center_single_weights& current = weights.heads[head];
        if (!load_sequence(file, prefix + "hm", hp.center_shared_channels,
                           hp.center_middle_convs, hp.center_class_counts[head],
                           current.heatmap, error) ||
            !load_sequence(file, prefix + "center", hp.center_shared_channels,
                           hp.center_middle_convs, 2, current.center, error) ||
            !load_sequence(file, prefix + "center_z", hp.center_shared_channels,
                           hp.center_middle_convs, 1, current.center_z,
                           error) ||
            !load_sequence(file, prefix + "dim", hp.center_shared_channels,
                           hp.center_middle_convs, 3, current.dimensions,
                           error) ||
            !load_sequence(file, prefix + "rot", hp.center_shared_channels,
                           hp.center_middle_convs, 2, current.rotation, error))
            return false;
        if (hp.center_has_velocity &&
            !load_sequence(file, prefix + "vel", hp.center_shared_channels,
                           hp.center_middle_convs, 2, current.velocity, error))
            return false;
    }
    return true;
}

std::vector<center_single_outputs> build_center_head_graph(
        ggml_context* ctx,
        const center_head_weights& weights,
        const hparams& hp,
        ggml_tensor* spatial_2d) {
    // Fuse every branch's middle convolutions into a single wide conv: all
    // heads x branches share the [3x3, C->C] kernel shape, so concatenating
    // the (conv, bn) weights along the output-channel axis turns N separate
    // small kernel launches into one op (C -> C*N). OpenPCDet CenterHead runs
    // up to 36 such kernels (6 heads x 6 branches). The fused output is
    // sliced back per branch with zero-copy contiguous views: the branch
    // stride equals the standard [W,H,C] layout, so ggml_is_contiguous holds
    // and the out convs consume the slices directly.
    struct branch_ref {
        const center_sequence_weights* seq;
    };
    std::vector<branch_ref> branches;
    for (const auto& head : weights.heads) {
        branches.emplace_back(branch_ref{&head.heatmap});
        branches.emplace_back(branch_ref{&head.center});
        branches.emplace_back(branch_ref{&head.center_z});
        branches.emplace_back(branch_ref{&head.dimensions});
        branches.emplace_back(branch_ref{&head.rotation});
        if (hp.center_has_velocity)
            branches.emplace_back(branch_ref{&head.velocity});
    }

    ggml_tensor* shared =
            build_conv_bn_relu(ctx, weights.shared, spatial_2d, 1, 1);
    ggml_tensor* fused_input = shared;
    const int middle_count = hp.center_middle_convs;
    for (int middle = 0; middle < middle_count; middle++) {
        ggml_tensor* fused_w = nullptr;
        ggml_tensor* fused_scale = nullptr;
        ggml_tensor* fused_bias = nullptr;
        for (const branch_ref& branch : branches) {
            const conv_bn_block_weights& block = branch.seq->middle[middle];
            fused_w = fused_w ? ggml_concat(ctx, fused_w, block.conv_w, 3)
                              : block.conv_w;
            fused_scale = fused_scale ? ggml_concat(ctx, fused_scale,
                                                    block.bn_scale, 0)
                                      : block.bn_scale;
            fused_bias =
                    fused_bias ? ggml_concat(ctx, fused_bias, block.bn_bias, 0)
                               : block.bn_bias;
        }
        conv_bn_block_weights fused_block;
        fused_block.conv_w = fused_w;
        fused_block.bn_scale = fused_scale;
        fused_block.bn_bias = fused_bias;
        fused_input = build_conv_bn_relu(ctx, fused_block, fused_input, 1, 1);
    }
    // fused_input: [W, H, C*N, 1] -> 4-D [W, H, C, N] for zero-copy slicing.
    ggml_tensor* fused4d =
            middle_count > 0
                    ? ggml_reshape_4d(ctx, fused_input, fused_input->ne[0],
                                      fused_input->ne[1],
                                      hp.center_shared_channels,
                                      (int64_t)branches.size())
                    : shared;

    std::vector<center_single_outputs> outputs(weights.heads.size());
    size_t branch_index = 0;
    for (size_t head = 0; head < weights.heads.size(); head++) {
        const center_single_weights& source = weights.heads[head];
        center_single_outputs& output = outputs[head];
        auto finish = [&](const center_sequence_weights& seq,
                          ggml_tensor*& out) {
            ggml_tensor* input = fused4d;
            if (middle_count > 0) {
                input = ggml_view_3d(ctx, fused4d, fused4d->ne[0],
                                     fused4d->ne[1], hp.center_shared_channels,
                                     fused4d->nb[1], fused4d->nb[2],
                                     branch_index * fused4d->nb[3]);
                branch_index++;
            }
            out = build_conv_bias_only(ctx, seq, input);
        };
        finish(source.heatmap, output.heatmap);
        finish(source.center, output.center);
        finish(source.center_z, output.center_z);
        finish(source.dimensions, output.dimensions);
        finish(source.rotation, output.rotation);
        if (hp.center_has_velocity) finish(source.velocity, output.velocity);
    }
    return outputs;
}

}  // namespace openpcdet

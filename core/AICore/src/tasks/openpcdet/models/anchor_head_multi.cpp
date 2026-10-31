// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "anchor_head_multi.h"

#include <ggml.h>

#include <numeric>

namespace openpcdet {

namespace {

bool check_weight(ggml_tensor* weight,
                  ggml_tensor* bias,
                  int in_channels,
                  int out_channels,
                  int kernel,
                  const std::string& name,
                  std::string& error) {
    if (!weight || !bias) {
        error = "missing multi-head tensor: " + name;
        return false;
    }
    const bool shape_ok =
            ggml_n_dims(weight) == 2
                    ? weight->ne[0] == (int64_t)in_channels * kernel * kernel &&
                              weight->ne[1] == out_channels
                    : weight->ne[0] == kernel && weight->ne[1] == kernel &&
                              weight->ne[2] == in_channels &&
                              weight->ne[3] == out_channels;
    if (!shape_ok || ggml_nelements(bias) != out_channels) {
        error = "multi-head tensor shape mismatch: " + name;
        return false;
    }
    return true;
}

bool load_sequence(model_file& file,
                   const std::string& prefix,
                   int in_channels,
                   int middle_count,
                   int middle_channels,
                   int out_channels,
                   int final_kernel,
                   dense_head_sequence_weights& sequence,
                   std::string& error) {
    sequence.middle.resize(middle_count);
    int current_channels = in_channels;
    for (int i = 0; i < middle_count; i++) {
        const std::string base = prefix + ".middle." + std::to_string(i) + ".";
        auto& block = sequence.middle[i];
        block.conv_w = file.require((base + "conv.weight").c_str());
        block.bn_scale = file.require((base + "bn_fold.scale").c_str());
        block.bn_bias = file.require((base + "bn_fold.bias").c_str());
        if (!file.error.empty()) {
            error = file.error;
            return false;
        }
        if (!check_weight(block.conv_w, block.bn_bias, current_channels,
                          middle_channels, 3, base + "conv.weight", error) ||
            ggml_nelements(block.bn_scale) != middle_channels)
            return false;
        current_channels = middle_channels;
    }
    sequence.output.weight = file.require((prefix + ".out.weight").c_str());
    sequence.output.bias = file.require((prefix + ".out.bias").c_str());
    if (!file.error.empty()) {
        error = file.error;
        return false;
    }
    return check_weight(sequence.output.weight, sequence.output.bias,
                        current_channels, out_channels, final_kernel,
                        prefix + ".out.weight", error);
}

ggml_tensor* build_sequence(ggml_context* ctx,
                            const dense_head_sequence_weights& sequence,
                            ggml_tensor* input,
                            int final_padding) {
    ggml_tensor* value = input;
    for (const auto& block : sequence.middle)
        value = build_conv_bn_relu(ctx, block, value, 1, 1);
    return build_conv_bias(ctx, sequence.output.weight, sequence.output.bias,
                           value, 1, final_padding);
}

}  // namespace

bool load_anchor_multi_weights(model_file& file,
                               const hparams& hp,
                               anchor_multi_weights& weights,
                               std::string& error) {
    if (hp.head_kind != dense_head_kind::anchor_multi ||
        hp.multi_num_heads <= 0) {
        error = "invalid AnchorHeadMulti metadata";
        return false;
    }
    const int backbone_channels =
            hp.bb_num_upsample_filters.empty()
                    ? hp.bb_num_filters.back()
                    : std::accumulate(hp.bb_num_upsample_filters.begin(),
                                      hp.bb_num_upsample_filters.end(), 0);
    if (file.tensor("head.multi.shared.conv.weight")) {
        weights.has_shared = true;
        weights.shared.conv_w = file.require("head.multi.shared.conv.weight");
        weights.shared.bn_scale =
                file.require("head.multi.shared.bn_fold.scale");
        weights.shared.bn_bias = file.require("head.multi.shared.bn_fold.bias");
        if (!file.error.empty()) {
            error = file.error;
            return false;
        }
        if (!check_weight(weights.shared.conv_w, weights.shared.bn_bias,
                          backbone_channels, hp.multi_shared_channels, 3,
                          "head.multi.shared", error) ||
            ggml_nelements(weights.shared.bn_scale) != hp.multi_shared_channels)
            return false;
    } else if (hp.multi_shared_channels != backbone_channels) {
        error = "AnchorHeadMulti shared convolution is missing";
        return false;
    }
    const int input_channels =
            weights.has_shared ? hp.multi_shared_channels : backbone_channels;
    weights.heads.resize(hp.multi_num_heads);
    for (int head = 0; head < hp.multi_num_heads; head++) {
        const std::string prefix = "head.multi." + std::to_string(head);
        const int anchors = hp.multi_head_anchor_counts[head];
        const int classes = hp.multi_head_class_counts[head];
        auto& current = weights.heads[head];
        if (!load_sequence(file, prefix + ".cls", input_channels,
                           hp.multi_middle_convs, hp.multi_middle_channels,
                           anchors * classes, hp.multi_separate_reg ? 3 : 1,
                           current.classification, error))
            return false;
        if (hp.multi_separate_reg) {
            current.regression.resize(hp.multi_reg_channels.size());
            for (size_t branch = 0; branch < hp.multi_reg_channels.size();
                 branch++) {
                if (!load_sequence(file,
                                   prefix + ".reg." + std::to_string(branch),
                                   input_channels, hp.multi_middle_convs,
                                   hp.multi_middle_channels,
                                   anchors * hp.multi_reg_channels[branch], 3,
                                   current.regression[branch], error))
                    return false;
            }
        } else {
            if (!load_sequence(file, prefix + ".box", input_channels, 0, 0,
                               anchors * hp.code_size, 1, current.box, error))
                return false;
        }
        if (hp.use_direction_classifier) {
            if (!load_sequence(file, prefix + ".dir", input_channels, 0, 0,
                               anchors * hp.num_dir_bins, 1, current.direction,
                               error))
                return false;
        }
    }
    return true;
}

std::vector<anchor_multi_single_outputs> build_anchor_multi_graph(
        ggml_context* ctx,
        const anchor_multi_weights& weights,
        const hparams& hp,
        ggml_tensor* spatial_2d) {
    ggml_tensor* shared =
            weights.has_shared
                    ? build_conv_bn_relu(ctx, weights.shared, spatial_2d, 1, 1)
                    : spatial_2d;
    std::vector<anchor_multi_single_outputs> outputs(weights.heads.size());
    for (size_t head = 0; head < weights.heads.size(); head++) {
        const auto& source = weights.heads[head];
        auto& target = outputs[head];
        target.classification =
                build_sequence(ctx, source.classification, shared,
                               hp.multi_separate_reg ? 1 : 0);
        if (hp.multi_separate_reg) {
            target.regression.reserve(source.regression.size());
            for (const auto& branch : source.regression)
                target.regression.push_back(
                        build_sequence(ctx, branch, shared, 1));
        } else {
            target.box = build_sequence(ctx, source.box, shared, 0);
        }
        if (hp.use_direction_classifier)
            target.direction = build_sequence(ctx, source.direction, shared, 0);
    }
    return outputs;
}

}  // namespace openpcdet

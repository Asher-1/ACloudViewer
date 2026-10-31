// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <array>

#include "../model.h"
#include "anchor_head.h"
#include "bev_backbone.h"

namespace openpcdet {

struct caddn_bottleneck_weights {
    conv_bn_block_weights conv1, conv2, conv3, downsample;
    bool has_downsample = false;
};

struct caddn final : model {
    conv_bn_block_weights image_stem;
    std::array<std::vector<caddn_bottleneck_weights>, 4> image_layers;
    std::array<conv_bn_block_weights, 5> aspp;
    conv_bn_block_weights aspp_project;
    conv_bn_block_weights classifier_hidden;
    ggml_tensor *classifier_output_weight = nullptr;
    ggml_tensor *classifier_output_bias = nullptr;
    conv_bn_block_weights channel_reduce;
    conv_bn_block_weights collapse;
    bev_backbone_weights backbone;
    anchor_head_weights head;

    ggml_tensor *weight(const std::string &name) const {
        return resolved(name.c_str());
    }

    bool load(const std::string &path,
              const std::string &device,
              int threads) override;
    bool forward(const float *,
                 int32_t,
                 int32_t,
                 std::vector<detection> &,
                 const tap_sink & = {}) override;
    bool forward_scene(const scene_input &scene,
                       std::vector<detection> &detections,
                       const tap_sink &sink = {}) override;
};

}  // namespace openpcdet

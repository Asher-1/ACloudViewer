// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <array>

#include "../model.h"
#include "sparse_backbone.h"

namespace openpcdet {

struct voxelnext_sequence_weights {
    std::vector<sparse_layer_weights> middle;
    sparse_layer_weights output;
};

struct voxelnext_single_head_weights {
    voxelnext_sequence_weights heatmap;
    voxelnext_sequence_weights center;
    voxelnext_sequence_weights center_z;
    voxelnext_sequence_weights dimensions;
    voxelnext_sequence_weights rotation;
    voxelnext_sequence_weights velocity;
    voxelnext_sequence_weights iou;
};

struct voxelnext : model {
    sparse_backbone_weights backbone;
    sparse_layer_weights bev_conv;
    sparse_layer_weights shared_conv;
    std::vector<voxelnext_single_head_weights> heads;

    bool load(const std::string& gguf_path,
              const std::string& device,
              int n_threads) override;
    bool forward(const float* points,
                 int32_t n_points,
                 int32_t point_dim,
                 std::vector<detection>& detections,
                 const tap_sink& sink = {}) override;

private:
    struct raw_head {
        std::vector<float> heatmap;
        std::vector<float> center;
        std::vector<float> center_z;
        std::vector<float> dimensions;
        std::vector<float> rotation;
        std::vector<float> velocity;
        std::vector<float> iou;
    };
    struct raw_result {
        std::vector<int32_t> coordinates;
        std::vector<int32_t> spatial_shape;
        std::vector<raw_head> heads;
        bool heatmap_is_probability = false;
        bool dimensions_are_linear = false;
    };

    bool forward_single(const float* points,
                        int32_t n_points,
                        int32_t point_dim,
                        raw_result& result,
                        const tap_sink& sink);
    bool merge_double_flip(const std::array<raw_result, 4>& inputs,
                           raw_result& output);
    bool decode(const raw_result& result,
                std::vector<detection>& detections,
                const tap_sink& sink);
};

std::vector<sparse_layer_spec> make_voxelnext_backbone_specs(
        int input_channels,
        const std::vector<int32_t>& channels,
        const std::vector<int32_t>& kernel_sizes);

}  // namespace openpcdet

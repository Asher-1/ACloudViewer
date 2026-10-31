// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "../model.h"
#include "bev_backbone.h"
#include "sparse_backbone.h"

namespace openpcdet {

struct transfusion_prediction_weights {
    conv_bn_block_weights hidden;
    ggml_tensor* output_weight = nullptr;
    ggml_tensor* output_bias = nullptr;
};

struct transfusion_weights {
    sparse_backbone_weights sparse;
    bev_backbone_weights backbone;

    ggml_tensor* shared_weight = nullptr;
    ggml_tensor* shared_bias = nullptr;
    conv_bn_block_weights heatmap_hidden;
    ggml_tensor* heatmap_weight = nullptr;
    ggml_tensor* heatmap_bias = nullptr;
    ggml_tensor* class_weight = nullptr;
    ggml_tensor* class_bias = nullptr;

    ggml_tensor* self_q_weight = nullptr;
    ggml_tensor* self_q_bias = nullptr;
    ggml_tensor* self_k_weight = nullptr;
    ggml_tensor* self_k_bias = nullptr;
    ggml_tensor* self_v_weight = nullptr;
    ggml_tensor* self_v_bias = nullptr;
    ggml_tensor* self_out_weight = nullptr;
    ggml_tensor* self_out_bias = nullptr;
    ggml_tensor* cross_q_weight = nullptr;
    ggml_tensor* cross_q_bias = nullptr;
    ggml_tensor* cross_k_weight = nullptr;
    ggml_tensor* cross_k_bias = nullptr;
    ggml_tensor* cross_v_weight = nullptr;
    ggml_tensor* cross_v_bias = nullptr;
    ggml_tensor* cross_out_weight = nullptr;
    ggml_tensor* cross_out_bias = nullptr;
    ggml_tensor* linear1_weight = nullptr;
    ggml_tensor* linear1_bias = nullptr;
    ggml_tensor* linear2_weight = nullptr;
    ggml_tensor* linear2_bias = nullptr;
    ggml_tensor* norm1_weight = nullptr;
    ggml_tensor* norm1_bias = nullptr;
    ggml_tensor* norm2_weight = nullptr;
    ggml_tensor* norm2_bias = nullptr;
    ggml_tensor* norm3_weight = nullptr;
    ggml_tensor* norm3_bias = nullptr;

    ggml_tensor* self_pos_weight1 = nullptr;
    ggml_tensor* self_pos_bias1 = nullptr;
    ggml_tensor* self_pos_scale = nullptr;
    ggml_tensor* self_pos_shift = nullptr;
    ggml_tensor* self_pos_weight2 = nullptr;
    ggml_tensor* self_pos_bias2 = nullptr;
    ggml_tensor* cross_pos_weight1 = nullptr;
    ggml_tensor* cross_pos_bias1 = nullptr;
    ggml_tensor* cross_pos_scale = nullptr;
    ggml_tensor* cross_pos_shift = nullptr;
    ggml_tensor* cross_pos_weight2 = nullptr;
    ggml_tensor* cross_pos_bias2 = nullptr;

    transfusion_prediction_weights center;
    transfusion_prediction_weights height;
    transfusion_prediction_weights dimensions;
    transfusion_prediction_weights rotation;
    transfusion_prediction_weights velocity;
    transfusion_prediction_weights heatmap;
};

struct transfusion : model {
    transfusion_weights weights;

    bool load(const std::string& path,
              const std::string& device,
              int threads) override;
    bool forward(const float* points,
                 int32_t count,
                 int32_t dimension,
                 std::vector<detection>& detections,
                 const tap_sink& sink = {}) override;

protected:
    // spatial_features is HWC-interleaved [(y*W + x)*C + c] unless
    // input_is_whc is set, in which case it is already GGML WHC planes
    // [x + W*y + W*H*c] and the host transpose is skipped entirely.
    bool forward_spatial(const std::vector<float>& spatial_features,
                         int width,
                         int height,
                         std::vector<detection>& detections,
                         const tap_sink& sink,
                         bool input_is_whc = false);
};

}  // namespace openpcdet

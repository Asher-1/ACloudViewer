// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "../backend.h"
#include "../extensions/sparse_ops.h"
#include "../gguf_loader.h"
#include "../model.h"

namespace openpcdet {

struct sparse_unet_layer {
    ggml_tensor* weight = nullptr;
    ggml_tensor* scale = nullptr;
    ggml_tensor* bias = nullptr;
};

struct sparse_unet_weights {
    std::unordered_map<std::string, sparse_unet_layer> layers;
    bool has_detection_output = false;
};

struct sparse_unet_result {
    std::vector<float> point_features;    // active rows x 16
    std::vector<float> point_coords;      // active rows x xyz
    std::vector<float> spatial_features;  // optional HWC encoded BEV
    int width = 0;
    int height = 0;
};

bool load_sparse_unet_weights(model_file& file,
                              bool detection_output,
                              sparse_unet_weights& weights,
                              std::string& error);
void remap_sparse_unet_weights(sparse_unet_weights& weights,
                               ggml_context* device_context);

bool run_sparse_unet(engine_backend& backend,
                     const hparams& hp,
                     const sparse_unet_weights& weights,
                     const float* points,
                     int32_t point_count,
                     int32_t point_dim,
                     sparse_unet_result& result,
                     const tap_sink& sink,
                     std::string& error);

}  // namespace openpcdet

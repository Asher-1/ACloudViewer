// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// ConvFuser for BEVFusion
// (pcdet.models.backbones_2d.fuser.convfuser.ConvFuser).
//
// Concatenates image BEV and LiDAR BEV features along the channel dimension,
// then applies Conv2d(3×3, pad=1) → BN → ReLU.  BN is folded at conversion.

#include <string>
#include <vector>

#include "../backend.h"
#include "../gguf_loader.h"
#include "../model.h"
#include "bev_backbone.h"

namespace openpcdet {

struct conv_fuser_weights {
    conv_bn_block_weights conv;  // 3×3 conv, BN folded
};

bool load_conv_fuser_weights(model_file& file,
                             const hparams& hp,
                             conv_fuser_weights& w,
                             std::string& error);

// Run the ConvFuser forward pass.
// img_bev / lidar_bev: [W, H, C, B] in GGML column-major layout.
// Output: [W, H, out_channels, B].
bool run_conv_fuser(engine_backend& backend,
                    const hparams& hp,
                    const conv_fuser_weights& w,
                    const float* img_bev,
                    int img_channels,
                    const float* lidar_bev,
                    int lidar_channels,
                    int width,
                    int height,
                    int batch_size,
                    std::vector<float>& output,
                    const tap_sink& sink,
                    std::string& error);

}  // namespace openpcdet

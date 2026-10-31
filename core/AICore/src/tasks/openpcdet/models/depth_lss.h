// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// DepthLSSTransform for BEVFusion
// (pcdet.models.view_transforms.depth_lss.DepthLSSTransform).
//
// Lifts 2-D image features into 3-D via depth prediction, then splats them
// onto a BEV grid using the camera geometry (LSS = Lift-Splat-Shoot).
//
// Pipeline:
//   1. dtransform: small conv chain to encode the sparse depth map
//   2. concat(dtransform_out, fpn_feature) → depthnet → depth softmax + feats
//   3. outer product depth ⊗ feats → 3-D volume [D, fH, fW, C]
//   4. bev_pool: scatter-add the volume to BEV grid using precomputed geometry
//   5. collapse-Z: sum over depth bins → BEV feature map
//   6. optional 2× downsample (conv chain)
//   7. permute (X,Y) → (Y,X) for the downstream BEV backbone
//
// Steps 1-2 and 6 are standard convolutions (GGML graph).
// Steps 3-5 and 7 are CPU operations (geometry depends on per-frame camera
// calibration, and bev_pool is an irregular scatter-add).

#include <string>
#include <vector>

#include "../backend.h"
#include "../gguf_loader.h"
#include "../model.h"
#include "bev_backbone.h"

namespace openpcdet {

// DepthLSS weights.
struct depth_lss_weights {
    // dtransform: 3 conv-BN-ReLU blocks
    conv_bn_block_weights dtrans_0;  // Conv2d(1, 8, 1)
    conv_bn_block_weights dtrans_1;  // Conv2d(8, 32, 5, stride=4, pad=2)
    conv_bn_block_weights dtrans_2;  // Conv2d(32, 64, 5, stride=2, pad=2)

    // depthnet: 3 conv-BN-ReLU blocks + final 1×1 conv
    conv_bn_block_weights dnet_0;  // Conv2d(in_ch+64, in_ch, 3, pad=1)
    conv_bn_block_weights dnet_1;  // Conv2d(in_ch, in_ch, 3, pad=1)
    // final 1×1 conv (no BN, has bias)
    ggml_tensor* dnet_final_w = nullptr;  // [in_ch, D+C]
    ggml_tensor* dnet_final_b = nullptr;  // [D+C]

    // optional downsample (3 conv-BN-ReLU blocks)
    bool has_downsample = false;
    conv_bn_block_weights down_0;  // Conv2d(C, C, 3, pad=1)
    conv_bn_block_weights down_1;  // Conv2d(C, C, 3, stride=2, pad=1)
    conv_bn_block_weights down_2;  // Conv2d(C, C, 3, pad=1)
};

bool load_depth_lss_weights(model_file& file,
                            const hparams& hp,
                            depth_lss_weights& w,
                            std::string& error);

// Run the DepthLSS forward pass.
//
// fpn_feature: first FPN output [W_f, H_f, in_channel, B] (B = num_cameras /
// batch) depth_input: [W_img, H_img, 1, B] sparse depth map (from LiDAR
// projection) camera params: per-camera matrices, each [3×3 or 3×4] × B ×
// num_cameras
//
// Returns BEV features: [W_bev, H_bev, out_channel, B_batch]
bool run_depth_lss(engine_backend& backend,
                   const hparams& hp,
                   const depth_lss_weights& w,
                   const float* fpn_feature,
                   int fpn_W,
                   int fpn_H,
                   const float* depth_input,
                   int img_W,
                   int img_H,
                   const float* camera_intrinsics,  // [B, N, 3, 3]
                   const float* camera2lidar,       // [B, N, 3, 4]
                   const float* img_aug_matrix,     // [B, N, 3, 4]
                   const float* lidar_aug_matrix,   // [B, N, 3, 4]
                   const float* lidar2image,        // [B, N, 3, 4]
                   const float* points,             // raw point cloud
                   int n_points,
                   int point_dim,
                   int batch_size,
                   std::vector<float>& bev_output,
                   int& bev_W,
                   int& bev_H,
                   const tap_sink& sink,
                   std::string& error);

}  // namespace openpcdet

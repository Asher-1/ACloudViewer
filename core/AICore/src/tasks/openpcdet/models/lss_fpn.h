#pragma once

// GeneralizedLSSFPN for BEVFusion
// (pcdet.models.backbones_image.img_neck.generalized_lss.GeneralizedLSSFPN).
//
// Takes multi-scale features from SwinTransformer and produces FPN features
// via a top-down path: upsample deeper level → concat with shallower → 1×1
// conv → 3×3 conv.  All convs are Conv-BN-ReLU (BN folded at conversion).
//
// For BEVFusion (IN_CHANNELS=[192,384,768], OUT_CHANNELS=256):
//   Input:  3 feature maps at strides 8/16/32
//   Output: 2 feature maps at strides 8/16 (used by DepthLSS)
//
// The 2× bilinear upsampling between levels is done on the CPU between
// per-level GGML graph computations (GGML has no built-in bilinear op).

#include "../backend.h"
#include "../gguf_loader.h"
#include "../model.h"
#include "bev_backbone.h"

#include <string>
#include <vector>

namespace openpcdet {

// One lateral (1×1) + one fpn_refine (3×3) conv-bn-relu block.
struct lss_fpn_level_weights {
    conv_bn_block_weights lateral;   // 1×1 conv
    conv_bn_block_weights fpn_conv;  // 3×3 conv
};

struct lss_fpn_weights {
    std::vector<lss_fpn_level_weights> levels;
};

struct lss_fpn_output {
    std::vector<std::vector<float>> features;  // one per output level, HWC layout
    std::vector<int> widths;
    std::vector<int> heights;
    std::vector<int> channels;
};

bool load_lss_fpn_weights(model_file & file, const hparams & hp,
                          lss_fpn_weights & w, std::string & error);

// Run the FPN forward pass.
// swin_features: one entry per SwinTransformer output (same order as out_indices).
// Each entry is [W*H*C*B] in GGML column-major order.
bool run_lss_fpn(engine_backend & backend, const hparams & hp,
                 const lss_fpn_weights & w,
                 const std::vector<std::vector<float>> & swin_features,
                 const std::vector<int> & widths,
                 const std::vector<int> & heights,
                 const std::vector<int> & channels,
                 int batch_size,
                 lss_fpn_output & result,
                 const tap_sink & sink, std::string & error);

} // namespace openpcdet

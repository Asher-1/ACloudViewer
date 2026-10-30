// BaseBEVBackbone — the 2D backbone for PointPillar/SECOND/CenterPoint
// (pcdet/models/backbones_2d/base_bev_backbone.py).
//
// Layout: a sequence of "levels". Each level is
//     ZeroPad2d(1) -> Conv2d(stride, k=3, bias=False) -> BN -> ReLU
//                  -> (layer_nums) x (Conv2d(k=3, pad=1, bias=False) -> BN -> ReLU)
// followed by an upsampling "deblock" per level:
//     ConvTranspose2d(stride, bias=False) -> BN -> ReLU
// Final: concat all deblock outputs along the channel dim.
//
// All Linears/Convs are bias-free; the only norm params are BN gamma/beta
// (running stats are pre-folded into bn_fold.scale / bn_fold.bias in the
// converter — eval-mode BN is an affine per-channel transform).
//
// In the ggml graph:
//   conv2d weight layout is [KW, KH, IC, OC] (the converter permutes PyTorch's
//   [OC, IC, KH, KW] -> [KW, KH, IC, OC] = permute(3, 2, 0, 1)).
//   conv_transpose_2d weight layout is [KW, KH, IC, OC] (PyTorch's
//   [IC, OC, KH, KW] -> [KW, KH, IC, OC] = permute(3, 2, 0, 1)).
//   Per-channel BN-fold scale/bias is applied via ggml_mul + ggml_add with
//   broadcast over the channel dim (ne[2]).
#pragma once

#include "../gguf_loader.h"
#include "../model.h"

#include <string>
#include <vector>

namespace openpcdet {

struct conv_bn_block_weights {
    ggml_tensor * conv_w    = nullptr;  // [KW, KH, IC, OC]
    ggml_tensor * bn_scale  = nullptr;  // [OC]
    ggml_tensor * bn_bias   = nullptr;  // [OC]
};

struct bev_backbone_weights {
    // one entry per level: the first block is the strided stem, then
    // `layer_nums[i]` non-strided follow-up blocks.
    std::vector<std::vector<conv_bn_block_weights>> levels;  // [level][block]
    // deblocks[i] is the upsample for level i (may be empty)
    std::vector<conv_bn_block_weights> deblocks;
    int num_bev_features = 0;  // sum of deblock out channels
};

// Reusable dense blocks for detector heads. They intentionally compose public
// GGML operations instead of adding operations to the GGML submodule.
ggml_tensor * build_conv_bn_relu(ggml_context * ctx,
                                 const conv_bn_block_weights & weights,
                                 ggml_tensor * input, int stride, int padding);
ggml_tensor * build_conv_bn_relu_dilated(ggml_context * ctx,
                                         const conv_bn_block_weights & weights,
                                         ggml_tensor * input, int stride,
                                         int padding, int dilation);
ggml_tensor * build_conv_bias(ggml_context * ctx, ggml_tensor * weight,
                              ggml_tensor * bias, ggml_tensor * input,
                              int stride, int padding);
ggml_tensor * build_conv_bias_dilated(ggml_context * ctx, ggml_tensor * weight,
                                      ggml_tensor * bias, ggml_tensor * input,
                                      int stride, int padding, int dilation);

bool load_bev_backbone_weights(model_file & file, const hparams & hp,
                               bev_backbone_weights & w, std::string & err);

// Build the backbone forward subgraph in `ctx`. `spatial` is the input BEV
// tensor [nx, ny, C_in, N=1] (ggml ne order). Returns the output tensor
// [nx_full, ny_full, num_bev_features, N] (concatenated deblock outputs at
// the upsampled resolution). All intermediate taps are emitted via `tap`.
ggml_tensor * build_bev_backbone_graph(ggml_context * ctx,
                                       const bev_backbone_weights & w,
                                       const hparams & hp,
                                       ggml_tensor * spatial,
                                       std::vector<ggml_tensor *> * taps = nullptr);

// Patch-count (OW*OH*N) ceiling for the ggml_conv_2d_direct fast path. The CPU
// direct conv batches patches through a fixed 16MB work buffer and currently
// scatters results across output cells beyond ~5240 patches per batch, so the
// CPU backend lowers this to route large convs through im2col+GEMM instead.
// CUDA keeps the fused direct path at any size. Set by the engine once the
// backend is known; OPCPD_CONV_DIRECT_MAX_PATCHES overrides at runtime.
void set_conv_direct_max_patches(int64_t max_patches);

#ifdef AICORE_OPENPCDET_CUDA_EXT
// CUDA-ext fused BN-fold+ReLU switch (ggml patch 0024): when enabled and the
// ggml op is available, conv_bn_relu emits one fused node instead of the
// mul + add + relu triple. Set once from model::realize_weights on CUDA;
// CPU/Vulkan keep the portable triple.
void set_bn_fuse_relu(bool enable);
#endif

} // namespace openpcdet

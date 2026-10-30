#pragma once

// SwinTransformer image backbone for BEVFusion
// (pcdet.models.backbones_image.swin.SwinTransformer).
//
// Processes camera images through patch embedding + 4 Swin Transformer stages
// with window attention. Outputs multi-scale features at strides 8/16/32
// (for OUT_INDICES=[1,2,3]) used by the FPN + DepthLSS view transform.
//
// In the GGML graph:
//   - Input:  [W, H, 3, B]  (NCHW in PyTorch → GGML column-major layout)
//   - Output: vector of [W', H', C, B] at requested stage indices
//   - All Linear weights stored as [in, out] (transposed from PyTorch)
//   - Conv2d weights stored as [KW, KH, IC, OC]

#include "../backend.h"
#include "../gguf_loader.h"
#include "../model.h"

#include <string>
#include <vector>

namespace openpcdet {

// --- Weight structures ---

struct swin_window_attn_weights {
    ggml_tensor * qkv_w = nullptr;   // [embed_dims, 3*embed_dims]
    ggml_tensor * qkv_b = nullptr;   // [3*embed_dims]
    ggml_tensor * proj_w = nullptr;  // [embed_dims, embed_dims]
    ggml_tensor * proj_b = nullptr;  // [embed_dims]
    // Precomputed relative position bias: [nH, Wh*Ww, Wh*Ww]
    // Computed during load from the bias table + fixed position indices.
    ggml_tensor * rel_pos_bias = nullptr;
};

struct swin_ffn_weights {
    ggml_tensor * fc1_w = nullptr;   // [embed_dims, ffn_channels]
    ggml_tensor * fc1_b = nullptr;   // [ffn_channels]
    ggml_tensor * fc2_w = nullptr;   // [ffn_channels, embed_dims]
    ggml_tensor * fc2_b = nullptr;   // [embed_dims]
};

struct swin_block_weights {
    // LayerNorm 1 (before attention)
    ggml_tensor * norm1_w = nullptr;  // [embed_dims]
    ggml_tensor * norm1_b = nullptr;  // [embed_dims]
    swin_window_attn_weights attn;

    // LayerNorm 2 (before FFN)
    ggml_tensor * norm2_w = nullptr;  // [embed_dims]
    ggml_tensor * norm2_b = nullptr;  // [embed_dims]
    swin_ffn_weights ffn;
};

struct swin_downsample_weights {
    ggml_tensor * norm_w = nullptr;    // [4*in_channels]
    ggml_tensor * norm_b = nullptr;    // [4*in_channels]
    ggml_tensor * reduction_w = nullptr;  // [4*in_channels, out_channels]
};

struct swin_stage_weights {
    std::vector<swin_block_weights> blocks;
    swin_downsample_weights downsample;  // unused in last stage
    bool has_downsample = false;
};

struct swin_weights {
    // Patch embedding
    ggml_tensor * patch_embed_w = nullptr;  // [KW, KH, 3, embed_dims]
    ggml_tensor * patch_embed_b = nullptr;  // [embed_dims]
    ggml_tensor * patch_norm_w = nullptr;   // [embed_dims]
    ggml_tensor * patch_norm_b = nullptr;   // [embed_dims]

    std::vector<swin_stage_weights> stages;

    // Output norms (one per out_index)
    std::vector<ggml_tensor *> out_norm_w;  // [num_features[i]]
    std::vector<ggml_tensor *> out_norm_b;  // [num_features[i]]
};

// Load SwinTransformer weights from the model file.
bool load_swin_weights(model_file & file, const hparams & hp,
                       swin_weights & w, std::string & error);

// Output from the SwinTransformer forward pass.
// Each entry is one multi-scale feature map at the requested out_index.
struct swin_output {
    std::vector<std::vector<float>> features;  // one per out_index, HWC layout
    std::vector<int> widths;
    std::vector<int> heights;
    std::vector<int> channels;
};

// Run the SwinTransformer forward pass.
//
// images: [B * H * W * 3] float array (B = num_cameras for a single scene)
// batch_size: number of images (cameras)
// Returns multi-scale features for each image.
bool run_swin_transformer(engine_backend & backend, const hparams & hp,
                          const swin_weights & w,
                          const float * images, int batch_size,
                          int image_height, int image_width,
                          swin_output & result,
                          const tap_sink & sink, std::string & error);

} // namespace openpcdet

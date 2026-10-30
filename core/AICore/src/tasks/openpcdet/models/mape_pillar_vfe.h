#pragma once

// MAPE (Max-and-Attention Pillar Encoding) VFE for FastPillars
// (pcdet.models.backbones_3d.vfe.mape_vfe.MAPEVFE).
//
// Pipeline:
//   1. CPU voxelization (reuse existing voxelization helpers)
//   2. Point feature augmentation (cluster offset + center offset)
//   3. Linear projection → BN → ReLU (GGML graph)
//   4. Max pooling + Attention pooling → average (host-side)
//   5. PillarScatter → [W, H, C, B] BEV tensor
//
// All Linear weights stored as [in, out] (transposed from PyTorch).

#include "../backend.h"
#include "../gguf_loader.h"
#include "../model.h"

#include <string>
#include <vector>

namespace openpcdet {

struct mape_layer_weights {
    ggml_tensor * linear_w = nullptr;   // [in_ch, out_ch]
    ggml_tensor * bn_scale = nullptr;   // [out_ch]  (eval_scale after BN fold)
    ggml_tensor * bn_bias  = nullptr;   // [out_ch]  (eval_bias after BN fold)
    // Attention MLP
    ggml_tensor * attn_fc1_w = nullptr; // [out_ch, out_ch]
    ggml_tensor * attn_fc1_b = nullptr; // [out_ch]
    ggml_tensor * attn_fc2_w = nullptr; // [out_ch, out_ch]
    ggml_tensor * attn_fc2_b = nullptr; // [out_ch]
};

struct mape_pillar_weights {
    std::vector<mape_layer_weights> layers;
    int in_channels = 0;
    int out_channels = 0;
};

struct mape_pillar_result {
    std::vector<float> pillar_features;  // [num_pillars, out_channels]
    std::vector<int64_t> coords;         // [num_pillars, 3] (batch_idx, y, x)
    int num_pillars = 0;
    int width = 0;   // grid width  (x direction)
    int height = 0;  // grid height (y direction)
};

bool load_mape_pillar_weights(model_file & file, const hparams & hp,
                              mape_pillar_weights & w, std::string & error);

// Run MAPE VFE + PillarScatter.
// Returns BEV features in pillar_features + coords for scatter, or
// directly produces a dense BEV tensor in bev_features [W, H, C, 1].
bool run_mape_pillar_vfe(engine_backend & backend, const hparams & hp,
                         const mape_pillar_weights & w,
                         const float * points, int n_points, int point_dim,
                         mape_pillar_result & result,
                         const tap_sink & sink, std::string & error);

} // namespace openpcdet

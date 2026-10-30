// PillarVFE — the VFE for PointPillar. Runs on CPU (it is point-wise Linear +
// BN-folded affine + ReLU + max-pool over the points dim; keeping it on CPU
// avoids needing a custom ggml max-over-dim op, and the VFE is a tiny fraction
// of total compute — the 2D conv backbone dominates).
//
// Replicates pcdet/models/backbones_3d/vfe/pillar_vfe.py in eval mode:
//   1. Expand point features: voxel_features (P dim) + f_cluster + f_center
//      -> [V, P, C_in_expanded]
//   2. Apply each PFNLayer: Linear (no bias when use_norm) -> BN1d (folded to
//      an affine y = scale*x + bias in eval mode) -> ReLU -> max over P dim ->
//      (non-last layer) concat with the repeated max along the channel dim.
//   3. Output: [V, C_out]
//
// BN folding: in eval mode BN1d is
//   y = (x - running_mean) / sqrt(running_var + eps) * gamma + beta
//     = x * (gamma / sqrt(var + eps)) + (beta - running_mean * gamma / sqrt(var + eps))
// We pre-compute scale=gamma/sqrt(var+eps) and bias=beta-mean*scale in the
// converter and store them as <pfn>.bn_fold.scale / <pfn>.bn_fold.bias. The
// engine never sees raw running_mean/running_var. Taps match the PyTorch
// reference because the math is identical.
#pragma once

#include "../gguf_loader.h"
#include "../model.h"

#include <string>
#include <vector>

namespace openpcdet {

struct pfn_layer_weights {
    ggml_tensor * linear_w = nullptr;    // [out_C, in_C]
    ggml_tensor * linear_b = nullptr;    // [out_C]  (nullptr if use_norm)
    ggml_tensor * bn_scale = nullptr;    // [out_C]  (gamma / sqrt(var+eps)); 1 if no norm
    ggml_tensor * bn_bias  = nullptr;    // [out_C]  (beta - mean*scale);    0 if no norm
    bool last_layer = false;
    int  out_channels = 0;
};

struct pillar_vfe_weights {
    std::vector<pfn_layer_weights> layers;
};

// Load PFNLayer weights from the GGUF. Returns false on missing tensors.
bool load_pillar_vfe_weights(model_file & file, const hparams & hp,
                             pillar_vfe_weights & w, std::string & err);

// Run the VFE on CPU. voxels: [V*P*point_dim] row-major (zero-padded P dim).
// num_points: [V]. coords: [V*4] = [batch_idx, z, y, x] per voxel.
// Output: pillar_features [V*out_C] row-major.
//
// `sink` (optional) gets per-layer taps in topological order, matching the
// PyTorch reference: "vfe.layer_i.linear", "vfe.layer_i.bn",
// "vfe.layer_i.relu", "vfe.layer_i.max", "vfe.out".
bool run_pillar_vfe_cpu(const pillar_vfe_weights & w,
                        const hparams & hp,
                        const float * voxels, const int32_t * num_points,
                        const int32_t * coords, int32_t V, int32_t P, int32_t point_dim,
                        std::vector<float> & out_pillar_features,
                        std::string & err,
                        const tap_sink & sink = {});

// DynamicPillarVFE counterpart. It keeps every in-range point, groups by the
// XY pillar coordinate, and returns sorted [batch,z,y,x] coordinates together
// with one max-aggregated feature row per pillar.
bool run_dynamic_pillar_vfe_cpu(const pillar_vfe_weights & w,
                                const hparams & hp,
                                const float * points, int32_t point_count,
                                int32_t point_dim,
                                std::vector<int32_t> & pillar_coords,
                                std::vector<float> & pillar_features,
                                std::string & err,
                                const tap_sink & sink = {});

} // namespace openpcdet

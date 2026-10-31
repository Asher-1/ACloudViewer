// AutoencoderKL (SD VAE, as configured by zero123plus-v1.2) — ggml port.
//
// Encoder: conv_in → 4 down levels (2 resnets each; stride-2 downsample on
// levels 0-2) → mid (resnet, vanilla spatial attention heads=1, resnet) →
// norm/silu/conv_out → quant_conv → [B, 2*4, h, w]; mode = first 4 channels.
// Decoder: post_quant_conv → conv_in → mid (same) → 4 up levels (3 resnets
// each, nearest-neighbor ×2 upsample on levels 0-2) → norm/silu/conv_out.
//
// Channel config: block_out_channels [128,256,512,512], latent 4,
// scaling_factor 0.18215, GroupNorm 32 @ eps 1e-6.
//
// Conventions (mirroring zero123plus/pipeline.py):
//   vae_encode_mode() returns the posterior MODE (deterministic; the pipeline
//   uses .sample(), whose posterior is near-delta). The returned latent is NOT
//   multiplied by scaling_factor (matches `encode_condition_image`).
//   vae_decode() divides its input by scaling_factor internally (matching
//   `vae.decode(latents / scaling_factor)`) and returns the raw image in the
//   same normalized domain the feature_extractor_vae produces (mean .5 / std
//   .8 space); unscale_image (x / 0.5 * 0.8) is the caller's job.
//
// Weights: `zero123pp_vae_<p>.gguf` from convert/convert_zero123pp.py, tensor
// names identical to the diffusers AutoencoderKL state_dict.
#ifndef INSTANTMESH_MODELS_VAE_HPP
#define INSTANTMESH_MODELS_VAE_HPP

#include <string>

#include "gguf_io.hpp"
#include "ggml.h"
#include "ggml-backend.h"

namespace instantmesh {

struct VaeHparams {
    int block_out_channels[4] = {128, 256, 512, 512};
    int latent_channels = 4;
    int norm_num_groups = 32;
    float norm_eps = 1e-6f;
    float scaling_factor = 0.18215f;
};

struct VaeModel {
    GgufModel gguf;
    ggml_backend_t backend = nullptr;
    VaeHparams hp;
};

bool vae_load(const std::string & path, ggml_backend_t backend,
              VaeModel & out, std::string * error);

// image: [B, 3, H, H] f32 in feature_extractor_vae normalized space
// ((x/255 - mean) / std, mean=.5 std=.8 → range ≈ [-0.625, 0.625]); H must be
// a multiple of 64 (4 levels of stride 2 → 8× downsample… encoder uses 4
// stride-2 ops total: 3 downsamplers + nothing else → H/8).
// Returns host [B, 4, H/8, W/8] f32 (posterior mode). Caller frees.
float * vae_encode_mode(const VaeModel & model, const float * image,
                        int B, int H, int W, int * out_h, int * out_w);

// latents: [B, 4, h, w] f32 (pipeline-scale latents, i.e. what the denoising
// loop produces). Divides by scaling_factor internally, decodes, and returns
// host [B, 3, 8h, 8w] f32 in feature_extractor_vae normalized space. Caller
// frees and applies unscale (x / 0.5 * 0.8) afterwards.
float * vae_decode(const VaeModel & model, const float * latents,
                   int B, int h, int w, int * out_h, int * out_w);

} // namespace instantmesh

#endif // INSTANTMESH_MODELS_VAE_HPP

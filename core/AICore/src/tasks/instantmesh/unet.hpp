// UNet2DConditionModel (SD2.1, as shipped with zero123plus-v1.2 + the
// InstantMesh white-background finetune) — ggml port with RefOnly attention.
//
// Structure (from unet/config.json):
//   time_emb: sinusoidal(320, flip_sin_to_cos, shift 0) → linear_1 → silu →
//             linear_2 → [B,1280]
//   conv_in(4→320) → down0..2 (CrossAttnDownBlock2D: 2 resnets + 1
//   transformer + downsample stride2-pad1) → down3 (DownBlock2D: 2 resnets)
//   → mid (resnet, transformer, resnet) → up0..3 (3 resnets each with skip
//   cat; up0 plain + upsample, up1..3 cross-attn + upsample on 0..2) →
//   norm/silu/conv_out(1280→4).
//   heads per level [5,10,20,20] (head_dim 64 everywhere), cross dim 1024,
//   use_linear_projection (Linear proj_in/out), transformer_layers 1.
//
// RefOnly (zero123plus pipeline.py `ReferenceOnlyAttnProc`, attn1 only):
//   w-pass records each self-attention block's post-norm hidden states
//   h_w[b] = [B,HW,C]; r-pass replaces attn1 with a cross-attention whose
//   K/V source is cat(h_r, h_w[b]) along the sequence axis (q stays r).
//   Both passes share one graph: the w-pass intermediate feeds the r-pass
//   K/V directly (no host round-trip).
//
// Layout contract: identical to vae.cpp — all activations contiguous
// [W,H,C,B] (== torch NCHW bytes); attention gathers [C,HW] via
// cont(permute(2,1,0,3)); convs via conv()/conv_f32().
#ifndef INSTANTMESH_MODELS_UNET_HPP
#define INSTANTMESH_MODELS_UNET_HPP

#include <string>

#include "gguf_io.hpp"
#include "ggml.h"
#include "ggml-backend.h"

namespace instantmesh {

struct UnetHparams {
    int in_channels = 4, out_channels = 4;
    int block_out_channels[4] = {320, 640, 1280, 1280};
    int num_heads[4] = {5, 10, 20, 20};       // heads per level (head_dim 64)
    int layers_per_block = 2;
    int cross_attention_dim = 1024;
    int norm_num_groups = 32;
    float norm_eps = 1e-5f;
    int context_len = 77;
    int time_emb_dim = 1280;                  // linear_2 output
    bool use_linear_projection = true;
    bool flip_sin_to_cos = true;
    int freq_shift = 0;
};

struct UnetModel {
    GgufModel gguf;
    ggml_backend_t backend = nullptr;
    UnetHparams hp;
};

bool unet_load(const std::string & path, ggml_backend_t backend,
               UnetModel & out, std::string * error);

// One RefOnly double-forward at a fixed integer timestep. Both passes run on
// batch B (`sample` is the [B,4,H,W] torch-NCHW r-pass input;
// `ref_sample` is the [B,4,ref_h,ref_w] torch-NCHW w-pass input — the
// official pipeline denoises at 120x80 while the condition latent is 64x64,
// so the two resolutions differ and RefOnly K/V is cat(h_r, h_w) along the
// sequence axis; `context` is the [B,L,1024] torch bytes fed to every
// cross-attention). Returns host [B,4,H,W] float32 (the eps prediction of
// the r-pass). Caller frees. This allocates a fresh graph per call.
float * unet_forward_refonly(const UnetModel & model,
                             const float * sample, const float * ref_sample,
                             const float * context, int context_len,
                             int B, int H, int W, int ref_h, int ref_w,
                             float timestep, int * out_h, int * out_w);

} // namespace instantmesh

#endif // INSTANTMESH_MODELS_UNET_HPP

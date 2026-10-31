// CLIPVisionModelWithProjection (zero123++ vision_encoder) — ggml port.
//
// Reference: transformers CLIPVisionModelWithProjection, as configured by
// zero123plus-v1.2/vision_encoder/config.json:
//   hidden 1280, 32 layers, 16 heads (head_dim 80), patch 14, image 224,
//   intermediate 5120, projection_dim 1024, hidden_act "gelu" (exact erf),
//   layer_norm_eps 1e-5. Note: patch_embedding / q/k/v / visual_projection
//   have NO bias (CLIP convention); out_proj / fc1 / fc2 have bias. The
//   embeddings layernorm key keeps the historic HF misspelling `pre_layrnorm`.
//
// Weights: `zero123pp_cond_f32.gguf` produced by convert/convert_zero123pp.py,
// tensor names prefixed `vis.` and otherwise identical to the HF state_dict.
#ifndef INSTANTMESH_MODELS_CLIP_VISION_HPP
#define INSTANTMESH_MODELS_CLIP_VISION_HPP

#include <string>

#include "gguf_io.hpp"
#include "ggml.h"
#include "ggml-backend.h"

namespace instantmesh {

struct ClipVisionHparams {
    int hidden_size = 1280;
    int num_hidden_layers = 32;
    int num_attention_heads = 16;
    int intermediate_size = 5120;
    int image_size = 224;
    int patch_size = 14;
    int projection_dim = 1024;
    float layer_norm_eps = 1e-5f;
};

struct ClipVisionModel {
    GgufModel gguf;
    ggml_backend_t backend = nullptr;
    ClipVisionHparams hp;
};

// Loads a cond GGUF onto `backend` (vision weights + pipeline constants).
bool clip_vision_load(const std::string & path, ggml_backend_t backend,
                      ClipVisionModel & out, std::string * error);

// Runs the vision encoder.
//   image: [B, 3, 224, 224] float32, RGB, CLIP-normalized (feature_extractor_clip).
// Returns a host buffer of [B, projection_dim] f32 (image_embeds). Caller frees.
float * clip_vision_encode(const ClipVisionModel & model, const float * image,
                           int B, int * out_dim);

} // namespace instantmesh

#endif // INSTANTMESH_MODELS_CLIP_VISION_HPP

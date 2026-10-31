// DINO ViT-B/16 encoder (adaLN camera-modulated) — ggml port.
//
// Reference: src/models/encoder/dino.py (Transformers-style ViTModel with a DiT
// adaLN modulation injected per layer). Forward:
//   camera_embedder(camera)         -> adaln_input [B, 768]
//   patch_embeddings(image) [+cls]  -> [B, 1+N, 768]
//   + interpolated pos_encoding
//   12x VITLayer(adaln)             -> [B, 1+N, 768]  (last hidden state)
//
// Weights come from the `dino_<p>.gguf` produced by convert/convert_lrm.py,
// whose tensor names mirror the PyTorch state_dict (after stripping the
// `encoder.` prefix): `model.embeddings.*`, `model.encoder.layer.N.*`,
// `camera_embedder.*`.
#ifndef INSTANTMESH_MODELS_DINO_HPP
#define INSTANTMESH_MODELS_DINO_HPP

#include <memory>
#include <string>

#include "gguf_io.hpp"
#include "ggml.h"
#include "ggml-backend.h"

namespace instantmesh {

struct DinoHparams {
    int hidden_size = 768;
    int num_hidden_layers = 12;
    int num_attention_heads = 12;
    int image_size = 224;
    int patch_size = 16;
    float layer_norm_eps = 1e-12f;
};

// Wrapper holding the loaded weights plus the backend they reside on.
struct DinoModel {
    GgufModel gguf;
    ggml_backend_t backend = nullptr;
    DinoHparams hp;
};

// Loads a dino GGUF onto `backend` and parses its hparams.
bool dino_load(const std::string & path, ggml_backend_t backend,
               DinoModel & out, std::string * error);

// Runs the DINO encoder.
//   image:   [B, 3, H, W] float32, RGB in [0,1], ImageNet-normalized.
//   camera:  [B, 16] float32, the 16-dim camera conditioning.
// Returns a host buffer of [B, 1+N, hidden_size] (contiguous f32). Caller frees.
float * dino_encode(const DinoModel & model, const float * image,
                    const float * camera, int B, int H, int W,
                    int * out_seq, int * out_hidden);

} // namespace instantmesh

#endif // INSTANTMESH_MODELS_DINO_HPP
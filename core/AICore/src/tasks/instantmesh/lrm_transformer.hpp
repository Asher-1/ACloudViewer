// TriplaneTransformer (InstantMesh LRM geometry branch) — ggml port.
//
// Reference: src/models/decoder/transformer.py (TriplaneTransformer). Forward:
//   x = pos_embed.repeat(N,1,1)                 -> [N, 3*low^2, inner_dim]
//   16x BasicTransformerBlock(x, image_feats)   -> [N, 3*low^2, inner_dim]
//   x = norm(x)
//   x.view(N,3,low,low,d).einsum('nihwd->indhw') -> [3,N,d,low,low]
//   .view(3N,d,low,low) -> ConvTranspose2d(d->triplane_dim,k=2,s=2) -> [3N,td,high,high]
//   .view(3,N,td,high,high).einsum('indhw->nidhw') -> [N,3,td,high,high]
//
// Each layer is norm1 -> cross_attn(cond=image_feats) | norm2 -> self_attn |
// norm3 -> MLP, all with residual connections (nn.MultiheadAttention, bias off,
// scale = 1/sqrt(head_dim)). Weights mirror the `transformer.` GGUF names
// (after stripping the prefix): pos_embed, layers.N.norm{1,2,3}.*,
// layers.N.{cross,self}_attn.{q,k,v}_proj_weight/out_proj.weight,
// layers.N.self_attn.in_proj_weight (q/k/v concatenated), layers.N.mlp.*,
// norm.*, deconv.{weight,bias}.
#ifndef INSTANTMESH_MODELS_LRM_TRANSFORMER_HPP
#define INSTANTMESH_MODELS_LRM_TRANSFORMER_HPP

#include <memory>
#include <string>

#include "gguf_io.hpp"
#include "ggml.h"
#include "ggml-backend.h"

namespace instantmesh {

struct LrmTransformerHparams {
    int inner_dim = 1024;
    int num_layers = 16;
    int num_heads = 16;
    int cond_dim = 768;
    int triplane_low_res = 32;
    int triplane_high_res = 64;
    int triplane_dim = 80;
    float layer_norm_eps = 1e-6f;
};

struct LrmTransformerModel {
    GgufModel gguf;
    ggml_backend_t backend = nullptr;
    LrmTransformerHparams hp;
};

// Loads an lrm_transformer GGUF onto `backend` and parses its hparams.
bool lrm_transformer_load(const std::string & path, ggml_backend_t backend,
                          LrmTransformerModel & out, std::string * error);

// Runs the TriplaneTransformer.
//   image_feats: [N, L_cond, cond_dim] float32 (the DINO encoder output).
//   n_cond: length of the condition sequence (L_cond).
// Returns a host buffer of [N, 3, triplane_dim, high, high] (contiguous f32,
// high fastest). Caller frees.
float * lrm_transformer_forward(const LrmTransformerModel & model,
                                const float * image_feats, int N, int n_cond,
                                int * out_planes, int * out_dim,
                                int * out_h, int * out_w);

} // namespace instantmesh

#endif // INSTANTMESH_MODELS_LRM_TRANSFORMER_HPP
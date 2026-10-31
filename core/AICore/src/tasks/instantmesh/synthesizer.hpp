// OSGDecoder (InstantMesh FlexiCubes geometry branch) — ggml port.
//
// Reference: src/models/renderer/synthesizer_mesh.py (TriplaneSynthesizer +
// OSGDecoder). This is the *neural* part of the isosurface extraction that
// carries weights; the voxel grid, triplane bilinear sampling and the
// FlexiCubes mesh extraction are fixed geometry (no weights).
//
// Forward (N==1; the triplane plane axes are the fixed permutation matrices):
//   planes [1,3,C,H,W] + grid verts [M,3] (already scaled by grid_scale)
//     -> sample_from_planes (bilinear, align_corners=False, box_warp=2.0)
//     -> sampled [3C, M]                            (row = plane*C + channel)
//   sdf        = net_sdf(sampled)                   -> [1, M]
//   deformation= net_deformation(sampled)           -> [3, M]
//   weight     = net_weight(grid_features) * 0.1    -> [21, n_cubes]
// where grid_features[c] = concat over the 8 cube corners of sampled[:,corn]
//   -> [8*3C, n_cubes].
//
// Each net is a 4-layer MLP: Linear(3C->64) ReLU Linear(64->64) ReLU
// Linear(64->64) ReLU Linear(64->out). Weights mirror the `decoder.` GGUF
// names (after stripping the `synthesizer.` prefix): net_sdf/net_rgb/
// net_deformation/net_weight.%d.{weight,bias}.
#ifndef INSTANTMESH_MODELS_SYNTHESIZER_HPP
#define INSTANTMESH_MODELS_SYNTHESIZER_HPP

#include <memory>
#include <string>
#include <functional>

#include "gguf_io.hpp"
#include "ggml.h"
#include "ggml-backend.h"

namespace instantmesh {

struct SynthesizerHparams {
    int plane_dim = 80;   // triplane feature channels (C)
    int hidden = 64;
    int num_layers = 4;
};

struct SynthesizerModel {
    GgufModel gguf;
    ggml_backend_t backend = nullptr;
    SynthesizerHparams hp;
};

// Loads a synthesizer GGUF onto `backend` and parses its hparams.
bool synthesizer_load(const std::string & path, ggml_backend_t backend,
                      SynthesizerModel & out, std::string * error);

// Runs the geometry prediction (triplane sampling + OSGDecoder).
//   planes:  [N, 3, C, H, W] float32 (the TriplaneTransformer output).
//   verts:   [M, 3] float32 grid vertices (already * grid_scale).
//   cubes:   [n_cubes, 8] int32 indices into verts (8 corners per cube).
// Returns host buffers (caller free()s each):
//   *sdf        : [N, M]
//   *deformation: [N, M, 3]
//   *weight     : [N, n_cubes, 21]
void synthesizer_forward(const SynthesizerModel & model,
                         const float * planes, int N, int H, int W,
                         const float * verts, int M,
                         const int32_t * cubes, int n_cubes,
                         float ** sdf, float ** deformation, float ** weight,
                         const std::function<void(int, int)>& progress = {});

// Runs the texture (RGB) branch: samples the triplane at `points` [M,3] and
// pushes them through net_rgb (sigmoid-clamped like the reference MipNeRF).
//   planes: [N, 3, C, H, W] float32.
//   points: [M, 3] float32 query positions (final mesh vertex coordinates).
//   rgb:    [N, M, 3] float32 in [0,1] (caller must free()).
// Returns true on success.
bool synthesizer_texture_forward(const SynthesizerModel & model,
                                 const float * planes, int N, int H, int W,
                                 const float * points, int M, float ** rgb);

} // namespace instantmesh

#endif // INSTANTMESH_MODELS_SYNTHESIZER_HPP
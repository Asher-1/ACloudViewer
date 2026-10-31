// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// InstantMesh end-to-end pipeline (in-tree port of
// https://github.com/Asher-1/InstantMesh-GGML, cpp_ggml/src/tools/
// {instantmesh,zero123pp}.cpp, MIT).
//
//   RGBA input (borrowed view)
//     -> (optional RMBG matting, in-tree aicore_rmbg_* task)
//     -> Zero123++ multi-view diffusion (75-step EulerAncestral, RefOnly
//        double-forward; or replicate the input 6x when the diffusion models
//        are absent)
//     -> DINO ViT-B/16 encoder with adaLN camera modulation
//     -> TriplaneTransformer -> OSGDecoder (sdf/deformation/flexi weights)
//     -> FlexiCubes isosurface extraction -> (optional UV texture bake)
//
// The stage implementations live next to this file (dino.cpp,
// lrm_transformer.cpp, synthesizer.cpp, flexicubes.cpp, texture_map.cpp,
// scheduler.cpp, unet.cpp, vae.cpp, clip_vision.cpp); this unit owns the
// context lifecycle, backend lease, stage orchestration and GLB export.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "aicore/pipeline_timing.h"

namespace instantmesh {

struct PipelineConfig {
    std::string cond_gguf;         // Zero123++ CLIP-Vision + constants (opt)
    std::string unet_gguf;         // Zero123++ UNet (opt)
    std::string vae_gguf;          // Zero123++ VAE (opt)
    std::string dino_gguf;         // DINO ViT-B/16 (required)
    std::string transformer_gguf;  // TriplaneTransformer (required)
    std::string synthesizer_gguf;  // OSGDecoder MLPs (required)
    std::string rmbg_gguf;         // RMBG-2.0 matting (optional)

    std::string device = "auto";
    int threads = 0;
    int steps = 75;          // Zero123++ denoising steps
    uint64_t seed = 42;      // diffusion noise seed
    float guidance = 4.0f;   // classifier-free guidance
    int grid_res = 64;       // FlexiCubes cubes per axis
    float grid_scale = 2.1f; // voxel-grid world scale
    int texture_res = 2048;  // UV atlas bake resolution
    bool bake_texture = true;
};

struct GenParams {
    int steps = 0;          // <= 0 -> config default
    uint64_t seed = 0;      // Explicit seed (0 is valid); C ABI callers resolve defaults
    float guidance = 0.0f;  // <= 0 -> config default
    int grid_res = 0;
    float grid_scale = 0.0f;
    int texture_res = 0;
    int bake_texture = -1;  // < 0 -> config default
};

struct PipelineMesh {
    std::vector<float> verts;    // [nv,3]
    std::vector<int32_t> faces;  // [nf,3]
    std::vector<float> vcolors;  // [nv,3] RGB in [0,1]
    std::vector<float> uvs;      // [3*nf,2] per-corner UVs (textured path)
    std::vector<uint8_t> tex_rgba; // [tex_w*tex_h*4] row 0 = top
    int tex_w = 0;
    int tex_h = 0;
};

using ProgressFn = void (*)(void* user, int stage, int step, int total);

class PipelineContext {
public:
    PipelineContext() = default;
    ~PipelineContext();
    PipelineContext(const PipelineContext &) = delete;
    PipelineContext &operator=(const PipelineContext &) = delete;

    bool load(const PipelineConfig &cfg, std::string *error);
    bool is_ready() const { return ready_; }
    bool has_multiview() const { return has_multiview_; }
    const std::string &backend_name() const { return backend_name_; }
    const PipelineConfig &config() const { return cfg_; }
    const std::string &last_error() const { return last_error_; }

    // rgba: borrowed, tightly packed RGBA8 (row_stride_bytes == 0 -> w*4).
    bool generate(const uint8_t *rgba, int w, int h, int row_stride_bytes,
                  const GenParams &params, ProgressFn progress, void *user,
                  PipelineMesh *out, aicore_pipeline_timings *timings,
                  std::string *error);

private:
    // Zero123++ multi-view diffusion (defined in the .cpp; needs the full
    // model types, so it lives on the .cpp-private Impl).
    struct Impl;
    Impl *impl_ = nullptr;
    PipelineConfig cfg_;
    bool ready_ = false;
    bool has_multiview_ = false;
    std::string backend_name_ = "CPU";
    std::string last_error_;
};

// Decode-free preprocessing helper shared with the ABI layer: bicubic
// shortest-edge resize + center crop of a packed RGB8 buffer into out_rgb
// (out_size*out_size*3). Returns false on invalid input.
bool preprocess_rgb(const uint8_t *rgb, int w, int h, int out_size,
                    unsigned char *out_rgb);

// glTF 2.0 binary export: POSITION, indices, optional per-corner UVs and RGBA texture.
// Split corner vertices for UV seams; convert v to glTF image conventions, retaining top-first PNG rows.
// Returns a malloc-allocated GLB buffer.
uint8_t *bake_glb_buffer(const float *verts, int n_verts, const int *tris,
                         int n_tris, const float *uvs,
                         const uint8_t *tex_rgba, int tex_w, int tex_h,
                         int *out_len, std::string *error);

} // namespace instantmesh

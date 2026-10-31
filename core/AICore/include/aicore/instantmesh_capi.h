// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// InstantMesh image-to-3D C API. In-tree port of
// https://github.com/Asher-1/InstantMesh-GGML (cpp_ggml, MIT).
//
// The engine lives in core/AICore/src/tasks/instantmesh/ and runs on the
// shared ggml runtime. AI background removal is delegated to the in-tree RMBG
// task (aicore_rmbg_*), so only one copy of the BiRefNet/RMBG-2.0 engine
// exists per process.
//
// Pipeline stages:
//   image -> (optional RMBG matting) -> Zero123++ multi-view diffusion
//   (VAE cond encode + CLIP-Vision embeds + UNet RefOnly double-forward,
//   EulerAncestral trailing v_prediction) -> 6-view grid
//   -> DINO ViT-B/16 (adaLN camera modulation) -> TriplaneTransformer
//   -> OSGDecoder (sdf / deformation / flexi-cubes weights)
//   -> FlexiCubes isosurface extraction -> (optional texture bake)
//
// Model selection (see aicore_instantmesh_model_paths):
//   - cond + unet + vae + dino + transformer + synthesizer  -> full pipeline
//   - dino + transformer + synthesizer only                 -> the input image
//     is replicated 6x as the multi-view condition (the upstream benchmark
//     path; geometry quality is lower than the diffused multi-views).

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"
#include "aicore/image_view.h"
#include "aicore/pipeline_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AICORE_INSTANTMESH_ABI_VERSION 2u

/** Returns the ABI version. Version 2 adds size and version checks to the
 * unreleased parameter structs. */
AICORE_CAPI int aicore_instantmesh_abi_version(void);

typedef struct aicore_instantmesh_ctx aicore_instantmesh_ctx;
typedef struct aicore_instantmesh_options aicore_instantmesh_options;
typedef struct aicore_instantmesh_mesh aicore_instantmesh_mesh;

/** Pipeline stages reported by the progress callback. */
enum aicore_instantmesh_stage {
    AICORE_INSTANTMESH_STAGE_PREPROCESS = 0, /* decode + (RMBG matting)     */
    AICORE_INSTANTMESH_STAGE_RMBG = 1,       /* AI background removal       */
    AICORE_INSTANTMESH_STAGE_MULTIVIEW = 2,  /* Zero123++ diffusion (steps) */
    AICORE_INSTANTMESH_STAGE_VAE_DECODE = 3, /* latent -> 6-view grid       */
    AICORE_INSTANTMESH_STAGE_ENCODE = 4,     /* DINO conditioning encoder   */
    AICORE_INSTANTMESH_STAGE_TRIPLANE = 5,   /* TriplaneTransformer          */
    AICORE_INSTANTMESH_STAGE_SYNTH = 6,      /* OSGDecoder fields            */
    AICORE_INSTANTMESH_STAGE_MESH = 7,       /* FlexiCubes extraction        */
    AICORE_INSTANTMESH_STAGE_TEXTURE = 8     /* UV unwrap + net_rgb bake     */
};

/** step/total are meaningful for the MULTIVIEW stage; other stages send 0/0
 *  at entry. Called from the generating thread. */
typedef void (*aicore_instantmesh_progress_fn)(void* user,
                                               int stage,
                                               int step,
                                               int total);

/** GGUF paths for the pipeline. dino/transformer/synthesizer are required;
 *  the Zero123++ trio (cond/unet/vae) is optional — when all three are NULL
 *  or "" the input image is replicated as the 6-view condition instead of
 *  running the multi-view diffusion. */
typedef struct aicore_instantmesh_model_paths {
    uint32_t
            struct_size; /**< At least sizeof(aicore_instantmesh_model_paths) */
    uint32_t abi_version;         /**< AICORE_INSTANTMESH_ABI_VERSION */
    const char* cond_gguf;        /**< Zero123++ CLIP-Vision + constants  */
    const char* unet_gguf;        /**< Zero123++ UNet                     */
    const char* vae_gguf;         /**< Zero123++ VAE (encode + decode)    */
    const char* dino_gguf;        /**< DINO ViT-B/16 (required)           */
    const char* transformer_gguf; /**< TriplaneTransformer (required)     */
    const char* synthesizer_gguf; /**< OSGDecoder MLPs (required)         */
} aicore_instantmesh_model_paths;

#define AICORE_INSTANTMESH_MODEL_PATHS_INIT  \
    {sizeof(aicore_instantmesh_model_paths), \
     AICORE_INSTANTMESH_ABI_VERSION,         \
     NULL,                                   \
     NULL,                                   \
     NULL,                                   \
     NULL,                                   \
     NULL,                                   \
     NULL}

/** Creates a default options struct (device "auto", threads 0 = backend
 *  default, RMBG disabled, steps 75, seed 42, guidance 4.0, grid_res 64,
 *  grid_scale 2.1, texture 2048, texture bake on). Release with
 *  aicore_instantmesh_options_free. */
AICORE_CAPI aicore_instantmesh_options* aicore_instantmesh_options_new(void);
/** Releases an options struct created by aicore_instantmesh_options_new. */
AICORE_CAPI void aicore_instantmesh_options_free(
        aicore_instantmesh_options* opts);
/** Selects the inference device: NULL or "auto", "cpu", "gpu", "vulkan"
 *  (optionally ":N"), "cuda" (Linux/Windows). */
AICORE_CAPI void aicore_instantmesh_options_set_device(
        aicore_instantmesh_options* opts, const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_instantmesh_options_set_threads(
        aicore_instantmesh_options* opts, int n_threads);
/** Attach an RMBG-2.0 GGUF for AI background removal before the Zero123++
 *  input preprocessing. NULL or "" disables it (the input alpha or an opaque
 *  frame is used as-is). */
AICORE_CAPI void aicore_instantmesh_options_set_rmbg_gguf(
        aicore_instantmesh_options* opts, const char* rmbg_gguf);
/** Zero123++ denoising steps (default 75, valid range [2, 1000]). */
AICORE_CAPI void aicore_instantmesh_options_set_steps(
        aicore_instantmesh_options* opts, int steps);
/** Zero123++ seed for the initial/ancestral noise (default 42). */
AICORE_CAPI void aicore_instantmesh_options_set_seed(
        aicore_instantmesh_options* opts, uint64_t seed);
/** Zero123++ classifier-free guidance scale (default 4.0). */
AICORE_CAPI void aicore_instantmesh_options_set_guidance(
        aicore_instantmesh_options* opts, float guidance);
/** FlexiCubes voxel-grid resolution per axis (default 64; upstream benchmark
 *  uses 88 for the highest quality). */
AICORE_CAPI void aicore_instantmesh_options_set_grid_res(
        aicore_instantmesh_options* opts, int grid_res);
/** FlexiCubes grid scale (default 2.1). */
AICORE_CAPI void aicore_instantmesh_options_set_grid_scale(
        aicore_instantmesh_options* opts, float grid_scale);
/** Texture atlas resolution for the bake (default 2048; higher = less bake
 *  aliasing). */
AICORE_CAPI void aicore_instantmesh_options_set_texture_res(
        aicore_instantmesh_options* opts, int texture_res);
/** 0 = skip the UV texture bake (vertex colors only), 1 = bake an RGBA
 *  texture atlas (default). */
AICORE_CAPI void aicore_instantmesh_options_set_bake_texture(
        aicore_instantmesh_options* opts, int bake_texture);

/** Returns NULL on load failure; the compatibility wrapper delegates to
 * load_opts_ex and logs diagnostics. */
AICORE_CAPI aicore_instantmesh_ctx* aicore_instantmesh_load_opts(
        const aicore_instantmesh_model_paths* paths,
        const aicore_instantmesh_options* opts);
/** Loads a pipeline; errors are written to err (NULL allowed), always
 * NUL-terminated. Zero123++ paths must all be supplied or all omitted; partial
 * sets are rejected. */
AICORE_CAPI aicore_instantmesh_ctx* aicore_instantmesh_load_opts_ex(
        const aicore_instantmesh_model_paths* paths,
        const aicore_instantmesh_options* opts,
        char* err,
        int err_len);
/** Releases a context returned by aicore_instantmesh_load_opts; safe on
 *  NULL. */
AICORE_CAPI void aicore_instantmesh_free(aicore_instantmesh_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded pipeline. */
AICORE_CAPI int aicore_instantmesh_is_ready(const aicore_instantmesh_ctx* ctx);
/** Returns the last error message of the context (empty when none). */
AICORE_CAPI const char* aicore_instantmesh_last_error(
        const aicore_instantmesh_ctx* ctx);
/** 1 when the Zero123++ multi-view models are loaded; 0 = replicate mode. */
AICORE_CAPI int aicore_instantmesh_has_multiview(
        const aicore_instantmesh_ctx* ctx);
/** Name of the compute backend the pipeline was loaded onto. */
AICORE_CAPI const char* aicore_instantmesh_backend(
        const aicore_instantmesh_ctx* ctx);
/** Releases any buffer returned by an aicore_instantmesh_* function (unified
 *  entry point). Safe on NULL. */
AICORE_CAPI void aicore_instantmesh_free_buffer(void* p);

/** Per-generation parameters. params=NULL uses context options; seed (including
 * 0) is always explicit. Other numeric fields <=0 use context defaults;
 * bake_texture=-1 uses the default. Valid ranges: steps [2,1000], guidance
 * (0,15], grid_res [2,256], finite grid_scale >0, texture_res [1,4096]. */
typedef struct aicore_instantmesh_generate_params {
    uint32_t struct_size; /**< At least
                             sizeof(aicore_instantmesh_generate_params) */
    uint32_t abi_version; /**< AICORE_INSTANTMESH_ABI_VERSION */
    int steps;            /**< <= 0 -> options value */
    uint64_t seed;    /**< Explicit noise seed; 0 is valid and never replaced by
                         the context default */
    float guidance;   /**< <= 0 -> options value */
    int grid_res;     /**< <= 0 -> options value */
    float grid_scale; /**< <= 0 -> options value */
    int texture_res;  /**< <= 0 -> options value */
    int bake_texture; /**< < 0 -> options value */
} aicore_instantmesh_generate_params;

#define AICORE_INSTANTMESH_GENERATE_PARAMS_INIT  \
    {sizeof(aicore_instantmesh_generate_params), \
     AICORE_INSTANTMESH_ABI_VERSION,             \
     0,                                          \
     42,                                         \
     0.0f,                                       \
     0,                                          \
     0.0f,                                       \
     0,                                          \
     -1}

/** Run image-to-3D generation. image_bytes must be an encoded image
 *  (PNG/JPEG/WebP/BMP/GIF/...). On success returns a mesh handle (free with
 *  aicore_instantmesh_mesh_free). The pipeline is NOT thread-safe: serialize
 *  calls per context. */
AICORE_CAPI aicore_instantmesh_mesh* aicore_instantmesh_generate(
        aicore_instantmesh_ctx* ctx,
        const void* image_bytes,
        int image_len,
        const aicore_instantmesh_generate_params* params,
        aicore_instantmesh_progress_fn progress,
        void* progress_user,
        char* err,
        int err_len);
/** Decoded RGB/RGBA/GRAY/BGR/BGRA input; pixels are borrowed read-only.
 *  row_stride_bytes must be the actual stride, at least the active row width; 0
 * is rejected. Input is limited to 16 Mi pixels; RGBA/BGRA retain alpha until
 * compositing over white. */
AICORE_CAPI aicore_instantmesh_mesh* aicore_instantmesh_generate_image_view(
        aicore_instantmesh_ctx* ctx,
        const aicore_image_view* image,
        const aicore_instantmesh_generate_params* params,
        aicore_instantmesh_progress_fn progress,
        void* progress_user,
        char* err,
        int err_len);
/** Copies the common stage timings of the last generate into *out. Returns 0
 *  on success. */
AICORE_CAPI int aicore_instantmesh_last_pipeline_timings(
        const aicore_instantmesh_ctx* ctx, aicore_pipeline_timings* out);

/** Mesh accessors. Vertices are in a centered, roughly unit-sized box
 *  ([-grid_scale/2, grid_scale/2]^3 scaled by grid_scale); faces are
 *  0-based triangles. For the textured path uv/texture carry the xatlas UV
 *  atlas and the baked RGBA texture (row 0 = top, v=1). Buffers stay valid
 *  until aicore_instantmesh_mesh_free. */
AICORE_CAPI int aicore_instantmesh_mesh_n_verts(
        const aicore_instantmesh_mesh* r);
AICORE_CAPI int aicore_instantmesh_mesh_n_tris(
        const aicore_instantmesh_mesh* r);
AICORE_CAPI const float* aicore_instantmesh_mesh_verts(
        const aicore_instantmesh_mesh* r);
AICORE_CAPI const int* aicore_instantmesh_mesh_tris(
        const aicore_instantmesh_mesh* r);
/** Per-vertex RGB in [0,1] (3*n_verts), sampled from net_rgb at the final
 *  vertex positions. Always present. */
AICORE_CAPI const float* aicore_instantmesh_mesh_vcolors(
        const aicore_instantmesh_mesh* r);
/** Textured path: per-corner UVs (3*n_tris * 2) in [0,1]^2, or NULL. */
AICORE_CAPI const float* aicore_instantmesh_mesh_uvs(
        const aicore_instantmesh_mesh* r);
/** Textured path: baked RGBA atlas (tex_w*tex_h*4), or NULL. */
AICORE_CAPI const uint8_t* aicore_instantmesh_mesh_texture(
        const aicore_instantmesh_mesh* r);
AICORE_CAPI int aicore_instantmesh_mesh_texture_w(
        const aicore_instantmesh_mesh* r);
AICORE_CAPI int aicore_instantmesh_mesh_texture_h(
        const aicore_instantmesh_mesh* r);
AICORE_CAPI void aicore_instantmesh_mesh_free(aicore_instantmesh_mesh* r);

/** Bake a mesh into a portable UV-atlas-textured GLB (glTF 2.0 binary).
 *  verts/tris/uvs/texture use the same layouts as the mesh accessors
 *  (texture_rgba row 0 = top). texture may be NULL for an untextured bake.
 *  On success returns a malloc'd GLB buffer (free with
 *  aicore_instantmesh_free_buffer) and writes its length into *out_len. */
AICORE_CAPI uint8_t* aicore_instantmesh_bake_glb(const float* verts,
                                                 int n_verts,
                                                 const int* tris,
                                                 int n_tris,
                                                 const float* uvs,
                                                 const uint8_t* texture_rgba,
                                                 int tex_w,
                                                 int tex_h,
                                                 int* out_len,
                                                 char* err,
                                                 int err_len);

/** Image decode + preprocessing only (no models): decodes image_bytes,
 *  composites alpha over white, and resizes the shortest edge to out_size
 *  with a center crop. out_rgb must hold out_size*out_size*3 bytes. Returns
 *  0 on success. */
AICORE_CAPI int aicore_instantmesh_preprocess_image_bytes(
        const void* image_bytes,
        int image_len,
        int out_size,
        unsigned char* out_rgb,
        char* err,
        int err_len);

/** Model-free decoded-image preprocessing with the same format, stride and
 * alpha contract as generation. out_size is in [1,4096]; the caller provides
 * out_size*out_size*3 bytes in out_rgb. Returns 0 on success; errors are
 * written to err. */
AICORE_CAPI int aicore_instantmesh_preprocess_image_view(
        const aicore_image_view* image,
        int out_size,
        unsigned char* out_rgb,
        char* err,
        int err_len);

/** Warms up the backend for `device`; returns 0 on success. */
AICORE_CAPI int aicore_instantmesh_warmup_backend(const char* device);
/** Releases process-wide InstantMesh backend resources (idempotent). */
AICORE_CAPI void aicore_instantmesh_shutdown(void);
/** Returns the local model cache directory. Caller frees with
 *  aicore_instantmesh_free_buffer. */
AICORE_CAPI char* aicore_instantmesh_model_cache_dir(void);
/** Returns a JSON summary of the loaded pipeline. Caller frees with
 *  aicore_instantmesh_free_buffer. */
AICORE_CAPI char* aicore_instantmesh_info_json(aicore_instantmesh_ctx* ctx);

/** Published GGUF catalog (Asher-1/InstantMesh-GUFF Hugging Face repository).
 *  The URL, size and SHA-256 identify the same immutable LFS object used by
 *  AICore's regression downloader and the qInstantMesh plugin. RMBG models
 *  are intentionally NOT part of this catalog: they are owned by the in-tree
 *  RMBG task catalog (same LFS objects, single owner). */
typedef struct aicore_instantmesh_model_entry {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
    /** Role tag: "cond", "unet", "vae", "dino", "transformer",
     *  "synthesizer". */
    const char* role;
    uint64_t size_bytes;
    const char* sha256;
} aicore_instantmesh_model_entry;

/** Number of published catalog entries. */
AICORE_CAPI int aicore_instantmesh_model_count(void);
/** Returns the catalog entry at `index` (NULL when out of range). */
AICORE_CAPI const aicore_instantmesh_model_entry* aicore_instantmesh_model_at(
        int index);
/** Returns the catalog entry whose filename matches (NULL when not found). */
AICORE_CAPI const aicore_instantmesh_model_entry*
aicore_instantmesh_model_by_filename(const char* filename);
/** Returns the base URL of the published model repository. */
AICORE_CAPI const char* aicore_instantmesh_model_download_base(void);

#ifdef __cplusplus
}
#endif

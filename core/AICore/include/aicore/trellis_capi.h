// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// TRELLIS.2 image-to-3D C API. In-tree port of
// https://github.com/Asher-1/trellis-ggml (trellis2.cpp, MIT).
//
// The engine lives in core/AICore/src/tasks/trellis/ and is accelerated by the
// shared ggml runtime (3rdparty_ggml v0.18.1). AI background removal is
// delegated to the in-tree RMBG task (aicore_rmbg_*), so only one copy of the
// BiRefNet/RMBG-2.0 engine exists per process.
//
// Pipeline stages:
//   image -> (optional RMBG matting) -> preprocess -> DINOv3 encode
//   -> sparse-structure flow (SS_FLOW) -> occupancy decode (SS_DEC, 64^3)
//   -> shape-SLAT flow -> shape decode (512^3 / 1024^3 dual grid)
//   -> mesh extraction -> (optional PBR texture stage)
//
// Model selection (see aicore_trellis_model_paths):
//   - dino + ss_flow + ss_dec                       -> coarse 64^3 preview
//   - + slat_flow + shape_dec                       -> 512 fine dual-grid
//   - + slat_hr_flow                                -> 1024 cascade
//   - + shape_enc + tex_dec + tex_flow (+hr)        -> PBR texturing

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the ABI version of the TRELLIS C API (bump on breaking ABI
 *  changes). */
AICORE_CAPI int aicore_trellis_abi_version(void);

typedef struct aicore_trellis_ctx aicore_trellis_ctx;
typedef struct aicore_trellis_options aicore_trellis_options;
typedef struct aicore_trellis_mesh aicore_trellis_mesh;

/** Pipeline stages reported by the progress callback. */
enum aicore_trellis_stage {
    AICORE_TRELLIS_STAGE_PREPROCESS =
            0,                     /* image decode + crop/premultiply/resize */
    AICORE_TRELLIS_STAGE_DINO = 1, /* conditioning encoder                   */
    AICORE_TRELLIS_STAGE_SS_FLOW =
            2, /* sparse-structure flow sampling (steps) */
    AICORE_TRELLIS_STAGE_SS_DEC = 3, /* occupancy decoder -> voxel scaffold */
    AICORE_TRELLIS_STAGE_SLAT_FLOW = 4, /* shape-SLAT flow sampling (steps) */
    AICORE_TRELLIS_STAGE_SHAPE_DEC = 5, /* shape decoder -> dual-grid fields */
    AICORE_TRELLIS_STAGE_MESH = 6, /* mesh extraction                        */
    AICORE_TRELLIS_STAGE_UPSAMPLE =
            7, /* cascade: LR slat -> HR voxel scaffold  */
    AICORE_TRELLIS_STAGE_SLAT_FLOW_HR =
            8, /* cascade: 1024 shape-SLAT flow (steps)  */
    AICORE_TRELLIS_STAGE_SHAPE_DEC_HR = 9, /* cascade: 1024^3 shape decoder */
    AICORE_TRELLIS_STAGE_TEXTURE = 10 /* PBR texture: flow + guided decode */
};

/** Pipeline type for aicore_trellis_generate. */
enum aicore_trellis_pipeline_type {
    AICORE_TRELLIS_PIPE_AUTO =
            0, /* cascade if available, else 512 fine, else coarse */
    AICORE_TRELLIS_PIPE_COARSE =
            1, /* 64^3 occupancy -> marching cubes preview        */
    AICORE_TRELLIS_PIPE_512 = 2, /* 512 fine dual-grid */
    AICORE_TRELLIS_PIPE_1024 = 3 /* 1024 cascade */
};

/** Solid-background handling before the alpha-bbox crop. */
enum aicore_trellis_background_mode {
    AICORE_TRELLIS_BG_AUTO =
            0, /* detect border-connected near-black/near-white */
    AICORE_TRELLIS_BG_KEEP = 1, /* preserve the decoded alpha exactly */
    AICORE_TRELLIS_BG_BLACK =
            2, /* force removal of border-connected near-black   */
    AICORE_TRELLIS_BG_WHITE =
            3 /* force removal of border-connected near-white   */
};

/** Capability bits reported by aicore_trellis_caps (loaded qualities). */
enum aicore_trellis_caps {
    AICORE_TRELLIS_CAP_COARSE = 1,
    AICORE_TRELLIS_CAP_512 = 2,
    AICORE_TRELLIS_CAP_1024 = 4,
    AICORE_TRELLIS_CAP_TEXTURE = 8
};

/** step/total are meaningful for the *_FLOW stages; other stages send 0/0 at
 *  entry. Called from the generating thread. */
typedef void (*aicore_trellis_progress_fn)(void* user,
                                           int stage,
                                           int step,
                                           int total);

/** GGUF paths for the pipeline. dino/ss_flow/ss_dec are required; the others
 *  select the available qualities (see file header). Any field may be NULL or
 *  "" to omit that model. */
typedef struct aicore_trellis_model_paths {
    const char* dino_gguf;      /**< DINOv3 conditioning encoder (required) */
    const char* ss_flow_gguf;   /**< sparse-structure flow DiT (required)  */
    const char* ss_dec_gguf;    /**< occupancy decoder (required)          */
    const char* slat_flow_gguf; /**< shape-SLAT flow 512 (optional)        */
    const char* slat_hr_flow_gguf; /**< shape-SLAT flow 1024 cascade (opt.)   */
    const char* shape_dec_gguf;    /**< shape decoder (optional)              */
    const char* shape_enc_gguf;    /**< PBR: shape encoder (optional)         */
    const char* tex_dec_gguf;      /**< PBR: texture decoder (optional)       */
    const char* tex_flow_gguf;     /**< PBR: texture flow 512 (optional)      */
    const char* tex_flow_hr_gguf;  /**< PBR: texture flow 1024 (optional)     */
} aicore_trellis_model_paths;

/** Creates a default options struct (device "auto", threads 0 = backend
 *  default, RMBG disabled). Release with aicore_trellis_options_free. */
AICORE_CAPI aicore_trellis_options* aicore_trellis_options_new(void);
/** Releases an options struct created by aicore_trellis_options_new. */
AICORE_CAPI void aicore_trellis_options_free(aicore_trellis_options* opts);
/** Selects the inference device: NULL or "auto", "cpu", "gpu", "vulkan"
 *  (optionally ":N"), "cuda" (Linux/Windows). */
AICORE_CAPI void aicore_trellis_options_set_device(aicore_trellis_options* opts,
                                                   const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_trellis_options_set_threads(
        aicore_trellis_options* opts, int n_threads);
/** Attach an RMBG-2.0 GGUF for AI background removal before the alpha-bbox
 *  crop. NULL or "" disables it (solid-color heuristic only). */
AICORE_CAPI void aicore_trellis_options_set_rmbg_gguf(
        aicore_trellis_options* opts, const char* rmbg_gguf);
/** Shape-decoder placement: "auto" (GPU when the card can hold the decode
 *  once the flow DiTs are freed, else CPU), "gpu", "cpu". Replaces the
 *  TRELLIS2_SHAPE_DEC_GPU / TRELLIS2_SHAPE_DEC_CPU environment variables of
 *  the upstream port. */
AICORE_CAPI void aicore_trellis_options_set_shape_dec_placement(
        aicore_trellis_options* opts, const char* placement);
/** Force the materialized (non-flash) scaled-dot-product attention path
 *  (debug; default 0). Replaces TRELLIS2_SDPA_EXACT upstream. */
AICORE_CAPI void aicore_trellis_options_set_sdpa_exact(
        aicore_trellis_options* opts, int exact);
/** Enable per-forward stage timing logs (debug; default 0). Replaces
 *  TRELLIS2_TIMING upstream. */
AICORE_CAPI void aicore_trellis_options_set_timing(aicore_trellis_options* opts,
                                                   int enabled);

/** Load the TRELLIS.2 pipeline. On failure returns NULL and the context's
 *  last error (or err) carries the reason. */
AICORE_CAPI aicore_trellis_ctx* aicore_trellis_load_opts(
        const aicore_trellis_model_paths* paths,
        const aicore_trellis_options* opts);
/** Releases a context returned by aicore_trellis_load_opts; safe on NULL. */
AICORE_CAPI void aicore_trellis_free(aicore_trellis_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded pipeline. */
AICORE_CAPI int aicore_trellis_is_ready(const aicore_trellis_ctx* ctx);
/** Returns the last error message of the context (empty when none). */
AICORE_CAPI const char* aicore_trellis_last_error(
        const aicore_trellis_ctx* ctx);
/** Bitmask of aicore_trellis_caps this pipeline can produce. */
AICORE_CAPI int aicore_trellis_caps(const aicore_trellis_ctx* ctx);
/** Name of the compute backend the pipeline was loaded onto. */
AICORE_CAPI const char* aicore_trellis_backend(const aicore_trellis_ctx* ctx);
/** Releases any buffer returned by an aicore_trellis_* function (unified
 *  entry point). Safe on NULL. */
AICORE_CAPI void aicore_trellis_free_buffer(void* p);

/** Per-generate parameters. 0 / negative values select the pipeline defaults
 *  (steps 12, guidance 7.5, texture_steps 12). */
typedef struct aicore_trellis_generate_params {
    int pipeline_type;   /**< aicore_trellis_pipeline_type */
    int background_mode; /**< aicore_trellis_background_mode */
    uint64_t seed;
    int steps;         /**< <= 0 -> 12 */
    float guidance;    /**< < 0 -> 7.5 */
    int texture_steps; /**< <= 0 -> 12 */
} aicore_trellis_generate_params;

/** Run image-to-3D generation. image_bytes must be an encoded image
 *  (PNG/JPEG/WebP/BMP/GIF/...). On success returns a mesh handle (free with
 *  aicore_trellis_mesh_free). The pipeline is NOT thread-safe: serialize
 *  calls per context. */
AICORE_CAPI aicore_trellis_mesh* aicore_trellis_generate(
        aicore_trellis_ctx* ctx,
        const void* image_bytes,
        int image_len,
        const aicore_trellis_generate_params* params,
        aicore_trellis_progress_fn progress,
        void* progress_user,
        char* err,
        int err_len);

/** Mesh accessors. Vertices are in a centered unit cube ([-0.5, 0.5]^3);
 *  normals are per-vertex unit vectors. Buffers stay valid until
 *  aicore_trellis_mesh_free. */
AICORE_CAPI int aicore_trellis_mesh_n_verts(const aicore_trellis_mesh* r);
AICORE_CAPI int aicore_trellis_mesh_n_tris(const aicore_trellis_mesh* r);
AICORE_CAPI const float* aicore_trellis_mesh_verts(
        const aicore_trellis_mesh* r);
AICORE_CAPI const float* aicore_trellis_mesh_normals(
        const aicore_trellis_mesh* r);
AICORE_CAPI const int* aicore_trellis_mesh_tris(const aicore_trellis_mesh* r);
/** Per-vertex PBR (6*n_verts: base_color rgb, metallic, roughness, alpha), or
 *  NULL when the mesh is untextured. */
AICORE_CAPI int aicore_trellis_mesh_has_pbr(const aicore_trellis_mesh* r);
AICORE_CAPI const float* aicore_trellis_mesh_pbr(const aicore_trellis_mesh* r);
/** Dual-grid sidecar from the fine path (empty on coarse). */
AICORE_CAPI int aicore_trellis_mesh_grid_res(const aicore_trellis_mesh* r);
AICORE_CAPI int aicore_trellis_mesh_grid_nvox(const aicore_trellis_mesh* r);
AICORE_CAPI const float* aicore_trellis_mesh_grid_feats(
        const aicore_trellis_mesh* r);
AICORE_CAPI const int* aicore_trellis_mesh_grid_coords(
        const aicore_trellis_mesh* r);
/** AI background-removal result (RGBA, decoded-input resolution), present only
 *  when an RMBG model was attached to the context and actually ran. Buffers
 *  stay valid until aicore_trellis_mesh_free. */
AICORE_CAPI int aicore_trellis_mesh_has_rmbg(const aicore_trellis_mesh* r);
AICORE_CAPI const uint8_t* aicore_trellis_mesh_rmbg_rgba(
        const aicore_trellis_mesh* r);
AICORE_CAPI int aicore_trellis_mesh_rmbg_w(const aicore_trellis_mesh* r);
AICORE_CAPI int aicore_trellis_mesh_rmbg_h(const aicore_trellis_mesh* r);
AICORE_CAPI void aicore_trellis_mesh_free(aicore_trellis_mesh* r);

/** Bake a mesh into a portable UV-atlas-textured GLB (glTF 2.0 binary):
 *  optional component cleanup -> UV unwrap -> per-texel PBR bake -> gutter
 *  inpaint -> glTF. pbr is 6*n_verts (base_color rgb, metallic, roughness,
 *  alpha) or NULL for an untextured grey bake. texture_size is a square atlas
 *  hint (2048 default; 0 -> default). component_filter: 0 remove tiny
 *  islands, 1 largest only, 2 keep all. On success returns a malloc'd GLB
 *  buffer (free with aicore_trellis_free_buffer) and writes its length into
 *  *out_len. */
AICORE_CAPI uint8_t* aicore_trellis_bake_glb(const float* verts,
                                             int n_verts,
                                             const int* tris,
                                             int n_tris,
                                             const float* pbr,
                                             int texture_size,
                                             int component_filter,
                                             int* out_len,
                                             char* err,
                                             int err_len);

/** Image decode + TRELLIS.2 preprocessing only (no models). out_rgb must hold
 *  out_size*out_size*3 bytes. Returns 0 on success. */
AICORE_CAPI int aicore_trellis_preprocess_image_bytes(const void* image_bytes,
                                                      int image_len,
                                                      int out_size,
                                                      unsigned char* out_rgb,
                                                      int background_mode,
                                                      char* err,
                                                      int err_len);

/** Warms up the backend for `device`; returns 0 on success. */
AICORE_CAPI int aicore_trellis_warmup_backend(const char* device);
/** Releases process-wide TRELLIS backend resources (idempotent). */
AICORE_CAPI void aicore_trellis_shutdown(void);
/** Returns the local model cache directory. Caller frees with
 *  aicore_trellis_free_buffer. */
AICORE_CAPI char* aicore_trellis_model_cache_dir(void);
/** Returns a JSON summary of the loaded pipeline. Caller frees with
 *  aicore_trellis_free_buffer. */
AICORE_CAPI char* aicore_trellis_info_json(aicore_trellis_ctx* ctx);

/** Published GGUF catalog (cloudViewer_downloads trellis2-ggml release). */
typedef struct aicore_trellis_model_entry {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
    /** Role tag: "dino", "ss_flow", "ss_dec", "slat_flow", "slat_flow_hr",
     *  "shape_dec", "shape_enc", "tex_dec", "tex_flow", "tex_flow_hr",
     *  "rmbg". */
    const char* role;
} aicore_trellis_model_entry;

/** Number of published catalog entries. */
AICORE_CAPI int aicore_trellis_model_count(void);
/** Returns the catalog entry at `index` (NULL when out of range). */
AICORE_CAPI const aicore_trellis_model_entry* aicore_trellis_model_at(
        int index);
/** Returns the catalog entry whose filename matches (NULL when not found). */
AICORE_CAPI const aicore_trellis_model_entry* aicore_trellis_model_by_filename(
        const char* filename);
/** Returns the base URL of the published model release. */
AICORE_CAPI const char* aicore_trellis_model_download_base(void);

#ifdef __cplusplus
}
#endif

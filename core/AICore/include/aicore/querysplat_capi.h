// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// QuerySplat C API: multi-view image-to-3DGS prediction over the in-tree
// ggml port in core/AICore/src/tasks/querysplat/ (upstream querysplat-ggml
// cpp_ggml). The chain delegates the VGGT-Omega encoder to the in-tree
// map_anything task (same builder, same published GGUFs) and runs this
// project's decoder graph + host activation heads + the gsplat-semantics
// CPU rasterizer + optional TTO (test-time optimization of the decoder KV
// memory against the input views' own pixels).
//
// One inference yields per-view camera outputs (pose_enc 9 floats, depth +
// confidence from the encoder's VGGT heads) and the scene's 3D-Gaussian set
// (N x stride: pos(3) | opacity | scale(3) | rot(4) | color(sh0: 3)).
// Rendering joins through the CPU rasterizer; PLY / cameras-JSON / depth
// point-cloud exports are compatibility boundaries on top of the typed
// buffers.
//
// Thread-safety: a context is single-threaded. Use one context per consumer,
// or serialize access externally.
//
// GGUF models: decoder family + LPIPS metric weights published at
//   https://huggingface.co/Asher-1/querysplat_vggto
// encoder (VGGT-Omega 1B/512) shared with map_anything at
//   https://huggingface.co/Asher-1/map_anything_vggt
// (see the model catalog entry points below and aicore/asset_digests.h).

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"
#include "aicore/image_view.h"
#include "aicore/pipeline_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the ABI version of the QuerySplat C API (bump on breaking ABI
 *  changes). */
AICORE_CAPI int aicore_querysplat_abi_version(void);

typedef struct aicore_querysplat_ctx aicore_querysplat_ctx;
typedef struct aicore_querysplat_options aicore_querysplat_options;

/** Creates a default options struct (device "auto", threads 0 = backend
 *  default, f32-accuracy ON = torch parity, image size 0 = decoder GGUF
 *  metadata resolution, TTO off, opacity threshold 0.05, grey background).
 *  Release with aicore_querysplat_options_free. */
AICORE_CAPI aicore_querysplat_options* aicore_querysplat_options_new(void);
/** Releases an options struct created by aicore_querysplat_options_new. */
AICORE_CAPI void aicore_querysplat_options_free(
        aicore_querysplat_options* opts);
/** Selects the inference device: NULL or "auto", "cpu", "gpu", "vulkan"
 *  (optionally ":N"), "cuda" (Linux/Windows), "metal" (macOS). */
AICORE_CAPI void aicore_querysplat_options_set_device(
        aicore_querysplat_options* opts, const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_querysplat_options_set_threads(
        aicore_querysplat_options* opts, int n_threads);
/** GPU f32-accuracy profile (default 1, torch parity): exact-F32 attention
 *  on CUDA, dequantized legacy-quant Linear weights on CPU/CUDA, and the
 *  Vulkan f32-accuracy instance profile. 0 opts into the fast (drift-prone)
 *  kernels; the choice is reported through the info JSON. */
AICORE_CAPI void aicore_querysplat_options_set_f32_accuracy(
        aicore_querysplat_options* opts, int enable);
/** Preprocess target resolution before inference. 0,0 (the default) uses
 *  the resolution stored in the decoder GGUF metadata (512x512 for the
 *  paper checkpoints; aicore_querysplat_default_image_size reports it). */
AICORE_CAPI void aicore_querysplat_options_set_image_size(
        aicore_querysplat_options* opts, int height, int width);
/** Rasterizer background color (default 0.5, 0.5, 0.5 = the official
 *  "grey"). Consumed by render and TTO. */
AICORE_CAPI void aicore_querysplat_options_set_bg_color(
        aicore_querysplat_options* opts, float r, float g, float b);
/** TTO: number of optimization steps (default 20; 0 disables TTO — the
 *  feed-forward chain runs instead). */
AICORE_CAPI void aicore_querysplat_options_set_tto_steps(
        aicore_querysplat_options* opts, int steps);
/** TTO AdamW learning rate (default 5e-3). */
AICORE_CAPI void aicore_querysplat_options_set_tto_lr(
        aicore_querysplat_options* opts, double lr);
/** TTO LPIPS term weight (default 0.05; 0 skips the LPIPS metric graph). */
AICORE_CAPI void aicore_querysplat_options_set_tto_lpips_weight(
        aicore_querysplat_options* opts, double weight);
/** Gaussian PLY export opacity threshold (default 0.05; the official
 *  --gaussian_save_opacity_threshold). */
AICORE_CAPI void aicore_querysplat_options_set_opacity_threshold(
        aicore_querysplat_options* opts, float threshold);
/** Log level: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR (default 1). */
AICORE_CAPI void aicore_querysplat_options_set_log_level(
        aicore_querysplat_options* opts, int log_level);
/** Debug: dump parity stage tensors as raw f32 files into this directory
 *  after every run call (effective only in MAP_ANYTHING_ENABLE_DUMP builds;
 *  also routes the map_anything encoder dump). Empty (default) disables. */
AICORE_CAPI void aicore_querysplat_options_set_dump_dir(
        aicore_querysplat_options* opts, const char* dir);

/** Load the dual-graph chain. Either path may be NULL or empty to resolve
 *  the catalog default (encoder: map_anything's vggt-omega-1b-512-q8_0,
 *  decoder: querysplat-paper-q8_0). Returns a context whose is_ready() may
 *  still be 0 (device admission failure); inspect
 *  aicore_querysplat_last_error(). NULL only on invalid input or
 *  out-of-memory. */
AICORE_CAPI aicore_querysplat_ctx* aicore_querysplat_load_opts(
        const char* encoder_gguf,
        const char* decoder_gguf,
        const aicore_querysplat_options* opts);
/** Releases a context; safe on NULL. Contexts must not be destroyed while a
 *  worker thread is inside a run call (serialize externally). */
AICORE_CAPI void aicore_querysplat_free(aicore_querysplat_ctx* ctx);
/** Returns 1 only when the context owns successfully loaded encoder +
 *  decoder models and a usable backend. */
AICORE_CAPI int aicore_querysplat_is_ready(const aicore_querysplat_ctx* ctx);
/** Returns the last error message of the context (NULL when none). */
AICORE_CAPI const char* aicore_querysplat_last_error(
        const aicore_querysplat_ctx* ctx);
/** Releases any buffer returned by an aicore_querysplat_* function
 *  (JSON/point-cloud helpers). Safe on NULL. */
AICORE_CAPI void aicore_querysplat_free_buffer(void* p);

/** Checkpoint-nominal preprocessing resolution from the decoder GGUF
 *  metadata (512 for the paper checkpoints; 0 when not ready). */
AICORE_CAPI int32_t
aicore_querysplat_default_image_size(const aicore_querysplat_ctx* ctx);

/** ---- typed inference ----
 *
 *  Multi-view inference from DECODED images. Every view must be non-NULL
 *  with positive dimensions, a known format and row_stride_bytes >=
 *  width * channels (else -1); the official preprocessing (center square
 *  crop, antialiased resize, ImageNet normalize) runs inside. n_views >= 1;
 *  the compute cost grows quadratically with the view count (full
 *  inter-frame attention), so keep batches modest. Returns 0 on success,
 *  -1 on contract/validation errors, -2 on runtime/inference errors and -3
 *  when the device cannot hold this geometry (nothing ran, the context
 *  stays usable, so retrying with fewer views is valid). Inspect
 *  aicore_querysplat_last_error.
 *
 *  The results are owned by the context and stay valid until the next run
 *  call or aicore_querysplat_free. All accessors below are borrowed,
 *  read-only views of that store — never freed by the caller. */
AICORE_CAPI int aicore_querysplat_run_images(aicore_querysplat_ctx* ctx,
                                             const aicore_image_view* views,
                                             int32_t n_views);

/** Parity feed: preprocessed frames, n_views*3*height*width floats in
 *  [0,1], torch (n,3,H,W) order — the bit-exact --bin contract of the
 *  upstream CLI. Same result store contract as run_images. */
AICORE_CAPI int aicore_querysplat_run_frames(aicore_querysplat_ctx* ctx,
                                             const float* frames_f32,
                                             int32_t n_views,
                                             int32_t height,
                                             int32_t width);

/** Number of views of the most recent successful run (0 before the first
 *  run or after a failed call left no result). */
AICORE_CAPI int32_t
aicore_querysplat_result_view_count(const aicore_querysplat_ctx* ctx);
AICORE_CAPI int32_t
aicore_querysplat_result_height(const aicore_querysplat_ctx* ctx);
AICORE_CAPI int32_t
aicore_querysplat_result_width(const aicore_querysplat_ctx* ctx);

/** Per-view camera head output [n*9]: pose_enc = absT 3 + quaternion 4
 *  (scalar-last xyzw) + FoV 2 (vertical/horizontal radians,
 *  FoV = relu(x) + 0.01). Borrowed; NULL until the first successful run. */
AICORE_CAPI const float* aicore_querysplat_result_pose(
        const aicore_querysplat_ctx* ctx);
/** Per-view camera-from-world extrinsics [n*16], row-major 4x4 (OpenCV
 *  w2c = [[R|T],[0,1]]), the render convention of scripts/rendering/gs.py.
 *  Borrowed. */
AICORE_CAPI const float* aicore_querysplat_result_w2c(
        const aicore_querysplat_ctx* ctx);
/** Per-view world-from-camera extrinsics [n*16], row-major 4x4 (the
 *  general inverse of w2c, as torch.linalg.inv). Borrowed. */
AICORE_CAPI const float* aicore_querysplat_result_cam2world(
        const aicore_querysplat_ctx* ctx);
/** Per-view intrinsics [n*4]: fx, fy, cx, cy over the preprocessed grid.
 *  Borrowed. */
AICORE_CAPI const float* aicore_querysplat_result_intrinsics(
        const aicore_querysplat_ctx* ctx);
/** Encoder dense-head outputs [n*H*W] view-major: depth = exp(logits),
 *  depth_conf = 1 + exp(logits) (>= 1). Borrowed. */
AICORE_CAPI const float* aicore_querysplat_result_depth(
        const aicore_querysplat_ctx* ctx);
AICORE_CAPI const float* aicore_querysplat_result_depth_conf(
        const aicore_querysplat_ctx* ctx);

/** Final gaussians [n_gaussians * stride], torch (N, stride) C-order:
 *  [pos(3) | opacity | scale(3) | rot(4) | color(sh0: 3)]. Borrowed;
 *  n_gaussians = num_gs_tokens * gaussians_per_token (524288 for the paper
 *  checkpoints), stride = 11 + color_dim (14 for sh0). */
AICORE_CAPI int64_t
aicore_querysplat_result_gaussian_count(const aicore_querysplat_ctx* ctx);
AICORE_CAPI int32_t
aicore_querysplat_result_gaussian_stride(const aicore_querysplat_ctx* ctx);
AICORE_CAPI const float* aicore_querysplat_result_gaussians(
        const aicore_querysplat_ctx* ctx);
/** SH degree of the loaded decoder checkpoint (0 for the paper family). */
AICORE_CAPI int32_t
aicore_querysplat_result_sh_degree(const aicore_querysplat_ctx* ctx);

/** Per-stage wall-clock timings of the most recent run as the common
 *  pipeline contract (preprocess = decode/resize/normalize + graph-input
 *  preparation; inference = encoder + camera math + decoder + readback;
 *  postprocess = gaussians assembly + typed-result construction; e2e =
 *  entry to ready result). Returns 0 on success, -1 when no inference has
 *  run yet. */
AICORE_CAPI int aicore_querysplat_last_pipeline_timings(
        const aicore_querysplat_ctx* ctx, aicore_pipeline_timings* out);

/** ---- rendering (CPU rasterizer, gsplat semantics) ----
 *
 *  Renders the most recent run's gaussians from ARBITRARY cameras.
 *  w2c: n*16 row-major world-to-camera (CameraData layout, the render
 *  convention of scripts/rendering/gs.py); intrinsics: n*4
 *  [fx, fy, cx, cy]. Renders at the result (height, width). On success
 *  *out_rgb receives a borrowed (n*3*H*W) plane-layout float buffer and
 *  *out_rgb8 an interleaved RGB8 buffer (n*H*W*3) — both owned by the
 *  context until the next render/run/free. Returns 0 on success, -1 on
 *  invalid arguments, -2 when no result is present. */
AICORE_CAPI int aicore_querysplat_render_views(aicore_querysplat_ctx* ctx,
                                               const float* w2c,
                                               const float* intrinsics,
                                               int32_t n_views,
                                               const float** out_rgb,
                                               unsigned char** out_rgb8);

/** ---- TTO (test-time optimization) ----
 *
 *  Fine-tunes the decoder KV memory (geom K/V + post_rgb K/V) against the
 *  input views' own pixels: L = L1 + 0.2*(1-SSIM)/2 + w*LPIPS, AdamW per
 *  step (the official _optimize_memory). Requires a previous successful
 *  run (the KV memories seed from it) and an sh0 checkpoint. On success
 *  *out_gaussians/*out_count expose the OPTIMIZED gaussian set (borrowed,
 *  same store contract as the result accessors) and *out_losses receives
 *  one (l1, ssim, lpips) triple per executed step (borrowed, n_steps
 *  entries). Returns 0 on success, -1 on invalid arguments, -2 on runtime
 *  errors. */
typedef struct aicore_querysplat_tto_step {
    double l1;
    double ssim;
    double lpips;
} aicore_querysplat_tto_step;
AICORE_CAPI int aicore_querysplat_run_tto(
        aicore_querysplat_ctx* ctx,
        int32_t n_steps,
        aicore_querysplat_tto_step** out_losses,
        int32_t* out_n_losses,
        const float** out_gaussians,
        int64_t* out_count);

/** ---- exports (compatibility/delivery boundaries) ---- */

/** Writes the most recent gaussians above \p threshold as a standard 3DGS
 *  PLY (binary_little_endian, sh0: f_dc + f_rest 0 coefficients; the same
 *  layout the official gaussian_save_opacity_threshold path produces).
 *  Returns 0 on success, -1 on invalid arguments, -2 on I/O failure. */
AICORE_CAPI int aicore_querysplat_export_gaussians_ply(
        const aicore_querysplat_ctx* ctx,
        const char* path,
        float opacity_threshold);

/** Serializes the most recent run's predicted input cameras as JSON
 *  (per-view pose_enc 9 floats + c2w 16 + intrinsics 4 + image size — the
 *  save_predicted_input_cameras contract). Caller frees with
 *  aicore_querysplat_free_buffer. NULL on failure (inspect last_error). */
AICORE_CAPI char* aicore_querysplat_export_cameras_json(
        const aicore_querysplat_ctx* ctx);

/** Reconstructs a colored point cloud from the most recent depth maps
 *  (save_vggt_depth_pointcloud semantics): up to target_points samples
 *  drawn proportionally to depth confidence, unprojected through the
 *  predicted cameras. *out_xyz/*out_rgb are borrowed context buffers until
 *  the next run/free; *out_n receives the sample count. Returns 0 on
 *  success, -1 on invalid arguments, -2 when no result is present. */
AICORE_CAPI int aicore_querysplat_depth_pointcloud(
        const aicore_querysplat_ctx* ctx,
        int32_t target_points,
        const float** out_xyz,
        unsigned char** out_rgb,
        int32_t* out_n);

/** Gaussian distribution statistics of the most recent run as JSON
 *  (save_gaussian_alpha_distribution / scale_distribution: histogram bins
 *  + min/max/mean per quantity — the data behind the official plots).
 *  Caller frees with aicore_querysplat_free_buffer. */
AICORE_CAPI char* aicore_querysplat_export_distribution_stats(
        const aicore_querysplat_ctx* ctx);

/** Compact machine-readable capability JSON (model paths, geometry,
 *  f32-accuracy mode, TTO config, resolved device). Caller frees with
 *  aicore_querysplat_free_buffer. */
AICORE_CAPI char* aicore_querysplat_info_json(const aicore_querysplat_ctx* ctx);

/** ---- model catalog ----
 *
 *  The decoder-family + LPIPS models (published at
 *  https://huggingface.co/Asher-1/querysplat_vggto). The VGGT-Omega
 *  encoder entries are NOT duplicated here: they live in the map_anything
 *  catalog (single source of truth for the shared GGUFs). */

typedef struct aicore_querysplat_model_entry {
    const char* filename;     /* e.g. "querysplat-paper-q8_0.gguf" */
    const char* display_name; /* e.g. "QuerySplat paper decoder" */
    const char* role;         /* "decoder" | "lpips" */
    const char* quant_note;
    const char* license_note;
    const char* variant; /* "paper" | "lpips" */
    int64_t size_bytes;
    const char* download_url; /* stable HF resolve URL */
} aicore_querysplat_model_entry;

AICORE_CAPI int32_t aicore_querysplat_model_count(void);
AICORE_CAPI const aicore_querysplat_model_entry* aicore_querysplat_model_at(
        int32_t index);
/** Catalog default for a role ("decoder" | "encoder" | "lpips"); the
 *  encoder default delegates to the map_anything catalog. NULL on an
 *  unknown role. */
AICORE_CAPI const char* aicore_querysplat_model_default(const char* role);
/** Resolves a catalog filename into its cache path under the shared data
 *  root (CLOUDVIEWER_DATA_ROOT or ~/cloudViewer_data/extract/
 *  querysplat_models). Caller frees with aicore_querysplat_free_buffer. */
AICORE_CAPI char* aicore_querysplat_model_resolve_cache_path(
        const char* filename);

#ifdef __cplusplus
} /* extern "C" */
#endif

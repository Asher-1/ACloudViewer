// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// MapAnything C API: universal feed-forward metric 3D reconstruction over the
// in-tree ggml port in core/AICore/src/tasks/map_anything/ (upstream
// map-anything-ggml cpp_ggml). ONE context type serves six model
// architectures routed by the GGUF general.architecture key:
//   vggt_omega (3 published variants) / vggt / pi3 / pi3x / mapanything /
//   dust3r.
//
// One multi-view inference yields per-view dense geometry (depth, confidence)
// plus camera outputs whose layout is an architecture property:
//   vggt_omega / vggt : pose_stride = 9, pose = pose_enc [n*9]
//                       (absT 3 + quaternion 4 scalar-last xyzw + FoV 2 =
//                       vertical/horizontal radians, FoV = relu(x) + 0.01)
//   pi3 / pi3x / mapanything : pose_stride = 16, pose = camera-to-world
//                       4x4 row-major [n*16]
//   dust3r : pose_stride = 0, pose = NULL (no pose head)
//
// Thread-safety: a context is single-threaded. Use one context per consumer,
// or serialize access externally.
//
// GGUF models: published pre-converted weights at
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

/** Returns the ABI version of the MapAnything C API (bump on breaking ABI
 *  changes). */
AICORE_CAPI int aicore_map_anything_abi_version(void);

typedef struct aicore_map_anything_ctx aicore_map_anything_ctx;
typedef struct aicore_map_anything_options aicore_map_anything_options;

/** Creates a default options struct (device "auto", threads 0 = backend
 *  default, image size 0 = checkpoint's nominal resolution, resize mode
 *  "balanced"). Release with aicore_map_anything_options_free. */
AICORE_CAPI aicore_map_anything_options* aicore_map_anything_options_new(void);
/** Releases an options struct created by aicore_map_anything_options_new. */
AICORE_CAPI void aicore_map_anything_options_free(
        aicore_map_anything_options* opts);
/** Selects the inference device: NULL or "auto", "cpu", "gpu", "vulkan"
 *  (optionally ":N"), "cuda" (Linux/Windows). */
AICORE_CAPI void aicore_map_anything_options_set_device(
        aicore_map_anything_options* opts, const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_map_anything_options_set_threads(
        aicore_map_anything_options* opts, int n_threads);
/** Preprocess longest-side target before inference. 0 (the default) uses the
 *  checkpoint's nominal resolution (aicore_map_anything_context_default_
 * image_size); non-zero values are clamped to a patch multiple. */
AICORE_CAPI void aicore_map_anything_options_set_image_size(
        aicore_map_anything_options* opts, int image_size);
/** Official preprocessing mode: "balanced" (center-crop extreme aspect
 *  ratios into [0.5, 2.0], patch-multiple target shape) or "max_size"
 *  (longest side = image size). NULL/empty keeps the current value. */
AICORE_CAPI void aicore_map_anything_options_set_resize_mode(
        aicore_map_anything_options* opts, const char* mode);
/** Log level: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR (default 1). */
AICORE_CAPI void aicore_map_anything_options_set_log_level(
        aicore_map_anything_options* opts, int log_level);
/** Debug: dump parity stage tensors as raw f32 files into this directory
 *  after every run call (effective only in MAP_ANYTHING_ENABLE_DUMP builds).
 *  The string is copied; empty (default) disables dumping. */
AICORE_CAPI void aicore_map_anything_options_set_dump_dir(
        aicore_map_anything_options* opts, const char* dir);

/** Load a MapAnything-family GGUF (map_anything_models catalog). Returns a
 *  context whose is_ready() may still be 0 (device admission failure);
 *  inspect aicore_map_anything_last_error(). NULL only on invalid input or
 *  out-of-memory. */
AICORE_CAPI aicore_map_anything_ctx* aicore_map_anything_load_opts(
        const char* gguf_path, const aicore_map_anything_options* opts);
/** Releases a context; safe on NULL. Contexts must not be destroyed while a
 *  worker thread is inside a run call (serialize externally). */
AICORE_CAPI void aicore_map_anything_free(aicore_map_anything_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model. */
AICORE_CAPI int aicore_map_anything_is_ready(
        const aicore_map_anything_ctx* ctx);
/** Returns the last error message of the context (NULL when none). */
AICORE_CAPI const char* aicore_map_anything_last_error(
        const aicore_map_anything_ctx* ctx);
/** Releases any buffer returned by an aicore_map_anything_* function
 *  (JSON/RGB helpers). Safe on NULL. */
AICORE_CAPI void aicore_map_anything_free_buffer(void* p);

/** Load an image file as tightly-packed RGB (HWC, 3 bytes/pixel).
 *  Compatibility helper for path-driven callers; the typed hot path is
 *  aicore_map_anything_run_images with borrowed aicore_image_view values.
 *  Caller frees \p out_rgb with aicore_map_anything_free_buffer. */
AICORE_CAPI int aicore_map_anything_load_path_rgb(const char* image_path,
                                                  uint8_t** out_rgb,
                                                  int32_t* out_width,
                                                  int32_t* out_height);

/** ---- typed inference ----
 *
 *  Multi-view inference from DECODED images. Every view must be non-NULL
 *  with positive dimensions; the official balanced/max_size preprocessing
 *  (including white padding of mixed sizes) runs inside. n_views >= 1;
 *  the compute cost grows quadratically with the view count (full
 *  inter-frame attention), so keep batches modest. Returns 0 on success,
 *  -1 on contract/validation errors and -2 on runtime/inference errors
 *  (inspect aicore_map_anything_last_error).
 *
 *  The result is owned by the context and stays valid until the next run
 *  call or aicore_map_anything_free. All accessors below are borrowed,
 *  read-only views of that store — never freed by the caller. */
AICORE_CAPI int aicore_map_anything_run_images(aicore_map_anything_ctx* ctx,
                                               const aicore_image_view* views,
                                               int32_t n_views);

/** Parity feed: preprocessed frames, n_views*3*height*width floats in
 *  [0,1], torch (n,3,H,W) order — the bit-exact --bin contract of the
 *  upstream CLI. Same result store contract as run_images. */
AICORE_CAPI int aicore_map_anything_run_frames(aicore_map_anything_ctx* ctx,
                                               const float* frames_f32,
                                               int32_t n_views,
                                               int32_t height,
                                               int32_t width);

/** Number of views of the most recent successful run (0 before the first
 *  run or after a failed call left no result). */
AICORE_CAPI int32_t
aicore_map_anything_result_view_count(const aicore_map_anything_ctx* ctx);
/** Dense-output geometry of the most recent run: vggt_omega resolves to the
 *  preprocessed input resolution; the other builders resolve to the
 *  patch-aligned grid. 0 before the first run. */
AICORE_CAPI int32_t
aicore_map_anything_result_height(const aicore_map_anything_ctx* ctx);
AICORE_CAPI int32_t
aicore_map_anything_result_width(const aicore_map_anything_ctx* ctx);
/** Normalized general.architecture of the loaded GGUF ("vggt_omega",
 *  "vggt", "pi3", "pi3x", "mapanything", "dust3r"; "" when not ready). */
AICORE_CAPI const char* aicore_map_anything_result_architecture(
        const aicore_map_anything_ctx* ctx);

/** Per-view RESIZED content size (w, h) BEFORE the centered white pad to
 *  the common grid. Pixels outside the valid rect are padded filler —
 *  consumers must skip them when unprojecting/fusing (the model sees
 *  white there and its predictions are not meaningful). Returns 0 on
 *  success; -1 on invalid arguments. */
AICORE_CAPI int aicore_map_anything_result_view_valid_size(
        const aicore_map_anything_ctx* ctx,
        int32_t view,
        int32_t* w,
        int32_t* h);

/** Dense depth [n*H*W] view-major. vggt family: the dedicated depth head
 *  (exp of logits). pi3 family: camera-frame z of the local points. dust3r:
 *  camera-frame z of the pointmaps. NULL until the first successful run. */
AICORE_CAPI const float* aicore_map_anything_result_depth(
        const aicore_map_anything_ctx* ctx);
/** Confidence [n*H*W]. vggt family: depth_conf = 1+exp(logits) (>= 1).
 *  pi3 family: RAW logits (apply sigmoid for probabilities). dust3r:
 *  1+exp. NULL when the model has none. */
AICORE_CAPI const float* aicore_map_anything_result_conf(
        const aicore_map_anything_ctx* ctx);
/** Per-view local pointmaps [n*H*W*3] (camera frame, xyz); dust3r views
 *  live in view1's frame. pi3/pi3x/mapanything/dust3r only. */
AICORE_CAPI const float* aicore_map_anything_result_local_points(
        const aicore_map_anything_ctx* ctx);
/** World/common-frame pointmaps [n*H*W*3]. mapanything: the common-frame
 *  points. vggt-1b: the official point head (view0 frame). dust3r: view2's
 *  pointmap in view1's frame. NULL when the model has no point head. */
AICORE_CAPI const float* aicore_map_anything_result_points(
        const aicore_map_anything_ctx* ctx);
/** mapanything only: non-ambiguous mask [n*H*W] (sigmoid probability). */
AICORE_CAPI const float* aicore_map_anything_result_mask(
        const aicore_map_anything_ctx* ctx);
/** mapanything only: metric scale (exp of the scale-token head, [1]). */
AICORE_CAPI const float* aicore_map_anything_result_scale(
        const aicore_map_anything_ctx* ctx);
/** Camera outputs of the most recent run; pose_stride is 9 (vggt family),
 *  16 (pi3 family) or 0 (dust3r, NULL pose). See the header comment for
 *  the per-family layout contract. */
AICORE_CAPI const float* aicore_map_anything_result_pose(
        const aicore_map_anything_ctx* ctx);
AICORE_CAPI int32_t
aicore_map_anything_result_pose_stride(const aicore_map_anything_ctx* ctx);
/** vggt-omega text-alignment variant only: L2-normalized language-aligned
 *  embedding; *out_dim receives the dimension (2048), 0 otherwise. */
AICORE_CAPI const float* aicore_map_anything_result_text_embedding(
        const aicore_map_anything_ctx* ctx, int32_t* out_dim);

/** Per-stage wall-clock timings of the most recent run as the common
 *  pipeline contract (preprocess = decode/resize/normalize + graph-input
 *  preparation; inference = upload + graph execution + readback;
 *  postprocess = typed-result construction; e2e = entry to ready result).
 *  Returns 0 on success, -1 when no inference has run yet. */
AICORE_CAPI int aicore_map_anything_last_pipeline_timings(
        const aicore_map_anything_ctx* ctx,
        aicore_pipeline_timings* out_timings);

/** Model introspection. */
/** GGUF patch size (14 for the vggt/pi3 families, 16 for dust3r). */
AICORE_CAPI int32_t
aicore_map_anything_context_patch_size(aicore_map_anything_ctx* ctx);
/** Checkpoint's nominal preprocessing resolution (512 for vggt-omega and
 *  dust3r, 518 for the others; 0 when not ready). */
AICORE_CAPI int32_t
aicore_map_anything_context_default_image_size(aicore_map_anything_ctx* ctx);
/** Backend-RESOLVED device name (surfaces silent CPU fallbacks). Owned by
 *  the context; copy before freeing. */
AICORE_CAPI const char* aicore_map_anything_context_device(
        const aicore_map_anything_ctx* ctx);
/** Effective CPU thread count after the auto (<=0) resolution. */
AICORE_CAPI int aicore_map_anything_context_threads(
        const aicore_map_anything_ctx* ctx);

/** Returns a JSON summary of the loaded model and the latest run.
 *  Caller frees with aicore_map_anything_free_buffer. */
AICORE_CAPI char* aicore_map_anything_info_json(aicore_map_anything_ctx* ctx);

/** Warms up the backend for `device` (thin wrapper of aicore_warmup_backend
 *  kept for task-symmetric plugin code). */
AICORE_CAPI int aicore_map_anything_warmup_backend(const char* device);
/** Releases process-wide MapAnything runtime resources it owns (idempotent;
 *  delegates to aicore_runtime_shutdown and never destroys live contexts). */
AICORE_CAPI void aicore_map_anything_shutdown(void);
/** Returns the local model cache directory (…/extract/map_anything_models).
 *  Caller frees with aicore_map_anything_free_buffer. */
AICORE_CAPI char* aicore_map_anything_model_cache_dir(void);

/** Published GGUF catalog (Hugging Face Asher-1/map_anything_vggt). */
typedef struct aicore_map_anything_model_entry {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
    /** Model-variant grouping key shared by the 4 quantization rows of one
     *  checkpoint family ("vggt-omega-1b-512", "mapanything", ...). The
     *  qMapAnything dialog renders one page per distinct variant, ordered by
     *  the catalog (first variant = the accuracy pick). */
    const char* variant;
    int64_t size_bytes; /**< exact published size (0 = unknown) */
    /** Official-parity inference defaults for this variant, sourced from
     *  the upstream repos' own reconstruction recipes:
     *  - resize mode: "balanced" (omega / mapanything / pi3 / dust3r
     *    load_and_preprocess_images) or "square" (original VGGT
     *    load_and_preprocess_images_square: center-pad to square, resize).
     *  - confidence filter: "normalized" (threshold on the family-unified
     *    sigmoid(logit) scale, [0,1)) or "percentile" (drop the lowest N
     *    percent of confidences, the official MapAnything gradio semantics;
     *    then default_min_conf is a percentage 0..99). */
    const char* default_resize_mode;
    const char* conf_filter;
    float default_min_conf;
} aicore_map_anything_model_entry;

/** Number of catalog entries (8 variants x 4 quantizations = 32). */
AICORE_CAPI int aicore_map_anything_model_count(void);
/** Catalog entry at `index` (NULL when out of range). Returned pointers are
 *  stable for the process lifetime. */
AICORE_CAPI const aicore_map_anything_model_entry* aicore_map_anything_model_at(
        int index);
/** Index of the entry the catalog declares as its default (the visible
 *  "(recommended)" row). Returns 0 for empty/unknown views. */
AICORE_CAPI int aicore_map_anything_model_default_index(void);
/** Returns the catalog entry whose filename matches (NULL when not
 *  found). */
AICORE_CAPI const aicore_map_anything_model_entry*
aicore_map_anything_model_by_filename(const char* filename);
/** Returns the base URL of the published model repo. */
AICORE_CAPI const char* aicore_map_anything_model_download_base(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

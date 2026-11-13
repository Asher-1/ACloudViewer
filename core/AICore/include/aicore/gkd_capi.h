// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// GKD C API: general keypoint detection (GKDT, ECCV 2026) over the in-tree
// ggml port in core/AICore/src/tasks/gkd/ (upstream
// General-Keypoint-Detection-GGML cpp_ggml).
//
// One prompt (text and/or 1-shot visual) yields one keypoint: position in the
// ROI's normalized -1..1 space plus the recovered source-image pixel
// coordinates and the peak heatmap score in 0..~1. Multi-object workflows
// compose this task with the YOLO open-vocabulary detectors
// (aicore_yolo_* WORLD models): per-class boxes -> per-box detect calls.
//
// Thread-safety: a context is single-threaded. Use one context per consumer,
// or serialize access externally.
//
// GGUF models: published pre-converted weights at
//   https://huggingface.co/Asher-1/GKD_GGUF
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

/** Returns the ABI version of the GKD C API (bump on breaking ABI
 *  changes). */
AICORE_CAPI int aicore_gkd_abi_version(void);

typedef struct aicore_gkd_ctx aicore_gkd_ctx;
typedef struct aicore_gkd_options aicore_gkd_options;

/** Creates a default options struct (device "auto", threads 0 = backend
 *  default). Release with aicore_gkd_options_free. */
AICORE_CAPI aicore_gkd_options* aicore_gkd_options_new(void);
/** Releases an options struct created by aicore_gkd_options_new. */
AICORE_CAPI void aicore_gkd_options_free(aicore_gkd_options* opts);
/** Selects the inference device: NULL or "auto", "cpu", "gpu", "vulkan"
 *  (optionally ":N"), "cuda" (Linux/Windows). */
AICORE_CAPI void aicore_gkd_options_set_device(aicore_gkd_options* opts,
                                               const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_gkd_options_set_threads(aicore_gkd_options* opts,
                                                int n_threads);
/** Log level: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR (default 1). */
AICORE_CAPI void aicore_gkd_options_set_log_level(aicore_gkd_options* opts,
                                                  int log_level);
/** Debug: dump parity taps (normalized inputs, token ids, prompts, fused
 *  heatmaps) as raw f32 files into this directory after every detect call.
 *  The string is copied; empty (default) disables dumping. */
AICORE_CAPI void aicore_gkd_options_set_dump_dir(aicore_gkd_options* opts,
                                                 const char* dir);

/** Load a GKDT GGUF (gkd_fullset-<quant>.gguf from the published catalog).
 *  Returns NULL on failure; inspect aicore_gkd_last_error() on a returned
 *  context for the reason. */
AICORE_CAPI aicore_gkd_ctx* aicore_gkd_load_opts(
        const char* gguf_path, const aicore_gkd_options* opts);
/** Releases a context returned by aicore_gkd_load_opts; safe on NULL. */
AICORE_CAPI void aicore_gkd_free(aicore_gkd_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model. */
AICORE_CAPI int aicore_gkd_is_ready(const aicore_gkd_ctx* ctx);
/** Returns the last error message of the context (NULL when none). */
AICORE_CAPI const char* aicore_gkd_last_error(const aicore_gkd_ctx* ctx);

/** Releases any buffer returned by an aicore_gkd_* function (string or
 *  float array; unified entry point). Safe on NULL. */
AICORE_CAPI void aicore_gkd_free_buffer(void* p);

/** Load an image file as tightly-packed RGB (HWC, 3 bytes/pixel).
 *  Compatibility helper for path-driven callers; the typed hot path is
 *  aicore_gkd_detect_image with a borrowed aicore_image_view. Caller frees
 *  \p out_rgb with aicore_gkd_free_buffer. */
AICORE_CAPI int aicore_gkd_load_path_rgb(const char* image_path,
                                         uint8_t** out_rgb,
                                         int32_t* out_width,
                                         int32_t* out_height);

/** One detection request. All pointers are borrowed for the duration of the
 *  aicore_gkd_detect_image call; ownership never transfers. */
typedef struct aicore_gkd_detect_request {
    uint32_t struct_size; /**< caller sets sizeof(aicore_gkd_detect_request) */
    /** Optional ROI [x1, y1, x2, y2] in image pixels (inclusive corners,
     *  may be fractional/negative; clamped internally). NULL = whole image. */
    const float* bbox_xyxy;
    /** Text prompts ("nose", "left eye", ...), one keypoint per prompt. */
    const char* const* kps_texts;
    int32_t n_kps_texts;
    /** Optional 1-shot visual prompt: support image view + keypoints in
     *  SUPPORT-image pixel coordinates. view may be NULL (text-only mode).
     *  Multimodal mode requires n_kps_texts == n_support_kps (the official
     *  fuse pairs text row i with visual row i). */
    const aicore_image_view* support_image;
    const float* support_kps_xy;    /**< n_support_kps * 2, x/y pairs */
    const uint8_t* support_kps_vis; /**< optional; NULL = all valid */
    int32_t n_support_kps;
} aicore_gkd_detect_request;

/** One detected keypoint. */
typedef struct aicore_gkd_keypoint {
    float x_norm; /**< normalized -1..1 in the ROI space */
    float y_norm;
    float x; /**< recovered source-image pixel coordinate */
    float y;
    float score; /**< heatmap peak value (0..~1) */
} aicore_gkd_keypoint;

/** Typed hot-path detection. Populates a context-owned result store without
 *  allocating a JSON envelope. At least one prompt source must be present;
 *  when both are present their counts must match (multimodal semantics).
 *  Returns 0 on success, -1 on contract/validation errors and -2 on
 *  runtime/inference errors (inspect aicore_gkd_last_error). The result is
 *  valid until the next detect call or aicore_gkd_free. */
AICORE_CAPI int aicore_gkd_detect_image(aicore_gkd_ctx* ctx,
                                        const aicore_image_view* image,
                                        const aicore_gkd_detect_request* req);
/** Packed-RGB (HWC, 3 bytes/pixel) equivalent of aicore_gkd_detect_image. */
AICORE_CAPI int aicore_gkd_detect_rgb(aicore_gkd_ctx* ctx,
                                      const uint8_t* rgb,
                                      int32_t width,
                                      int32_t height,
                                      const aicore_gkd_detect_request* req);

/** Multi-ROI batch detection (the official top-down pipeline: N boxes, one
 *  shared-prompt batched forward). \p base_req carries the texts and the
 *  optional support prompt; its bbox_xyxy is ignored in favor of \p
 *  bboxes_xyxy (N boxes in image pixels). Results land in the per-ROI
 *  accessors below; the single-ROI accessors keep reading ROI 0. Returns 0
 *  on success, -1 on contract errors, -2 on runtime errors. */
AICORE_CAPI int aicore_gkd_detect_image_multi(
        aicore_gkd_ctx* ctx,
        const aicore_image_view* image,
        const aicore_gkd_detect_request* base_req,
        const float* bboxes_xyxy,
        int32_t n_bboxes);
/** Packed-RGB equivalent of aicore_gkd_detect_image_multi. */
AICORE_CAPI int aicore_gkd_detect_rgb_multi(
        aicore_gkd_ctx* ctx,
        const uint8_t* rgb,
        int32_t width,
        int32_t height,
        const aicore_gkd_detect_request* base_req,
        const float* bboxes_xyxy,
        int32_t n_bboxes);

/** Number of keypoints in the most recent successful detect call. */
AICORE_CAPI int aicore_gkd_result_keypoint_count(const aicore_gkd_ctx* ctx);
/** Keypoint at \p index (zeroed when out of range). */
AICORE_CAPI aicore_gkd_keypoint
aicore_gkd_result_keypoint_at(const aicore_gkd_ctx* ctx, int index);
/** Text of the i-th prompt when one was given (owned by the context, valid
 *  until the next detect call or aicore_gkd_free; NULL for visual-only
 *  results or out-of-range indexes). */
AICORE_CAPI const char* aicore_gkd_result_prompt_at(const aicore_gkd_ctx* ctx,
                                                    int index);
/** ROI of the most recent call in image pixels [x1,y1,x2,y2] (the whole
 *  image when the request carried no bbox). NULL-safe: returns 0 and zeroes
 *  \p out_bbox when no inference ran. */
AICORE_CAPI int aicore_gkd_result_roi(const aicore_gkd_ctx* ctx,
                                      float out_bbox[4]);

/** Number of ROIs held by the most recent successful detect call (1 for the
 *  single-ROI entry points, N for the multi-ROI batch). */
AICORE_CAPI int32_t aicore_gkd_result_roi_count(const aicore_gkd_ctx* ctx);
/** Per-ROI keypoint count (zeroed when out of range). */
AICORE_CAPI int aicore_gkd_result_keypoint_count_at(const aicore_gkd_ctx* ctx,
                                                    int32_t roi);
/** Per-ROI keypoint (zeroed when out of range). */
AICORE_CAPI aicore_gkd_keypoint aicore_gkd_result_keypoint_at_roi(
        const aicore_gkd_ctx* ctx, int32_t roi, int index);
/** Per-ROI prompt text (NULL when out of range or visual-only). */
AICORE_CAPI const char* aicore_gkd_result_prompt_at_roi(
        const aicore_gkd_ctx* ctx, int32_t roi, int index);
/** Per-ROI used box in image pixels [x1,y1,x2,y2]. */
AICORE_CAPI int aicore_gkd_result_roi_bbox_at(const aicore_gkd_ctx* ctx,
                                              int32_t roi,
                                              float out_bbox[4]);

/** Per-stage wall-clock timings of the most recent aicore_gkd_detect_* call,
 *  in milliseconds (preprocess / vision / text / prompt / detect / decode /
 *  e2e). All fields stay 0 until the first call. */
typedef struct aicore_gkd_timings {
    double preprocess_ms;
    double vision_ms;
    double text_ms;
    double prompt_prep_ms;
    double detect_ms;
    double decode_ms;
    double e2e_ms;
} aicore_gkd_timings;

/** Copy the most recent successful inference timings into out_timings.
 *  Returns 0 on success, -1 when ctx has never run an inference. */
AICORE_CAPI int aicore_gkd_last_timings(const aicore_gkd_ctx* ctx,
                                        aicore_gkd_timings* out_timings);
/** Common timing contract adapter for the most recent inference:
 *  preprocess = preprocess + prompt_prep, inference = vision + text +
 *  detect, postprocess = decode, e2e = total. */
AICORE_CAPI int aicore_gkd_last_pipeline_timings(
        const aicore_gkd_ctx* ctx, aicore_pipeline_timings* out_timings);

/** Model introspection. */
/** GGUF-declared model name ("" when not ready). */
AICORE_CAPI const char* aicore_gkd_context_model_name(aicore_gkd_ctx* ctx);
/** Model inference square (from GGUF metadata; 384 for GKDT-L). */
AICORE_CAPI int32_t aicore_gkd_context_image_size(const aicore_gkd_ctx* ctx);
/** Backend-RESOLVED device name. Differs from the requested device when the
 *  GPU lease can't be acquired — surfaces silent CPU fallbacks. Owned by
 *  ctx; copy before freeing. */
AICORE_CAPI const char* aicore_gkd_context_device(const aicore_gkd_ctx* ctx);
/** Effective CPU thread count after the auto (<=0) resolution. */
AICORE_CAPI int aicore_gkd_context_threads(const aicore_gkd_ctx* ctx);

/** Returns a JSON summary of the loaded model and the latest result.
 *  Caller frees with aicore_gkd_free_buffer. */
AICORE_CAPI char* aicore_gkd_info_json(aicore_gkd_ctx* ctx);

/** Warms up the backend for `device`; returns 0 on success (thin wrapper of
 *  aicore_warmup_backend kept for task-symmetric plugin code). */
AICORE_CAPI int aicore_gkd_warmup_backend(const char* device);
/** Releases process-wide GKD runtime resources it owns (idempotent;
 *  delegates to aicore_runtime_shutdown and never destroys live contexts). */
AICORE_CAPI void aicore_gkd_shutdown(void);
/** Returns the local model cache directory (…/extract/gkd_models). Caller
 *  frees with aicore_gkd_free_buffer. */
AICORE_CAPI char* aicore_gkd_model_cache_dir(void);

/** Published GGUF catalog (Hugging Face Asher-1/GKD_GGUF). */
typedef struct aicore_gkd_model_entry {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
    int64_t size_bytes; /**< exact published size (0 = unknown) */
} aicore_gkd_model_entry;

/** Number of catalog entries. */
AICORE_CAPI int aicore_gkd_model_count(void);
/** Catalog entry at `index` (NULL when out of range). Returned pointers are
 *  stable for the process lifetime. */
AICORE_CAPI const aicore_gkd_model_entry* aicore_gkd_model_at(int index);
/** Index of the entry the catalog declares as its default (the visible
 *  "(recommended)" row). Returns 0 for empty/unknown views. */
AICORE_CAPI int aicore_gkd_model_default_index(void);
/** Returns the catalog entry whose filename matches (NULL when not
 *  found). */
AICORE_CAPI const aicore_gkd_model_entry* aicore_gkd_model_by_filename(
        const char* filename);
/** Returns the base URL of the published model repo. */
AICORE_CAPI const char* aicore_gkd_model_download_base(void);

#ifdef __cplusplus
}
#endif

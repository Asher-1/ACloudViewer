// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// YOLO object detection / instance segmentation / metric depth C API.
//
// The ggml engine under core/AICore/src/tasks/yolo/ is an in-tree port of
// ultralytics-ggml cpp_ggml (https://github.com/Asher-1/ultralytics-ggml),
// extended with typed segment results and yolo26n-depth absolute-depth
// support. The upstream source is AGPL-3.0; this port keeps the license
// until a written relicensing decision is recorded (see
// ultralytics-ggml-integration-plan.md §5).

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the ABI version of the YOLO C API (bump on breaking ABI
 *  changes). */
AICORE_CAPI int aicore_yolo_abi_version(void);

typedef struct aicore_yolo_ctx aicore_yolo_ctx;
typedef struct aicore_yolo_options aicore_yolo_options;

/** Creates a default options struct (device "auto", threads 0 = backend
 *  default). Release with aicore_yolo_options_free. */
AICORE_CAPI aicore_yolo_options* aicore_yolo_options_new(void);
/** Releases an options struct created by aicore_yolo_options_new. */
AICORE_CAPI void aicore_yolo_options_free(aicore_yolo_options* opts);
/** Selects the inference device: NULL or "auto", "cpu", "gpu", "vulkan"
 *  (optionally ":N"), "cuda" (Linux/Windows). */
AICORE_CAPI void aicore_yolo_options_set_device(aicore_yolo_options* opts,
                                                const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_yolo_options_set_threads(aicore_yolo_options* opts,
                                                 int n_threads);
/** Detection confidence threshold (default 0.25). */
AICORE_CAPI void aicore_yolo_options_set_conf_thres(aicore_yolo_options* opts,
                                                    float conf_thres);
/** NMS IoU threshold (default 0.7). */
AICORE_CAPI void aicore_yolo_options_set_iou_thres(aicore_yolo_options* opts,
                                                   float iou_thres);
/** Caps the returned detections (0 = model default max_det). */
AICORE_CAPI void aicore_yolo_options_set_top_k(aicore_yolo_options* opts,
                                               uint32_t top_k);
/* Session-level debug/tuning knobs (default: model metadata / off). 0 clears
 * back to the default. log_level: 0=DEBUG,1=INFO,2=WARN,3=ERROR. */
AICORE_CAPI void aicore_yolo_options_set_log_level(aicore_yolo_options* opts,
                                                   int log_level);
/* Override the inference input size (0 = use the square imgsz from the GGUF
 * metadata; both dimensions must be > 0 once set). */
AICORE_CAPI void aicore_yolo_options_set_input_size(aicore_yolo_options* opts,
                                                    int width,
                                                    int height);
/* Debug: keep every op output alive / print a per-op wall-time table /
 * per-stage timing on stderr (default off). */
AICORE_CAPI void aicore_yolo_options_set_keep_all_ops(aicore_yolo_options* opts,
                                                      int enabled);
/** Debug: print a per-op wall-time table on stderr (default off). */
AICORE_CAPI void aicore_yolo_options_set_profile_ops(aicore_yolo_options* opts,
                                                     int enabled);
/** Debug: print per-stage upload/compute/readback timing gaps on stderr
 *  (default off). */
AICORE_CAPI void aicore_yolo_options_set_profile_gaps(aicore_yolo_options* opts,
                                                      int enabled);

/** Get the recommended default confidence threshold for this model. */
AICORE_CAPI float aicore_yolo_options_get_conf_thres(
        const aicore_yolo_options* opts);
/** Get the recommended default IoU threshold for this model. */
AICORE_CAPI float aicore_yolo_options_get_iou_thres(
        const aicore_yolo_options* opts);

/** Load a YOLO GGUF (detection or depth variant; the task is read from the
 *  GGUF's yolo.task metadata). Returns NULL on failure; inspect
 *  aicore_yolo_last_error() for the reason. */
AICORE_CAPI aicore_yolo_ctx* aicore_yolo_load_opts(
        const char* gguf_path, const aicore_yolo_options* opts);
/** Releases a context returned by aicore_yolo_load_opts; safe on NULL. */
AICORE_CAPI void aicore_yolo_free(aicore_yolo_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model. */
AICORE_CAPI int aicore_yolo_is_ready(const aicore_yolo_ctx* ctx);
/** Returns the last error message of the context (empty when none). */
AICORE_CAPI const char* aicore_yolo_last_error(const aicore_yolo_ctx* ctx);

/** Releases any buffer returned by an aicore_yolo_* function (string or
 *  float array; unified entry point). Safe on NULL. */
AICORE_CAPI void aicore_yolo_free_buffer(void* p);

/** Load an image file as tightly-packed RGB (HWC, 3 bytes/pixel). Caller frees
 *  \p out_rgb with aicore_yolo_free_buffer. */
AICORE_CAPI int aicore_yolo_load_path_rgb(const char* image_path,
                                          uint8_t** out_rgb,
                                          int32_t* out_width,
                                          int32_t* out_height);

/** Run detection on a borrowed RGB buffer (HWC, 3 bytes/pixel; no copy).
 *  JSON output:
 *  {"model": "<name>", "task": "detect", "image_size": N,
 *   "num_classes": N, "end2end": 0|1,
 *   "image": {"width": W, "height": H},
 *   "detections":[{"class_id":N,"class_name":"..","score":F,
 *                  "box":[x1,y1,x2,y2]}, ...]}
 *  Boxes are in original-image pixel coordinates. Detection thresholds are
 *  the single configuration point of the context: seed them via
 *  aicore_yolo_options_set_conf_thres / _iou_thres / _top_k at load, or
 *  adjust live via aicore_yolo_set_detect_thresholds.
 *  For depth models the call fails; use aicore_yolo_depth_rgb(). */
AICORE_CAPI char* aicore_yolo_detect_path_json(aicore_yolo_ctx* ctx,
                                               const char* image_path);
/** Same as aicore_yolo_detect_path_json but on a borrowed RGB buffer. */
AICORE_CAPI char* aicore_yolo_detect_rgb_json(aicore_yolo_ctx* ctx,
                                              const uint8_t* rgb,
                                              int32_t width,
                                              int32_t height);
/** Update detection thresholds at runtime without rebuilding the context.
 *  Out-of-range values keep the previous value (0 for top_k = model
 *  max_det). */
AICORE_CAPI void aicore_yolo_set_detect_thresholds(aicore_yolo_ctx* ctx,
                                                   float conf_thres,
                                                   float iou_thres,
                                                   uint32_t top_k);

/** Drop the host-side copies of the model weights to halve the session's
 *  host memory footprint. The device weight buffer is untouched, so
 *  inference and canvas rebuilds keep working. Reload the copies on demand
 *  with aicore_yolo_ensure_host_weights. Returns 0 on success, -1 when the
 *  context has no loaded engine. */
AICORE_CAPI int aicore_yolo_release_host_weights(aicore_yolo_ctx* ctx);
/** Reload released host weight copies from the GGUF file (no-op when they
 *  are present). Returns 0 on success, -1 on failure. */
AICORE_CAPI int aicore_yolo_ensure_host_weights(aicore_yolo_ctx* ctx);

/** Run metric depth estimation on a borrowed RGB buffer. Returns a malloc'd
 *  float array (meters, row-major [out_height, out_width] at the ORIGINAL
 *  image resolution — the map is already restored from the letterbox canvas)
 *  or NULL on failure; free with aicore_yolo_free_buffer(). Summary statistics
 *  of the returned map are available via aicore_yolo_last_depth_json(). */
AICORE_CAPI float* aicore_yolo_depth_path(aicore_yolo_ctx* ctx,
                                          const char* image_path,
                                          int32_t* out_width,
                                          int32_t* out_height);
/** Same as aicore_yolo_depth_path but on a borrowed RGB buffer. */
AICORE_CAPI float* aicore_yolo_depth_rgb(aicore_yolo_ctx* ctx,
                                         const uint8_t* rgb,
                                         int32_t width,
                                         int32_t height,
                                         int32_t* out_width,
                                         int32_t* out_height);

/** Statistics of the most recent aicore_yolo_depth_* call:
 *  {"model": "..", "task": "depth", "image_size": N,
 *   "image": {"width": W, "height": H},
 *   "depth_width": W, "depth_height": H,
 *   "min_depth": F, "max_depth": F, "mean_depth": F, "p95_depth": F,
 *   "valid_pixels": N}
 *  Returns NULL when no depth call has run. */
AICORE_CAPI char* aicore_yolo_last_depth_json(aicore_yolo_ctx* ctx);

/** Model introspection. */
/** Task of the loaded model: "detect", "segment" or "depth" ("" when
 *  not ready). */
AICORE_CAPI const char* aicore_yolo_context_task(aicore_yolo_ctx* ctx);
/** GGUF-declared model name. */
AICORE_CAPI const char* aicore_yolo_context_model_name(aicore_yolo_ctx* ctx);
/** Model input resolution (square, from GGUF metadata). */
AICORE_CAPI uint32_t aicore_yolo_context_image_size(aicore_yolo_ctx* ctx);
/** Number of classes the model was trained on. */
AICORE_CAPI uint32_t aicore_yolo_context_num_classes(aicore_yolo_ctx* ctx);
/** 1 when the head already emits NMS-free detections (yolo26, reg_max=1). */
AICORE_CAPI int aicore_yolo_context_end2end(aicore_yolo_ctx* ctx);
/** Backend-RESOLVED device name ("Vulkan0 (…)", "cpu", ...). Differs from
 *  the requested device when the GPU lease can't be acquired — surfaces
 *  silent CPU fallbacks. Owned by ctx; copy before freeing. */
AICORE_CAPI const char* aicore_yolo_context_device(aicore_yolo_ctx* ctx);
/** Effective CPU thread count after the auto (<=0) resolution. */
AICORE_CAPI int aicore_yolo_context_threads(aicore_yolo_ctx* ctx);

/** Returns a JSON summary of the loaded model. Caller frees with
 *  aicore_yolo_free_buffer. */
AICORE_CAPI char* aicore_yolo_info_json(aicore_yolo_ctx* ctx);
/** Warms up the backend for `device`; returns 0 on success. */
AICORE_CAPI int aicore_yolo_warmup_backend(const char* device);
/** Releases process-wide YOLO backend resources (idempotent). */
AICORE_CAPI void aicore_yolo_shutdown(void);
/** Returns the local model cache directory. Caller frees with
 *  aicore_yolo_free_buffer. */
AICORE_CAPI char* aicore_yolo_model_cache_dir(void);

/** Per-stage wall-clock timings of the most recent aicore_yolo_* inference,
 *  in milliseconds. Mirrors the upstream ultralytics-ggml bench fields so
 *  integrated latency can be compared 1:1 with the upstream matrix:
 *  preprocess (letterbox) / inference (upload + graph + readback) /
 *  postprocess (NMS, mask compose, depth restore) / e2e (all of the above).
 *  json_ms covers only the optional JSON serialization (detect_*_json) and
 *  is NOT part of e2e_ms. All fields stay 0 until the first call. */
typedef struct aicore_yolo_timings {
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double json_ms;
    double e2e_ms;
} aicore_yolo_timings;

/** Copy the most recent successful inference timings into out_timings.
 *  Returns 0 on success, -1 when ctx has never run an inference. */
AICORE_CAPI int aicore_yolo_last_timings(const aicore_yolo_ctx* ctx,
                                         aicore_yolo_timings* out_timings);

/** Published GGUF catalog (cloudViewer_downloads yolo_gguf_models release). */
typedef struct aicore_yolo_model_entry {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
    const char* task;   // "detect" | "segment" | "depth"
    int depth_capable;  // 1 for the yolo26n-depth absolute-depth variants
    int end2end;        // 1 for the yolo26 family (NMS-free head)
} aicore_yolo_model_entry;

/** Catalog role filter for the unified query entry points. */
enum aicore_yolo_model_role {
    AICORE_YOLO_ROLE_ANY = 0,       /**< every catalog entry */
    AICORE_YOLO_ROLE_DETECTION = 1, /**< detection-capable (incl. segment) */
    AICORE_YOLO_ROLE_DEPTH = 2,     /**< absolute-depth variants */
    AICORE_YOLO_ROLE_SEGMENT = 3,   /**< models with a segmentation head */
};

/** Number of catalog entries matching the role filter. */
AICORE_CAPI int aicore_yolo_model_count(enum aicore_yolo_model_role role);
/** Returns the role-filtered catalog entry at `index` (NULL when out of
 *  range). Returned pointers are stable for the process lifetime. */
AICORE_CAPI const aicore_yolo_model_entry* aicore_yolo_model_at(
        int index, enum aicore_yolo_model_role role);
/** Returns the catalog entry whose filename matches (NULL when not
 *  found). */
AICORE_CAPI const aicore_yolo_model_entry* aicore_yolo_model_by_filename(
        const char* filename);
/** Returns the base URL of the published model release. */
AICORE_CAPI const char* aicore_yolo_model_download_base(void);

/** Per-layer result of aicore_yolo_verify_model: 1 = passed, 0 = failed.
 *  Layers without an official catalog baseline (currently the segment
 *  assets) report 1 — their byte count / SHA-256 are not published yet. */
typedef struct aicore_yolo_verify_report {
    int filename_ok; /* basename matches a catalog entry */
    int size_ok;     /* exact byte count (no baseline -> 1) */
    int hash_ok;     /* SHA-256 match (no baseline -> 1) */
    int magic_ok;    /* GGUF magic */
    int task_ok;     /* GGUF yolo.task matches the catalog entry */
} aicore_yolo_verify_report;

/** Verify a downloaded model file against the official catalog: basename,
 *  exact byte count, SHA-256, GGUF magic and yolo.task metadata. Returns 0
 *  when every layer passes, -1 on the first failing layer (out->*_ok shows
 *  which one). Layers without an official baseline are skipped. NULL-safe
 *  (NULL path / NULL out returns -1). */
AICORE_CAPI int aicore_yolo_verify_model(const char* path,
                                         aicore_yolo_verify_report* out);

// ---- Segment API (typed results, not JSON) ----

typedef struct aicore_yolo_segment_result aicore_yolo_segment_result;

/** Typed detection (used by segment result accessors). */
typedef struct aicore_yolo_detection {
    float x1, y1, x2, y2;
    float score;
    int32_t class_id;
} aicore_yolo_detection;

/** Non-owning view of a plane (mask, depth). */
typedef struct aicore_yolo_plane_view {
    const void* data;
    int32_t width;
    int32_t height;
    size_t row_stride_bytes;
} aicore_yolo_plane_view;

/** Run segment inference on a borrowed RGB buffer. Detection thresholds come
 *  from the context (see aicore_yolo_set_detect_thresholds).
 *  Returns NULL on failure; inspect aicore_yolo_last_error() for the reason.
 *  The result is valid until aicore_yolo_seg_result_free(). */
AICORE_CAPI aicore_yolo_segment_result* aicore_yolo_seg_rgb(
        aicore_yolo_ctx* ctx,
        const uint8_t* rgb,
        int32_t width,
        int32_t height);

/** Number of detections in the segment result. */
AICORE_CAPI int aicore_yolo_seg_det_count(
        const aicore_yolo_segment_result* res);

/** Get the i-th detection (shallow copy). */
AICORE_CAPI aicore_yolo_detection
aicore_yolo_seg_det_at(const aicore_yolo_segment_result* res, int index);

/** Get the mask for the i-th detection (borrowed; valid while res lives). */
AICORE_CAPI aicore_yolo_plane_view
aicore_yolo_seg_mask_at(const aicore_yolo_segment_result* res, int index);

/** Release a segment result. Safe on NULL. */
AICORE_CAPI void aicore_yolo_seg_result_free(aicore_yolo_segment_result* res);

#ifdef __cplusplus
}
#endif

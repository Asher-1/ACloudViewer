// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"

#ifdef __cplusplus
extern "C" {
#endif

AICORE_CAPI int aicore_yolo_abi_version(void);

typedef struct aicore_yolo_ctx aicore_yolo_ctx;
typedef struct aicore_yolo_options aicore_yolo_options;

AICORE_CAPI aicore_yolo_options* aicore_yolo_options_new(void);
AICORE_CAPI void aicore_yolo_options_free(aicore_yolo_options* opts);
AICORE_CAPI void aicore_yolo_options_set_device(aicore_yolo_options* opts,
                                                const char* device);
AICORE_CAPI void aicore_yolo_options_set_threads(aicore_yolo_options* opts,
                                                 int n_threads);

/** Load a YOLO GGUF (detection or depth variant; the task is read from the
 *  GGUF's yolo.task metadata). Returns NULL on failure; inspect
 *  aicore_yolo_last_error() for the reason. */
AICORE_CAPI aicore_yolo_ctx* aicore_yolo_load_opts(
        const char* gguf_path, const aicore_yolo_options* opts);
AICORE_CAPI void aicore_yolo_free(aicore_yolo_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model. */
AICORE_CAPI int aicore_yolo_is_ready(const aicore_yolo_ctx* ctx);
AICORE_CAPI const char* aicore_yolo_last_error(const aicore_yolo_ctx* ctx);

AICORE_CAPI void aicore_yolo_free_string(char* s);
AICORE_CAPI void aicore_yolo_free_vec(float* v);

/** Load an image file as tightly-packed RGB (HWC, 3 bytes/pixel). Caller frees
 *  \p out_rgb with aicore_yolo_free_vec (same allocator). */
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
 *  Boxes are in original-image pixel coordinates. \p conf_thres /
 *  \p iou_thres follow the ultralytics predict defaults (0.25 / 0.7);
 *  \p top_k caps the returned detections (0 = model default max_det).
 *  For depth models the call fails; use aicore_yolo_depth_rgb(). */
AICORE_CAPI char* aicore_yolo_detect_path_json(aicore_yolo_ctx* ctx,
                                               const char* image_path,
                                               float conf_thres,
                                               float iou_thres,
                                               uint32_t top_k);
AICORE_CAPI char* aicore_yolo_detect_rgb_json(aicore_yolo_ctx* ctx,
                                              const uint8_t* rgb,
                                              int32_t width,
                                              int32_t height,
                                              float conf_thres,
                                              float iou_thres,
                                              uint32_t top_k);

/** Run metric depth estimation on a borrowed RGB buffer. Returns a malloc'd
 *  float array (meters, row-major [out_height, out_width] at the ORIGINAL
 *  image resolution — the map is already restored from the letterbox canvas)
 *  or NULL on failure; free with aicore_yolo_free_vec(). Summary statistics
 *  of the returned map are available via aicore_yolo_last_depth_json(). */
AICORE_CAPI float* aicore_yolo_depth_path(aicore_yolo_ctx* ctx,
                                          const char* image_path,
                                          int32_t* out_width,
                                          int32_t* out_height);
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
/** Task of the loaded model: "detect" or "depth" ("" when not ready). */
AICORE_CAPI const char* aicore_yolo_context_task(aicore_yolo_ctx* ctx);
AICORE_CAPI const char* aicore_yolo_context_model_name(aicore_yolo_ctx* ctx);
AICORE_CAPI uint32_t aicore_yolo_context_image_size(aicore_yolo_ctx* ctx);
AICORE_CAPI uint32_t aicore_yolo_context_num_classes(aicore_yolo_ctx* ctx);
/** 1 when the head already emits NMS-free detections (yolo26, reg_max=1). */
AICORE_CAPI int aicore_yolo_context_end2end(aicore_yolo_ctx* ctx);
/** Backend-RESOLVED device name ("Vulkan0 (…)", "cpu", ...). Differs from
 *  the requested device when the GPU lease can't be acquired — surfaces
 *  silent CPU fallbacks. Owned by ctx; copy before freeing. */
AICORE_CAPI const char* aicore_yolo_context_device(aicore_yolo_ctx* ctx);
/** Effective CPU thread count after the auto (<=0) resolution. */
AICORE_CAPI int aicore_yolo_context_threads(aicore_yolo_ctx* ctx);

AICORE_CAPI char* aicore_yolo_info_json(aicore_yolo_ctx* ctx);
AICORE_CAPI int aicore_yolo_warmup_backend(const char* device);
AICORE_CAPI void aicore_yolo_shutdown(void);
AICORE_CAPI char* aicore_yolo_model_cache_dir(void);

/** Published GGUF catalog (cloudViewer_downloads yolo_gguf_models release). */
typedef struct aicore_yolo_model_entry {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
    int depth_capable;  // 1 for the yolo26n-depth absolute-depth variants
    int end2end;        // 1 for the yolo26 family (NMS-free head)
} aicore_yolo_model_entry;

AICORE_CAPI int aicore_yolo_model_count(void);
AICORE_CAPI const aicore_yolo_model_entry* aicore_yolo_model_at(int index);
AICORE_CAPI int aicore_yolo_detection_model_count(void);
AICORE_CAPI const aicore_yolo_model_entry* aicore_yolo_detection_model_at(
        int index);
AICORE_CAPI int aicore_yolo_depth_model_count(void);
AICORE_CAPI const aicore_yolo_model_entry* aicore_yolo_depth_model_at(
        int index);
AICORE_CAPI const aicore_yolo_model_entry* aicore_yolo_model_by_filename(
        const char* filename);
AICORE_CAPI const char* aicore_yolo_model_download_base(void);

#ifdef __cplusplus
}
#endif

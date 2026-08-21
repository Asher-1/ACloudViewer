// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// RF-DETR object detection / segmentation C API (in-tree port of
// rf-detr.cpp: https://github.com/mudler/rf-detr.cpp).
//
// The engine lives in core/AICore/src/tasks/rfdetr/ and is accelerated by
// ggml with CPU / CUDA / Vulkan / Metal backends (device auto-pick follows
// the AICore runtime order, e.g. CUDA -> Vulkan -> CPU on Linux).

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the ABI version of the RF-DETR C API (bump on breaking ABI
 *  changes). */
AICORE_CAPI int aicore_rfdetr_abi_version(void);

typedef struct aicore_rfdetr_ctx aicore_rfdetr_ctx;
typedef struct aicore_rfdetr_options aicore_rfdetr_options;

/** Creates a default options struct (device "auto", threads 0 = backend
 *  default). Release with aicore_rfdetr_options_free. */
AICORE_CAPI aicore_rfdetr_options* aicore_rfdetr_options_new(void);
/** Releases an options struct created by aicore_rfdetr_options_new. */
AICORE_CAPI void aicore_rfdetr_options_free(aicore_rfdetr_options* opts);
/** Selects the inference device: NULL or "auto", "cpu", "gpu", "vulkan"
 *  (optionally ":N"), "cuda" (Linux/Windows). */
AICORE_CAPI void aicore_rfdetr_options_set_device(aicore_rfdetr_options* opts,
                                                  const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_rfdetr_options_set_threads(aicore_rfdetr_options* opts,
                                                   int n_threads);

/** Load an RF-DETR GGUF (detection or segmentation variant). Returns NULL on
 *  failure; inspect aicore_rfdetr_last_error() for the reason. */
AICORE_CAPI aicore_rfdetr_ctx* aicore_rfdetr_load_opts(
        const char* gguf_path, const aicore_rfdetr_options* opts);
/** Releases a context returned by aicore_rfdetr_load_opts; safe on NULL. */
AICORE_CAPI void aicore_rfdetr_free(aicore_rfdetr_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model. */
AICORE_CAPI int aicore_rfdetr_is_ready(const aicore_rfdetr_ctx* ctx);
/** Returns the last error message of the context (empty when none). */
AICORE_CAPI const char* aicore_rfdetr_last_error(const aicore_rfdetr_ctx* ctx);

/** Releases any buffer returned by an aicore_rfdetr_* function (string or
 *  float array; unified entry point). Safe on NULL. */
AICORE_CAPI void aicore_rfdetr_free_buffer(void* p);

/** Load an image file as tightly-packed RGB (HWC, 3 bytes/pixel). Caller frees
 *  \p out_rgb with aicore_rfdetr_free_buffer. */
AICORE_CAPI int aicore_rfdetr_load_path_rgb(const char* image_path,
                                            uint8_t** out_rgb,
                                            int32_t* out_width,
                                            int32_t* out_height);

/** Run detection. JSON output:
 *  {"model": "<variant>", "segmentation": 0|1,
 *   "image_size": N, "num_classes": N, "num_queries": N,
 *   "detections":[{"class_id":N,"class_name":"..","score":F,
 *                  "box":[x1,y1,x2,y2]}, ...]}
 *  Boxes are in original-image pixel coordinates. For segmentation models the
 *  per-detection binary mask can be fetched via aicore_rfdetr_detection_mask()
 *  (raw bytes, preferred) or aicore_rfdetr_detection_mask_png() after this
 *  call. */
AICORE_CAPI char* aicore_rfdetr_detect_path_json(aicore_rfdetr_ctx* ctx,
                                                 const char* image_path,
                                                 float threshold,
                                                 uint32_t top_k);
/** Same as aicore_rfdetr_detect_path_json but on a borrowed RGB buffer
 *  (HWC, 3 bytes/pixel). */
AICORE_CAPI char* aicore_rfdetr_detect_rgb_json(aicore_rfdetr_ctx* ctx,
                                                const uint8_t* rgb,
                                                int32_t width,
                                                int32_t height,
                                                float threshold,
                                                uint32_t top_k);

/** Number of detections from the most recent detect call. Returns 0 when no
 *  detect has run or the model has no segmentation head; -1 on invalid ctx. */
AICORE_CAPI int aicore_rfdetr_detection_count(const aicore_rfdetr_ctx* ctx);

/** Raw thresholded binary mask (0/255, row-major) of detection \p index, at
 *  the MODEL resolution (e.g. 640x640 — masks are no longer upsampled to the
 *  image size; display code stretches them over the frame). Two-call sizing:
 *  pass NULL/0 to get the required byte length (>=1), then pass a buffer of
 *  that size. \p out_width / \p out_height receive the mask dimensions and
 *  may be NULL. Returns the required size, 0 when the model has no
 *  segmentation head or the detection has no mask, and -1 on invalid args. */
AICORE_CAPI int aicore_rfdetr_detection_mask(aicore_rfdetr_ctx* ctx,
                                             int index,
                                             unsigned char* buf,
                                             int buf_size,
                                             int32_t* out_width,
                                             int32_t* out_height);

/** PNG-encoded binary mask (0=background, 255=foreground) of detection
 *  \p index, at the MODEL resolution (same geometry as
 *  aicore_rfdetr_detection_mask; the PNG form is encoded on demand for
 *  metadata/export callers — the hot video path should use the raw API).
 *  Two-call sizing: pass NULL/0 to get the required byte length (>=1), then
 *  pass a buffer of that size. Returns the required size, 0 when the model
 *  has no segmentation head or the detection has no mask, and -1 on invalid
 *  arguments. */
AICORE_CAPI int aicore_rfdetr_detection_mask_png(aicore_rfdetr_ctx* ctx,
                                                 int index,
                                                 unsigned char* buf,
                                                 int buf_size);

/** Model introspection. */
/** GGUF-declared model variant (e.g. "rf-detr" / "rf-detr-seg"). */
AICORE_CAPI const char* aicore_rfdetr_context_variant(
        const aicore_rfdetr_ctx* ctx);
/** Model input resolution (square, from GGUF metadata). */
AICORE_CAPI uint32_t
aicore_rfdetr_context_image_size(const aicore_rfdetr_ctx* ctx);
/** Number of classes the model was trained on. */
AICORE_CAPI uint32_t
aicore_rfdetr_context_num_classes(const aicore_rfdetr_ctx* ctx);
/** 1 when the model has a segmentation head. */
AICORE_CAPI int aicore_rfdetr_context_has_segmentation(
        const aicore_rfdetr_ctx* ctx);
/** Backend-RESOLVED device name ("CUDA0", "Vulkan0", "cpu", ...).
 * Differs from the requested device when the GPU lease can't be acquired —
 * surfaces silent CPU fallbacks. Owned by ctx; copy before freeing. */
AICORE_CAPI const char* aicore_rfdetr_context_device(aicore_rfdetr_ctx* ctx);
/** Effective CPU thread count after the auto (<=0) resolution. */
AICORE_CAPI int aicore_rfdetr_context_threads(aicore_rfdetr_ctx* ctx);

/** Returns a JSON summary of the loaded model. Caller frees with
 *  aicore_rfdetr_free_buffer. */
AICORE_CAPI char* aicore_rfdetr_info_json(aicore_rfdetr_ctx* ctx);
/** Warms up the backend for `device`; returns 0 on success. */
AICORE_CAPI int aicore_rfdetr_warmup_backend(const char* device);
/** Releases process-wide RF-DETR backend resources (idempotent). */
AICORE_CAPI void aicore_rfdetr_shutdown(void);
/** Returns the local model cache directory. Caller frees with
 *  aicore_rfdetr_free_buffer. */
AICORE_CAPI char* aicore_rfdetr_model_cache_dir(void);

/** Published GGUF catalog (cloudViewer_downloads RF-DETR-GGUF release). */
typedef struct aicore_rfdetr_model_entry {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
    int segmentation_capable;
} aicore_rfdetr_model_entry;

/** Number of published catalog entries (all model roles). */
AICORE_CAPI int aicore_rfdetr_model_count(void);
/** Returns the catalog entry at `index` (NULL when out of range). */
AICORE_CAPI const aicore_rfdetr_model_entry* aicore_rfdetr_model_at(int index);
/** Number of catalog entries usable for detection. */
AICORE_CAPI int aicore_rfdetr_detection_model_count(void);
/** Returns the detection-capable catalog entry at `index` (NULL when out
 *  of range). */
AICORE_CAPI const aicore_rfdetr_model_entry* aicore_rfdetr_detection_model_at(
        int index);
/** Number of catalog entries with a segmentation head. */
AICORE_CAPI int aicore_rfdetr_segmentation_model_count(void);
/** Returns the segmentation-capable catalog entry at `index` (NULL when
 *  out of range). */
AICORE_CAPI const aicore_rfdetr_model_entry*
aicore_rfdetr_segmentation_model_at(int index);
/** Returns the catalog entry whose filename matches (NULL when not
 *  found). */
AICORE_CAPI const aicore_rfdetr_model_entry* aicore_rfdetr_model_by_filename(
        const char* filename);
/** Returns the base URL of the published model release. */
AICORE_CAPI const char* aicore_rfdetr_model_download_base(void);

#ifdef __cplusplus
}
#endif

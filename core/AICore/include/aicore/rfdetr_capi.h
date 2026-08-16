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

AICORE_CAPI int aicore_rfdetr_abi_version(void);

typedef struct aicore_rfdetr_ctx aicore_rfdetr_ctx;
typedef struct aicore_rfdetr_options aicore_rfdetr_options;

AICORE_CAPI aicore_rfdetr_options* aicore_rfdetr_options_new(void);
AICORE_CAPI void aicore_rfdetr_options_free(aicore_rfdetr_options* opts);
AICORE_CAPI void aicore_rfdetr_options_set_device(aicore_rfdetr_options* opts,
                                                 const char* device);
AICORE_CAPI void aicore_rfdetr_options_set_threads(aicore_rfdetr_options* opts,
                                                  int n_threads);

/** Load an RF-DETR GGUF (detection or segmentation variant). Returns NULL on
 *  failure; inspect aicore_rfdetr_last_error() for the reason. */
AICORE_CAPI aicore_rfdetr_ctx* aicore_rfdetr_load_opts(
        const char* gguf_path, const aicore_rfdetr_options* opts);
AICORE_CAPI void aicore_rfdetr_free(aicore_rfdetr_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model. */
AICORE_CAPI int aicore_rfdetr_is_ready(const aicore_rfdetr_ctx* ctx);
AICORE_CAPI const char* aicore_rfdetr_last_error(const aicore_rfdetr_ctx* ctx);

AICORE_CAPI void aicore_rfdetr_free_string(char* s);
AICORE_CAPI void aicore_rfdetr_free_vec(float* v);

/** Load an image file as tightly-packed RGB (HWC, 3 bytes/pixel). Caller frees
 *  \p out_rgb with aicore_rfdetr_free_vec (same allocator). */
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
 *  per-detection binary mask can be fetched via
 *  aicore_rfdetr_detection_mask_png() after this call. */
AICORE_CAPI char* aicore_rfdetr_detect_path_json(aicore_rfdetr_ctx* ctx,
                                                 const char* image_path,
                                                 float threshold,
                                                 uint32_t top_k);
AICORE_CAPI char* aicore_rfdetr_detect_rgb_json(aicore_rfdetr_ctx* ctx,
                                                const uint8_t* rgb,
                                                int32_t width,
                                                int32_t height,
                                                float threshold,
                                                uint32_t top_k);

/** Number of detections from the most recent detect call. Returns 0 when no
 *  detect has run or the model has no segmentation head; -1 on invalid ctx. */
AICORE_CAPI int aicore_rfdetr_detection_count(const aicore_rfdetr_ctx* ctx);

/** PNG-encoded binary mask (0=background, 255=foreground) of detection
 *  \p index, at the ORIGINAL image resolution. Two-call sizing: pass NULL/0 to
 *  get the required byte length (>=1), then pass a buffer of that size.
 *  Returns the required size, 0 when the model has no segmentation head or the
 *  detection has no mask, and -1 on invalid arguments. */
AICORE_CAPI int aicore_rfdetr_detection_mask_png(aicore_rfdetr_ctx* ctx,
                                                 int index,
                                                 unsigned char* buf,
                                                 int buf_size);

/** Model introspection. */
AICORE_CAPI const char* aicore_rfdetr_context_variant(
        const aicore_rfdetr_ctx* ctx);
AICORE_CAPI uint32_t aicore_rfdetr_context_image_size(
        const aicore_rfdetr_ctx* ctx);
AICORE_CAPI uint32_t aicore_rfdetr_context_num_classes(
        const aicore_rfdetr_ctx* ctx);
AICORE_CAPI int aicore_rfdetr_context_has_segmentation(
        const aicore_rfdetr_ctx* ctx);

AICORE_CAPI char* aicore_rfdetr_info_json(aicore_rfdetr_ctx* ctx);
AICORE_CAPI int aicore_rfdetr_warmup_backend(const char* device);
AICORE_CAPI void aicore_rfdetr_shutdown(void);
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

AICORE_CAPI int aicore_rfdetr_model_count(void);
AICORE_CAPI const aicore_rfdetr_model_entry* aicore_rfdetr_model_at(
        int index);
AICORE_CAPI int aicore_rfdetr_detection_model_count(void);
AICORE_CAPI const aicore_rfdetr_model_entry*
aicore_rfdetr_detection_model_at(int index);
AICORE_CAPI int aicore_rfdetr_segmentation_model_count(void);
AICORE_CAPI const aicore_rfdetr_model_entry*
aicore_rfdetr_segmentation_model_at(int index);
AICORE_CAPI const aicore_rfdetr_model_entry*
aicore_rfdetr_model_by_filename(const char* filename);
AICORE_CAPI const char* aicore_rfdetr_model_download_base(void);

#ifdef __cplusplus
}
#endif

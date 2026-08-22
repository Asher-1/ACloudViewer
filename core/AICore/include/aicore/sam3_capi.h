// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// SAM 2 / SAM 2.1 / SAM 3 image & video segmentation C API.
//
// The ggml engine under core/AICore/src/tasks/sam3/ is an in-tree port of
// sam3-ggml (https://github.com/Asher-1/sam3-ggml), a single-file C++14
// library running Segment Anything 2 / 2.1 / 3 on CPU, CUDA, Vulkan and
// Metal through ggml v0.18.1 (see 3rdparty/ggml/patches/sam3_merged/).
// The upstream source is MIT licensed.
//
// Supported models (all GGUF, published in the cloudViewer_downloads "sam"
// release): sam3-* (full text+detector), sam3-visual-* (no text encoder),
// sam2* / sam2.1* (Hiera backbone, visual-only). sam3-f32 is intentionally
// not published (too large) and not in the catalog.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the ABI version of the SAM3 C API (bump on breaking ABI
 *  changes). */
AICORE_CAPI int aicore_sam3_abi_version(void);

typedef struct aicore_sam3_ctx aicore_sam3_ctx;
typedef struct aicore_sam3_options aicore_sam3_options;
typedef struct aicore_sam3_tracker_ctx aicore_sam3_tracker_ctx;

/** Model family, mirroring sam3_model_type upstream. */
enum aicore_sam3_model_type {
    AICORE_SAM3_MODEL_SAM3        = 0, /**< full SAM3 (ViT + text detector) */
    AICORE_SAM3_MODEL_SAM3_VISUAL = 1, /**< SAM3 visual-only (no text) */
    AICORE_SAM3_MODEL_SAM2        = 2, /**< SAM2 / SAM2.1 Hiera */
};

/** Creates a default options struct (device "auto", threads 0 = backend
 *  default). Release with aicore_sam3_options_free. */
AICORE_CAPI aicore_sam3_options* aicore_sam3_options_new(void);
/** Releases an options struct created by aicore_sam3_options_new. */
AICORE_CAPI void aicore_sam3_options_free(aicore_sam3_options* opts);
/** Selects the inference device: NULL or "auto" (CUDA -> Vulkan -> CPU),
 *  "cpu", "cuda", "vulkan" (Linux/Windows), "metal" (macOS). */
AICORE_CAPI void aicore_sam3_options_set_device(aicore_sam3_options* opts,
                                                const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_sam3_options_set_threads(aicore_sam3_options* opts,
                                                 int n_threads);
/** Override the encoder input resolution (0 = model default, e.g. 1008 for
 *  SAM3, 1024 for SAM2.1). */
AICORE_CAPI void aicore_sam3_options_set_encode_img_size(
        aicore_sam3_options* opts, int img_size);
/** Text-detection score threshold (default 0.5). */
AICORE_CAPI void aicore_sam3_options_set_score_threshold(
        aicore_sam3_options* opts, float score_threshold);
/** Text-detection NMS IoU threshold (default 0.1). */
AICORE_CAPI void aicore_sam3_options_set_nms_threshold(
        aicore_sam3_options* opts, float nms_threshold);
/** Video tracker association IoU threshold (default 0.1). */
AICORE_CAPI void aicore_sam3_options_set_assoc_iou_threshold(
        aicore_sam3_options* opts, float iou_threshold);
/** Video tracker hotstart delay in frames (default 15). */
AICORE_CAPI void aicore_sam3_options_set_hotstart_delay(
        aicore_sam3_options* opts, int frames);
/** Video tracker max keep-alive for lost instances (default 30). */
AICORE_CAPI void aicore_sam3_options_set_max_keep_alive(
        aicore_sam3_options* opts, int frames);
/** Video tracker recondition interval (default 16). */
AICORE_CAPI void aicore_sam3_options_set_recondition_every(
        aicore_sam3_options* opts, int frames);
/** Video tracker hole-fill area threshold (default 16). */
AICORE_CAPI void aicore_sam3_options_set_fill_hole_area(
        aicore_sam3_options* opts, int area);

/** Load a SAM3 GGUF (sam3 / sam3-visual / sam2 / sam2.1 family; the model
 *  type is detected from GGUF metadata). Returns NULL on failure; inspect
 *  aicore_sam3_last_error() for the reason. */
AICORE_CAPI aicore_sam3_ctx* aicore_sam3_load_opts(
        const char* gguf_path, const aicore_sam3_options* opts);
/** Releases a context returned by aicore_sam3_load_opts; safe on NULL. */
AICORE_CAPI void aicore_sam3_free(aicore_sam3_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model. */
AICORE_CAPI int aicore_sam3_is_ready(const aicore_sam3_ctx* ctx);
/** Returns the last error message of the context (empty when none). */
AICORE_CAPI const char* aicore_sam3_last_error(const aicore_sam3_ctx* ctx);
/** Releases any buffer returned by an aicore_sam3_* function (unified entry
 *  point). Safe on NULL. */
AICORE_CAPI void aicore_sam3_free_buffer(void* p);

/** Model introspection. */
/** Model family of the loaded model (see aicore_sam3_model_type). */
AICORE_CAPI int aicore_sam3_context_model_type(const aicore_sam3_ctx* ctx);
/** 1 when the model has no text/detector path (sam3-visual / sam2 family). */
AICORE_CAPI int aicore_sam3_context_visual_only(const aicore_sam3_ctx* ctx);
/** Backend-RESOLVED name ("CUDA", "Vulkan", "Metal", "CPU"); owned by ctx. */
AICORE_CAPI const char* aicore_sam3_context_backend_name(
        const aicore_sam3_ctx* ctx);
/** Effective CPU thread count after the auto (<=0) resolution. */
AICORE_CAPI int aicore_sam3_context_threads(const aicore_sam3_ctx* ctx);

/** Encode an image through the ViT backbone (+ FPN neck when \p pvs_only is
 *  0). Call once per image before aicore_sam3_segment_pcs / _pvs.
 *  \p rgb is a borrowed RGB24 view (row stride >= 3*width) valid during the
 *  call. Returns 0 on success, -1 on failure. */
AICORE_CAPI int aicore_sam3_encode_rgb(aicore_sam3_ctx* ctx,
                                       const uint8_t* rgb,
                                       int32_t width,
                                       int32_t height,
                                       size_t row_stride_bytes,
                                       int pvs_only);
/** 1 when the context has a successfully encoded image. */
AICORE_CAPI int aicore_sam3_has_encoded_image(const aicore_sam3_ctx* ctx);

/** Geometry primitives (image pixel coordinates). */
typedef struct aicore_sam3_point {
    float x;
    float y;
} aicore_sam3_point;

typedef struct aicore_sam3_box {
    float x0; /**< top-left x */
    float y0; /**< top-left y */
    float x1; /**< bottom-right x */
    float y1; /**< bottom-right y */
} aicore_sam3_box;

/** Non-owning view of a binary mask (0/255, 1 byte per pixel). */
typedef struct aicore_sam3_plane_view {
    const void* data;
    int32_t width;
    int32_t height;
    size_t row_stride_bytes;
} aicore_sam3_plane_view;

/** Text-prompt detection (PCS) prompt: text + optional exemplar boxes. */
typedef struct aicore_sam3_pcs_prompt {
    const char* text;                     /**< prompt, e.g. "cat" */
    const aicore_sam3_box* pos_exemplars; /**< NULL when none */
    int n_pos_exemplars;
    const aicore_sam3_box* neg_exemplars; /**< NULL when none */
    int n_neg_exemplars;
    float score_threshold; /**< <= 0 uses the context default */
    float nms_threshold;   /**< <= 0 uses the context default */
} aicore_sam3_pcs_prompt;

/** Point/box segmentation (PVS) prompt. */
typedef struct aicore_sam3_pvs_prompt {
    const aicore_sam3_point* pos_points; /**< NULL when none */
    int n_pos_points;
    const aicore_sam3_point* neg_points; /**< NULL when none */
    int n_neg_points;
    aicore_sam3_box box;
    int use_box;   /**< 1 = box prompt active */
    int multimask; /**< 1 = return the best of 3 masks */
} aicore_sam3_pvs_prompt;

/** Typed segmentation result (detections + masks). */
typedef struct aicore_sam3_seg_result aicore_sam3_seg_result;

/** Run text-prompted segmentation on a borrowed RGB buffer. The image is
 *  encoded if needed (pvs_only=0 so the detector neck runs). Returns NULL on
 *  failure; inspect aicore_sam3_last_error() for the reason. The result is
 *  valid until aicore_sam3_seg_result_free(). */
AICORE_CAPI aicore_sam3_seg_result* aicore_sam3_segment_pcs_rgb(
        aicore_sam3_ctx* ctx,
        const aicore_sam3_pcs_prompt* prompt,
        const uint8_t* rgb,
        int32_t width,
        int32_t height,
        size_t row_stride_bytes);
/** Run point/box segmentation on a borrowed RGB buffer (encode with
 *  pvs_only=1 when the detector neck is not needed). */
AICORE_CAPI aicore_sam3_seg_result* aicore_sam3_segment_pvs_rgb(
        aicore_sam3_ctx* ctx,
        const aicore_sam3_pvs_prompt* prompt,
        const uint8_t* rgb,
        int32_t width,
        int32_t height,
        size_t row_stride_bytes);

/** Number of detections in the segment result. */
AICORE_CAPI int aicore_sam3_seg_det_count(const aicore_sam3_seg_result* res);
/** Detection box of the i-th detection (shallow copy). */
AICORE_CAPI aicore_sam3_box aicore_sam3_seg_det_box_at(
        const aicore_sam3_seg_result* res, int index);
/** Detection score of the i-th detection. */
AICORE_CAPI float aicore_sam3_seg_det_score_at(
        const aicore_sam3_seg_result* res, int index);
/** IoU score of the i-th detection. */
AICORE_CAPI float aicore_sam3_seg_det_iou_at(
        const aicore_sam3_seg_result* res, int index);
/** Instance id of the i-th detection (-1 for one-shot image results). */
AICORE_CAPI int aicore_sam3_seg_det_instance_id_at(
        const aicore_sam3_seg_result* res, int index);
/** Mask of the i-th detection (borrowed; valid while res lives; 0/255 at
 *  original image resolution). */
AICORE_CAPI aicore_sam3_plane_view aicore_sam3_seg_mask_at(
        const aicore_sam3_seg_result* res, int index);
/** Release a segment result. Safe on NULL. */
AICORE_CAPI void aicore_sam3_seg_result_free(aicore_sam3_seg_result* res);

/** ---- Video tracking ---- */

/** Create a tracker bound to the context's model. The tracker owns its own
 *  inference state (independent from the image-mode state). Returns NULL on
 *  failure. */
AICORE_CAPI aicore_sam3_tracker_ctx* aicore_sam3_tracker_create(
        aicore_sam3_ctx* ctx);
/** Set (or clear, with an empty string) the text prompt of a text-prompted
 *  tracker. Must be called before the first track_frame; the tracker
 *  re-reads it on every frame. No-op on visual-only trackers. */
AICORE_CAPI void aicore_sam3_tracker_set_text_prompt(
        aicore_sam3_tracker_ctx* tracker, const char* text);
/** Releases a tracker (safe on NULL). */
AICORE_CAPI void aicore_sam3_tracker_free(aicore_sam3_tracker_ctx* tracker);
/** Track a frame: encode, detect (text models) or propagate (visual-only),
 *  and update the memory bank. Returns NULL on failure. */
AICORE_CAPI aicore_sam3_seg_result* aicore_sam3_track_frame(
        aicore_sam3_tracker_ctx* tracker,
        const uint8_t* rgb,
        int32_t width,
        int32_t height,
        size_t row_stride_bytes);
/** Visual-only propagation: encode + propagate all tracked instances
 *  (no detection step). Returns NULL on failure. */
AICORE_CAPI aicore_sam3_seg_result* aicore_sam3_propagate_frame(
        aicore_sam3_tracker_ctx* tracker,
        const uint8_t* rgb,
        int32_t width,
        int32_t height,
        size_t row_stride_bytes);
/** Add a tracked instance from PVS prompts on the current frame. The frame
 *  must have been processed by track_frame/propagate_frame or encoded via
 *  aicore_sam3_encode_rgb. Returns the assigned instance id, or -1. */
AICORE_CAPI int aicore_sam3_tracker_add_instance(
        aicore_sam3_tracker_ctx* tracker,
        const aicore_sam3_pvs_prompt* prompt);
/** Add a tracked instance from an existing binary mask (0/255 view) on the
 *  current frame. Returns the assigned instance id, or -1. */
AICORE_CAPI int aicore_sam3_tracker_add_instance_from_mask(
        aicore_sam3_tracker_ctx* tracker,
        const aicore_sam3_plane_view* mask);
/** Refine a tracked instance with interactive points. Returns 0 on success,
 *  -1 on failure. */
AICORE_CAPI int aicore_sam3_refine_instance(
        aicore_sam3_tracker_ctx* tracker,
        int instance_id,
        const aicore_sam3_point* pos_points,
        int n_pos_points,
        const aicore_sam3_point* neg_points,
        int n_neg_points);
/** Current frame index of the tracker. */
AICORE_CAPI int aicore_sam3_tracker_frame_index(
        const aicore_sam3_tracker_ctx* tracker);
/** Reset the tracker, clearing all instances and memory. */
AICORE_CAPI void aicore_sam3_tracker_reset(aicore_sam3_tracker_ctx* tracker);

/** ---- Timings ---- */

/** Per-stage wall-clock timings of the most recent inference, in
 *  milliseconds. preprocess covers the letterbox/CHW conversion, inference
 *  covers the ggml graph compute, postprocess covers mask/box extraction,
 *  e2e covers all of the above. All fields stay 0 until the first call. */
typedef struct aicore_sam3_timings {
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double e2e_ms;
} aicore_sam3_timings;

/** Copy the most recent successful inference timings into out_timings.
 *  Returns 0 on success, -1 when ctx has never run an inference. */
AICORE_CAPI int aicore_sam3_last_timings(const aicore_sam3_ctx* ctx,
                                         aicore_sam3_timings* out_timings);

/** ---- Published model catalog (cloudViewer_downloads "sam" release) ---- */

typedef struct aicore_sam3_model_entry {
    const char* filename;      /**< e.g. "sam3-f16.gguf" */
    const char* download_url;  /**< full download URL */
    const char* display_name;  /**< human-readable */
    const char* quant_note;    /**< quantization description */
    const char* model_family;  /**< "sam3" | "sam3-visual" | "sam2.1" | "sam2" */
    int64_t size_bytes;        /**< published asset size */
    int visual_only;           /**< 1 = no text prompt support */
} aicore_sam3_model_entry;

/** Number of catalog entries. */
AICORE_CAPI int aicore_sam3_model_count(void);
/** Returns the catalog entry at `index` (NULL when out of range). Returned
 *  pointers are stable for the process lifetime. */
AICORE_CAPI const aicore_sam3_model_entry* aicore_sam3_model_at(int index);
/** Returns the catalog entry whose filename matches (NULL when not
 *  found). */
AICORE_CAPI const aicore_sam3_model_entry* aicore_sam3_model_by_filename(
        const char* filename);
/** Returns the base URL of the published model release. */
AICORE_CAPI const char* aicore_sam3_model_download_base(void);

/** ---- Process-wide helpers ---- */

/** Warms up the backend for `device`; returns 0 on success. */
AICORE_CAPI int aicore_sam3_warmup_backend(const char* device);
/** Releases process-wide SAM3 backend resources (idempotent). */
AICORE_CAPI void aicore_sam3_shutdown(void);

#ifdef __cplusplus
}
#endif

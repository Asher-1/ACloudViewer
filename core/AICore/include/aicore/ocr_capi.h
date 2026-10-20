// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef AICORE_OCR_CAPI_H
#define AICORE_OCR_CAPI_H

#include "aicore/export.h"
#include "aicore/image_view.h"
#include "aicore/pipeline_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

/** OCR model catalog.
 *
 * This header publishes the PP-OCRv4 catalog entries converted by the
 * first-party converter (core/AICore/tools/convert_paddle_ocr_gguf.py) and
 * hosted in the PP-OCR release of cloudViewer_downloads. The catalog and
 * inference session/result APIs below are the public OCR contract; graph,
 * preprocessing, postprocessing, model state, and backend details remain
 * task-private and are validated by the manifest-owned accuracy gates.
 */

typedef struct aicore_ocr_model_entry {
    /** Release asset name, e.g. "ppocr-det-ch-f32.gguf". */
    const char* filename;
    /** Stable download URL. */
    const char* download_url;
    /** User-facing display name. */
    const char* display_name;
    /** "det", "rec", or "layout". */
    const char* role;
    /** "ch", "en", or "cdla". */
    const char* language;
} aicore_ocr_model_entry;

/** Number of published OCR catalog entries. */
AICORE_CAPI int aicore_ocr_model_count(void);

/** Returns the catalog entry at `index` (NULL when out of range). The
 *  returned pointer is valid until the next call on this thread. */
AICORE_CAPI const aicore_ocr_model_entry* aicore_ocr_model_at(int index);

/** Lookup by release asset name (NULL when unknown). */
AICORE_CAPI const aicore_ocr_model_entry* aicore_ocr_model_by_filename(
        const char* filename);

/** Release asset base URL (stable, ends with '/'). */
AICORE_CAPI const char* aicore_ocr_model_download_base(void);

/** Task-owned model cache directory (~/cloudViewer_data/extract/
 *  ocr_models). Caller frees with aicore_ocr_free_buffer. */
AICORE_CAPI char* aicore_ocr_model_cache_dir(void);

/** Free a buffer returned by this API (NULL is a no-op). */
AICORE_CAPI void aicore_ocr_free_buffer(char* buffer);

/** X-AnyLabeling pipeline recipes, independent of release availability.
 * Stems become <stem>-{f32,f16,q8_0}.gguf. A recipe does not assert that its
 * weights are published; use model_by_filename() for downloadable assets.
 * Unpublished recipes accept reproducibly converted local GGUFs only.
 * Strings and entries have process lifetime. */
typedef struct aicore_ocr_preset_entry {
    const char* id;
    const char* display_name;
    const char* det_stem;
    const char* rec_stem;
    const char* cls_stem;
    float det_thresh;
    float det_box_thresh;
    float det_unclip_ratio;
    int32_t det_max_candidates;
    int32_t det_max_side;
    int32_t rec_batch;
    int32_t cls_batch;
    float cls_thresh;
    float drop_score;
} aicore_ocr_preset_entry;

AICORE_CAPI int aicore_ocr_preset_count(void);
AICORE_CAPI const aicore_ocr_preset_entry* aicore_ocr_preset_at(int index);
AICORE_CAPI const aicore_ocr_preset_entry* aicore_ocr_preset_by_rec_filename(
        const char* filename);

/** ---- Inference contract (implemented by the `ocr` task runtime) ----
 *
 * Lifecycle: create a context from resolved catalog or converted local GGUF
 * paths, poll readiness, run images, release every result, destroy the context
 * on the owning thread, and let process shutdown delegate to
 * aicore_runtime_shutdown(). All failure paths surface through
 * aicore_ocr_last_error(). Images cross as borrowed aicore_image_view
 * values with their actual stride and RGB/RGBA/GRAY/BGR/BGRA format.
 *
 * One run executes the audited det graph (DB probability map), crops text
 * regions, optionally classifies/rotates their text direction, and executes
 * the rec graph per crop, returning typed
 * result arrays owned by the result container:
 *   - boxes:      text quadrilateral + UTF-8 transcription + score + script
 *   - regions:    layout class + reading order (when a layout model loads)
 *   - relations:  reserved KIE index pairs; empty in ABI v1
 *   - orientations: optional text-line angle/probability/rotation per box
 */

typedef struct aicore_ocr_options {
    size_t struct_size;            /**< sizeof(aicore_ocr_options) */
    uint32_t abi_version;          /**< AICORE_OCR_ABI_VERSION */
    const char* det_model_path;    /**< det GGUF (required) */
    const char* rec_model_path;    /**< rec GGUF (required) */
    const char* layout_model_path; /**< layout GGUF (optional, NULL) */
    /** Reserved for a future KIE contract. Must be NULL in ABI v1; a non-empty
     *  value is rejected instead of being silently ignored. */
    const char* kie_model_path;
    int32_t det_max_side;   /**< det resize limit (0 = model default) */
    float det_box_thresh;   /**< DB binarization threshold (0 = default) */
    float det_unclip_ratio; /**< polygon expansion (0 = default) */
    /** Official text_recognition_batch_size: crops are sorted by aspect
     *  ratio and every group of this many is padded to the group's widest
     *  crop, exactly like the PaddleOCR pipeline. The pad width changes the
     *  recognized text, so this is a semantic option, not a speed knob; the
     *  forward chunking is independent of it (0 = official 6). */
    int32_t rec_max_batch;
    /** Device request: NULL or "auto" follows the platform order (CUDA ->
     *  Vulkan -> CPU on Linux/Windows, Metal -> CPU on macOS); "cpu" or an
     *  explicit backend id ("cuda", "vulkan:1", ...) selects one. Appended
     *  after rec_max_batch; callers that keep struct_size at the previous
     *  sizeof get the historical auto behavior. */
    const char* device;
    /** CPU thread count for the CPU backend (<= 0 = backend default). */
    int32_t n_threads;
    /** DB box-score filter: a candidate whose mean probability inside the
     *  polygon falls below this is dropped (official PP-OCR det
     *  box_thresh, default 0.6; <= 0 keeps the official default). Appended
     *  in v1.1; callers with the older struct_size get the default. */
    float det_box_score_thresh;
    /** det resize mode: 0 = PaddleOCR 3.0.1 OCR default ("min": scale the
     *  SHORT side up to det_max_side when it is below it; det_max_side is
     *  limit_side_len and defaults to 64), 1 = "max" cap (never upscale,
     *  long side <= det_max_side; det_max_side 960 matches the official
     *  table_recognition pipeline). Appended in v1.1. */
    int32_t det_limit_type;
    /** Optional 0/180-degree text-line classifier GGUF. NULL preserves the
     *  historical detection/recognition pipeline. Applied to both run() and
     *  recognize(), after perspective crop/rot90 and before recognition. */
    const char* cls_model_path;
    /** Rotate only when the 180-degree probability is strictly greater than
     *  this threshold (0 = upstream 0.9). */
    float cls_thresh;
    /** Aspect-sorted classifier batch size (0 = upstream 6). */
    int32_t cls_max_batch;
    /** DB contour candidate limit (0 = upstream v4/v5 1000; v6 uses 3000). */
    int32_t det_max_candidates;
} aicore_ocr_options;

#define AICORE_OCR_ABI_VERSION 1u

typedef struct aicore_ocr_box {
    float quad[4][2]; /**< corner polygon, image pixel coordinates */
    const char* text; /**< UTF-8 transcription (owned by the result) */
    float score;      /**< recognition confidence */
    char language[8]; /**< script tag from the rec catalog entry */
} aicore_ocr_box;

typedef struct aicore_ocr_region {
    int32_t class_id;      /**< layout region class */
    int32_t reading_order; /**< -1 when the layout head is absent */
    float quad[4][2];
    float score;
} aicore_ocr_region;

/** Reserved KIE relation shape. ABI v1 returns zero relations because KIE
 *  model execution is not part of the published OCR contract. */
typedef struct aicore_ocr_relation {
    int32_t head_index; /**< box index */
    int32_t tail_index; /**< box index */
    int32_t role;       /**< KIE role class id */
    float score;
} aicore_ocr_relation;

/** Optional OCR post-process breakdown for the last result. These wall-clock
 *  intervals include parallel worker lifetime and are diagnostic/benchmark
 *  evidence, not a replacement for aicore_pipeline_timings. */
#define AICORE_OCR_POST_TIMINGS_ABI_VERSION 2u

enum {
    AICORE_OCR_POST_TIMING_DB_BOXES = 1u << 0,
    AICORE_OCR_POST_TIMING_CROP_RESIZE_PACK = 1u << 1,
    AICORE_OCR_POST_TIMING_CTC_DECODE = 1u << 2,
    AICORE_OCR_POST_TIMING_RESULT_ASSEMBLY = 1u << 3,
    AICORE_OCR_POST_TIMING_OUTPUT_READBACK = 1u << 4,
    AICORE_OCR_POST_TIMING_INPUT_UPLOAD = 1u << 5,
    /** 独立图计算时间；用于定位 det 与 rec 的最大执行热点。 */
    AICORE_OCR_POST_TIMING_DET_GRAPH_COMPUTE = 1u << 6,
    AICORE_OCR_POST_TIMING_REC_GRAPH_COMPUTE = 1u << 7,
};

typedef struct aicore_ocr_post_timing_breakdown {
    uint32_t abi_version;
    uint32_t valid_fields;
    double db_boxes_ms;
    double crop_resize_pack_ms;
    double ctc_decode_ms;
    double result_assembly_ms;
    /** Device-to-host output transfer; graph compute remains in inference. */
    double output_readback_ms;
    /** Host-to-device graph-input transfer; excludes graph compute. */
    double input_upload_ms;
    /** DBNet 图计算时间；不包含传输或后处理。v2 追加字段。 */
    double det_graph_compute_ms;
    /** 所有 SVTR rec 图计算时间；不包含传输或 CTC。v2 追加字段。 */
    double rec_graph_compute_ms;
} aicore_ocr_post_timing_breakdown;

typedef struct aicore_ocr_result aicore_ocr_result;
typedef struct aicore_ocr_ctx aicore_ocr_ctx;

/** Classifier decision corresponding to one returned text box. angle is
 *  the predicted label (0 or 180); rotated indicates that the strict
 *  confidence gate actually rotated the recognition crop. */
typedef struct aicore_ocr_orientation {
    int32_t angle;
    float score;
    int32_t rotated;
} aicore_ocr_orientation;

/** NULL when no classifier was requested, index is invalid, or result is
 *  NULL. The pointer is borrowed from result. */
AICORE_CAPI const aicore_ocr_orientation* aicore_ocr_orientation_at(
        const aicore_ocr_result* result, int index);

/** Create a context. Returns NULL with aicore_ocr_last_error() set when a
 *  model file is missing, a digest mismatch is detected, or the embedded
 *  graph topology is unsupported. */
AICORE_CAPI aicore_ocr_ctx* aicore_ocr_context_create(
        const aicore_ocr_options* options);

/** Destroy a context created on this thread. NULL is a no-op. */
AICORE_CAPI void aicore_ocr_context_destroy(aicore_ocr_ctx* ctx);

/** Backend-resolved device name of the context ("CUDA0", "Vulkan0",
 *  "cpu", ...). Owned by the context; valid until
 *  aicore_ocr_context_destroy. Empty string for NULL. */
AICORE_CAPI const char* aicore_ocr_context_device(const aicore_ocr_ctx* ctx);

/** Non-zero once every requested graph has its weights resident and the
 *  backends are ready. */
AICORE_CAPI int aicore_ocr_context_ready(const aicore_ocr_ctx* ctx);

/** Run OCR over a borrowed image. Returns NULL on failure (check
 *  aicore_ocr_last_error()); the result stays valid until
 *  aicore_ocr_result_release(). */
AICORE_CAPI aicore_ocr_result* aicore_ocr_run(aicore_ocr_ctx* ctx,
                                              const aicore_image_view* image);

/** Recognize text inside caller-provided quads (skip-detection path).
 *  `quads` holds `count` clockwise-from-top-left corner polygons in image
 *  pixel coordinates, row-major (count * 4 * 2 floats). Each quad is
 *  clipped to the image and cropped exactly like a det box (direct quad
 *  for ONNX-lowered models, integer minimum-area rectangle for PaddleX),
 *  so feeding back the quads of
 *  aicore_ocr_run() reproduces its texts. Returns exactly `count` boxes in
 *  input order (no det pass); a degenerate or non-finite quad yields an
 *  empty text with score 0. Same ownership rules as aicore_ocr_run(). */
AICORE_CAPI aicore_ocr_result* aicore_ocr_recognize(
        aicore_ocr_ctx* ctx,
        const aicore_image_view* image,
        const float* quads,
        int32_t count);

/** Borrowed selected annotation polygon. points_xy contains point_count
 *  consecutive x,y pairs; storage is needed only for the synchronous call. */
typedef struct aicore_ocr_text_region {
    const float* points_xy;
    int32_t point_count;
} aicore_ocr_text_region;

/** Recognize selected annotation polygons with the X-AnyLabeling contract:
 *  truncate coordinates to integer pixels, fit a minimum-area rectangle,
 *  order top/bottom pairs, then warp with border replication (no clipping).
 *  Returns count boxes in input order, containing the normalized quads;
 *  callers retain the original polygon and annotation metadata. Regions
 *  need at least three finite points representable as int32. Malformed
 *  inputs fail the call; a degenerate region yields empty text/score zero.
 *  Uses the same classifier, batching, timings and ownership as recognize(). */
AICORE_CAPI aicore_ocr_result* aicore_ocr_recognize_regions(
        aicore_ocr_ctx* ctx,
        const aicore_image_view* image,
        const aicore_ocr_text_region* regions,
        int32_t count);

AICORE_CAPI int aicore_ocr_box_count(const aicore_ocr_result* result);
AICORE_CAPI const aicore_ocr_box* aicore_ocr_box_at(
        const aicore_ocr_result* result, int index);
/** Layout class-table accessors: class names embedded in the layout
 *  GGUF ("ocr.labels"); count is 0 without a layout model. */
AICORE_CAPI int aicore_ocr_label_count(const aicore_ocr_ctx* ctx);

AICORE_CAPI const char* aicore_ocr_label_at(const aicore_ocr_ctx* ctx,
                                            int index);

AICORE_CAPI int aicore_ocr_region_count(const aicore_ocr_result* result);
AICORE_CAPI const aicore_ocr_region* aicore_ocr_region_at(
        const aicore_ocr_result* result, int index);
AICORE_CAPI int aicore_ocr_relation_count(const aicore_ocr_result* result);
AICORE_CAPI const aicore_ocr_relation* aicore_ocr_relation_at(
        const aicore_ocr_result* result, int index);

/** Free a result (NULL is a no-op). */
AICORE_CAPI void aicore_ocr_result_release(aicore_ocr_result* result);

/** Common stage timings of the last run (valid_fields marks truth). */
AICORE_CAPI aicore_pipeline_timings
aicore_ocr_timings(const aicore_ocr_result* result);

/** OCR post-process substage timings of the last run (valid_fields marks
 *  available stages; NULL returns a zero-initialized breakdown). */
AICORE_CAPI aicore_ocr_post_timing_breakdown
aicore_ocr_post_timings(const aicore_ocr_result* result);

/** Thread-local error string of the last failing call (never NULL). */
AICORE_CAPI const char* aicore_ocr_last_error(void);

/** Idempotent process-level shutdown; delegates to
 *  aicore_runtime_shutdown() and never destroys live contexts. */
AICORE_CAPI void aicore_ocr_runtime_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* AICORE_OCR_CAPI_H */

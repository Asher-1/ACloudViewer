// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef AICORE_TABLE_CAPI_H
#define AICORE_TABLE_CAPI_H

#include "aicore/export.h"
#include "aicore/image_view.h"
#include "aicore/ocr_capi.h"
#include "aicore/pipeline_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Table structure recognition (SLANet_plus) and table reconstruction.
 *
 * The `table` task predicts the HTML structure tokens of a table image and
 * one 4-point polygon per cell. Reconstruction composes it with a caller
 * owned OCR result through the public OCR accessors only:
 *
 *   image + optional aicore_ocr_result (boxes + layout regions)
 *     -> table regions (layout "table" class, or the whole image)
 *     -> SLANet structure tokens + cell polygons per region
 *     -> OCR boxes assigned to cells (IoU, then L1 distance)
 *     -> typed tables/cells + HTML + CSV owned by aicore_table_result
 *
 * Lifecycle follows the OCR task: create a context on the worker thread,
 * release every result, destroy the context on the same thread, and let
 * process shutdown delegate to aicore_runtime_shutdown().
 */

typedef struct aicore_table_model_entry {
    /** Release asset name, e.g. "slanet-plus-table-f16.gguf". */
    const char* filename;
    /** Stable download URL. */
    const char* download_url;
    /** User-facing display name. */
    const char* display_name;
    /** Model role: "table". */
    const char* role;
    /** Source architecture: "slanet_plus". */
    const char* arch;
} aicore_table_model_entry;

AICORE_CAPI int aicore_table_model_count(void);
/** Catalog entry at `index` (NULL when out of range); valid until the next
 *  call on this thread. */
AICORE_CAPI const aicore_table_model_entry* aicore_table_model_at(int index);
AICORE_CAPI const aicore_table_model_entry* aicore_table_model_by_filename(
        const char* filename);
/** Release asset base URL (stable, ends with '/'). */
AICORE_CAPI const char* aicore_table_model_download_base(void);
/** Model cache directory (shared with OCR: ~/cloudViewer_data/extract/
 *  ocr_models). Caller frees with aicore_table_free_buffer. */
AICORE_CAPI char* aicore_table_model_cache_dir(void);
AICORE_CAPI void aicore_table_free_buffer(char* buffer);

typedef struct aicore_table_options {
    size_t struct_size;     /**< sizeof(aicore_table_options) */
    uint32_t abi_version;   /**< AICORE_TABLE_ABI_VERSION */
    const char* model_path; /**< SLANet table GGUF (required) */
    /** Device request: NULL/"auto", "cpu", "cuda", "vulkan:1", ... */
    const char* device;
    int32_t n_threads; /**< CPU threads (<= 0 = default) */
} aicore_table_options;

#define AICORE_TABLE_ABI_VERSION 1u

/** Per-run reconstruction options. */
typedef struct aicore_table_run_options {
    size_t struct_size;   /**< sizeof(aicore_table_run_options) */
    uint32_t abi_version; /**< AICORE_TABLE_ABI_VERSION */
    /** 1: one table per layout region whose class label is "table" (needs
     *  an OCR result produced with a layout model); 0: the whole image is
     *  one table. */
    int32_t use_layout_regions;
    /** Minimum layout region score for a table crop (<= 0 keeps every
     *  region the OCR layout stage returned; that stage already applies
     *  the PP-Structure layout_score_threshold 0.5). */
    float min_region_score;
    /** Bit i = layout class id i is accepted as a table crop when the page
     *  contains no "table"-class region. The CDLA layout model labels some
     *  synthetic-border tables "figure" (measured: 36/100 composed pages,
     *  region IoU 0.54-0.99, zero "table" regions); the fallback recovers
     *  them. At most one fallback crop per page (highest region score).
     *  0 keeps the upstream behavior ("table" class only). Callers that do
     *  not extend their struct to this field get 0; class ids index
     *  aicore_ocr_label_at of the OCR context's layout labels. */
    uint32_t fallback_region_mask;
} aicore_table_run_options;

typedef struct aicore_table_cell {
    /** Predicted cell polygon (clockwise from top-left), image pixels. */
    float quad[4][2];
    /** Axis-aligned [xmin, ymin, xmax, ymax], clipped to the image. */
    float box[4];
    int32_t row;      /**< grid row of the top-left slot */
    int32_t col;      /**< grid column of the top-left slot */
    int32_t row_span; /**< >= 1 */
    int32_t col_span; /**< >= 1 */
    /** UTF-8 cell text assembled from the assigned OCR boxes (owned by the
     *  result; empty without an OCR result). */
    const char* text;
    int32_t ocr_box_count; /**< OCR boxes assigned to this cell */
} aicore_table_cell;

typedef struct aicore_table {
    /** Table region [xmin, ymin, xmax, ymax] in image pixels. */
    float box[4];
    /** Mean max-probability of the emitted structure tokens. */
    float structure_score;
    /** Source layout region index, -1 for the whole-image table. */
    int32_t layout_region;
    int32_t rows;
    int32_t cols;
    /** Cells of this table: [cell_offset, cell_offset + cell_count) in the
     *  result-wide cell array. */
    int32_t cell_offset;
    int32_t cell_count;
    /** Structure token count (excluding the html/body/table wrapper). */
    int32_t token_count;
    /** `<html><body><table>...</table></body></html>` with cell text. */
    const char* html;
    /** RFC 4180 CSV of the grid; a spanned cell's text sits in its top-left
     *  slot. */
    const char* csv;
} aicore_table;

typedef struct aicore_table_result aicore_table_result;
typedef struct aicore_table_ctx aicore_table_ctx;

/** Create a context. Returns NULL with aicore_table_last_error() set when
 *  the model is missing, invalid, or the device is unavailable. */
AICORE_CAPI aicore_table_ctx* aicore_table_context_create(
        const aicore_table_options* options);
AICORE_CAPI void aicore_table_context_destroy(aicore_table_ctx* ctx);
AICORE_CAPI int aicore_table_context_ready(const aicore_table_ctx* ctx);
/** Backend-resolved device name ("CUDA0", "Vulkan0", "CPU", ...). */
AICORE_CAPI const char* aicore_table_context_device(
        const aicore_table_ctx* ctx);

/** Reconstruct tables in a borrowed image. `ocr_result` (optional, NULL)
 *  supplies text boxes and layout regions, and `ocr_ctx` its layout class
 *  labels; both are borrowed for the call. Without an OCR result the cells
 *  carry structure only. `options` may be NULL (whole image, no layout).
 *  Returns NULL on failure (see aicore_table_last_error()). */
AICORE_CAPI aicore_table_result* aicore_table_run(
        aicore_table_ctx* ctx,
        const aicore_image_view* image,
        const aicore_ocr_ctx* ocr_ctx,
        const aicore_ocr_result* ocr_result,
        const aicore_table_run_options* options);

AICORE_CAPI int aicore_table_count(const aicore_table_result* result);
AICORE_CAPI const aicore_table* aicore_table_at(
        const aicore_table_result* result, int index);
AICORE_CAPI int aicore_table_cell_count(const aicore_table_result* result);
AICORE_CAPI const aicore_table_cell* aicore_table_cell_at(
        const aicore_table_result* result, int index);
/** Structure token `index` of table `table_index` (NULL when out of
 *  range); owned by the result. */
AICORE_CAPI const char* aicore_table_token_at(const aicore_table_result* result,
                                              int table_index,
                                              int index);

AICORE_CAPI void aicore_table_result_release(aicore_table_result* result);

/** Common stage timings: preprocess = crop/resize/pack, inference =
 *  encoder + structure decoder, postprocess = decode + OCR assignment +
 *  HTML/CSV. */
AICORE_CAPI aicore_pipeline_timings
aicore_table_timings(const aicore_table_result* result);

/** Thread-local error string of the last failing call (never NULL). */
AICORE_CAPI const char* aicore_table_last_error(void);

/** Idempotent process-level shutdown; delegates to
 *  aicore_runtime_shutdown() and never destroys live contexts. */
AICORE_CAPI void aicore_table_runtime_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* AICORE_TABLE_CAPI_H */

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
#include "aicore/image_view.h"
#include "aicore/pipeline_timing.h"
#ifdef __cplusplus
extern "C" {
#endif
#define AICORE_RELATION_ABI_VERSION 1u
typedef struct aicore_relation_ctx aicore_relation_ctx;
/** Borrow strings during create. Serialize calls on one context. All resident
 * weights, graphs and caches belong to the context. Devices use BackendLease.
 */
typedef struct aicore_relation_options {
    uint32_t struct_size, abi_version;
    const char *model_path, *device;
    int32_t threads;
    uint32_t max_boxes, topk;
    /** -1 (default): upstream automatic precision, strict for CUDA F32
     * matrices; 0: optimized arithmetic; 1: always strict F32 arithmetic. */
    int32_t strict_f32;
    int32_t quant_f32_input, raw_logits, decompose;
    int32_t flash_attention, compact_pairs, fused_vocabulary, cache_positions,
            retain_features;
    const char *vocabulary_path, *vocabulary_names_path,
            *predicate_type_map_path;
    /** Optional native text student + official CLIP tokenizer. When predicate
     * texts are supplied, encode directly into the relation vocabulary in
     * memory. */
    const char *text_student_path, *tokenizer_path;
    const char *const *predicate_texts;
    size_t predicate_text_count;
    /** Text-only context: model_path names a text student; use encode_text. */
    int32_t text_only;
    /** Optional diagnostic tensor dumps; unset in normal inference/benchmarks.
     */
    const char *diagnostic_directory;
} aicore_relation_options;
typedef struct aicore_relation_request {
    uint32_t struct_size, abi_version;
    const float *boxes_xyxy; /**< Original-image pixel coordinates [N,4]. */
    size_t object_count;
    const float *box_scores; /**< Optional [N]; borrowed through return. */
    const uint8_t *masks;    /**< Optional contiguous binary [N,H,W]. */
    size_t mask_bytes;
} aicore_relation_request;
typedef struct aicore_relation_edge {
    uint32_t subject, object, predicate;
    float score;
} aicore_relation_edge;
/** Independent ownership, including predicate names and raw tensors; valid
 * after context destruction. Release exactly once with result_free.
 * Raw arrays are [pair_count,predicate_count], in returned candidate-ID order.
 */
typedef struct aicore_relation_result {
    uint32_t struct_size, abi_version;
    size_t relation_count, spatial_count, semantic_count, pair_count,
            predicate_count;
    const aicore_relation_edge *relations, *spatial_relations,
            *semantic_relations;
    const float *predicate_logits, *pair_logits;
    const int32_t *subject_indices, *object_indices, *valid_mask;
    const char *const *predicate_names;
    aicore_pipeline_timings timings;
    size_t backend_buffer_bytes;
    double forward_ms; /**< Upstream FullModel::infer duration, including graph
                          construction/ranking. */
    void *private_owner;
    /** [predicate_count] 1 = spatial predicate (corpus type map, else the
     * head's routing gate); NULL unless the context was created with
     * decompose. Appended after private_owner: older readers ignore it. */
    const uint8_t *predicate_spatial;
} aicore_relation_result;
/** Independently owned normalized row-major [count,512] native student output.
 */
typedef struct aicore_relation_embeddings {
    uint32_t struct_size, abi_version;
    size_t count, dimension;
    const float *values;
    aicore_pipeline_timings timings;
    void *private_owner;
} aicore_relation_embeddings;
enum aicore_relation_model_role {
    AICORE_RELATION_MODEL_RELATIONS = 0,
    AICORE_RELATION_MODEL_TEXT = 1,
    AICORE_RELATION_MODEL_TOKENIZER = 2
};
typedef struct aicore_relation_model_entry {
    const char *filename, *download_url, *display_name, *architecture, *sha256;
    int64_t size_bytes;
    int32_t role;
} aicore_relation_model_entry;
AICORE_CAPI aicore_relation_options aicore_relation_default_options(void);
AICORE_CAPI aicore_relation_request aicore_relation_default_request(void);
/** Failed load returns an error context; NULL only on allocation failure. */
AICORE_CAPI aicore_relation_ctx *aicore_relation_create(
        const aicore_relation_options *options);
AICORE_CAPI void aicore_relation_free(aicore_relation_ctx *ctx);
AICORE_CAPI int aicore_relation_ready(const aicore_relation_ctx *ctx);
AICORE_CAPI const char *aicore_relation_last_error(
        const aicore_relation_ctx *ctx);
AICORE_CAPI const char *aicore_relation_context_device(
        const aicore_relation_ctx *ctx);
/** Score calibration stored in the model (sigmoid(a * logit + b)). The
 * decomposed spatial/semantic streams report sigmoid(logit) exactly as
 * upstream's native path does; callers apply a/b to match the calibrated
 * flat list and upstream's Python contract. Returns 0, or -1 when the
 * context is not a loaded relation model. */
AICORE_CAPI int aicore_relation_calibration(const aicore_relation_ctx *ctx,
                                            float *a,
                                            float *b);
/** Borrow image/regions through return. Clear output on failure. Return 0 for
 * success, 1 for cooperative cancellation, -1 for error. Shared cancel token
 * scopes apply; cancellation is checked before and after graph execution. */
AICORE_CAPI int aicore_relation_infer_image(
        aicore_relation_ctx *ctx,
        const aicore_image_view *image,
        const aicore_relation_request *request,
        aicore_relation_result **output);
AICORE_CAPI void aicore_relation_result_free(aicore_relation_result *result);
/** Retains student weights and tokenizer in this context after first use.
 * First-call e2e includes loading; later calls encode with the resident data.
 * Embedding results own their values and remain valid after context teardown.
 * Clears output/timings on error or cancellation, with the same return codes
 * and cancel-token scopes as infer_image. */
AICORE_CAPI int aicore_relation_encode_text(
        aicore_relation_ctx *ctx,
        const char *const *texts,
        size_t count,
        aicore_relation_embeddings **output);
AICORE_CAPI void aicore_relation_embeddings_free(
        aicore_relation_embeddings *result);
AICORE_CAPI int aicore_relation_last_pipeline_timings(
        const aicore_relation_ctx *ctx, aicore_pipeline_timings *output);
AICORE_CAPI int aicore_relation_model_count(void);
AICORE_CAPI const aicore_relation_model_entry *aicore_relation_model_at(
        int index);
AICORE_CAPI int aicore_relation_model_default_index(void);
/** The 497 object categories (MegaSG / RA-4M) the official RelateAnything
 * detectors are re-parameterised to. Feeding them to an open-vocabulary
 * detector (YOLO-World, YOLOE) reproduces upstream's detector vocabulary.
 * Strings are static; `at` returns NULL out of range. */
AICORE_CAPI int aicore_relation_object_vocabulary_count(void);
AICORE_CAPI const char *aicore_relation_object_vocabulary_at(int index);
AICORE_CAPI char *aicore_relation_model_cache_dir(void);
AICORE_CAPI void aicore_relation_free_buffer(void *buffer);
AICORE_CAPI void aicore_relation_shutdown(void);
#ifdef __cplusplus
}
#endif

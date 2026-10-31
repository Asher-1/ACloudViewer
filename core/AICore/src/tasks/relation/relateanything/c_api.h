// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <stddef.h>
#include <stdint.h>

#include "relateanything/backend_binding.h"
#ifdef __cplusplus
extern "C" {
#endif
#define RA_ABI_VERSION 1u
#define RA_STRICT_F32_AUTO (-1)
typedef struct ra_context ra_context;
/* Strings are borrowed during create. Calls on a context must be serialized.
 * Runtime tuning is configured by the owning runtime before creating sessions.
 */
typedef struct ra_options {
    uint32_t struct_size, abi_version;
    const char *diagnostic_directory;
    const ra_backend_binding *backend_binding; /* optional host BackendLease */
    const char *model_path;
    const char *backend; /* cpu, cuda, vulkan; unavailable devices are errors */
    uint32_t device_index, threads, max_boxes, topk;
    int32_t strict_f32, quant_f32_input, raw_logits, decompose;
    int32_t flash_attention, compact_pairs, fused_vocabulary, cache_positions,
            retain_features;
    const char *text_student_path, *tokenizer_path;
    const char *const *predicate_texts;
    size_t predicate_text_count;
    const char *vocabulary_path;
    const char *vocabulary_names_path;
    const char *predicate_type_map_path;
} ra_options;
typedef enum ra_image_format {
    RA_RGB8 = 1,
    RA_RGBA8 = 2,
    RA_GRAY8 = 3,
    RA_BGR8 = 4,
    RA_BGRA8 = 5
} ra_image_format;
typedef struct ra_image_view {
    const uint8_t *data;
    int32_t width, height;
    size_t row_stride_bytes, data_bytes;
    ra_image_format format;
} ra_image_view;
typedef struct ra_request {
    uint32_t struct_size, abi_version;
    const float *boxes_xyxy; /* absolute original-image coordinates [N,4] */
    size_t object_count;
    const float *box_scores; /* optional [N] */
    const uint8_t *masks;    /* optional binary [N,H,W], contiguous */
    size_t mask_bytes;
    int (*cancelled)(void *user); /* checked before and after execution */
    void *user;
} ra_request;
typedef struct ra_relation {
    uint32_t subject, object, predicate;
    float score;
} ra_relation;
enum ra_timing_field {
    RA_TIMING_PREPROCESS = 1u,
    RA_TIMING_INFERENCE = 2u,
    RA_TIMING_POSTPROCESS = 4u,
    RA_TIMING_SERIALIZATION = 8u,
    RA_TIMING_E2E = 16u
};
/* Layout and valid-field meanings match AICore pipeline timings. Unmeasurable
 * subdivisions stay invalid, rather than labelling the whole graph as
 * inference. */
typedef struct ra_timings {
    uint32_t abi_version, valid_fields;
    double preprocess_ms, inference_ms, postprocess_ms, serialization_ms,
            e2e_ms;
} ra_timings;
/* Independent ownership: valid after context destruction. Release exactly once.
 * Raw arrays are [pair_count,predicate_count], in returned candidate-ID order.
 */
typedef struct ra_result {
    uint32_t struct_size, abi_version;
    size_t relation_count, spatial_count, semantic_count, pair_count,
            predicate_count;
    const ra_relation *relations, *spatial_relations, *semantic_relations;
    const float *predicate_logits, *pair_logits;
    const int32_t *subject_indices, *object_indices, *valid_mask;
    const char *const *predicate_names;
    ra_timings timings;
    size_t backend_buffer_bytes;
    double forward_ms;
    void *private_owner;
    /* [predicate_count] 1 = spatial predicate; NULL unless decompose. */
    const uint8_t *predicate_spatial;
} ra_result;
ra_options ra_default_options(void);
ra_request ra_default_request(void);
/* Failed model creation returns an error context, or NULL on allocation
 * failure. */
ra_context *ra_create(const ra_options *options);
void ra_free(ra_context *context);
int ra_ready(const ra_context *context);
const char *ra_last_error(const ra_context *context);
/* 0 success, 1 cooperative cancellation, -1 error; output is NULL on failure.
 */
int ra_infer_image(ra_context *context,
                   const ra_image_view *image,
                   const ra_request *request,
                   ra_result **output);
void ra_result_free(ra_result *result);
/* Read-only GGUF score calibration; 0 on success, -1 if not loaded. */
int ra_calibration(const ra_context *context, float *a, float *b);
/* No process-wide backend teardown: the host owns runtime shutdown. */
#ifdef __cplusplus
}
#endif

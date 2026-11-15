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

#define AICORE_ATTRIBUTE_ABI_VERSION 1u

typedef struct aicore_attribute_ctx aicore_attribute_ctx;

/** Strings are borrowed only during create. Accept local PULC GGUFs with
 * verified source identity; local-file support does not qualify downloads,
 * quantization or release. */
typedef struct aicore_attribute_options {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* model_path;
    const char* device;
    int32_t threads;
} aicore_attribute_options;

/** One attribute selected by a strict threshold or group argmax.
 *  score is the selected branch probability; a negative binary branch uses 1-p.
 *  probability_index addresses raw probabilities, without softmax or
 * recalibration. */
typedef struct aicore_attribute_item {
    const char* name;
    const char* value;
    float score;
    int32_t probability_index;
} aicore_attribute_item;

/** Independently owned result, valid after context release. Geometry is the
 * full source-image rectangle (0,0)-(width,height); raw probabilities retain
 * source order. Do not modify members/private_owner; release exactly once with
 * result_free. */
typedef struct aicore_attribute_result {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* label;
    int32_t width;
    int32_t height;
    const aicore_attribute_item* attributes;
    size_t attribute_count;
    const float* probabilities;
    size_t probability_count;
    void* private_owner;
} aicore_attribute_result;

/** Local conversion recipe, not a published download catalog; lists verified
 * F32 paths only. source_sha256 identifies original ONNX, not the GGUF
 * download. Members/strings are process-lifetime borrows. */
typedef struct aicore_attribute_preset_entry {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* filename;
    const char* display_name;
    const char* label;
    const char* source_sha256;
    size_t probability_count;
    size_t attribute_count;
} aicore_attribute_preset_entry;

AICORE_CAPI int aicore_attribute_preset_count(void);
/** Return NULL for out-of-range indices; perform no model loading or inference.
 */
AICORE_CAPI const aicore_attribute_preset_entry* aicore_attribute_preset_at(
        int index);
/** Return extract/attribute_models under the shared data root, or NULL on
 * failure. Release the owned buffer with free_buffer; creates no directory and
 * downloads no file. */
AICORE_CAPI char* aicore_attribute_model_cache_dir(void);
AICORE_CAPI void aicore_attribute_free_buffer(char* buffer);

AICORE_CAPI aicore_attribute_options aicore_attribute_default_options(void);
/** Failure returns an error context; ready/last_error remain queryable except
 * on allocation failure. */
AICORE_CAPI aicore_attribute_ctx* aicore_attribute_create(
        const aicore_attribute_options* options);
AICORE_CAPI void aicore_attribute_free(aicore_attribute_ctx* ctx);
AICORE_CAPI int aicore_attribute_ready(const aicore_attribute_ctx* ctx);
AICORE_CAPI const char* aicore_attribute_last_error(
        const aicore_attribute_ctx* ctx);
AICORE_CAPI const char* aicore_attribute_context_device(
        const aicore_attribute_ctx* ctx);
/** Read-only borrowed input preserves actual stride/format; serialize calls on
 * one context. Return 0 on success, -1 on error; every error clears *output. */
AICORE_CAPI int aicore_attribute_run(aicore_attribute_ctx* ctx,
                                     const aicore_image_view* image,
                                     aicore_attribute_result** output);
AICORE_CAPI void aicore_attribute_result_free(aicore_attribute_result* result);
/** Return -1 before successful inference, on NULL or timing ABI mismatch;
 * otherwise fill timing and return 0. */
AICORE_CAPI int aicore_attribute_last_pipeline_timings(
        const aicore_attribute_ctx* ctx, aicore_pipeline_timings* output);
/** Delegate to public runtime shutdown without destroying live contexts. */
AICORE_CAPI void aicore_attribute_shutdown(void);

#ifdef __cplusplus
}
#endif

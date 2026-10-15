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

#define AICORE_TAGGING_ABI_VERSION 1u
typedef struct aicore_tagging_ctx aicore_tagging_ctx;

typedef enum aicore_tagging_cpu_precision {
    AICORE_TAGGING_CPU_AUTO = 0,
    AICORE_TAGGING_CPU_UPSTREAM = 1
} aicore_tagging_cpu_precision;

/** Strings are borrowed for create/infer only. A context owns resident weights
 * and graphs. Calls on one context must be serialized. */
typedef struct aicore_tagging_options {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* model_path;
    const char* device;       /**< auto, cpu, CUDA0, Vulkan0, Metal, etc. */
    int32_t threads;          /**< <=0 selects the runtime CPU default. */
    int32_t generate_caption; /**< Tag2Text only; loads text decoder weights. */
    int32_t max_length;       /**< 10..512, default 30. */
    int32_t beam_batch;       /**< 0=backend default, 1 or 3. */
    const char* tag_list_path; /**< optional upstream label-space adapter */
    const char* threshold_path;
    const char* label_embedding_path;
    int32_t cpu_precision; /**< AUTO preserves floating QKV activations in
                            * RAM Q8's first two high-resolution stages and
                            * keeps CUDA RAM++ Q8 on its faster MMQ path;
                            * UPSTREAM retains the original CPU activation
                            * quantization and the weight-only CUDA Q8 graph.
                            * Caption math is unchanged.
                            */
} aicore_tagging_options;

typedef struct aicore_tagging_request {
    uint32_t struct_size;
    uint32_t abi_version;
    float threshold; /**< -1 uses the GGUF per-class thresholds; else 0..1. */
    const char* specified_tags; /**< optional Tag2Text caption conditioning */
    float threshold_epsilon;    /**< Tag2Text only: subtract 0..1 from the class
                                     threshold; default 0 matches upstream. */
} aicore_tagging_request;

typedef struct aicore_tagging_tag {
    int32_t index;
    const char* name;
    float logit;
    float probability;
    float threshold; /**< effective selection threshold, including epsilon */
    int32_t selected;
} aicore_tagging_tag;

/** Independently owned result; remains valid after the context is freed.
 * Release exactly once with aicore_tagging_result_free. Every label is returned
 * in GGUF order, including labels below threshold. */
typedef struct aicore_tagging_result {
    uint32_t struct_size;
    uint32_t abi_version;
    size_t tag_count;
    const aicore_tagging_tag* tags;
    const char* caption;
    aicore_pipeline_timings timings;
    void* private_owner; /**< opaque owner; callers must not inspect or modify
                          */
} aicore_tagging_result;

typedef struct aicore_tagging_model_entry {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* architecture;
    const char* sha256;
    int64_t size_bytes;
    int32_t supports_caption;
} aicore_tagging_model_entry;

AICORE_CAPI aicore_tagging_options aicore_tagging_default_options(void);
AICORE_CAPI aicore_tagging_request aicore_tagging_default_request(void);
/** Returns an error context on load failure; inspect ready/last_error. */
AICORE_CAPI aicore_tagging_ctx* aicore_tagging_create(
        const aicore_tagging_options* options);
AICORE_CAPI void aicore_tagging_free(aicore_tagging_ctx* ctx);
AICORE_CAPI int aicore_tagging_ready(const aicore_tagging_ctx* ctx);
AICORE_CAPI const char* aicore_tagging_last_error(
        const aicore_tagging_ctx* ctx);
AICORE_CAPI const char* aicore_tagging_context_device(
        const aicore_tagging_ctx* ctx);
/** Borrow image storage through return. Output is set to NULL on failure.
 * Returns 0 on success, 1 for cooperative cancellation, -1 for error. */
AICORE_CAPI int aicore_tagging_infer_image(
        aicore_tagging_ctx* ctx,
        const aicore_image_view* image,
        const aicore_tagging_request* request,
        aicore_tagging_result** output);
AICORE_CAPI void aicore_tagging_result_free(aicore_tagging_result* result);
AICORE_CAPI int aicore_tagging_last_pipeline_timings(
        const aicore_tagging_ctx* ctx, aicore_pipeline_timings* output);
AICORE_CAPI int aicore_tagging_model_count(void);
AICORE_CAPI const aicore_tagging_model_entry* aicore_tagging_model_at(
        int index);
AICORE_CAPI int aicore_tagging_model_default_index(void);
AICORE_CAPI char* aicore_tagging_model_cache_dir(void);
AICORE_CAPI void aicore_tagging_free_buffer(void* buffer);
AICORE_CAPI void aicore_tagging_shutdown(void);

#ifdef __cplusplus
}
#endif

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// DeepLSD (deep line segment detection) C API.
//
// The ggml engine under core/AICore/src/tasks/deeplsd/ implements the DeepLSD
// architecture ("DeepLSD: Line Segment Detection and Refinement with Deep
// Image Gradients", Pautrat et al., CVPR 2023); the classic LSD post-processor
// is the original C implementation by von Gioi et al. (lsd.cpp/lsd.h, vendored
// as-is with its own header docs).

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the ABI version of the DeepLSD C API (bump on breaking ABI
 *  changes). */
AICORE_CAPI int aicore_deeplsd_abi_version(void);

typedef struct aicore_deeplsd_ctx aicore_deeplsd_ctx;
typedef struct aicore_deeplsd_options aicore_deeplsd_options;

/** Creates a default options struct (device "auto", threads 0 = backend
 *  default). Release with aicore_deeplsd_options_free. */
AICORE_CAPI aicore_deeplsd_options* aicore_deeplsd_options_new(void);
/** Releases an options struct created by aicore_deeplsd_options_new. */
AICORE_CAPI void aicore_deeplsd_options_free(aicore_deeplsd_options* opts);
/** Selects the inference device: NULL or "auto", "cpu", "gpu", "vulkan"
 *  (optionally ":N"), "cuda" (Linux/Windows). */
AICORE_CAPI void aicore_deeplsd_options_set_device(aicore_deeplsd_options* opts,
                                                   const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_deeplsd_options_set_threads(
        aicore_deeplsd_options* opts, int n_threads);

/** Loads a DeepLSD line-segment GGUF model. Returns NULL on failure (see
 *  aicore_deeplsd_last_error). opts may be NULL for defaults. */
AICORE_CAPI aicore_deeplsd_ctx* aicore_deeplsd_load_opts(
        const char* gguf_path, const aicore_deeplsd_options* opts);
/** Releases a context returned by aicore_deeplsd_load_opts; safe on NULL. */
AICORE_CAPI void aicore_deeplsd_free(aicore_deeplsd_ctx* ctx);
/** True only after a context owns a successfully initialized extractor. */
AICORE_CAPI int aicore_deeplsd_is_ready(const aicore_deeplsd_ctx* ctx);
/** Returns the last error message of the context (empty when none). */
AICORE_CAPI const char* aicore_deeplsd_last_error(
        const aicore_deeplsd_ctx* ctx);

/** Extract distance + angle fields (row-major, size width*height). Caller frees
 * with free(). */
AICORE_CAPI int aicore_deeplsd_extract_gray(aicore_deeplsd_ctx* ctx,
                                            const uint8_t* gray,
                                            int32_t width,
                                            int32_t height,
                                            int32_t row_stride,
                                            float** out_distance,
                                            float** out_angle,
                                            int32_t* out_width,
                                            int32_t* out_height);

/** Returns a JSON summary of the loaded model. Caller frees with
 *  aicore_deeplsd_free_buffer. */
AICORE_CAPI char* aicore_deeplsd_info_json(aicore_deeplsd_ctx* ctx);
/** Releases any buffer returned by an aicore_deeplsd_* function (unified
 *  entry point). Safe on NULL. */
AICORE_CAPI void aicore_deeplsd_free_buffer(void* p);
/** Warms up the backend for `device` (pre-allocates pipeline state);
 *  returns 0 on success. */
AICORE_CAPI int aicore_deeplsd_warmup_backend(const char* device);
/** Returns the local model cache directory. Caller frees with
 *  aicore_deeplsd_free_buffer. */
AICORE_CAPI char* aicore_deeplsd_model_cache_dir(void);

typedef struct aicore_deeplsd_segment {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
} aicore_deeplsd_segment;

/** Optional segment outputs (caller frees *out_segments with free()). */
AICORE_CAPI int aicore_deeplsd_extract_segments(
        aicore_deeplsd_ctx* ctx,
        const uint8_t* gray,
        int32_t width,
        int32_t height,
        int32_t row_stride,
        aicore_deeplsd_segment** out_segments,
        int32_t* out_segment_count,
        float** out_distance,
        float** out_angle,
        int32_t* out_width,
        int32_t* out_height);

/** Quantize conv weights in F32 GGUF to f16 or q8_0. */
AICORE_CAPI int aicore_deeplsd_quantize(const char* input_gguf,
                                        const char* output_gguf,
                                        const char* type);

#ifdef __cplusplus
}
#endif

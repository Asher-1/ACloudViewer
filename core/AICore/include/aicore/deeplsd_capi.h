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

#ifdef __cplusplus
extern "C" {
#endif

AICORE_CAPI int aicore_deeplsd_abi_version(void);

typedef struct aicore_deeplsd_ctx aicore_deeplsd_ctx;
typedef struct aicore_deeplsd_options aicore_deeplsd_options;

AICORE_CAPI aicore_deeplsd_options* aicore_deeplsd_options_new(void);
AICORE_CAPI void aicore_deeplsd_options_free(aicore_deeplsd_options* opts);
AICORE_CAPI void aicore_deeplsd_options_set_device(aicore_deeplsd_options* opts,
                                                   const char* device);
AICORE_CAPI void aicore_deeplsd_options_set_threads(
        aicore_deeplsd_options* opts, int n_threads);

AICORE_CAPI aicore_deeplsd_ctx* aicore_deeplsd_load_opts(
        const char* gguf_path, const aicore_deeplsd_options* opts);
AICORE_CAPI void aicore_deeplsd_free(aicore_deeplsd_ctx* ctx);
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

AICORE_CAPI char* aicore_deeplsd_info_json(aicore_deeplsd_ctx* ctx);
AICORE_CAPI void aicore_deeplsd_free_string(char* s);
AICORE_CAPI int aicore_deeplsd_warmup_backend(const char* device);
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

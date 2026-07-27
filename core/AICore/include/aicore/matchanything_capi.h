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

AICORE_CAPI int aicore_matchanything_abi_version(void);

typedef struct aicore_matchanything_ctx aicore_matchanything_ctx;
typedef struct aicore_matchanything_options aicore_matchanything_options;

/** Model variant: "eloftr" (default) or "roma". */
typedef enum {
    AICORE_MATCHANYTHING_ELOFTR = 0,
    AICORE_MATCHANYTHING_ROMA = 1,
} aicore_matchanything_variant;

typedef struct {
    float x0;
    float y0;
    float x1;
    float y1;
    float score;
} aicore_matchanything_match;

AICORE_CAPI aicore_matchanything_options* aicore_matchanything_options_new(
        void);
AICORE_CAPI void aicore_matchanything_options_free(
        aicore_matchanything_options* opts);
AICORE_CAPI void aicore_matchanything_options_set_device(
        aicore_matchanything_options* opts, const char* device);
AICORE_CAPI void aicore_matchanything_options_set_threads(
        aicore_matchanything_options* opts, int n_threads);
AICORE_CAPI void aicore_matchanything_options_set_variant(
        aicore_matchanything_options* opts,
        aicore_matchanything_variant variant);

AICORE_CAPI aicore_matchanything_ctx* aicore_matchanything_load_opts(
        const char* gguf_path, const aicore_matchanything_options* opts);
AICORE_CAPI void aicore_matchanything_free(aicore_matchanything_ctx* ctx);
AICORE_CAPI const char* aicore_matchanything_last_error(
        const aicore_matchanything_ctx* ctx);

/** End-to-end match on two grayscale images (same width/height, padded square).
 */
AICORE_CAPI int aicore_matchanything_match_gray(
        aicore_matchanything_ctx* ctx,
        const uint8_t* img0,
        const uint8_t* img1,
        int32_t width,
        int32_t height,
        int32_t row_stride,
        aicore_matchanything_match** out_matches,
        int32_t* out_count);

AICORE_CAPI void aicore_matchanything_free_matches(
        aicore_matchanything_match* matches);

AICORE_CAPI char* aicore_matchanything_info_json(aicore_matchanything_ctx* ctx);
AICORE_CAPI void aicore_matchanything_free_string(char* s);
AICORE_CAPI int aicore_matchanything_warmup_backend(const char* device);
AICORE_CAPI char* aicore_matchanything_model_cache_dir(void);

/** Quantize conv weights in F32 GGUF to f16 or q8_0. */
AICORE_CAPI int aicore_matchanything_quantize(const char* input_gguf,
                                              const char* output_gguf,
                                              const char* type);

#ifdef __cplusplus
}
#endif

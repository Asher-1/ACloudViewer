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

AICORE_CAPI int aicore_eloftr_abi_version(void);

typedef struct aicore_eloftr_ctx aicore_eloftr_ctx;
typedef struct aicore_eloftr_options aicore_eloftr_options;

typedef struct {
    float x0;
    float y0;
    float x1;
    float y1;
    float score;
} aicore_eloftr_match;

AICORE_CAPI aicore_eloftr_options* aicore_eloftr_options_new(void);
AICORE_CAPI void aicore_eloftr_options_free(aicore_eloftr_options* opts);
AICORE_CAPI void aicore_eloftr_options_set_device(aicore_eloftr_options* opts,
                                                  const char* device);
AICORE_CAPI void aicore_eloftr_options_set_threads(aicore_eloftr_options* opts,
                                                   int n_threads);

AICORE_CAPI aicore_eloftr_ctx* aicore_eloftr_load_opts(
        const char* gguf_path, const aicore_eloftr_options* opts);
AICORE_CAPI void aicore_eloftr_free(aicore_eloftr_ctx* ctx);
AICORE_CAPI const char* aicore_eloftr_last_error(const aicore_eloftr_ctx* ctx);

/** End-to-end match on two grayscale images (same width/height). */
AICORE_CAPI int aicore_eloftr_match_gray(aicore_eloftr_ctx* ctx,
                                         const uint8_t* img0,
                                         const uint8_t* img1,
                                         int32_t width,
                                         int32_t height,
                                         int32_t row_stride,
                                         aicore_eloftr_match** out_matches,
                                         int32_t* out_count);

AICORE_CAPI void aicore_eloftr_free_matches(aicore_eloftr_match* matches);

AICORE_CAPI char* aicore_eloftr_info_json(aicore_eloftr_ctx* ctx);
AICORE_CAPI void aicore_eloftr_free_string(char* s);
AICORE_CAPI int aicore_eloftr_warmup_backend(const char* device);
AICORE_CAPI char* aicore_eloftr_model_cache_dir(void);

/** Quantize conv weights in F32 GGUF to f16 or q8_0. */
AICORE_CAPI int aicore_eloftr_quantize(const char* input_gguf,
                                       const char* output_gguf,
                                       const char* type);

#ifdef __cplusplus
}
#endif

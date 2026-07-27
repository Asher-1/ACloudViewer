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
#include "aicore/lightglue_capi.h"

#ifdef __cplusplus
extern "C" {
#endif

AICORE_CAPI int aicore_aliked_abi_version(void);

typedef struct aicore_aliked_ctx aicore_aliked_ctx;
typedef struct aicore_aliked_options aicore_aliked_options;

AICORE_CAPI aicore_aliked_options* aicore_aliked_options_new(void);
AICORE_CAPI void aicore_aliked_options_free(aicore_aliked_options* opts);
AICORE_CAPI void aicore_aliked_options_set_device(aicore_aliked_options* opts,
                                                  const char* device);
AICORE_CAPI void aicore_aliked_options_set_threads(aicore_aliked_options* opts,
                                                   int n_threads);
AICORE_CAPI void aicore_aliked_options_set_max_keypoints(
        aicore_aliked_options* opts, int32_t max_keypoints);
AICORE_CAPI void aicore_aliked_options_set_resize_long_edge(
        aicore_aliked_options* opts, int32_t px);

AICORE_CAPI aicore_aliked_ctx* aicore_aliked_load_opts(
        const char* gguf_path, const aicore_aliked_options* opts);
AICORE_CAPI void aicore_aliked_free(aicore_aliked_ctx* ctx);
AICORE_CAPI const char* aicore_aliked_last_error(const aicore_aliked_ctx* ctx);

/** Extract from RGB888 row-major image. Output packed into lightglue features.
 */
AICORE_CAPI int aicore_aliked_extract_rgb(aicore_aliked_ctx* ctx,
                                          const uint8_t* rgb,
                                          int32_t width,
                                          int32_t height,
                                          int32_t row_stride,
                                          aicore_lightglue_features* out);

AICORE_CAPI char* aicore_aliked_info_json(aicore_aliked_ctx* ctx);
AICORE_CAPI void aicore_aliked_free_string(char* s);
AICORE_CAPI char* aicore_aliked_model_cache_dir(void);

#ifdef __cplusplus
}
#endif

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// ALIKED (adaptive local keypoint detector) C API.
//
// The ggml engine under core/AICore/src/tasks/aliked/ implements the ALIKED
// architecture described in "ALIKED: A Lighter Keypoint and Descriptor
// Extraction Network via Deformable Attention" (Zhao et al.); its output
// contract matches the cvg/LightGlue ALIKED variant
// (https://github.com/cvg/LightGlue). This is an original implementation on
// ggml — no upstream source files are vendored.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"
#include "aicore/lightglue_capi.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the ABI version of the ALIKED C API (bump on breaking ABI
 *  changes; clients should verify it matches the header they compiled
 *  against). */
AICORE_CAPI int aicore_aliked_abi_version(void);

typedef struct aicore_aliked_ctx aicore_aliked_ctx;
typedef struct aicore_aliked_options aicore_aliked_options;

/** Creates a default options struct. Fields default to device "auto",
 *  threads 0 (backend default), max_keypoints 0 (model default) and
 *  resize_long_edge 0 (no resize). Release with
 *  aicore_aliked_options_free. */
AICORE_CAPI aicore_aliked_options* aicore_aliked_options_new(void);
/** Releases an options struct created by aicore_aliked_options_new. */
AICORE_CAPI void aicore_aliked_options_free(aicore_aliked_options* opts);
/** Selects the inference device: NULL or "auto" (runtime pick), "cpu",
 *  "gpu", "vulkan" (optionally ":N"), "cuda" (Linux/Windows). */
AICORE_CAPI void aicore_aliked_options_set_device(aicore_aliked_options* opts,
                                                  const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_aliked_options_set_threads(aicore_aliked_options* opts,
                                                   int n_threads);
/** Maximum number of keypoints returned by the extractor; <= 0 keeps the
 *  model default. */
AICORE_CAPI void aicore_aliked_options_set_max_keypoints(
        aicore_aliked_options* opts, int32_t max_keypoints);
/** Resizes the input so its long edge is `px` pixels before extraction
 *  (0 disables resizing). */
AICORE_CAPI void aicore_aliked_options_set_resize_long_edge(
        aicore_aliked_options* opts, int32_t px);

/** Loads an ALIKED feature-extractor GGUF model. Returns NULL on failure
 *  (see aicore_aliked_last_error). opts may be NULL for defaults. */
AICORE_CAPI aicore_aliked_ctx* aicore_aliked_load_opts(
        const char* gguf_path, const aicore_aliked_options* opts);
/** Releases a context returned by aicore_aliked_load_opts (also releases
 *  its backend lease; safe on NULL). */
AICORE_CAPI void aicore_aliked_free(aicore_aliked_ctx* ctx);
/** True only after a context owns a successfully initialized extractor. */
AICORE_CAPI int aicore_aliked_is_ready(const aicore_aliked_ctx* ctx);
/** Returns the last error message of the context (empty string when none).
 *  The pointer stays valid until the next failing call on the same ctx. */
AICORE_CAPI const char* aicore_aliked_last_error(const aicore_aliked_ctx* ctx);

/** Extracts ALIKED features from an RGB888 row-major image. Output is
 *  packed into an aicore_lightglue_features struct (keypoints + float
 *  descriptors). Returns 0 on success, non-zero on failure (see
 *  aicore_aliked_last_error). */
AICORE_CAPI int aicore_aliked_extract_rgb(aicore_aliked_ctx* ctx,
                                          const uint8_t* rgb,
                                          int32_t width,
                                          int32_t height,
                                          int32_t row_stride,
                                          aicore_lightglue_features* out);

/** Returns a JSON summary of the loaded model (name, geometry, ...).
 *  Caller frees with aicore_aliked_free_buffer. */
AICORE_CAPI char* aicore_aliked_info_json(aicore_aliked_ctx* ctx);
/** Releases any buffer returned by an aicore_aliked_* function (string or
 *  feature arrays; unified entry point). Safe on NULL. */
AICORE_CAPI void aicore_aliked_free_buffer(void* p);
/** Returns the local model cache directory (download base for ALIKED
 *  GGUF assets). Caller frees with aicore_aliked_free_buffer. */
AICORE_CAPI char* aicore_aliked_model_cache_dir(void);
/** Quantize eligible convolution / linear weights to f16 or q8_0. */
AICORE_CAPI int aicore_aliked_quantize_gguf(const char* input_gguf,
                                            const char* output_gguf,
                                            const char* type);

#ifdef __cplusplus
}
#endif

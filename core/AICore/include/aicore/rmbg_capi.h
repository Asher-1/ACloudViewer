// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// RMBG-2.0 (BRIA background removal) C API. In-tree port of
// https://github.com/Asher-1/RMBG-2.0-GGML.
//
// The engine lives in core/AICore/src/tasks/rmbg/ and is accelerated by ggml
// with CPU / CUDA / Vulkan / Metal backends. This is the foundational
// background-removal module other plugins may build on: the raw 8-bit alpha
// matte is exposed separately from the RGBA composite so downstream consumers
// can threshold / feather / re-composite at their own resolution.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"

#ifdef __cplusplus
extern "C" {
#endif

AICORE_CAPI int aicore_rmbg_abi_version(void);

typedef struct aicore_rmbg_ctx aicore_rmbg_ctx;
typedef struct aicore_rmbg_options aicore_rmbg_options;

/** Timings for the most recent successful inference request. inference_ms is
 *  the graph->forward() interval used by the upstream RMBG benchmark; total_ms
 *  additionally includes input decoding/preprocessing and output encoding. */
typedef struct aicore_rmbg_timings {
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double total_ms;
} aicore_rmbg_timings;

AICORE_CAPI aicore_rmbg_options* aicore_rmbg_options_new(void);
AICORE_CAPI void aicore_rmbg_options_free(aicore_rmbg_options* opts);
AICORE_CAPI void aicore_rmbg_options_set_device(aicore_rmbg_options* opts,
                                                const char* device);
AICORE_CAPI void aicore_rmbg_options_set_threads(aicore_rmbg_options* opts,
                                                 int n_threads);

/** Load the unified RMBG-2.0 GGUF (encoder + decoder in one file). Returns
 *  NULL on failure; inspect aicore_rmbg_last_error() for the reason. */
AICORE_CAPI aicore_rmbg_ctx* aicore_rmbg_load_opts(
        const char* gguf_path, const aicore_rmbg_options* opts);
AICORE_CAPI void aicore_rmbg_free(aicore_rmbg_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model. */
AICORE_CAPI int aicore_rmbg_is_ready(const aicore_rmbg_ctx* ctx);
AICORE_CAPI const char* aicore_rmbg_last_error(const aicore_rmbg_ctx* ctx);
/** Copy the most recent successful request timings into out_timings. */
AICORE_CAPI int aicore_rmbg_last_timings(const aicore_rmbg_ctx* ctx,
                                         aicore_rmbg_timings* out_timings);

AICORE_CAPI void aicore_rmbg_free_string(char* s);
AICORE_CAPI void aicore_rmbg_free_buffer(void* p);

/** Remove background from an image file. \p out_png receives PNG-encoded RGBA
 *  bytes at the ORIGINAL image resolution (caller frees with
 *  aicore_rmbg_free_buffer). Returns 0 on success, -1 on failure. */
AICORE_CAPI int aicore_rmbg_remove_background_path(aicore_rmbg_ctx* ctx,
                                                   const char* image_path,
                                                   uint8_t** out_png,
                                                   int* out_len);

/** Remove background from an in-memory RGB buffer (HWC, 3 bytes/pixel).
 *  Same output contract as aicore_rmbg_remove_background_path. */
AICORE_CAPI int aicore_rmbg_remove_background_rgb(aicore_rmbg_ctx* ctx,
                                                  const uint8_t* rgb,
                                                  int32_t width,
                                                  int32_t height,
                                                  uint8_t** out_png,
                                                  int* out_len);

/** Remove background from an in-memory RGB buffer and return the raw RGBA
 *  composite (HWC, 4 bytes/pixel, alpha blended, at the ORIGINAL resolution)
 *  instead of PNG bytes — the in-memory consumer path (GUI preview), which
 *  skips a PNG encode/decode round-trip. \p out_rgba is allocated by the
 *  callee (size \p out_len = out_width*out_height*4) and must be released
 *  with aicore_rmbg_free_buffer. Returns 0 on success. */
AICORE_CAPI int aicore_rmbg_remove_background_rgba(aicore_rmbg_ctx* ctx,
                                                   const uint8_t* rgb,
                                                   int32_t width,
                                                   int32_t height,
                                                   uint8_t** out_rgba,
                                                   int32_t* out_width,
                                                   int32_t* out_height,
                                                   int* out_len);

/** Raw 8-bit alpha matte (0 = background, 255 = foreground) at the ORIGINAL
 *  image resolution, row-major. out_alpha is allocated by the callee and must
 *  be released with aicore_rmbg_free_buffer. Returns 0 on success. This is the
 *  primitive future plugins use to build their own composites. */
AICORE_CAPI int aicore_rmbg_alpha_mat_rgb(aicore_rmbg_ctx* ctx,
                                          const uint8_t* rgb,
                                          int32_t width,
                                          int32_t height,
                                          uint8_t** out_alpha,
                                          int32_t* out_width,
                                          int32_t* out_height);

AICORE_CAPI char* aicore_rmbg_info_json(aicore_rmbg_ctx* ctx);
AICORE_CAPI int aicore_rmbg_warmup_backend(const char* device);
AICORE_CAPI void aicore_rmbg_shutdown(void);
AICORE_CAPI char* aicore_rmbg_model_cache_dir(void);

/** Published GGUF catalog (cloudViewer_downloads trellis2-ggml release). */
typedef struct aicore_rmbg_model_entry {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
} aicore_rmbg_model_entry;

AICORE_CAPI int aicore_rmbg_model_count(void);
AICORE_CAPI const aicore_rmbg_model_entry* aicore_rmbg_model_at(int index);
AICORE_CAPI const aicore_rmbg_model_entry* aicore_rmbg_model_by_filename(
        const char* filename);
AICORE_CAPI const char* aicore_rmbg_model_download_base(void);

#ifdef __cplusplus
}
#endif

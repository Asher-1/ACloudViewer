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

/** Returns the ABI version of the RMBG C API (bump on breaking ABI
 *  changes). */
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

/** Creates a default options struct (device "auto", threads 0 = backend
 *  default). Release with aicore_rmbg_options_free. */
AICORE_CAPI aicore_rmbg_options* aicore_rmbg_options_new(void);
/** Releases an options struct created by aicore_rmbg_options_new. */
AICORE_CAPI void aicore_rmbg_options_free(aicore_rmbg_options* opts);
/** Selects the inference device: NULL or "auto", "cpu", "gpu", "vulkan"
 *  (optionally ":N"), "cuda" (Linux/Windows). */
AICORE_CAPI void aicore_rmbg_options_set_device(aicore_rmbg_options* opts,
                                                const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_rmbg_options_set_threads(aicore_rmbg_options* opts,
                                                 int n_threads);

/** Select the math profile (default "optimized"): one of "default",
 *  "optimized", "strict", "fast", "unsafe-fast".
 *  "optimized" — historical default: Vulkan coopmat whitelist + scalar
 *                 direct conv + F32 flash attention, cuBLAS TF32 on.
 *  "strict"    — bit-stable FP32 (disables coopmat/dot-product/TF32).
 *  "fast"/"unsafe-fast" — every fast path enabled (no F16 guard rails).
 *  Replaces the RMBG_VULKAN_MODE / RMBG_STRICT_MATH / RMBG_VULKAN_*
 *  environment variables of the upstream port. */
AICORE_CAPI void aicore_rmbg_options_set_math_profile(aicore_rmbg_options* opts,
                                                      const char* profile);

/** Fine-tuning switches (all default to the historical behavior; the math
 *  profile always wins for the flow-defining switches these cannot
 *  override). enable != 0 turns the switch on. */
/** CUDA: F16-in/FP32-accumulate GEMMs for the Swin MLP layers (opt-in;
 *  was RMBG_CUDA_F16_GEMM). */
AICORE_CAPI void aicore_rmbg_options_set_cuda_f16_gemm(
        aicore_rmbg_options* opts, int enable);
/** CUDA: minimum Swin stage the F16 GEMM path applies to (default 2; was
 *  RMBG_CUDA_F16_MIN_STAGE). */
AICORE_CAPI void aicore_rmbg_options_set_cuda_f16_min_stage(
        aicore_rmbg_options* opts, int stage);
/** CUDA: pre-transposed NN weights for the Swin QKV/projection GEMMs
 *  (opt-in; was RMBG_CUDA_NN_GEMM). */
AICORE_CAPI void aicore_rmbg_options_set_cuda_nn_gemm(aicore_rmbg_options* opts,
                                                      int enable);
/** Vulkan: fused QKV layout conv (default on; was RMBG_VK_QKV_LAYOUT). */
AICORE_CAPI void aicore_rmbg_options_set_vulkan_qkv_layout(
        aicore_rmbg_options* opts, int enable);
/** Vulkan: fused deformable projection conv. mode: "off" (default), "on",
 *  "coop" (was RMBG_VK_DEFORM_PROJECT). */
AICORE_CAPI void aicore_rmbg_options_set_vulkan_deform_project(
        aicore_rmbg_options* opts, const char* mode);
/** Vulkan: Swin attention kernel. mode: "off" (materialized scores),
 *  "scalar" (F32 flash, default), "coop" / "coop0".."coop3" (F16
 *  cooperative flash for all / one stage; was RMBG_VK_FLASH_ATTN). */
AICORE_CAPI void aicore_rmbg_options_set_vulkan_flash_attn(
        aicore_rmbg_options* opts, const char* mode);

/** Load the unified RMBG-2.0 GGUF (encoder + decoder in one file). Returns
 *  NULL on failure; inspect aicore_rmbg_last_error() for the reason. */
AICORE_CAPI aicore_rmbg_ctx* aicore_rmbg_load_opts(
        const char* gguf_path, const aicore_rmbg_options* opts);
/** Releases a context returned by aicore_rmbg_load_opts; safe on NULL. */
AICORE_CAPI void aicore_rmbg_free(aicore_rmbg_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model. */
AICORE_CAPI int aicore_rmbg_is_ready(const aicore_rmbg_ctx* ctx);
/** Returns the last error message of the context (empty when none). */
AICORE_CAPI const char* aicore_rmbg_last_error(const aicore_rmbg_ctx* ctx);
/** Copy the most recent successful request timings into out_timings. */
AICORE_CAPI int aicore_rmbg_last_timings(const aicore_rmbg_ctx* ctx,
                                         aicore_rmbg_timings* out_timings);

/** Releases any buffer returned by an aicore_rmbg_* function (string, PNG
 *  bytes, RGBA or alpha matte; unified entry point). Safe on NULL. */
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

/** Returns a JSON summary of the loaded model. Caller frees with
 *  aicore_rmbg_free_buffer. */
AICORE_CAPI char* aicore_rmbg_info_json(aicore_rmbg_ctx* ctx);
/** Warms up the backend for `device`; returns 0 on success. */
AICORE_CAPI int aicore_rmbg_warmup_backend(const char* device);
/** Releases process-wide RMBG backend resources (idempotent). */
AICORE_CAPI void aicore_rmbg_shutdown(void);
/** Returns the local model cache directory. Caller frees with
 *  aicore_rmbg_free_buffer. */
AICORE_CAPI char* aicore_rmbg_model_cache_dir(void);

/** Published GGUF catalog (cloudViewer_downloads trellis2-ggml release). */
typedef struct aicore_rmbg_model_entry {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
} aicore_rmbg_model_entry;

/** Number of published catalog entries. */
AICORE_CAPI int aicore_rmbg_model_count(void);
/** Returns the catalog entry at `index` (NULL when out of range). */
AICORE_CAPI const aicore_rmbg_model_entry* aicore_rmbg_model_at(int index);
/** Returns the catalog entry whose filename matches (NULL when not
 *  found). */
AICORE_CAPI const aicore_rmbg_model_entry* aicore_rmbg_model_by_filename(
        const char* filename);
/** Returns the base URL of the published model release. */
AICORE_CAPI const char* aicore_rmbg_model_download_base(void);

#ifdef __cplusplus
}
#endif

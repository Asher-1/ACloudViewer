// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// VLX-Seek C API: full-chain fine-grained VLM inference over the in-tree
// ggml port (task under core/AICore/src/tasks/vlx_seek/, ported from the
// upstream Asher-1/VLX-Seek-GGML cpp_ggml).
//
// Pipeline: WeDetect-Uni proposals -> Qwen3.5-VL main tower -> C-RADIOv4 aux
// tower + SimpleFPN -> HFRE region encoding -> Qwen3.5 LLM (the embedded
// pinned llama.cpp, qwen3_5 arch) -> referring answer. One context owns the
// detector, the vision stack and the LLM.
//
// Thread-safety: a context is single-threaded. Use one context per consumer,
// or serialize access externally.
//
// Weights: GGUF files published on the Asher-1/VLX-Seek-1.5-10B-GGUF
// Hugging Face repository (wedetect-base-uni-f32, the merged
// vlx-vision-10b-<precision>, vlx-10b-llm-<precision>); the task model
// catalog is the single URL/digest owner.
//
// The C API never exposes ggml, STL containers or exceptions across the
// boundary; errors are return codes plus a per-context last_error message,
// exactly like the other AICore task APIs.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"
#include "aicore/image_view.h"
#include "aicore/pipeline_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AICORE_VLX_SEEK_ABI_VERSION 1u

/** Returns the ABI version of the VLX-Seek C API (bump on breaking ABI
 *  changes). */
AICORE_CAPI int aicore_vlx_seek_abi_version(void);

typedef struct aicore_vlx_seek_ctx aicore_vlx_seek_ctx;
typedef struct aicore_vlx_seek_options aicore_vlx_seek_options;

/** Creates a default options struct (device "auto" — AICore runtime order,
 *  threads 0 = backend default, precision "f16", max new tokens 2048).
 *  Release with aicore_vlx_seek_options_free. */
AICORE_CAPI aicore_vlx_seek_options* aicore_vlx_seek_options_new(void);
/** Releases an options struct created by aicore_vlx_seek_options_new. */
AICORE_CAPI void aicore_vlx_seek_options_free(aicore_vlx_seek_options* opts);
/** Selects the inference device: NULL/"auto", "cpu", "cuda", "vulkan",
 *  "metal" (optionally ":N"). */
AICORE_CAPI void aicore_vlx_seek_options_set_device(
        aicore_vlx_seek_options* opts, const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_vlx_seek_options_set_threads(
        aicore_vlx_seek_options* opts, int n_threads);
/** Vision weight precision: "f16" (default) or "q8_0". The detector GGUF is
 *  always f32 (the official .pth dtype — quantization changes the proposal
 *  set); the LLM follows the vision precision unless overridden. */
AICORE_CAPI void aicore_vlx_seek_options_set_precision(
        aicore_vlx_seek_options* opts, const char* precision);
/** Directory holding the GGUF files (required; the model catalog cache dir
 *  is the canonical value). */
AICORE_CAPI void aicore_vlx_seek_options_set_models_dir(
        aicore_vlx_seek_options* opts, const char* dir);
/** Explicit per-model overrides; empty (default) derives the filename from
 *  models_dir + precision (wedetect is always f32). */
AICORE_CAPI void aicore_vlx_seek_options_set_wedetect_gguf(
        aicore_vlx_seek_options* opts, const char* path);
AICORE_CAPI void aicore_vlx_seek_options_set_vision_gguf(
        aicore_vlx_seek_options* opts, const char* path);
AICORE_CAPI void aicore_vlx_seek_options_set_llm_gguf(
        aicore_vlx_seek_options* opts, const char* path);
/** Maximum greedy generation length (default 2048). */
AICORE_CAPI void aicore_vlx_seek_options_set_max_new_tokens(
        aicore_vlx_seek_options* opts, int n_tokens);
/** Log level: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR (default 1). */
AICORE_CAPI void aicore_vlx_seek_options_set_log_level(
        aicore_vlx_seek_options* opts, int log_level);
/** Debug: dump parity stage tensors as raw f32 files into this directory
 *  after every run call. The string is copied; empty (default) disables
 *  dumping. */
AICORE_CAPI void aicore_vlx_seek_options_set_dump_dir(
        aicore_vlx_seek_options* opts, const char* dir);

/** Loads the detector, the vision stack and the LLM. Returns a context whose
 *  is_ready() may still be 0 (device admission failure); inspect
 *  aicore_vlx_seek_last_error(). NULL only on invalid input or out-of-memory.
 */
AICORE_CAPI aicore_vlx_seek_ctx* aicore_vlx_seek_load_opts(
        const aicore_vlx_seek_options* opts);
/** Releases a context; safe on NULL. Contexts must not be destroyed while a
 *  worker thread is inside a run call (serialize externally). */
AICORE_CAPI void aicore_vlx_seek_free(aicore_vlx_seek_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model set. */
AICORE_CAPI int aicore_vlx_seek_is_ready(const aicore_vlx_seek_ctx* ctx);
/** Returns the last error message of the context (NULL when none). */
AICORE_CAPI const char* aicore_vlx_seek_last_error(
        const aicore_vlx_seek_ctx* ctx);
/** Releases any buffer returned by an aicore_vlx_seek_* function (JSON/RGB
 *  helpers). Safe on NULL. */
AICORE_CAPI void aicore_vlx_seek_free_buffer(void* p);

/** Load an image file as tightly-packed RGB (HWC, 3 bytes/pixel).
 *  Compatibility helper for path-driven callers; the typed hot path is
 *  aicore_vlx_seek_run_image with a borrowed aicore_image_view. Caller frees
 *  \p out_rgb with aicore_vlx_seek_free_buffer. */
AICORE_CAPI int aicore_vlx_seek_load_path_rgb(const char* image_path,
                                              uint8_t** out_rgb,
                                              int32_t* out_width,
                                              int32_t* out_height);

/** ---- typed inference ----
 *
 *  One request = [optional WeDetect-Uni proposals] + vision + region
 *  encoding + LLM greedy decode. \p img must be non-NULL with positive
 *  dimensions, a supported format (RGB8 native; RGBA8/BGRA8/GRAY8/BGR8 are
 *  converted) and row_stride_bytes >= width * bytes-per-pixel (else -1).
 *
 *  \p task: "detection" | "grounding_single" | "counting" |
 *  "reasoning_detection" | "brief_region_caption" | "region_ocr" | "vqa".
 *  \p text: categories / referring phrase / question (UTF-8).
 *  \p lang: "en" or "zh" (task template language; "vqa" ignores it).
 *  \p boxes: borrowed [x1, y1, x2, y2] float quadruples in original-image
 *  pixels; NULL runs WeDetect-Uni first (100 proposals). Region tasks
 *  require non-NULL boxes. \p n_boxes <= 100.
 *  \p target_indexes / \p n_targets: zero-based proposal indexes for
 *  brief_region_caption / region_ocr (required there, ignored otherwise).
 *
 *  Returns 0 on success, -1 on contract/validation errors, -2 on
 *  runtime/inference errors and -3 when the device cannot hold the models
 *  plus the graph workspace (see aicore_vlx_seek_last_error). On success the
 *  typed result getters below expose the answer until the next run call. */
AICORE_CAPI int aicore_vlx_seek_run_image(aicore_vlx_seek_ctx* ctx,
                                          const aicore_image_view* img,
                                          const char* task,
                                          const char* text,
                                          const char* lang,
                                          const float* boxes,
                                          int32_t n_boxes,
                                          const int32_t* target_indexes,
                                          int32_t n_targets);

/** ---- typed result of the most recent run (valid until the next run) ---- */

/** Model answer with <objN> references remapped to caller ordering. */
AICORE_CAPI const char* aicore_vlx_seek_result_answer(
        const aicore_vlx_seek_ctx* ctx);
/** Full result as the official inference.py JSON (answer, result_bbox_list,
 *  prompt_tokens, completion_tokens). Caller frees with
 *  aicore_vlx_seek_free_buffer. */
AICORE_CAPI char* aicore_vlx_seek_result_json(const aicore_vlx_seek_ctx* ctx);
/** Number of resolved result boxes (0 when the answer references none). */
AICORE_CAPI int32_t
aicore_vlx_seek_result_box_count(const aicore_vlx_seek_ctx* ctx);
/** Box \p index in original-image pixels (out[4] = x1, y1, x2, y2).
 *  Returns 0 on success, -1 when out of range. */
AICORE_CAPI int aicore_vlx_seek_result_box(const aicore_vlx_seek_ctx* ctx,
                                           int32_t index,
                                           float* out);
/** Model-generated label of box \p index (ASCII when encodable, else the
 *  <objN> index — the official fallback). NULL when out of range. */
AICORE_CAPI const char* aicore_vlx_seek_result_box_label(
        const aicore_vlx_seek_ctx* ctx, int32_t index);
/** Token counts of the most recent run. */
AICORE_CAPI int32_t
aicore_vlx_seek_result_prompt_tokens(const aicore_vlx_seek_ctx* ctx);
AICORE_CAPI int32_t
aicore_vlx_seek_result_completion_tokens(const aicore_vlx_seek_ctx* ctx);
/** WeDetect proposal count of the most recent run (0 when the caller passed
 *  boxes). */
AICORE_CAPI int32_t
aicore_vlx_seek_result_proposal_count(const aicore_vlx_seek_ctx* ctx);

/** Renders the most recent result on \p img (borrowed view; the image is
 *  copied) in the official style: red width-3 rectangles plus the
 *  model-generated label at (xmin+2, ymin+2) — non-ASCII labels render as
 *  the <objN> index, exactly like the official visualize_result_boxes.
 *  Debug/compat path; the plugin renders through Qt instead.
 *  Returns 0 on success, -1 on invalid arguments, -2 on write failure. */
AICORE_CAPI int aicore_vlx_seek_result_render_png(
        const aicore_vlx_seek_ctx* ctx,
        const aicore_image_view* img,
        const char* out_png_path);

/** Per-stage wall-clock timings of the most recent run (preprocess includes
 *  the optional WeDetect pass; inference = vision graphs + LLM;
 *  postprocess = answer parsing; serialization = PNG render). Returns 0 on
 *  success, -1 when no inference has run yet. */
AICORE_CAPI int aicore_vlx_seek_last_pipeline_timings(
        const aicore_vlx_seek_ctx* ctx, aicore_pipeline_timings* out_timings);

/** Model introspection. */
/** Backend-RESOLVED device name (surfaces silent CPU fallbacks). Owned by
 *  the context; copy before freeing. */
AICORE_CAPI const char* aicore_vlx_seek_context_device(
        const aicore_vlx_seek_ctx* ctx);
/** Effective CPU thread count after the auto (<=0) resolution. */
AICORE_CAPI int aicore_vlx_seek_context_threads(const aicore_vlx_seek_ctx* ctx);
/** Vision weight precision of the loaded context ("f16"/"q8_0"). */
AICORE_CAPI const char* aicore_vlx_seek_context_precision(
        const aicore_vlx_seek_ctx* ctx);
/** Returns a JSON summary of the loaded models and the latest run. Caller
 *  frees with aicore_vlx_seek_free_buffer. */
AICORE_CAPI char* aicore_vlx_seek_info_json(aicore_vlx_seek_ctx* ctx);

/** Releases process-wide VLX-Seek runtime resources it owns (idempotent;
 *  never destroys live contexts). */
AICORE_CAPI void aicore_vlx_seek_shutdown(void);

/** Returns the local model cache directory (extract/vlx_seek_models under
 *  the shared data root). Caller frees with aicore_vlx_seek_free_buffer. */
AICORE_CAPI char* aicore_vlx_seek_model_cache_dir(void);

/** Published GGUF catalog (Asher-1/VLX-Seek-1.5-10B-GGUF revision).
 *  Roles pair the three-model set a context loads; wedetect is always f32. */
typedef enum aicore_vlx_seek_model_role {
    AICORE_VLX_SEEK_ROLE_WEDETECT = 0,
    AICORE_VLX_SEEK_ROLE_VISION = 1,
    AICORE_VLX_SEEK_ROLE_LLM = 2,
} aicore_vlx_seek_model_role;

typedef struct aicore_vlx_seek_model_entry {
    const char* filename;
    const char* download_url; /**< revision-pinned resolver URL */
    const char* display_name;
    const char* family;       /**< the three-model set of one checkpoint */
    int32_t role;             /**< aicore_vlx_seek_model_role */
    const char* quantization; /**< f32, f16, q8_0 */
    const char* sha256;
    int64_t size_bytes;
} aicore_vlx_seek_model_entry;

/** Number of published catalog entries. */
AICORE_CAPI int aicore_vlx_seek_model_count(void);
/** Returns the catalog entry at `index` (NULL when out of range). */
AICORE_CAPI const aicore_vlx_seek_model_entry* aicore_vlx_seek_model_at(
        int index);
/** Index of the entry the catalog declares as its default — the row
 *  carrying the visible "(recommended)" marker. */
AICORE_CAPI int aicore_vlx_seek_model_default_index(void);
/** Returns the catalog entry whose filename matches (NULL when not found). */
AICORE_CAPI const aicore_vlx_seek_model_entry*
aicore_vlx_seek_model_by_filename(const char* filename);
/** Returns the base URL of the published model repository. */
AICORE_CAPI const char* aicore_vlx_seek_model_download_base(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

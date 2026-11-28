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

#define AICORE_WEVISONE_ABI_VERSION 1u

/** Returns the ABI version of the WeVisOne-VL C API (bump on breaking ABI
 *  changes). */
AICORE_CAPI int aicore_wevisone_abi_version(void);

typedef struct aicore_wevisone_ctx aicore_wevisone_ctx;
typedef struct aicore_wevisone_options aicore_wevisone_options;

/** Creates a default options struct (device "auto" — AICore runtime order,
 *  threads 0 = backend default, size "2b", precision "f16", decoder gguf
 *  precision "f32", models_dir empty = shared wevisone cache dir, max new
 *  tokens 4096, matte decoding enabled).
 *  Release with aicore_wevisone_options_free. */
AICORE_CAPI aicore_wevisone_options* aicore_wevisone_options_new(void);
/** Releases an options struct created by aicore_wevisone_options_new. */
AICORE_CAPI void aicore_wevisone_options_free(aicore_wevisone_options* opts);
/** Selects the inference device: NULL/"auto", "cpu", "cuda", "vulkan",
 *  "metal" (optionally ":N"). */
AICORE_CAPI void aicore_wevisone_options_set_device(
        aicore_wevisone_options* opts, const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_wevisone_options_set_threads(
        aicore_wevisone_options* opts, int n_threads);
/** Checkpoint family: "2b" (default) or "4b". Selects the published
 *  wevisone-vlm-<size>-*, mmproj-wevisone-vlm-<size>-f16 and
 *  wevisone-mask-decoder-<size>-* GGUF rows of the model catalog. */
AICORE_CAPI void aicore_wevisone_options_set_size(aicore_wevisone_options* opts,
                                                  const char* size);
/** VLM weight precision: "f16" (default), "q8_0" or "q4_K_M" (the published
 *  tiers). Selects the wevisone-vlm-<size>-<precision>.gguf file; the
 *  mmproj stays f16 (the only published projector tier). */
AICORE_CAPI void aicore_wevisone_options_set_precision(
        aicore_wevisone_options* opts, const char* precision);
/** Pixel-decoder weight precision: "f32" (default, the checkpoint dtype and
 *  the only parity-gated tier) or "f16". */
AICORE_CAPI void aicore_wevisone_options_set_decoder_precision(
        aicore_wevisone_options* opts, const char* precision);
/** Directory holding the GGUF files. Empty (default) uses the shared model
 *  cache dir (extract/wevisone_models under the shared data root — see
 *  aicore_wevisone_model_cache_dir). */
AICORE_CAPI void aicore_wevisone_options_set_models_dir(
        aicore_wevisone_options* opts, const char* dir);
/** Explicit per-model overrides; empty (default) derives the filenames from
 *  models_dir + size + precision:
 *    wevisone-vlm-<size>-<precision>.gguf        (LLM)
 *    mmproj-wevisone-vlm-<size>-f16.gguf         (ViT projector, always f16)
 *    wevisone-mask-decoder-<size>-<prec>.gguf    (pixel-decoder sidecar) */
AICORE_CAPI void aicore_wevisone_options_set_vlm_gguf(
        aicore_wevisone_options* opts, const char* path);
AICORE_CAPI void aicore_wevisone_options_set_mmproj_gguf(
        aicore_wevisone_options* opts, const char* path);
AICORE_CAPI void aicore_wevisone_options_set_decoder_gguf(
        aicore_wevisone_options* opts, const char* path);
/** Explicit libtorch_cpu.so path for the bit-exact-with-torch float log
 *  (vmsLn) used by the decoder's BoxFourierEncoder. Empty (default) keeps
 *  the standard pip-torch layout auto-discovery; when unavailable the
 *  double-precision log fallback runs. POSIX-only discovery. */
AICORE_CAPI void aicore_wevisone_options_set_mkl_log_lib(
        aicore_wevisone_options* opts, const char* path);
/** Maximum greedy generation length (default 4096). */
AICORE_CAPI void aicore_wevisone_options_set_max_new_tokens(
        aicore_wevisone_options* opts, int n_tokens);
/** Enables (default) or disables the pixel-decoder pass when the answer
 *  contains <OBJ> tokens. Disabling turns the context into a pure object-
 *  level detector (boxes / points / counts / OCR). */
AICORE_CAPI void aicore_wevisone_options_set_decode_masks(
        aicore_wevisone_options* opts, int enable);
/** Log level: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR (default 1). */
AICORE_CAPI void aicore_wevisone_options_set_log_level(
        aicore_wevisone_options* opts, int log_level);
/** Debug: dump parity stage tensors as raw f32 files into this directory
 *  after every run call. The string is copied; empty (default) disables
 *  dumping. */
AICORE_CAPI void aicore_wevisone_options_set_dump_dir(
        aicore_wevisone_options* opts, const char* dir);

/** Loads the VLM (mmproj + LLM) and, lazily, the pixel decoder. Returns a
 *  context whose is_ready() may still be 0 (device admission failure);
 *  inspect aicore_wevisone_last_error(). NULL only on invalid input or
 *  out-of-memory. */
AICORE_CAPI aicore_wevisone_ctx* aicore_wevisone_load_opts(
        const aicore_wevisone_options* opts);
/** Releases a context; safe on NULL. Contexts must not be destroyed while a
 *  worker thread is inside a run call (serialize externally). */
AICORE_CAPI void aicore_wevisone_free(aicore_wevisone_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model set. */
AICORE_CAPI int aicore_wevisone_is_ready(const aicore_wevisone_ctx* ctx);
/** Returns the last error message of the context (NULL when none). */
AICORE_CAPI const char* aicore_wevisone_last_error(
        const aicore_wevisone_ctx* ctx);
/** Releases any buffer returned by an aicore_wevisone_* function (JSON/RGB
 *  helpers). Safe on NULL. */
AICORE_CAPI void aicore_wevisone_free_buffer(void* p);

/** Load an image file as tightly-packed RGB (HWC, 3 bytes/pixel).
 *  Compatibility helper for path-driven callers; the typed hot path is
 *  aicore_wevisone_run_image with a borrowed aicore_image_view. Caller
 *  frees \p out_rgb with aicore_wevisone_free_buffer. */
AICORE_CAPI int aicore_wevisone_load_path_rgb(const char* image_path,
                                              uint8_t** out_rgb,
                                              int32_t* out_width,
                                              int32_t* out_height);

/** ---- typed inference ----
 *
 *  One request = vision encode + LLM greedy decode (+ optional pixel
 *  decoder over every <OBJ>). \p img must be non-NULL with positive
 *  dimensions, a supported format (RGB8 native; RGBA8/BGRA8/GRAY8/BGR8 are
 *  converted) and row_stride_bytes >= width * channels (else -1).
 *
 *  \p task: "detection" | "referring" | "point" | "count_box" |
 *  "count_point" | "ocr" | "ocr_grounding" | "segmentation" | "matting" |
 *  "visual_prompt" | "freeform" (the released prompt family, prompts.py).
 *  \p query: newline-separated category list or a single referring phrase
 *  (UTF-8); ignored by "ocr" and "freeform".
 *  \p ref_boxes: borrowed [x1, y1, x2, y2] float quadruples in original-
 *  image pixels, used only by "visual_prompt" (the <ref>/<box> tokens).
 *
 *  Returns 0 on success, -1 on contract/validation errors, -2 on
 *  runtime/inference errors and -3 when the device cannot hold the models
 *  plus the graph workspace (see aicore_wevisone_last_error). On success
 *  the typed result getters below expose the answer until the next run
 *  call. */
AICORE_CAPI int aicore_wevisone_run_image(aicore_wevisone_ctx* ctx,
                                          const aicore_image_view* img,
                                          const char* task,
                                          const char* query,
                                          const float* ref_boxes,
                                          int32_t n_ref_boxes);

/** ---- batch inference (engine.predict_batch semantics) ----
 *
 *  One call = N images through the SAME loaded model set with a shared
 *  task/query/prompt construction and per-image mask decoding, one result
 *  per image (the python engine.predict_batch contract). Scheduling note:
 *  the ggml runtime executes samples sequentially — llama.cpp multimodal
 *  contexts are single-sequence, and the python padded-batch generate is
 *  itself only an approximation of per-sample generation. The interface and
 *  per-image outputs are what this API aligns.
 *
 *  All images share one task/query; \p ref_boxes applies only to
 *  "visual_prompt" and every image must carry the same reference boxes
 *  (pixel-space of that image — use identical normalized values when the
 *  batch shares a framing).
 *
 *  On success returns 0 and the results live in the context until the next
 *  run/run_batch; the per-index getters below read them. A single image
 *  failure aborts the batch with that image's error code (already-computed
 *  results stay readable at their indices). */
AICORE_CAPI int aicore_wevisone_run_batch(aicore_wevisone_ctx* ctx,
                                          const aicore_image_view* imgs,
                                          int32_t n_images,
                                          const char* task,
                                          const char* query,
                                          const float* ref_boxes,
                                          int32_t n_ref_boxes);

/** ---- GT-conditioned pixel-decoder pass (engine.decode_masks contract) ----
 *
 *  Teacher-forces \p prompt + \p response (NO LLM generation) and decodes
 *  one matte per query from the <OBJ> hidden states. \p boxes01 (when
 *  non-NULL with n_boxes > 0) conditions the decoder on those UNROUNDED
 *  normalized xyxy boxes — the evaluation GT protocol
 *  (inference_matting.py box_source=gt: the synthetic response carries the
 *  ROUNDED box while the decoder conditions on the unrounded one); when
 *  NULL the boxes are parsed from the response tokens (rounded, the
 *  generated-path semantics). The query count is
 *  min(n_boxes|parsed, <OBJ> count), exactly like python; \p response
 *  must contain <OBJ> when boxes are supplied (else -1, the python
 *  RuntimeError).
 *
 *  Returns 0 on success with the same result getters as
 *  aicore_wevisone_run_image (answer = \p response; boxes = the conditioning
 *  boxes in original-image pixels; first matte exposed as usual). */
AICORE_CAPI int aicore_wevisone_run_decode_masks(aicore_wevisone_ctx* ctx,
                                                 const aicore_image_view* img,
                                                 const char* prompt,
                                                 const char* response,
                                                 const float* boxes01,
                                                 int32_t n_boxes);

/** ---- indexed results of the most recent run/run_batch ---- */

/** Number of results held (1 after run_image, N after run_batch). */
AICORE_CAPI int32_t
aicore_wevisone_result_count(const aicore_wevisone_ctx* ctx);
/** Per-index answer; NULL when \p index is out of range. */
AICORE_CAPI const char* aicore_wevisone_result_answer_at(
        const aicore_wevisone_ctx* ctx, int32_t index);
/** Full per-index result as JSON (same schema as aicore_wevisone_result_json).
 *  Caller frees with aicore_wevisone_free_buffer. */
AICORE_CAPI char* aicore_wevisone_result_json_at(const aicore_wevisone_ctx* ctx,
                                                 int32_t index);
/** Per-index box count (0 when the answer rejected the target). */
AICORE_CAPI int32_t aicore_wevisone_result_box_count_at(
        const aicore_wevisone_ctx* ctx, int32_t index);
/** Per-index box in original-image pixels (out[4]). 0 on success. */
AICORE_CAPI int aicore_wevisone_result_box_at(const aicore_wevisone_ctx* ctx,
                                              int32_t index,
                                              int32_t box_index,
                                              float* out);
/** Per-index: 1 when the answer carried <OBJ> tokens and the decoder ran. */
AICORE_CAPI int aicore_wevisone_result_has_masks_at(
        const aicore_wevisone_ctx* ctx, int32_t index);
/** Per-index matte count. */
AICORE_CAPI int32_t aicore_wevisone_result_matte_count_at(
        const aicore_wevisone_ctx* ctx, int32_t index);
/** Per-index matte dimensions. 0 on success, -1 out of range. Only matte
 *  index 0 is retained (the upstream RunResult contract). */
AICORE_CAPI int aicore_wevisone_result_matte_size_at(
        const aicore_wevisone_ctx* ctx,
        int32_t index,
        int32_t matte_index,
        int32_t* out_width,
        int32_t* out_height);
/** Borrowed row-major f32 alpha of matte \p matte_index of result \p index;
 *  valid until the next run/run_batch. NULL out of range (only matte index
 *  0 is retained — the upstream RunResult contract). */
AICORE_CAPI const float* aicore_wevisone_result_matte_data_at(
        const aicore_wevisone_ctx* ctx, int32_t index, int32_t matte_index);

/** ---- typed result of the most recent run (valid until the next run) ---- */

/** Model answer after <|im_end|>/stripping, before parsing. */
AICORE_CAPI const char* aicore_wevisone_result_answer(
        const aicore_wevisone_ctx* ctx);
/** Full result as JSON (answer, boxes, has_masks, prompt_tokens,
 *  completion_tokens). Caller frees with aicore_wevisone_free_buffer. */
AICORE_CAPI char* aicore_wevisone_result_json(const aicore_wevisone_ctx* ctx);
/** Number of parsed result boxes in original-image pixels (0 when the
 *  answer rejected the target). */
AICORE_CAPI int32_t
aicore_wevisone_result_box_count(const aicore_wevisone_ctx* ctx);
/** Box \p index in original-image pixels (out[4] = x1, y1, x2, y2).
 *  Returns 0 on success, -1 when out of range. */
AICORE_CAPI int aicore_wevisone_result_box(const aicore_wevisone_ctx* ctx,
                                           int32_t index,
                                           float* out);
/** 1 when the answer carried <OBJ> tokens and the pixel decoder ran. */
AICORE_CAPI int aicore_wevisone_result_has_masks(
        const aicore_wevisone_ctx* ctx);
/** Number of decoded mattes (== box count when masks were decoded). */
AICORE_CAPI int32_t
aicore_wevisone_result_matte_count(const aicore_wevisone_ctx* ctx);
/** Matte \p index dimensions. Returns 0 on success, -1 when out of range.
 *  Only matte index 0 is retained (the upstream RunResult contract). */
AICORE_CAPI int aicore_wevisone_result_matte_size(
        const aicore_wevisone_ctx* ctx,
        int32_t index,
        int32_t* out_width,
        int32_t* out_height);
/** Borrowed row-major f32 alpha of matte \p index in [0, 1] (w*h floats);
 *  valid until the next run call. NULL when out of range (only matte index
 *  0 is retained — the upstream RunResult contract). */
AICORE_CAPI const float* aicore_wevisone_result_matte_data(
        const aicore_wevisone_ctx* ctx, int32_t index);
/** Token counts of the most recent run. */
AICORE_CAPI int32_t
aicore_wevisone_result_prompt_tokens(const aicore_wevisone_ctx* ctx);
AICORE_CAPI int32_t
aicore_wevisone_result_completion_tokens(const aicore_wevisone_ctx* ctx);

/** Renders the most recent result on \p img (borrowed view; the image is
 *  copied) in the official style: red width-3 rectangles for boxes plus,
 *  when mattes exist, a green alpha composite of the first matte. Returns 0
 *  on success, -1 on invalid arguments, -2 on write failure. */
AICORE_CAPI int aicore_wevisone_result_render_png(
        const aicore_wevisone_ctx* ctx,
        const aicore_image_view* img,
        const char* out_png_path);

/** Per-stage wall-clock timings of the most recent run. Returns 0 on
 *  success, -1 when no inference has run yet. */
AICORE_CAPI int aicore_wevisone_last_pipeline_timings(
        const aicore_wevisone_ctx* ctx, aicore_pipeline_timings* out);

/** Model introspection. */
/** Backend-RESOLVED device name (surfaces silent CPU fallbacks). Owned by
 *  the context; copy before freeing. */
AICORE_CAPI const char* aicore_wevisone_context_device(
        const aicore_wevisone_ctx* ctx);
/** Effective CPU thread count after the auto (<=0) resolution. */
AICORE_CAPI int aicore_wevisone_context_threads(const aicore_wevisone_ctx* ctx);
/** VLM weight precision of the loaded context ("f16"/"q8_0"/"q4_K_M"). */
AICORE_CAPI const char* aicore_wevisone_context_precision(
        const aicore_wevisone_ctx* ctx);
/** Returns a JSON summary of the loaded models and the latest run. Caller
 *  frees with aicore_wevisone_free_buffer. */
AICORE_CAPI char* aicore_wevisone_info_json(aicore_wevisone_ctx* ctx);

/** Warms up the backend for the context's device (compiles backend kernels
 *  so the first real request does not pay JIT costs). */
AICORE_CAPI int aicore_wevisone_warmup_backend(aicore_wevisone_ctx* ctx);
/** Releases process-wide WeVisOne-VL runtime resources it owns (idempotent;
 *  never destroys live contexts). Delegates to aicore_runtime_shutdown(). */
AICORE_CAPI void aicore_wevisone_shutdown(void);

/** Returns the local model cache directory (extract/wevisone_models under
 *  the shared data root). Caller frees with aicore_wevisone_free_buffer. */
AICORE_CAPI char* aicore_wevisone_model_cache_dir(void);

/** Published GGUF catalog (Asher-1/WeVisOne-VL-GGML revision). Roles pair
 *  the three-file set a context loads per family; the mmproj is always f16
 *  and the parity-gated decoder tier is f32. */
typedef enum aicore_wevisone_model_role {
    AICORE_WEVISONE_ROLE_VLM = 0,
    AICORE_WEVISONE_ROLE_MMPROJ = 1,
    AICORE_WEVISONE_ROLE_DECODER = 2,
} aicore_wevisone_model_role;

typedef struct aicore_wevisone_model_entry {
    const char* filename;
    const char* download_url; /**< revision-pinned resolver URL */
    const char* display_name;
    const char* family;       /**< checkpoint family: "2b" or "4b" */
    int32_t role;             /**< aicore_wevisone_model_role */
    const char* quantization; /**< f16, q8_0, q4_K_M, f32 */
    const char* sha256;
    int64_t size_bytes;
} aicore_wevisone_model_entry;

/** Number of published catalog entries. */
AICORE_CAPI int aicore_wevisone_model_count(void);
/** Returns the catalog entry at `index` (NULL when out of range). */
AICORE_CAPI const aicore_wevisone_model_entry* aicore_wevisone_model_at(
        int index);
/** Index of the entry the catalog declares as its default — the row
 *  carrying the visible "(recommended)" marker. */
AICORE_CAPI int aicore_wevisone_model_default_index(void);
/** Returns the catalog entry whose filename matches (NULL when not found). */
AICORE_CAPI const aicore_wevisone_model_entry*
aicore_wevisone_model_by_filename(const char* filename);
/** Returns the base URL of the published model repository. */
AICORE_CAPI const char* aicore_wevisone_model_download_base(void);

#ifdef __cplusplus
} /* extern "C" */
#endif
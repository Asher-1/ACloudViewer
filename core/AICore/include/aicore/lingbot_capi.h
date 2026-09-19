// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// LingBot-Map (Geometric Context Transformer) streaming 3D reconstruction
// C API. In-tree port of the cpp_ggml runtime from
// https://github.com/Asher-1/lingbot-map-ggml (upstream Apache-2.0).
//
// The engine lives in core/AICore/src/tasks/lingbot/ and is accelerated by
// ggml with CPU / CUDA / Vulkan / Metal backends (device auto-pick follows
// the AICore runtime order, e.g. CUDA -> Vulkan -> CPU on Linux).
//
// Streaming contract: a context owns one persistent GCT KV cache. Feed
// frames of one video/sequence in order through aicore_lingbot_infer_stream
// (the engine runs the official scale pass automatically on the first call
// of a stream); call aicore_lingbot_stream_reset to start a new stream.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"
#include "aicore/image_view.h"
#include "aicore/pipeline_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the ABI version of the LingBot-Map C API (bump on breaking ABI
 *  changes). */
AICORE_CAPI int aicore_lingbot_abi_version(void);

typedef struct aicore_lingbot_ctx aicore_lingbot_ctx;
typedef struct aicore_lingbot_options aicore_lingbot_options;

/** Creates a default options struct (device "auto", threads 0 = backend
 *  default, image size 518, official scale=8/window=64 persistent F16 KV
 *  cache profile). Release with aicore_lingbot_options_free. */
AICORE_CAPI aicore_lingbot_options* aicore_lingbot_options_new(void);
/** Releases an options struct created by aicore_lingbot_options_new. */
AICORE_CAPI void aicore_lingbot_options_free(aicore_lingbot_options* opts);
/** Selects the inference device: NULL or "auto", "cpu", "gpu", "vulkan"
 *  (optionally ":N"), "cuda" (Linux/Windows). */
AICORE_CAPI void aicore_lingbot_options_set_device(aicore_lingbot_options* opts,
                                                   const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_lingbot_options_set_threads(
        aicore_lingbot_options* opts, int n_threads);
/** Processing resolution (official crop width, snapped to the patch grid;
 *  518 by default). Height follows the source aspect ratio. */
AICORE_CAPI void aicore_lingbot_options_set_image_size(
        aicore_lingbot_options* opts, int size);
/** Persistent KV-cache profile (official release default: 8 / 64). */
AICORE_CAPI void aicore_lingbot_options_set_kv_profile(
        aicore_lingbot_options* opts, int scale_frames, int window_frames);
/** Total frames of the stream, as a device-resident cache capacity hint
 *  (0 = unknown; the engine then sizes from the first stream call). */
AICORE_CAPI void aicore_lingbot_options_set_stream_capacity(
        aicore_lingbot_options* opts, int total_frames);

/** Load a LingBot-Map GGUF (lingbot-map q8/f16/f32). Returns NULL on
 *  failure; inspect aicore_lingbot_last_error() for the reason. */
AICORE_CAPI aicore_lingbot_ctx* aicore_lingbot_load_opts(
        const char* gguf_path, const aicore_lingbot_options* opts);
/** Releases a context returned by aicore_lingbot_load_opts; safe on NULL. */
AICORE_CAPI void aicore_lingbot_free(aicore_lingbot_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model. */
AICORE_CAPI int aicore_lingbot_is_ready(const aicore_lingbot_ctx* ctx);
/** Returns the last error message of the context (empty when none). */
AICORE_CAPI const char* aicore_lingbot_last_error(
        const aicore_lingbot_ctx* ctx);

/** Releases any buffer returned by an aicore_lingbot_* function. Safe on
 *  NULL. */
AICORE_CAPI void aicore_lingbot_free_buffer(void* p);

/** Typed streaming frame result. All pointers are BORROWED from the
 *  context and remain valid only until the next stream call, reset, or
 *  context free. Row-major, tightly packed. */
typedef struct aicore_lingbot_result {
    const float* depth;      /**< [height * width], meters. */
    const float* depth_conf; /**< [height * width], confidence. */
    const float* c2w;        /**< 4x4 row-major camera-to-world matrix. */
    const float* intrinsics; /**< [fx, fy, cx, cy] in pixels. */
    const float* pose_enc;   /**< 9 floats, raw pose encoding. */
    int32_t width;
    int32_t height;
    int32_t frame_index; /**< Zero-based index inside the current call. */
} aicore_lingbot_result;

/** Per-frame streaming callback. Invoked exactly once per completed output
 *  frame while aicore_lingbot_infer_stream runs. The \p result is valid
 *  only during the call; copy anything that must outlive it. Return 0 to
 *  continue, non-zero to abort the stream (the infer call then returns -1
 *  with error "frame callback aborted"). */
typedef int (*aicore_lingbot_frame_cb)(void* user,
                                       const aicore_lingbot_result* result);

/** Streaming reconstruction over a prepared float frame buffer.
 *
 *  \p frames_nchw holds \p n_frames tightly packed, normalized [0,1]
 *  NCHW RGB frames of \p width x \p height (both multiples of the patch
 *  grid, i.e. the output of aicore_lingbot_preprocess_image). The engine
 *  runs the official scale pass automatically when the KV cache is empty,
 *  then streams frame by frame, invoking \p cb once per frame.
 *
 *  Returns 0 on success (all frames delivered), -1 on error (inspect
 *  aicore_lingbot_last_error). */
AICORE_CAPI int aicore_lingbot_infer_stream(aicore_lingbot_ctx* ctx,
                                            const float* frames_nchw,
                                            int n_frames,
                                            int width,
                                            int height,
                                            aicore_lingbot_frame_cb cb,
                                            void* user);

/** Official preprocessing of one decoded image (mode="crop" of the
 *  upstream load_and_preprocess_images): aspect-preserving bicubic resize
 *  to \p image_size width, height snapped to the patch grid, center crop
 *  when it exceeds \p image_size, output as normalized [0,1] NCHW RGB.
 *
 *  Two-call sizing: pass NULL/0 for \p out_nchw to get the required float
 *  count, then pass a buffer of that size. \p out_width / \p out_height
 *  (optional) receive the processed dimensions. Returns the required/
 *  written float count, -1 on invalid arguments. The image view is
 *  borrowed and never modified. */
AICORE_CAPI int aicore_lingbot_preprocess_image(const aicore_image_view* image,
                                                int image_size,
                                                float* out_nchw,
                                                int out_size,
                                                int32_t* out_width,
                                                int32_t* out_height);

/** Clears the persistent KV cache: the next stream call starts a new
 *  stream (with a fresh scale pass). Returns 0, -1 on invalid ctx. */
AICORE_CAPI int aicore_lingbot_stream_reset(aicore_lingbot_ctx* ctx);

/** Number of frames delivered by the most recent stream (or so far on this
 *  stream, when called from the frame callback). -1 on invalid ctx. */
AICORE_CAPI int aicore_lingbot_last_stream_frames(
        const aicore_lingbot_ctx* ctx);

/** Pipeline timings of the most recent infer_stream call (aggregated over
 *  its frames: preprocess = image preparation, inference = graph work,
 *  postprocess = result decode/materialization). */
AICORE_CAPI int aicore_lingbot_last_pipeline_timings(
        const aicore_lingbot_ctx* ctx, aicore_pipeline_timings* out);

/** Optional native sky segmentation (lingbot-map-skyseg GGUF). When loaded,
 *  infer_stream masks sky pixels per frame: sky depth confidence is zeroed
 *  (matching the official viewer's --mask_sky semantics) and the raw
 *  keep-mask is retrievable via aicore_lingbot_last_sky_mask. */
AICORE_CAPI int aicore_lingbot_skyseg_load(aicore_lingbot_ctx* ctx,
                                           const char* skyseg_gguf_path);
/** 1 when a sky segmentation model is loaded on this context. */
AICORE_CAPI int aicore_lingbot_skyseg_ready(const aicore_lingbot_ctx* ctx);
/** Provides external per-frame sky keep-masks (255 = keep, 0 = sky) for
 *  the NEXT aicore_lingbot_infer_stream call, tightly packed at
 *  [frame_count][width * height] in the stream's processed resolution.
 *  When set, the native skyseg pass is skipped and these masks drive the
 *  official --mask_sky confidence zeroing instead (upstream
 *  <scene>_sky_masks cache semantics). The masks are consumed once.
 *  Returns 0 on success, -1 on invalid arguments. */
AICORE_CAPI int aicore_lingbot_set_external_sky_masks(
        aicore_lingbot_ctx* ctx,
        const unsigned char* masks,
        int frame_count,
        int width,
        int height);

/** Keep-mask (0 = sky, 255 = keep) of the most recent frame processed by
 *  infer_stream, at the frame's processed resolution. Two-call sizing:
 *  pass NULL/0 to get the required byte length, then a buffer of that
 *  size. Returns the required size, 0 when no sky model is loaded, -1 on
 *  invalid arguments. */
AICORE_CAPI int aicore_lingbot_last_sky_mask(aicore_lingbot_ctx* ctx,
                                             unsigned char* buf,
                                             int buf_size);

/** Model introspection. */
/** Backend-RESOLVED device name ("CUDA0", "Vulkan0", "cpu", ...). Owned by
 *  ctx; copy before freeing. */
AICORE_CAPI const char* aicore_lingbot_context_device(
        const aicore_lingbot_ctx* ctx);
/** Effective CPU thread count after the auto (<=0) resolution. */
AICORE_CAPI int aicore_lingbot_context_threads(const aicore_lingbot_ctx* ctx);
/** Processing width (from GGUF metadata / options; 518 by default). */
AICORE_CAPI int aicore_lingbot_context_image_size(
        const aicore_lingbot_ctx* ctx);
/** Patch size (14 by default). */
AICORE_CAPI int aicore_lingbot_context_patch_size(
        const aicore_lingbot_ctx* ctx);
/** Returns a JSON summary of the loaded model. Caller frees with
 *  aicore_lingbot_free_buffer. */
AICORE_CAPI char* aicore_lingbot_info_json(aicore_lingbot_ctx* ctx);

/** Warms up the backend for `device`; returns 0 on success. */
AICORE_CAPI int aicore_lingbot_warmup_backend(const char* device);

/** Total memory of the GPU device that `device` resolves to, in bytes
 *  (0 when the device is "cpu" or no accelerator matches). "auto"/"gpu"
 *  resolve to the first accelerator of any family in runtime order;
 *  "cuda"/"vulkan" (optionally ":N") select that family's device. Read-only
 *  query — safe before any context is created; lets callers auto-profile
 *  options (e.g. the KV-cache profile) to the machine's VRAM. */
AICORE_CAPI uint64_t aicore_lingbot_device_total_memory(const char* device);

/** Releases process-wide LingBot-Map resources (idempotent; delegates to
 *  aicore_runtime_shutdown). */
AICORE_CAPI void aicore_lingbot_shutdown(void);
/** Returns the local model cache directory. Caller frees with
 *  aicore_lingbot_free_buffer. */
AICORE_CAPI char* aicore_lingbot_model_cache_dir(void);

/** Published GGUF catalog (Hugging Face Asher-1/lingbot-map-gguf). */
typedef struct aicore_lingbot_model_entry {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
    const char* role; /**< "map" or "skyseg". */
    uint64_t size_bytes;
    const char* sha256; /**< hex-encoded, canonical HF LFS oid. */
} aicore_lingbot_model_entry;

/** Number of published catalog entries (map + skyseg roles). */
AICORE_CAPI int aicore_lingbot_model_count(void);
/** Returns the catalog entry at `index` (NULL when out of range). */
AICORE_CAPI const aicore_lingbot_model_entry* aicore_lingbot_model_at(
        int index);
/** Index of the map-role entry declared as the default (the visible
 *  "(recommended)" row for UI combos). */
AICORE_CAPI int aicore_lingbot_model_default_index(void);
/** Index of the default skyseg-role entry (-1 when none). */
AICORE_CAPI int aicore_lingbot_skyseg_model_default_index(void);
/** Returns the catalog entry whose filename matches (NULL when not
 *  found). */
AICORE_CAPI const aicore_lingbot_model_entry* aicore_lingbot_model_by_filename(
        const char* filename);
/** Returns the base URL of the published Hugging Face model repo. */
AICORE_CAPI const char* aicore_lingbot_model_download_base(void);

#ifdef __cplusplus
}
#endif

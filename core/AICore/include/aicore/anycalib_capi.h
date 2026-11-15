// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// AnyCalib task public C ABI (single-image intrinsics regression).
//
// The task wraps the AnyCalib v1.0.0 feed-forward network (DINOv2 ViT-L/14
// backbone + light DPT decoder + convex tangent head) exported by upstream
// `scripts/anycalib/export_onnx.py` and converted to GGUF by
// `core/AICore/tools/convert_anycalib_onnx_gguf.py`. This header currently
// publishes the model catalog surface. The inference surface lands with the
// task graph implementation and is documented here as the settled contract:
//
//   aicore_anycalib_abi_version() -> 1
//   aicore_anycalib_options_new/free/set_device/set_threads
//   aicore_anycalib_load_opts(gguf_path, opts) -> ctx (opaque)
//   aicore_anycalib_is_ready / aicore_anycalib_last_error
//   aicore_anycalib_run(ctx, const float* image_rgb_chw, out)
//       image_rgb_chw: borrowed read-only network input, 3x322x322 RGB in
//           [0, 1], HWC-of-channels = CHW, exactly the exported ONNX
//           "image" input (the caller owns the crop/rotate/scale geometry
//           bookkeeping, mirroring upstream AnyCalibInput). A prepared
//           network-input tensor crosses the ABI as a borrowed typed float
//           buffer (same boundary class as the LightGlue/LoMa matcher
//           descriptor inputs), not as a decoded image view.
//       out: typed result — rays [322*322, 3] unit camera-frame rays and
//           tangent_coords [322*322, 2] FoV tangent coordinates (row-major
//           over the 322x322 grid), malloc'd, released by
//           aicore_anycalib_free_buffer.
//   aicore_anycalib_last_pipeline_timings (common timing struct)
//   aicore_anycalib_free / aicore_anycalib_shutdown (idempotent)
//
// Consumers read model URLs and cache folders only through the catalog
// accessors below plus aicore/asset_digests.h; the release assets live in
// the `anycalib` tag of the cloudViewer_downloads release repository.
//
// Inference contract (implemented):
//   - aicore_anycalib_run takes the prepared network input (borrowed,
//     read-only 3x322x322 RGB float CHW in [0, 1], exactly the exported
//     ONNX "image" input; the caller owns the crop/rotate/scale geometry
//     bookkeeping, mirroring upstream AnyCalibInput). A prepared
//     network-input tensor crosses the ABI as a borrowed typed float
//     buffer (same boundary class as the LightGlue/LoMa matcher
//     descriptor inputs), not as a decoded image view.
//   - Typed results: rays [322*322, 3] unit camera-frame rays and
//     tangent_coords [322*322, 2] FoV tangent coordinates, row-major over
//     the grid, malloc'd and released via aicore_anycalib_free_buffer.
//   - Common timing struct, readiness, last_error, idempotent shutdown.

#pragma once

#include <stdint.h>

#include "aicore/export.h"
#include "aicore/pipeline_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aicore_anycalib_model_entry {
    /** Release asset name, e.g. "anycalib_gen_v1.0.0_322x322.f16.gguf". */
    const char* filename;
    /** Stable download URL. */
    const char* download_url;
    /** User-facing display name. */
    const char* display_name;
    /** "gen" — the dense per-pixel ray/tangent feed-forward network. */
    const char* role;
    /** "square322" — the fixed square training-resolution export. */
    const char* resolution;
} aicore_anycalib_model_entry;

/** Number of published AnyCalib catalog entries. */
AICORE_CAPI int aicore_anycalib_model_count(void);

/** Catalog entry by index (nullptr when out of range). */
AICORE_CAPI const aicore_anycalib_model_entry* aicore_anycalib_model_at(
        int index);

/** Catalog entry by release asset filename (nullptr when unknown). */
AICORE_CAPI const aicore_anycalib_model_entry*
aicore_anycalib_model_by_filename(const char* filename);

/** Default cross-platform GGUF model cache directory (UTF-8). Free with
 *  aicore_anycalib_free_buffer. */
AICORE_CAPI char* aicore_anycalib_model_cache_dir(void);

typedef struct aicore_anycalib_ctx aicore_anycalib_ctx;
typedef struct aicore_anycalib_options aicore_anycalib_options;

AICORE_CAPI int aicore_anycalib_abi_version(void);

AICORE_CAPI aicore_anycalib_options* aicore_anycalib_options_new(void);
AICORE_CAPI void aicore_anycalib_options_free(aicore_anycalib_options* opts);
/** "auto", "cpu", ... (runtime backend registry). Default "auto". */
AICORE_CAPI void aicore_anycalib_options_set_device(
        aicore_anycalib_options* opts, const char* device);
/** 0 = backend default. */
AICORE_CAPI void aicore_anycalib_options_set_threads(
        aicore_anycalib_options* opts, int n_threads);

/** Load the anycalib GGUF and build the inference session. Returns a
 *  context even on failure; check aicore_anycalib_is_ready/last_error. */
AICORE_CAPI aicore_anycalib_ctx* aicore_anycalib_load_opts(
        const char* gguf_path, const aicore_anycalib_options* opts);
AICORE_CAPI void aicore_anycalib_free(aicore_anycalib_ctx* ctx);
AICORE_CAPI int aicore_anycalib_is_ready(const aicore_anycalib_ctx* ctx);
AICORE_CAPI const char* aicore_anycalib_last_error(
        const aicore_anycalib_ctx* ctx);

/** One forward pass. image_rgb_chw is the borrowed read-only network input
 *  (3*322*322 floats, RGB CHW in [0, 1]); the outputs are typed results
 *  (rows = 322*322) released via aicore_anycalib_free_buffer. */
AICORE_CAPI int aicore_anycalib_run(aicore_anycalib_ctx* ctx,
                                    const float* image_rgb_chw,
                                    float** out_rays,
                                    float** out_tangent,
                                    int32_t* out_count);

AICORE_CAPI int aicore_anycalib_last_pipeline_timings(
        const aicore_anycalib_ctx* ctx, aicore_pipeline_timings* out_timings);
AICORE_CAPI void aicore_anycalib_free_buffer(void* p);
/** Idempotent process-runtime cleanup; never destroys live contexts. */
AICORE_CAPI void aicore_anycalib_shutdown(void);

#ifdef __cplusplus
}  // extern "C"
#endif

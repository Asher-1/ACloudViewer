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

#pragma once

#include "aicore/export.h"

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

#ifdef __cplusplus
}  // extern "C"
#endif

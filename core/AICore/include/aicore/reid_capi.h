// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// ReID C API: appearance-embedding extraction for multi-object tracking.
//
// The context wraps a classify-task YOLO GGUF (the official model="auto"
// fallback encoder of the Ultralytics track mode) and exposes the pooled
// feature feeding the final linear as the per-detection embedding vector.
// Crops follow the official save_one_box semantics (gain 1.02, pad 10,
// boundary clip) and are stretched to the model's square input, mirroring
// the upstream ReID preprocessing.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"
#include "aicore/image_view.h"
#include "aicore/pipeline_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the ABI version of the ReID C API (bump on breaking ABI
 *  changes). */
AICORE_CAPI int aicore_reid_abi_version(void);

typedef struct aicore_reid_ctx aicore_reid_ctx;
typedef struct aicore_reid_options aicore_reid_options;

/** Creates a default options struct (device "auto", threads 0 = backend
 *  default). Release with aicore_reid_options_free. */
AICORE_CAPI aicore_reid_options* aicore_reid_options_new(void);
/** Releases an options struct created by aicore_reid_options_new. */
AICORE_CAPI void aicore_reid_options_free(aicore_reid_options* opts);
/** Selects the inference device: NULL or "auto", "cpu", "gpu", "vulkan"
 *  (optionally ":N"), "cuda" (Linux/Windows). */
AICORE_CAPI void aicore_reid_options_set_device(aicore_reid_options* opts,
                                                const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_reid_options_set_threads(aicore_reid_options* opts,
                                                 int n_threads);

/** Load a classify-task YOLO GGUF as a ReID encoder. Returns NULL on
 *  failure; inspect aicore_reid_last_error() for the reason. */
AICORE_CAPI aicore_reid_ctx* aicore_reid_load_opts(
        const char* gguf_path, const aicore_reid_options* opts);
/** Releases a context returned by aicore_reid_load_opts; safe on NULL. */
AICORE_CAPI void aicore_reid_free(aicore_reid_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded encoder. */
AICORE_CAPI int aicore_reid_is_ready(const aicore_reid_ctx* ctx);
/** Returns the last error message of the context (empty when none). */
AICORE_CAPI const char* aicore_reid_last_error(const aicore_reid_ctx* ctx);

/** Releases any buffer returned by an aicore_reid_* function. Safe on
 *  NULL. */
AICORE_CAPI void aicore_reid_free_buffer(void* p);

/** Embedding dimension of the loaded encoder (the pooled feature width;
 *  0 before the first successful embed call validates the graph output).
 *  -1 on invalid arguments. */
AICORE_CAPI int aicore_reid_embed_dim(const aicore_reid_ctx* ctx);

/** Extract appearance embeddings for a batch of detection boxes.
 *
 *  image: borrowed, stride-aware view (RGB8/BGR8/RGBA8/BGRA8/GRAY8);
 *  boxes_xyxy: [count, 4] boxes in original-image pixel coordinates
 *  (x1, y1, x2, y2); count may be 0 (returns an empty result).
 *
 *  On success (0) *out_embed receives a malloc'd [count, dim] row-major
 *  float array (count rows, aicore_reid_embed_dim values each, unnor-
 *  malized — the tracker's EMA owns normalization) and *out_count /
 *  *out_dim mirror the shape; free with aicore_reid_free_buffer. The
 *  result is valid until the next embed call or aicore_reid_free.
 *  Returns -1 on failure (inspect aicore_reid_last_error). */
AICORE_CAPI int aicore_reid_embed_image(aicore_reid_ctx* ctx,
                                        const aicore_image_view* image,
                                        const float* boxes_xyxy,
                                        int32_t count,
                                        float** out_embed,
                                        int32_t* out_count,
                                        int32_t* out_dim);

/** Common timing contract adapter for the most recent embed call. */
AICORE_CAPI int aicore_reid_last_pipeline_timings(
        const aicore_reid_ctx* ctx, aicore_pipeline_timings* out_timings);

/** Releases process-wide ReID backend resources (idempotent; delegates to
 *  the shared runtime shutdown). */
AICORE_CAPI void aicore_reid_shutdown(void);

#ifdef __cplusplus
}
#endif

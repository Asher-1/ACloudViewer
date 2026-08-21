// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// LightGlue local-feature matcher C API.
//
// The ggml engine under core/AICore/src/tasks/lightglue/ implements the
// LightGlue architecture from "LightGlue: Local Feature Matching at Light
// Speed" (Lindenberger et al., ICCV 2023); the ALIKED/SIFT front ends and
// descriptor contracts follow cvg/LightGlue
// (https://github.com/cvg/LightGlue). Original implementation on ggml — no
// upstream source files are vendored.

#pragma once
#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the ABI version of the LightGlue C API (bump on breaking ABI
 *  changes). */
AICORE_CAPI int aicore_lightglue_abi_version(void);

typedef struct aicore_lightglue_ctx aicore_lightglue_ctx;
typedef struct aicore_lightglue_options aicore_lightglue_options;

typedef struct {
    float x;
    float y;
    float scale;
    float orientation;
} aicore_lightglue_keypoint;

typedef struct {
    aicore_lightglue_keypoint* keypoints;
    int32_t n_keypoints;
    float* descriptors;
    int32_t descriptor_dim;
    int32_t image_width;
    int32_t image_height;
} aicore_lightglue_features;

typedef struct {
    int32_t idx1;
    int32_t idx2;
    float score;
} aicore_lightglue_match;

typedef struct {
    int32_t input_dim;
    int32_t descriptor_dim;
    int32_t num_heads;
    int32_t num_layers;
    int32_t feature_type;
    int32_t add_scale_orientation;
} aicore_lightglue_geometry;

/* ---- options ---- */
/** Creates a default options struct. Release with
 *  aicore_lightglue_options_free. */
AICORE_CAPI aicore_lightglue_options* aicore_lightglue_options_new(void);
/** Releases an options struct created by aicore_lightglue_options_new. */
AICORE_CAPI void aicore_lightglue_options_free(aicore_lightglue_options* opts);
/** Selects the inference device: NULL or "auto", "cpu", "gpu", "vulkan"
 *  (optionally ":N"), "cuda" (Linux/Windows). */
AICORE_CAPI void aicore_lightglue_options_set_device(
        aicore_lightglue_options* opts, const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_lightglue_options_set_threads(
        aicore_lightglue_options* opts, int n_threads);
/** Minimum accepted match score (lower = more matches, more outliers). */
AICORE_CAPI void aicore_lightglue_options_set_min_score(
        aicore_lightglue_options* opts, double min_score);
/* matcher_type: 0=auto, 1=sift_lightglue, 2=aliked_lightglue */
AICORE_CAPI void aicore_lightglue_options_set_matcher_type(
        aicore_lightglue_options* opts, int matcher_type);

/* ---- lifecycle ---- */
/* Options-based loading (the only load entry; abi 2 removed the former
 * threads-only aicore_lightglue_load variant). opts may be NULL. */
AICORE_CAPI aicore_lightglue_ctx* aicore_lightglue_load_opts(
        const char* gguf_path, const aicore_lightglue_options* opts);
/** Releases a context returned by aicore_lightglue_load_opts; safe on
 *  NULL. */
AICORE_CAPI void aicore_lightglue_free(aicore_lightglue_ctx* ctx);
/** True only after a context owns a successfully initialized matcher. */
AICORE_CAPI int aicore_lightglue_is_ready(const aicore_lightglue_ctx* ctx);
/** Returns the last error message of the context (empty when none). */
AICORE_CAPI const char* aicore_lightglue_last_error(
        const aicore_lightglue_ctx* ctx);

/* ---- model info ---- */
/** Returns the model geometry (input dim, descriptor dim, heads, layers). */
AICORE_CAPI int aicore_lightglue_geometry_of(const aicore_lightglue_ctx* ctx,
                                             aicore_lightglue_geometry* out);
/** Returns a JSON summary of the loaded model. Caller frees with
 *  aicore_lightglue_free_buffer. */
AICORE_CAPI char* aicore_lightglue_info_json(aicore_lightglue_ctx* ctx);
/** Releases any plain buffer returned by an aicore_lightglue_* function
 *  (unified entry point). Safe on NULL. */
AICORE_CAPI void aicore_lightglue_free_buffer(void* p);

/* ---- inference ---- */
AICORE_CAPI int aicore_lightglue_run_match(
        aicore_lightglue_ctx* ctx,
        const aicore_lightglue_features* image1,
        const aicore_lightglue_features* image2,
        aicore_lightglue_match** out_matches,
        int32_t* n_matches);

/** Releases a matches array returned by aicore_lightglue_run_match. */
AICORE_CAPI void aicore_lightglue_free_matches(aicore_lightglue_match* matches);

/* Load LGINP01 binary fixture (two feature sets). Caller frees with
 * aicore_lightglue_free_features. Returns 0 on success. */
AICORE_CAPI int aicore_lightglue_load_fixture(
        const char* path,
        aicore_lightglue_features* image0,
        aicore_lightglue_features* image1);
/** Releases a features struct returned by aicore_lightglue_load_fixture. */
AICORE_CAPI void aicore_lightglue_free_features(
        aicore_lightglue_features* features);

/* ---- quantize ---- */
/** Quantize eligible matcher weights to f16 or q8_0. */
AICORE_CAPI int aicore_lightglue_quantize_gguf(const char* input_gguf,
                                               const char* output_gguf,
                                               const char* type);

/* ---- backend / cache ---- */
/** Warms up the backend for `device`; returns 0 on success. */
AICORE_CAPI int aicore_lightglue_warmup_backend(const char* device);
/** Returns the local model cache directory. Caller frees with
 *  aicore_lightglue_free_buffer. */
AICORE_CAPI char* aicore_lightglue_model_cache_dir(void);

#ifdef __cplusplus
}
#endif

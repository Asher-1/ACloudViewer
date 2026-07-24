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

#ifndef AICORE_LIGHTGLUE_CAPI_H
#define AICORE_LIGHTGLUE_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

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
AICORE_CAPI aicore_lightglue_options* aicore_lightglue_options_new(void);
AICORE_CAPI void aicore_lightglue_options_free(aicore_lightglue_options* opts);
AICORE_CAPI void aicore_lightglue_options_set_device(
        aicore_lightglue_options* opts, const char* device);
AICORE_CAPI void aicore_lightglue_options_set_threads(
        aicore_lightglue_options* opts, int n_threads);
AICORE_CAPI void aicore_lightglue_options_set_min_score(
        aicore_lightglue_options* opts, double min_score);
/* matcher_type: 0=auto, 1=sift_lightglue, 2=aliked_lightglue */
AICORE_CAPI void aicore_lightglue_options_set_matcher_type(
        aicore_lightglue_options* opts, int matcher_type);

/* ---- lifecycle ---- */
AICORE_CAPI aicore_lightglue_ctx* aicore_lightglue_load(const char* gguf_path,
                                                        int n_threads);
AICORE_CAPI aicore_lightglue_ctx* aicore_lightglue_load_opts(
        const char* gguf_path, const aicore_lightglue_options* opts);
AICORE_CAPI void aicore_lightglue_free(aicore_lightglue_ctx* ctx);
AICORE_CAPI const char* aicore_lightglue_last_error(
        const aicore_lightglue_ctx* ctx);

/* ---- model info ---- */
AICORE_CAPI int aicore_lightglue_geometry_of(const aicore_lightglue_ctx* ctx,
                                             aicore_lightglue_geometry* out);
AICORE_CAPI char* aicore_lightglue_info_json(aicore_lightglue_ctx* ctx);
AICORE_CAPI void aicore_lightglue_free_string(char* s);

/* ---- inference ---- */
AICORE_CAPI int aicore_lightglue_run_match(
        aicore_lightglue_ctx* ctx,
        const aicore_lightglue_features* image1,
        const aicore_lightglue_features* image2,
        aicore_lightglue_match** out_matches,
        int32_t* n_matches);

AICORE_CAPI void aicore_lightglue_free_matches(aicore_lightglue_match* matches);

/* Load LGINP01 binary fixture (two feature sets). Caller frees with
 * aicore_lightglue_free_features. Returns 0 on success. */
AICORE_CAPI int aicore_lightglue_load_fixture(
        const char* path,
        aicore_lightglue_features* image0,
        aicore_lightglue_features* image1);
AICORE_CAPI void aicore_lightglue_free_features(
        aicore_lightglue_features* features);

/* ---- quantize ---- */
AICORE_CAPI int aicore_lightglue_quantize(const char* input_gguf,
                                          const char* output_gguf,
                                          const char* type);

/* ---- backend / cache ---- */
AICORE_CAPI int aicore_lightglue_warmup_backend(const char* device);
AICORE_CAPI char* aicore_lightglue_model_cache_dir(void);

#ifdef __cplusplus
}
#endif

#endif

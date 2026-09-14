// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// LoMa feature C API. The detector, descriptor, and matcher implementations
// are ggml-backed and never link an ONNX runtime.
#pragma once

#include <stdint.h>

#include "aicore/export.h"
#include "aicore/pipeline_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aicore_loma_matcher_ctx aicore_loma_matcher_ctx;
typedef struct aicore_loma_matcher_options aicore_loma_matcher_options;
typedef struct aicore_loma_detector_ctx aicore_loma_detector_ctx;
typedef struct aicore_loma_detector_options aicore_loma_detector_options;
typedef struct aicore_loma_descriptor_ctx aicore_loma_descriptor_ctx;
typedef struct aicore_loma_descriptor_options aicore_loma_descriptor_options;

typedef struct {
    float x;
    float y;
} aicore_loma_keypoint;

/* RGB is borrowed and row_stride_bytes may be greater than width * 3. */
typedef struct {
    const uint8_t* rgb;
    int32_t width;
    int32_t height;
    int32_t row_stride_bytes;
} aicore_loma_rgb_image;

/* Owned by the caller after a successful detector run; release it with
 * aicore_loma_detected_features_free(). Coordinates are in source-image
 * pixels, matching COLMAP's LoMa feature contract. */
typedef struct {
    aicore_loma_keypoint* keypoints;
    float* scores;
    int32_t count;
    int32_t image_width;
    int32_t image_height;
} aicore_loma_detected_features;

typedef struct {
    float* descriptors;
    int32_t count;
    int32_t descriptor_dim;
} aicore_loma_described_features;

typedef struct {
    const aicore_loma_keypoint* keypoints;
    int32_t n_keypoints;
    const float* descriptors;
    int32_t descriptor_dim;
    int32_t image_width;
    int32_t image_height;
} aicore_loma_features;

typedef struct {
    int32_t idx0;
    int32_t idx1;
    float score;
} aicore_loma_match;

typedef enum {
    AICORE_LOMA_MODEL_ROLE_DETECTOR = 0,
    AICORE_LOMA_MODEL_ROLE_DESCRIPTOR = 1,
    AICORE_LOMA_MODEL_ROLE_MATCHER = 2,
} aicore_loma_model_role;

/* A model variant identifies a compatible graph family within a role. The
 * default triplet remains DaD + DeDoDe-G + LoMa-B. */
typedef enum {
    AICORE_LOMA_MODEL_VARIANT_DAD = 0,
    AICORE_LOMA_MODEL_VARIANT_DEDODE_B = 1,
    AICORE_LOMA_MODEL_VARIANT_DEDODE_G = 2,
    AICORE_LOMA_MODEL_VARIANT_MATCHER_B = 3,
    AICORE_LOMA_MODEL_VARIANT_MATCHER_B128 = 4,
    AICORE_LOMA_MODEL_VARIANT_MATCHER_R = 5,
    AICORE_LOMA_MODEL_VARIANT_MATCHER_L = 6,
    AICORE_LOMA_MODEL_VARIANT_MATCHER_G = 7,
} aicore_loma_model_variant;

/* The default triplet exactly matches COLMAP: DaD detector, DeDoDe-G
 * descriptor, and LoMa-B matcher. */
typedef struct {
    const char* filename;
    const char* download_url;
    const char* display_name;
    aicore_loma_model_role role;
    aicore_loma_model_variant variant;
} aicore_loma_model_entry;

AICORE_CAPI int aicore_loma_abi_version(void);

AICORE_CAPI aicore_loma_detector_options* aicore_loma_detector_options_new(
        void);
AICORE_CAPI void aicore_loma_detector_options_free(
        aicore_loma_detector_options* options);
AICORE_CAPI void aicore_loma_detector_options_set_device(
        aicore_loma_detector_options* options, const char* device);
AICORE_CAPI void aicore_loma_detector_options_set_threads(
        aicore_loma_detector_options* options, int32_t threads);
/* DaD selects this many points from its dense score map. COLMAP recommends
 * 2048 or 4096, but the exported graph accepts any positive count. */
AICORE_CAPI void aicore_loma_detector_options_set_max_keypoints(
        aicore_loma_detector_options* options, int32_t max_keypoints);
AICORE_CAPI aicore_loma_detector_ctx* aicore_loma_detector_load(
        const char* gguf_path, const aicore_loma_detector_options* options);
AICORE_CAPI void aicore_loma_detector_free(aicore_loma_detector_ctx* ctx);
AICORE_CAPI int aicore_loma_detector_is_ready(
        const aicore_loma_detector_ctx* ctx);
AICORE_CAPI const char* aicore_loma_detector_last_error(
        const aicore_loma_detector_ctx* ctx);
AICORE_CAPI int aicore_loma_detector_run(
        aicore_loma_detector_ctx* ctx,
        const aicore_loma_rgb_image* image,
        aicore_loma_detected_features* out_features);
AICORE_CAPI void aicore_loma_detected_features_free(
        aicore_loma_detected_features* features);

AICORE_CAPI aicore_loma_descriptor_options* aicore_loma_descriptor_options_new(
        void);
AICORE_CAPI void aicore_loma_descriptor_options_free(
        aicore_loma_descriptor_options* options);
AICORE_CAPI void aicore_loma_descriptor_options_set_device(
        aicore_loma_descriptor_options* options, const char* device);
AICORE_CAPI void aicore_loma_descriptor_options_set_threads(
        aicore_loma_descriptor_options* options, int32_t threads);
AICORE_CAPI aicore_loma_descriptor_ctx* aicore_loma_descriptor_load(
        const char* gguf_path, const aicore_loma_descriptor_options* options);
AICORE_CAPI void aicore_loma_descriptor_free(aicore_loma_descriptor_ctx* ctx);
AICORE_CAPI int aicore_loma_descriptor_is_ready(
        const aicore_loma_descriptor_ctx* ctx);
AICORE_CAPI const char* aicore_loma_descriptor_last_error(
        const aicore_loma_descriptor_ctx* ctx);
/* image is the descriptor's already-resized RGB input. keypoint_image_width
 * and keypoint_image_height describe the detector/source coordinate frame. */
AICORE_CAPI int aicore_loma_descriptor_run(
        aicore_loma_descriptor_ctx* ctx,
        const aicore_loma_rgb_image* image,
        const aicore_loma_keypoint* keypoints,
        int32_t count,
        int32_t keypoint_image_width,
        int32_t keypoint_image_height,
        aicore_loma_described_features* out_features);
AICORE_CAPI void aicore_loma_described_features_free(
        aicore_loma_described_features* features);

AICORE_CAPI aicore_loma_matcher_options* aicore_loma_matcher_options_new(void);
AICORE_CAPI void aicore_loma_matcher_options_free(
        aicore_loma_matcher_options* options);
AICORE_CAPI void aicore_loma_matcher_options_set_device(
        aicore_loma_matcher_options* options, const char* device);
AICORE_CAPI void aicore_loma_matcher_options_set_threads(
        aicore_loma_matcher_options* options, int32_t threads);
/* COLMAP default is 0.1. Valid range is [0, 1]. */
AICORE_CAPI void aicore_loma_matcher_options_set_min_score(
        aicore_loma_matcher_options* options, double min_score);

AICORE_CAPI aicore_loma_matcher_ctx* aicore_loma_matcher_load(
        const char* gguf_path, const aicore_loma_matcher_options* options);
AICORE_CAPI void aicore_loma_matcher_free(aicore_loma_matcher_ctx* ctx);
AICORE_CAPI int aicore_loma_matcher_is_ready(
        const aicore_loma_matcher_ctx* ctx);
AICORE_CAPI const char* aicore_loma_matcher_last_error(
        const aicore_loma_matcher_ctx* ctx);

/* Pixel coordinates are normalized exactly as COLMAP LoMa: x/y map to [-1,1]
 * independently by image width/height before entering the matcher graph. */
AICORE_CAPI int aicore_loma_matcher_run(aicore_loma_matcher_ctx* ctx,
                                        const aicore_loma_features* image0,
                                        const aicore_loma_features* image1,
                                        aicore_loma_match** out_matches,
                                        int32_t* out_count);
AICORE_CAPI void aicore_loma_free_matches(aicore_loma_match* matches);
AICORE_CAPI int aicore_loma_matcher_last_pipeline_timings(
        const aicore_loma_matcher_ctx* ctx, aicore_pipeline_timings* out);
AICORE_CAPI int aicore_loma_warmup_backend(const char* device);

/* Create a standard LoMa GGUF with F16 or Q8_0 weights. Q8_0 is deliberately
 * limited to linear tensors because DaD/DeDoDe convolution kernels require
 * dense weights. Returns zero only for a non-empty, runnable conversion. */
AICORE_CAPI int aicore_loma_quantize_gguf(const char* input_gguf,
                                          const char* output_gguf,
                                          const char* type);

AICORE_CAPI int aicore_loma_model_count(void);
AICORE_CAPI int aicore_loma_model_default_index(aicore_loma_model_role role);
AICORE_CAPI const aicore_loma_model_entry* aicore_loma_model_at(int index);
AICORE_CAPI const aicore_loma_model_entry* aicore_loma_model_by_role(
        aicore_loma_model_role role);
AICORE_CAPI const aicore_loma_model_entry* aicore_loma_model_by_variant(
        aicore_loma_model_variant variant);
AICORE_CAPI const char* aicore_loma_model_download_base(void);

/* Stable model-cache directory used by the shared asset downloader. The
 * returned string is heap allocated and must be released with free(). */
AICORE_CAPI char* aicore_loma_model_cache_dir(void);

#ifdef __cplusplus
}
#endif

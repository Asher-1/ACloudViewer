// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Face detection / landmark / embedding / anti-spoofing C API.
//
// The ggml engine under core/AICore/src/tasks/facedetect/ reproduces the
// insightface SCRFD detector, ArcFace / SFace embeddings and MiniFASNet
// anti-spoofing heads (https://github.com/deepinsight/insightface) on ggml;
// alignment and cosine-distance semantics follow the insightface conventions.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"
#include "aicore/image_view.h"
#include "aicore/pipeline_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the ABI version of the FaceDetect C API (bump on breaking
 *  ABI changes). */
AICORE_CAPI int aicore_facedetect_abi_version(void);

typedef struct aicore_facedetect_ctx aicore_facedetect_ctx;
typedef struct aicore_facedetect_options aicore_facedetect_options;

typedef struct aicore_facedetect_detection {
    float score;
    float x1;
    float y1;
    float x2;
    float y2;
    float landmarks_xy10[10];
} aicore_facedetect_detection;

typedef struct aicore_facedetect_analysis {
    aicore_facedetect_detection detection;
    int32_t age;
    int32_t gender;    /**< 0 unknown, 1 female, 2 male. */
    float spoof_score; /**< -1 when the model has no anti-spoof result. */
} aicore_facedetect_analysis;

typedef struct aicore_facedetect_landmark_point {
    float x;
    float y;
    float z;
} aicore_facedetect_landmark_point;

typedef struct aicore_facedetect_verify_options {
    float threshold;           /**< <= 0 uses the model default. */
    float min_detection_score; /**< <= 0 uses the model default. */
    int32_t anti_spoof;
} aicore_facedetect_verify_options;

typedef struct aicore_facedetect_verify_result {
    float distance;
    float threshold;
    int32_t verified;
    /** -1 when not requested, unavailable, or skipped after distance rejection;
     * otherwise 0/1. */
    int32_t anti_spoof_passed;
} aicore_facedetect_verify_result;

/** Creates a default options struct (device "auto", threads 0 = backend
 *  default). Release with aicore_facedetect_options_free. */
AICORE_CAPI aicore_facedetect_options* aicore_facedetect_options_new(void);
/** Releases an options struct created by aicore_facedetect_options_new. */
AICORE_CAPI void aicore_facedetect_options_free(
        aicore_facedetect_options* opts);
/** Selects the inference device: NULL or "auto", "cpu", "gpu", "vulkan"
 *  (optionally ":N"), "cuda" (Linux/Windows). */
AICORE_CAPI void aicore_facedetect_options_set_device(
        aicore_facedetect_options* opts, const char* device);
/** CPU thread count; <= 0 picks the backend default. */
AICORE_CAPI void aicore_facedetect_options_set_threads(
        aicore_facedetect_options* opts, int n_threads);

/** Loads a FaceDetect GGUF model (detector, landmark or embedding pack).
 *  Returns NULL on failure (see aicore_facedetect_last_error). opts may be
 *  NULL for defaults. */
AICORE_CAPI aicore_facedetect_ctx* aicore_facedetect_load_opts(
        const char* gguf_path, const aicore_facedetect_options* opts);
/** Releases a context returned by aicore_facedetect_load_opts; safe on
 *  NULL. Also invalidates its graph-cache entries. */
AICORE_CAPI void aicore_facedetect_free(aicore_facedetect_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model. */
AICORE_CAPI int aicore_facedetect_is_ready(const aicore_facedetect_ctx* ctx);
/** Returns the last error message of the context (empty when none). */
AICORE_CAPI const char* aicore_facedetect_last_error(
        const aicore_facedetect_ctx* ctx);

/** Releases any buffer returned by an aicore_facedetect_* function (string
 *  or float array; unified entry point). Safe on NULL. */
AICORE_CAPI void aicore_facedetect_free_buffer(void* p);

/** Load an image through Qt as tightly-packed RGB. Caller frees \p out_rgb
 *  with aicore_facedetect_free_buffer. */
AICORE_CAPI int aicore_facedetect_load_path_rgb(const char* image_path,
                                                uint8_t** out_rgb,
                                                int32_t* out_width,
                                                int32_t* out_height);

/** Detect all faces on a borrowed RGB buffer; JSON: {"faces":[{"score", "box",
 *  "landmarks"}, ...]}. */
AICORE_CAPI char* aicore_facedetect_detect_rgb_json(aicore_facedetect_ctx* ctx,
                                                    const uint8_t* rgb,
                                                    int32_t width,
                                                    int32_t height);

/** Detect faces into a context-owned typed result store. The image view is
 * borrowed for the duration of the call and may have Qt row padding. */
AICORE_CAPI int aicore_facedetect_detect_image(aicore_facedetect_ctx* ctx,
                                               const aicore_image_view* image);
/** Compatibility wrapper for tightly packed RGB input. */
AICORE_CAPI int aicore_facedetect_detect_rgb(aicore_facedetect_ctx* ctx,
                                             const uint8_t* rgb,
                                             int32_t width,
                                             int32_t height);
AICORE_CAPI size_t
aicore_facedetect_detection_count(const aicore_facedetect_ctx* ctx);
AICORE_CAPI int aicore_facedetect_detection_at(
        const aicore_facedetect_ctx* ctx,
        size_t index,
        aicore_facedetect_detection* out);

/** Age/gender JSON for every detected face on a borrowed RGB buffer.
 *  When min_score > 0, faces below that detection score are omitted (same rule
 * as dense_landmarks). Pass 0 to return every face the detector found. */
AICORE_CAPI char* aicore_facedetect_analyze_rgb_json(aicore_facedetect_ctx* ctx,
                                                     const uint8_t* rgb,
                                                     int32_t width,
                                                     int32_t height,
                                                     float min_score);
/** Typed, stride-aware analyze entry point. Results are owned by ctx until
 * the next analyze call or context destruction. */
AICORE_CAPI int aicore_facedetect_analyze_image(aicore_facedetect_ctx* ctx,
                                                const aicore_image_view* image,
                                                float min_score);
AICORE_CAPI size_t
aicore_facedetect_analysis_count(const aicore_facedetect_ctx* ctx);
AICORE_CAPI int aicore_facedetect_analysis_at(const aicore_facedetect_ctx* ctx,
                                              size_t index,
                                              aicore_facedetect_analysis* out);
/** Borrowed embedding for an analyzed face. Empty embeddings report NULL/0. */
AICORE_CAPI const float* aicore_facedetect_analysis_embedding(
        const aicore_facedetect_ctx* ctx, size_t index, int32_t* out_dim);

/** Detect + dense landmarks (106 2D + 68 3D) using detector_ctx + landmark_ctx.
 *  Faces below min_score are skipped. JSON:
 *  {"faces":[{"score","box","landmarks_5", "landmarks_2d":[[x,y],...],
 *   "landmarks_3d":[[x,y,z],...]}, ...]} */
AICORE_CAPI char* aicore_facedetect_dense_landmarks_rgb_json(
        aicore_facedetect_ctx* detector_ctx,
        aicore_facedetect_ctx* landmark_ctx,
        const uint8_t* rgb,
        int32_t width,
        int32_t height,
        float min_score);
/** Typed, stride-aware dense landmark entry point. Results are owned by
 * detector_ctx until its next dense landmark call or destruction. */
AICORE_CAPI int aicore_facedetect_dense_landmarks_image(
        aicore_facedetect_ctx* detector_ctx,
        aicore_facedetect_ctx* landmark_ctx,
        const aicore_image_view* image,
        float min_score);
AICORE_CAPI size_t
aicore_facedetect_dense_face_count(const aicore_facedetect_ctx* detector_ctx);
AICORE_CAPI int aicore_facedetect_dense_detection_at(
        const aicore_facedetect_ctx* detector_ctx,
        size_t face_index,
        aicore_facedetect_detection* out);
AICORE_CAPI size_t
aicore_facedetect_dense_point_count(const aicore_facedetect_ctx* detector_ctx,
                                    size_t face_index,
                                    int three_d);
AICORE_CAPI int aicore_facedetect_dense_point_at(
        const aicore_facedetect_ctx* detector_ctx,
        size_t face_index,
        int three_d,
        size_t point_index,
        aicore_facedetect_landmark_point* out);

/** Primary-face L2-normalized embedding (512-d ArcFace or 128-d SFace).
 *  Faces below min_detection_score are ignored when picking the primary face
 *  (0 = only the pack's built-in detector threshold applies). */
AICORE_CAPI int aicore_facedetect_embed_path(aicore_facedetect_ctx* ctx,
                                             const char* image_path,
                                             float min_detection_score,
                                             float** out_vec,
                                             int* out_dim);
/** Same as aicore_facedetect_embed_path but on a borrowed RGB buffer. */
AICORE_CAPI int aicore_facedetect_embed_rgb(aicore_facedetect_ctx* ctx,
                                            const uint8_t* rgb,
                                            int32_t width,
                                            int32_t height,
                                            float min_detection_score,
                                            float** out_vec,
                                            int* out_dim);

/** L2-normalized embedding from a full RGB frame and SCRFD 5-point landmarks
 *  (10 floats: x0,y0,...,x4,y4 in pixel coordinates). Same alignment path as
 *  group-photo authentication when detection landmarks are available. */
AICORE_CAPI int aicore_facedetect_embed_rgb_landmarks(
        aicore_facedetect_ctx* ctx,
        const uint8_t* rgb,
        int32_t width,
        int32_t height,
        const float* landmarks_xy10,
        float** out_vec,
        int* out_dim);

/** Compute a row-major query_count x gallery_count cosine-distance matrix.
 *  Every input row must already be finite and L2-normalized. This allocation-
 *  free API is intended for cached face registries and frame-level multi-face
 *  assignment. Returns 0 on success, -1 for invalid input. */
AICORE_CAPI int aicore_facedetect_cosine_distance_matrix(const float* queries,
                                                         int query_count,
                                                         const float* gallery,
                                                         int gallery_count,
                                                         int dim,
                                                         float* out_distances);

/** Cosine distance + match between two images (threshold <=0 → pack default).
 */
AICORE_CAPI int aicore_facedetect_verify_paths(aicore_facedetect_ctx* ctx,
                                               const char* a,
                                               const char* b,
                                               float threshold,
                                               int anti_spoof,
                                               float* out_distance,
                                               int* out_verified);
/** Typed in-memory verification with explicit threshold/detection/liveness
 * options. Both views are borrowed only for the duration of the call. */
AICORE_CAPI int aicore_facedetect_verify_images(
        aicore_facedetect_ctx* ctx,
        const aicore_image_view* image_a,
        const aicore_image_view* image_b,
        const aicore_facedetect_verify_options* options,
        aicore_facedetect_verify_result* out_result);
/** Common timing contract for the latest typed FaceDetect operation. FaceDetect
 * currently reports only the observable E2E boundary. */
AICORE_CAPI int aicore_facedetect_last_pipeline_timings(
        const aicore_facedetect_ctx* ctx, aicore_pipeline_timings* out_timings);

/** Returns a JSON summary of the loaded model. Caller frees with
 *  aicore_facedetect_free_buffer. */
AICORE_CAPI char* aicore_facedetect_info_json(aicore_facedetect_ctx* ctx);
/** Warms up the backend for `device`; returns 0 on success. */
AICORE_CAPI int aicore_facedetect_warmup_backend(const char* device);
/** Releases process-wide FaceDetect backend resources (idempotent). */
AICORE_CAPI void aicore_facedetect_shutdown(void);
/** Returns the local model cache directory. Caller frees with
 *  aicore_facedetect_free_buffer. */
AICORE_CAPI char* aicore_facedetect_model_cache_dir(void);

/** Published GGUF catalog (cloudViewer_downloads qFaceDetect release). */
typedef struct aicore_facedetect_model_entry {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
    int detector_capable;
} aicore_facedetect_model_entry;

/** Number of published catalog entries (all model roles). */
AICORE_CAPI int aicore_facedetect_model_count(void);
/** Returns the catalog entry at `index` (NULL when out of range). */
AICORE_CAPI const aicore_facedetect_model_entry* aicore_facedetect_model_at(
        int index);
/** Number of catalog entries usable as detectors. */
AICORE_CAPI int aicore_facedetect_detector_model_count(void);
/** Returns the detector-capable catalog entry at `index` (NULL when out
 *  of range). */
AICORE_CAPI const aicore_facedetect_model_entry*
aicore_facedetect_detector_model_at(int index);
/** Number of catalog entries usable as landmark models. */
AICORE_CAPI int aicore_facedetect_landmark_model_count(void);
/** Returns the landmark-capable catalog entry at `index` (NULL when out
 *  of range). */
AICORE_CAPI const aicore_facedetect_model_entry*
aicore_facedetect_landmark_model_at(int index);
/** Returns the catalog entry whose filename matches (NULL when not
 *  found). */
AICORE_CAPI const aicore_facedetect_model_entry*
aicore_facedetect_model_by_filename(const char* filename);
/** Returns the base URL of the published model release. */
AICORE_CAPI const char* aicore_facedetect_model_download_base(void);

#ifdef __cplusplus
}
#endif

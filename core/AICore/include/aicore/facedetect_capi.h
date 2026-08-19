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

#ifdef __cplusplus
extern "C" {
#endif

AICORE_CAPI int aicore_facedetect_abi_version(void);

typedef struct aicore_facedetect_ctx aicore_facedetect_ctx;
typedef struct aicore_facedetect_options aicore_facedetect_options;

AICORE_CAPI aicore_facedetect_options* aicore_facedetect_options_new(void);
AICORE_CAPI void aicore_facedetect_options_free(
        aicore_facedetect_options* opts);
AICORE_CAPI void aicore_facedetect_options_set_device(
        aicore_facedetect_options* opts, const char* device);
AICORE_CAPI void aicore_facedetect_options_set_threads(
        aicore_facedetect_options* opts, int n_threads);

AICORE_CAPI aicore_facedetect_ctx* aicore_facedetect_load_opts(
        const char* gguf_path, const aicore_facedetect_options* opts);
AICORE_CAPI void aicore_facedetect_free(aicore_facedetect_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model. */
AICORE_CAPI int aicore_facedetect_is_ready(const aicore_facedetect_ctx* ctx);
AICORE_CAPI const char* aicore_facedetect_last_error(
        const aicore_facedetect_ctx* ctx);

AICORE_CAPI void aicore_facedetect_free_string(char* s);
AICORE_CAPI void aicore_facedetect_free_vec(float* v);

/** Load an image through Qt as tightly-packed RGB. Caller frees \p out_rgb
 *  with aicore_facedetect_free_vec (same allocator). */
AICORE_CAPI int aicore_facedetect_load_path_rgb(const char* image_path,
                                                uint8_t** out_rgb,
                                                int32_t* out_width,
                                                int32_t* out_height);

/** Detect all faces; JSON: {"faces":[{"score", "box", "landmarks"}, ...]}. */
AICORE_CAPI char* aicore_facedetect_detect_path_json(aicore_facedetect_ctx* ctx,
                                                     const char* image_path);
AICORE_CAPI char* aicore_facedetect_detect_rgb_json(aicore_facedetect_ctx* ctx,
                                                    const uint8_t* rgb,
                                                    int32_t width,
                                                    int32_t height);

/** Age/gender JSON for every detected face.
 *  When min_score > 0, faces below that detection score are omitted (same rule
 * as dense_landmarks). Pass 0 to return every face the detector found. */
AICORE_CAPI char* aicore_facedetect_analyze_path_json(
        aicore_facedetect_ctx* ctx, const char* image_path, float min_score);
AICORE_CAPI char* aicore_facedetect_analyze_rgb_json(aicore_facedetect_ctx* ctx,
                                                     const uint8_t* rgb,
                                                     int32_t width,
                                                     int32_t height,
                                                     float min_score);

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

/** Primary-face L2-normalized embedding (512-d ArcFace or 128-d SFace).
 *  Faces below min_detection_score are ignored when picking the primary face
 *  (0 = only the pack's built-in detector threshold applies). */
AICORE_CAPI int aicore_facedetect_embed_path(aicore_facedetect_ctx* ctx,
                                             const char* image_path,
                                             float min_detection_score,
                                             float** out_vec,
                                             int* out_dim);
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

AICORE_CAPI char* aicore_facedetect_info_json(aicore_facedetect_ctx* ctx);
AICORE_CAPI int aicore_facedetect_warmup_backend(const char* device);
AICORE_CAPI void aicore_facedetect_shutdown(void);
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

AICORE_CAPI int aicore_facedetect_model_count(void);
AICORE_CAPI const aicore_facedetect_model_entry* aicore_facedetect_model_at(
        int index);
AICORE_CAPI int aicore_facedetect_detector_model_count(void);
AICORE_CAPI const aicore_facedetect_model_entry*
aicore_facedetect_detector_model_at(int index);
AICORE_CAPI int aicore_facedetect_landmark_model_count(void);
AICORE_CAPI const aicore_facedetect_model_entry*
aicore_facedetect_landmark_model_at(int index);
AICORE_CAPI const aicore_facedetect_model_entry*
aicore_facedetect_model_by_filename(const char* filename);
AICORE_CAPI const char* aicore_facedetect_model_download_base(void);

#ifdef __cplusplus
}
#endif

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "aicore/backend_capi.h"
#include "aicore/facedetect_capi.h"
#include "aicore/runtime_capi.h"
#include "common/capi_utils.hpp"
#include "common/ggml_backend_utils.hpp"
#include "common/model_cache.hpp"
#include "tasks/facedetect/backend.hpp"
#include "tasks/facedetect/image_io.hpp"
#include "tasks/facedetect/model.hpp"

namespace {

float* dup_vec(const std::vector<float>& v) {
    if (v.empty()) return nullptr;
    float* buf = static_cast<float*>(std::malloc(v.size() * sizeof(float)));
    if (buf == nullptr) return nullptr;
    std::memcpy(buf, v.data(), v.size() * sizeof(float));
    return buf;
}

std::string detections_to_json(const std::vector<fd::Detection>& dets) {
    std::string out = "{\"faces\":[";
    for (size_t i = 0; i < dets.size(); ++i) {
        const fd::Detection& d = dets[i];
        if (i) out += ',';
        char b[256];
        std::snprintf(b, sizeof(b),
                      "{\"score\":%.4f,\"box\":[%.2f,%.2f,%.2f,%.2f],"
                      "\"landmarks\":[",
                      d.score, d.x1, d.y1, d.x2, d.y2);
        out += b;
        for (int k = 0; k < 5; ++k) {
            if (k) out += ',';
            std::snprintf(b, sizeof(b), "[%.2f,%.2f]", d.landmarks[k][0],
                          d.landmarks[k][1]);
            out += b;
        }
        out += "]}";
    }
    out += "]}";
    return out;
}

std::string dense_landmarks_to_json(
        const std::vector<fd::DenseLandmarkFace>& faces) {
    std::string out = "{\"faces\":[";
    for (size_t i = 0; i < faces.size(); ++i) {
        const fd::DenseLandmarkFace& f = faces[i];
        const fd::Detection& d = f.det;
        if (i) out += ',';
        char b[256];
        std::snprintf(b, sizeof(b),
                      "{\"score\":%.4f,\"box\":[%.2f,%.2f,%.2f,%.2f],"
                      "\"landmarks_5\":[",
                      d.score, d.x1, d.y1, d.x2, d.y2);
        out += b;
        for (int k = 0; k < 5; ++k) {
            if (k) out += ',';
            std::snprintf(b, sizeof(b), "[%.2f,%.2f]", d.landmarks[k][0],
                          d.landmarks[k][1]);
            out += b;
        }
        out += "],\"landmarks_2d\":[";
        for (size_t j = 0; j < f.points_2d.size(); ++j) {
            if (j) out += ',';
            std::snprintf(b, sizeof(b), "[%.2f,%.2f]", f.points_2d[j].x,
                          f.points_2d[j].y);
            out += b;
        }
        out += "],\"landmarks_3d\":[";
        for (size_t j = 0; j < f.points_3d.size(); ++j) {
            if (j) out += ',';
            std::snprintf(b, sizeof(b), "[%.2f,%.2f,%.3f]", f.points_3d[j].x,
                          f.points_3d[j].y, f.points_3d[j].z);
            out += b;
        }
        out += "]}";
    }
    out += "]}";
    return out;
}

std::string faces_to_analyze_json(const std::vector<fd::Face>& faces) {
    std::string out = "{\"faces\":[";
    for (size_t i = 0; i < faces.size(); ++i) {
        const fd::Face& f = faces[i];
        if (i) out += ',';
        char b[256];
        std::snprintf(b, sizeof(b),
                      "{\"score\":%.4f,\"box\":[%.2f,%.2f,%.2f,%.2f],"
                      "\"age\":%d,\"gender\":\"%c\"}",
                      f.det.score, f.det.x1, f.det.y1, f.det.x2, f.det.y2,
                      f.age, f.gender);
        out += b;
    }
    out += "]}";
    return out;
}

void copy_detection(const fd::Detection& d, aicore_facedetect_detection* out) {
    out->score = d.score;
    out->x1 = d.x1;
    out->y1 = d.y1;
    out->x2 = d.x2;
    out->y2 = d.y2;
    for (int i = 0; i < 5; ++i) {
        out->landmarks_xy10[2 * i] = d.landmarks[i][0];
        out->landmarks_xy10[2 * i + 1] = d.landmarks[i][1];
    }
}

void filter_analyze_faces(std::vector<fd::Face>* faces, float min_score) {
    if (faces == nullptr || min_score <= 0.0f) return;
    faces->erase(std::remove_if(faces->begin(), faces->end(),
                                [min_score](const fd::Face& f) {
                                    return f.det.score < min_score;
                                }),
                 faces->end());
}

bool load_rgb_image(const uint8_t* rgb,
                    int32_t width,
                    int32_t height,
                    fd::Image& img,
                    std::string* err) {
    if (rgb == nullptr || width <= 0 || height <= 0) {
        if (err) *err = "invalid rgb buffer";
        return false;
    }
    if (!fd::image_from_rgb(rgb, width, height, img)) {
        if (err) *err = "failed to wrap rgb buffer";
        return false;
    }
    return true;
}

}  // namespace

using aicore::capi::dup_cstr;

struct aicore_facedetect_options {
    std::string device = "auto";
    int32_t threads = 0;
};

struct aicore_facedetect_ctx {
    // Keep the lease before the model so explicit teardown can invalidate its
    // graph entries while the compatible backend is still bound.
    fd::BackendLease backend;
    std::unique_ptr<fd::Model> model;
    std::string model_path;
    std::string device;
    int32_t threads = 0;
    std::string last_error;
    std::vector<fd::Detection> last_detections;
    std::vector<fd::Face> last_analysis;
    std::vector<fd::DenseLandmarkFace> last_dense_faces;
    aicore_pipeline_timings timings{};
    bool has_timings = false;
};

namespace {

using FaceDetectClock = std::chrono::steady_clock;

void record_facedetect_e2e(aicore_facedetect_ctx* ctx,
                           FaceDetectClock::time_point start) {
    ctx->timings = {};
    ctx->timings.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    ctx->timings.valid_fields = AICORE_TIMING_E2E;
    ctx->timings.e2e_ms = std::chrono::duration<double, std::milli>(
                                  FaceDetectClock::now() - start)
                                  .count();
    ctx->has_timings = true;
}

}  // namespace

AICORE_CAPI int aicore_facedetect_abi_version(void) { return 2; }

AICORE_CAPI aicore_facedetect_options* aicore_facedetect_options_new(void) {
    return new (std::nothrow) aicore_facedetect_options();
}

AICORE_CAPI void aicore_facedetect_options_free(
        aicore_facedetect_options* opts) {
    delete opts;
}

AICORE_CAPI void aicore_facedetect_options_set_device(
        aicore_facedetect_options* opts, const char* device) {
    if (opts != nullptr && device != nullptr) {
        opts->device = device;
    }
}

AICORE_CAPI void aicore_facedetect_options_set_threads(
        aicore_facedetect_options* opts, int n_threads) {
    if (opts != nullptr) {
        opts->threads = n_threads;
    }
}

AICORE_CAPI aicore_facedetect_ctx* aicore_facedetect_load_opts(
        const char* gguf_path, const aicore_facedetect_options* opts) {
    if (gguf_path == nullptr) return nullptr;
    auto* ctx = new (std::nothrow) aicore_facedetect_ctx();
    if (ctx == nullptr) return nullptr;

    ctx->model_path = gguf_path;
    ctx->device = opts != nullptr ? opts->device : "auto";
    ctx->threads = opts != nullptr ? opts->threads : 0;

    try {
        ctx->backend = fd::acquire_backend_lease(ctx->device, ctx->threads);
        fd::ScopedBackendBinding bind(ctx->backend);
        ctx->model = fd::Model::load(ctx->model_path);
        if (!ctx->model) {
            ctx->last_error =
                    "failed to load face-detect GGUF: " + ctx->model_path;
        }
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
    }
    return ctx;
}

AICORE_CAPI void aicore_facedetect_free(aicore_facedetect_ctx* ctx) {
    if (ctx == nullptr) return;
    // ModelLoader invalidates graph-cache entries in its destructor. Bind the
    // owning registry entry first so that invalidation targets the right cache.
    {
        fd::ScopedBackendBinding bind(ctx->backend);
        ctx->model.reset();
    }
    delete ctx;
}

AICORE_CAPI int aicore_facedetect_is_ready(const aicore_facedetect_ctx* ctx) {
    return ctx != nullptr && ctx->model != nullptr ? 1 : 0;
}

AICORE_CAPI const char* aicore_facedetect_last_error(
        const aicore_facedetect_ctx* ctx) {
    return ctx != nullptr && !ctx->last_error.empty() ? ctx->last_error.c_str()
                                                      : nullptr;
}

AICORE_CAPI void aicore_facedetect_free_buffer(void* p) { std::free(p); }

AICORE_CAPI int aicore_facedetect_load_path_rgb(const char* image_path,
                                                uint8_t** out_rgb,
                                                int32_t* out_width,
                                                int32_t* out_height) {
    if (image_path == nullptr || out_rgb == nullptr || out_width == nullptr ||
        out_height == nullptr) {
        return -1;
    }
    *out_rgb = nullptr;
    *out_width = 0;
    *out_height = 0;
    fd::Image img;
    if (!fd::load_image_rgb(image_path, img)) {
        return -1;
    }
    const size_t nbytes = static_cast<size_t>(img.width) *
                          static_cast<size_t>(img.height) * 3;
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(nbytes));
    if (buf == nullptr) {
        return -1;
    }
    std::memcpy(buf, img.data(), nbytes);
    *out_rgb = buf;
    *out_width = img.width;
    *out_height = img.height;
    return 0;
}

AICORE_CAPI char* aicore_facedetect_detect_rgb_json(aicore_facedetect_ctx* ctx,
                                                    const uint8_t* rgb,
                                                    int32_t width,
                                                    int32_t height) {
    if (ctx == nullptr || ctx->model == nullptr) return nullptr;
    fd::Image img;
    if (!load_rgb_image(rgb, width, height, img, &ctx->last_error)) {
        return nullptr;
    }
    try {
        fd::ScopedBackendBinding bind(ctx->backend);
        return dup_cstr(detections_to_json(ctx->model->detect(img)));
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return nullptr;
    }
}

AICORE_CAPI int aicore_facedetect_detect_image(aicore_facedetect_ctx* ctx,
                                               const aicore_image_view* image) {
    if (ctx == nullptr || ctx->model == nullptr || image == nullptr) return -1;
    const auto request_start = FaceDetectClock::now();
    fd::Image img;
    if (!fd::image_from_view(*image, img)) {
        ctx->last_error = "invalid image view";
        return -1;
    }
    try {
        fd::ScopedBackendBinding bind(ctx->backend);
        ctx->last_detections = ctx->model->detect(img);
        record_facedetect_e2e(ctx, request_start);
        return 0;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        ctx->last_detections.clear();
        return -1;
    }
}

AICORE_CAPI int aicore_facedetect_detect_rgb(aicore_facedetect_ctx* ctx,
                                             const uint8_t* rgb,
                                             int32_t width,
                                             int32_t height) {
    aicore_image_view view{rgb, width, height,
                           static_cast<size_t>(width > 0 ? width : 0) * 3,
                           AICORE_IMAGE_RGB8};
    return aicore_facedetect_detect_image(ctx, &view);
}

AICORE_CAPI size_t
aicore_facedetect_detection_count(const aicore_facedetect_ctx* ctx) {
    return ctx == nullptr ? 0 : ctx->last_detections.size();
}

AICORE_CAPI int aicore_facedetect_detection_at(
        const aicore_facedetect_ctx* ctx,
        size_t index,
        aicore_facedetect_detection* out) {
    if (ctx == nullptr || out == nullptr ||
        index >= ctx->last_detections.size()) {
        return -1;
    }
    copy_detection(ctx->last_detections[index], out);
    return 0;
}

AICORE_CAPI int aicore_facedetect_analyze_image(aicore_facedetect_ctx* ctx,
                                                const aicore_image_view* image,
                                                float min_score) {
    if (ctx == nullptr || ctx->model == nullptr || image == nullptr) return -1;
    const auto request_start = FaceDetectClock::now();
    fd::Image img;
    if (!fd::image_from_view(*image, img)) {
        ctx->last_error = "invalid image view";
        ctx->last_analysis.clear();
        return -1;
    }
    try {
        fd::ScopedBackendBinding bind(ctx->backend);
        ctx->last_analysis = ctx->model->analyze(img);
        filter_analyze_faces(&ctx->last_analysis, min_score);
        record_facedetect_e2e(ctx, request_start);
        return 0;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        ctx->last_analysis.clear();
        return -1;
    }
}

AICORE_CAPI size_t
aicore_facedetect_analysis_count(const aicore_facedetect_ctx* ctx) {
    return ctx == nullptr ? 0 : ctx->last_analysis.size();
}

AICORE_CAPI int aicore_facedetect_analysis_at(const aicore_facedetect_ctx* ctx,
                                              size_t index,
                                              aicore_facedetect_analysis* out) {
    if (ctx == nullptr || out == nullptr ||
        index >= ctx->last_analysis.size()) {
        return -1;
    }
    const fd::Face& face = ctx->last_analysis[index];
    copy_detection(face.det, &out->detection);
    out->age = face.age;
    out->gender = face.gender == 'F' ? 1 : (face.gender == 'M' ? 2 : 0);
    out->spoof_score = face.spoof_score;
    return 0;
}

AICORE_CAPI const float* aicore_facedetect_analysis_embedding(
        const aicore_facedetect_ctx* ctx, size_t index, int32_t* out_dim) {
    if (out_dim != nullptr) *out_dim = 0;
    if (ctx == nullptr || out_dim == nullptr ||
        index >= ctx->last_analysis.size()) {
        return nullptr;
    }
    const std::vector<float>& embedding = ctx->last_analysis[index].embedding;
    *out_dim = static_cast<int32_t>(embedding.size());
    return embedding.empty() ? nullptr : embedding.data();
}

AICORE_CAPI char* aicore_facedetect_analyze_rgb_json(aicore_facedetect_ctx* ctx,
                                                     const uint8_t* rgb,
                                                     int32_t width,
                                                     int32_t height,
                                                     float min_score) {
    const aicore_image_view view{rgb, width, height,
                                 static_cast<size_t>(width > 0 ? width : 0) * 3,
                                 AICORE_IMAGE_RGB8};
    if (aicore_facedetect_analyze_image(ctx, &view, min_score) != 0) {
        return nullptr;
    }
    return dup_cstr(faces_to_analyze_json(ctx->last_analysis));
}

AICORE_CAPI int aicore_facedetect_dense_landmarks_image(
        aicore_facedetect_ctx* detector_ctx,
        aicore_facedetect_ctx* landmark_ctx,
        const aicore_image_view* image,
        float min_score) {
    if (detector_ctx == nullptr || detector_ctx->model == nullptr ||
        landmark_ctx == nullptr || landmark_ctx->model == nullptr ||
        image == nullptr) {
        if (detector_ctx != nullptr) {
            detector_ctx->last_error = "null detector/landmark ctx or image";
            detector_ctx->last_dense_faces.clear();
        }
        return -1;
    }
    const auto request_start = FaceDetectClock::now();
    fd::Image img;
    if (!fd::image_from_view(*image, img)) {
        detector_ctx->last_error = "invalid image view";
        detector_ctx->last_dense_faces.clear();
        return -1;
    }
    try {
        std::vector<fd::Detection> detections;
        {
            fd::ScopedBackendBinding bind(detector_ctx->backend);
            detections = detector_ctx->model->detect(img);
        }
        if (min_score > 0.0f) {
            detections.erase(
                    std::remove_if(detections.begin(), detections.end(),
                                   [min_score](const fd::Detection& detection) {
                                       return detection.score < min_score;
                                   }),
                    detections.end());
        }
        if (detections.empty()) {
            detector_ctx->last_dense_faces.clear();
            record_facedetect_e2e(detector_ctx, request_start);
            return 0;
        }
        fd::ScopedBackendBinding bind(landmark_ctx->backend);
        detector_ctx->last_dense_faces =
                landmark_ctx->model->dense_landmarks(img, detections);
        record_facedetect_e2e(detector_ctx, request_start);
        return 0;
    } catch (const std::exception& e) {
        detector_ctx->last_error = e.what();
        detector_ctx->last_dense_faces.clear();
        return -1;
    }
}

AICORE_CAPI size_t
aicore_facedetect_dense_face_count(const aicore_facedetect_ctx* detector_ctx) {
    return detector_ctx == nullptr ? 0 : detector_ctx->last_dense_faces.size();
}

AICORE_CAPI int aicore_facedetect_dense_detection_at(
        const aicore_facedetect_ctx* detector_ctx,
        size_t face_index,
        aicore_facedetect_detection* out) {
    if (detector_ctx == nullptr || out == nullptr ||
        face_index >= detector_ctx->last_dense_faces.size()) {
        return -1;
    }
    copy_detection(detector_ctx->last_dense_faces[face_index].det, out);
    return 0;
}

AICORE_CAPI size_t
aicore_facedetect_dense_point_count(const aicore_facedetect_ctx* detector_ctx,
                                    size_t face_index,
                                    int three_d) {
    if (detector_ctx == nullptr ||
        face_index >= detector_ctx->last_dense_faces.size()) {
        return 0;
    }
    const fd::DenseLandmarkFace& face =
            detector_ctx->last_dense_faces[face_index];
    return three_d != 0 ? face.points_3d.size() : face.points_2d.size();
}

AICORE_CAPI int aicore_facedetect_dense_point_at(
        const aicore_facedetect_ctx* detector_ctx,
        size_t face_index,
        int three_d,
        size_t point_index,
        aicore_facedetect_landmark_point* out) {
    if (detector_ctx == nullptr || out == nullptr ||
        face_index >= detector_ctx->last_dense_faces.size()) {
        return -1;
    }
    const fd::DenseLandmarkFace& face =
            detector_ctx->last_dense_faces[face_index];
    const std::vector<fd::LandmarkPoint>& points =
            three_d != 0 ? face.points_3d : face.points_2d;
    if (point_index >= points.size()) return -1;
    out->x = points[point_index].x;
    out->y = points[point_index].y;
    out->z = points[point_index].z;
    return 0;
}

AICORE_CAPI char* aicore_facedetect_dense_landmarks_rgb_json(
        aicore_facedetect_ctx* detector_ctx,
        aicore_facedetect_ctx* landmark_ctx,
        const uint8_t* rgb,
        int32_t width,
        int32_t height,
        float min_score) {
    const aicore_image_view view{rgb, width, height,
                                 static_cast<size_t>(width > 0 ? width : 0) * 3,
                                 AICORE_IMAGE_RGB8};
    if (aicore_facedetect_dense_landmarks_image(detector_ctx, landmark_ctx,
                                                &view, min_score) != 0) {
        return nullptr;
    }
    return dup_cstr(dense_landmarks_to_json(detector_ctx->last_dense_faces));
}

AICORE_CAPI int aicore_facedetect_embed_path(aicore_facedetect_ctx* ctx,
                                             const char* image_path,
                                             float min_detection_score,
                                             float** out_vec,
                                             int* out_dim) {
    // The embed entry points must also report truthful pipeline timings so
    // the validation probe can query them after the run.
    const auto request_start = FaceDetectClock::now();
    if (ctx == nullptr || ctx->model == nullptr || image_path == nullptr ||
        out_vec == nullptr || out_dim == nullptr) {
        return -1;
    }
    *out_vec = nullptr;
    try {
        fd::ScopedBackendBinding bind(ctx->backend);
        fd::Image img;
        if (!fd::load_image_rgb(image_path, img)) {
            ctx->last_error =
                    std::string("failed to load image: ") + image_path;
            return -1;
        }
        const std::vector<float> emb =
                ctx->model->embed(img, min_detection_score);
        float* buf = dup_vec(emb);
        if (buf == nullptr) {
            ctx->last_error = "out of memory";
            return -1;
        }
        *out_vec = buf;
        *out_dim = static_cast<int>(emb.size());
        record_facedetect_e2e(ctx, request_start);
        return 0;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return -1;
    }
}

AICORE_CAPI int aicore_facedetect_embed_rgb(aicore_facedetect_ctx* ctx,
                                            const uint8_t* rgb,
                                            int32_t width,
                                            int32_t height,
                                            float min_detection_score,
                                            float** out_vec,
                                            int* out_dim) {
    const auto request_start = FaceDetectClock::now();
    if (ctx == nullptr || ctx->model == nullptr || rgb == nullptr ||
        width <= 0 || height <= 0 || out_vec == nullptr || out_dim == nullptr) {
        return -1;
    }
    *out_vec = nullptr;
    try {
        fd::ScopedBackendBinding bind(ctx->backend);
        fd::Image img;
        if (!load_rgb_image(rgb, width, height, img, &ctx->last_error)) {
            return -1;
        }
        const std::vector<float> emb =
                ctx->model->embed(img, min_detection_score);
        float* buf = dup_vec(emb);
        if (buf == nullptr) {
            ctx->last_error = "out of memory";
            return -1;
        }
        *out_vec = buf;
        *out_dim = static_cast<int>(emb.size());
        record_facedetect_e2e(ctx, request_start);
        return 0;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return -1;
    }
}

AICORE_CAPI int aicore_facedetect_embed_rgb_landmarks(
        aicore_facedetect_ctx* ctx,
        const uint8_t* rgb,
        int32_t width,
        int32_t height,
        const float* landmarks_xy10,
        float** out_vec,
        int* out_dim) {
    const auto request_start = FaceDetectClock::now();
    if (ctx == nullptr || ctx->model == nullptr || rgb == nullptr ||
        width <= 0 || height <= 0 || landmarks_xy10 == nullptr ||
        out_vec == nullptr || out_dim == nullptr) {
        return -1;
    }
    *out_vec = nullptr;
    try {
        fd::ScopedBackendBinding bind(ctx->backend);
        fd::Image img;
        if (!load_rgb_image(rgb, width, height, img, &ctx->last_error)) {
            return -1;
        }
        fd::Detection det{};
        for (int k = 0; k < 5; ++k) {
            det.landmarks[k][0] = landmarks_xy10[k * 2 + 0];
            det.landmarks[k][1] = landmarks_xy10[k * 2 + 1];
        }
        const std::vector<float> emb = ctx->model->embed(img, det);
        float* buf = dup_vec(emb);
        if (buf == nullptr) {
            ctx->last_error = "out of memory";
            return -1;
        }
        *out_vec = buf;
        *out_dim = static_cast<int>(emb.size());
        record_facedetect_e2e(ctx, request_start);
        return 0;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return -1;
    }
}

AICORE_CAPI int aicore_facedetect_cosine_distance_matrix(const float* queries,
                                                         int query_count,
                                                         const float* gallery,
                                                         int gallery_count,
                                                         int dim,
                                                         float* out_distances) {
    if (queries == nullptr || gallery == nullptr || out_distances == nullptr ||
        query_count <= 0 || gallery_count <= 0 || dim <= 0) {
        return -1;
    }

    const int64_t pair_count =
            static_cast<int64_t>(query_count) * gallery_count;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (pair_count >= 64)
#endif
    for (int64_t pair = 0; pair < pair_count; ++pair) {
        const int q = static_cast<int>(pair / gallery_count);
        const int g = static_cast<int>(pair % gallery_count);
        const float* query = queries + static_cast<size_t>(q) * dim;
        const float* row = gallery + static_cast<size_t>(g) * dim;
        float dot = 0.0f;
#ifdef _OPENMP
#pragma omp simd reduction(+ : dot)
#endif
        for (int d = 0; d < dim; ++d) {
            dot += query[d] * row[d];
        }
        out_distances[pair] = 1.0f - std::clamp(dot, -1.0f, 1.0f);
    }
    return 0;
}

AICORE_CAPI int aicore_facedetect_verify_images(
        aicore_facedetect_ctx* ctx,
        const aicore_image_view* image_a,
        const aicore_image_view* image_b,
        const aicore_facedetect_verify_options* options,
        aicore_facedetect_verify_result* out_result) {
    if (ctx == nullptr || ctx->model == nullptr || image_a == nullptr ||
        image_b == nullptr || out_result == nullptr) {
        return -1;
    }
    const auto request_start = FaceDetectClock::now();
    fd::Image first;
    fd::Image second;
    if (!fd::image_from_view(*image_a, first) ||
        !fd::image_from_view(*image_b, second)) {
        ctx->last_error = "invalid image view";
        return -1;
    }

    const float requested_threshold =
            options != nullptr ? options->threshold : 0.0f;
    const float min_detection_score =
            options != nullptr ? options->min_detection_score : 0.0f;
    const bool request_anti_spoof =
            options != nullptr && options->anti_spoof != 0;
    try {
        fd::ScopedBackendBinding bind(ctx->backend);
        const float threshold = requested_threshold > 0.0f
                                        ? requested_threshold
                                        : ctx->model->config().verify_threshold;
        const std::vector<float> first_embedding =
                ctx->model->embed(first, min_detection_score);
        const std::vector<float> second_embedding =
                ctx->model->embed(second, min_detection_score);
        if (first_embedding.empty() ||
            first_embedding.size() != second_embedding.size()) {
            ctx->last_error = "invalid or incompatible face embeddings";
            return -1;
        }

        double dot = 0.0;
        for (size_t i = 0; i < first_embedding.size(); ++i) {
            dot += static_cast<double>(first_embedding[i]) *
                   second_embedding[i];
        }
        const float distance = static_cast<float>(1.0 - dot);
        int verified = distance <= threshold ? 1 : 0;
        int anti_spoof_passed = -1;

        if (verified != 0 && request_anti_spoof &&
            ctx->model->config().antispoof_present) {
            auto image_is_live = [&](const fd::Image& image) {
                std::vector<fd::Detection> detections =
                        ctx->model->detect(image);
                if (min_detection_score > 0.0f) {
                    detections.erase(
                            std::remove_if(
                                    detections.begin(), detections.end(),
                                    [min_detection_score](
                                            const fd::Detection& detection) {
                                        return detection.score <
                                               min_detection_score;
                                    }),
                            detections.end());
                }
                if (detections.empty()) return false;
                const fd::Detection& primary = *std::max_element(
                        detections.begin(), detections.end(),
                        [](const fd::Detection& lhs, const fd::Detection& rhs) {
                            return (lhs.x2 - lhs.x1) * (lhs.y2 - lhs.y1) <
                                   (rhs.x2 - rhs.x1) * (rhs.y2 - rhs.y1);
                        });
                return ctx->model->is_real(image, primary);
            };
            anti_spoof_passed =
                    image_is_live(first) && image_is_live(second) ? 1 : 0;
            if (anti_spoof_passed == 0) verified = 0;
        }

        out_result->distance = distance;
        out_result->threshold = threshold;
        out_result->verified = verified;
        out_result->anti_spoof_passed = anti_spoof_passed;
        record_facedetect_e2e(ctx, request_start);
        return 0;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return -1;
    }
}

AICORE_CAPI int aicore_facedetect_last_pipeline_timings(
        const aicore_facedetect_ctx* ctx,
        aicore_pipeline_timings* out_timings) {
    if (ctx == nullptr || out_timings == nullptr || !ctx->has_timings) {
        return -1;
    }
    *out_timings = ctx->timings;
    return 0;
}

AICORE_CAPI int aicore_facedetect_verify_paths(aicore_facedetect_ctx* ctx,
                                               const char* a,
                                               const char* b,
                                               float threshold,
                                               int anti_spoof,
                                               float* out_distance,
                                               int* out_verified) {
    if (ctx == nullptr || ctx->model == nullptr || a == nullptr ||
        b == nullptr || out_distance == nullptr || out_verified == nullptr) {
        return -1;
    }
    fd::Image first;
    fd::Image second;
    if (!fd::load_image_rgb(a, first)) {
        ctx->last_error = std::string("failed to load image: ") + a;
        return -1;
    }
    if (!fd::load_image_rgb(b, second)) {
        ctx->last_error = std::string("failed to load image: ") + b;
        return -1;
    }
    const aicore_image_view first_view{first.data(), first.width, first.height,
                                       first.stride(), AICORE_IMAGE_RGB8};
    const aicore_image_view second_view{second.data(), second.width,
                                        second.height, second.stride(),
                                        AICORE_IMAGE_RGB8};
    const aicore_facedetect_verify_options options{threshold, 0.0f, anti_spoof};
    aicore_facedetect_verify_result result{};
    if (aicore_facedetect_verify_images(ctx, &first_view, &second_view,
                                        &options, &result) != 0) {
        return -1;
    }
    *out_distance = result.distance;
    *out_verified = result.verified;
    return 0;
}

AICORE_CAPI char* aicore_facedetect_info_json(aicore_facedetect_ctx* ctx) {
    if (ctx == nullptr || ctx->model == nullptr) {
        return dup_cstr("{\"architecture\":\"facedetect\"}");
    }
    const fd::FaceConfig& c = ctx->model->config();
    const std::string resolved_device = ctx->backend.device_name();
    const std::string json =
            std::string(
                    "{\n  \"architecture\": \"facedetect\",\n  \"pack\": \"") +
            c.arch + "\",\n  \"detector\": \"" + c.detector +
            "\",\n  \"recognizer\": \"" + c.recognizer +
            "\",\n  \"embed_dim\": " + std::to_string(c.embed_dim) +
            ",\n  \"device\": \"" + resolved_device +
            "\",\n  \"device_request\": \"" + ctx->device +
            "\",\n  \"model\": \"" + ctx->model_path + "\"\n}";
    return dup_cstr(json);
}

AICORE_CAPI int aicore_facedetect_warmup_backend(const char* device) {
    // Warmup is intentionally lease-scoped: callers that need persistence
    // retain a model context, while this compatibility API proves availability
    // without changing any active context's selected backend.
    const fd::BackendLease lease =
            fd::acquire_backend_lease(device != nullptr ? device : "auto", 0);
    if (!lease || lease.backend().handle() == nullptr) return -1;
    return 0;
}

AICORE_CAPI void aicore_facedetect_shutdown(void) { aicore_runtime_shutdown(); }

AICORE_CAPI char* aicore_facedetect_model_cache_dir(void) {
    return dup_cstr(aicore::facedetect_model_cache_dir());
}

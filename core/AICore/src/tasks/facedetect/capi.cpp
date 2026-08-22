// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "aicore/backend_capi.h"
#include "aicore/facedetect_capi.h"
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
};

AICORE_CAPI int aicore_facedetect_abi_version(void) { return 1; }

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
    std::memcpy(buf, img.rgb.data(), nbytes);
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

AICORE_CAPI char* aicore_facedetect_analyze_rgb_json(aicore_facedetect_ctx* ctx,
                                                     const uint8_t* rgb,
                                                     int32_t width,
                                                     int32_t height,
                                                     float min_score) {
    if (ctx == nullptr || ctx->model == nullptr) return nullptr;
    fd::Image img;
    if (!load_rgb_image(rgb, width, height, img, &ctx->last_error)) {
        return nullptr;
    }
    try {
        fd::ScopedBackendBinding bind(ctx->backend);
        std::vector<fd::Face> faces = ctx->model->analyze(img);
        filter_analyze_faces(&faces, min_score);
        return dup_cstr(faces_to_analyze_json(faces));
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return nullptr;
    }
}

AICORE_CAPI char* aicore_facedetect_dense_landmarks_rgb_json(
        aicore_facedetect_ctx* detector_ctx,
        aicore_facedetect_ctx* landmark_ctx,
        const uint8_t* rgb,
        int32_t width,
        int32_t height,
        float min_score) {
    if (detector_ctx == nullptr || detector_ctx->model == nullptr ||
        landmark_ctx == nullptr || landmark_ctx->model == nullptr) {
        if (detector_ctx)
            detector_ctx->last_error = "null detector/landmark ctx";
        return nullptr;
    }
    fd::Image img;
    if (!load_rgb_image(rgb, width, height, img, &detector_ctx->last_error)) {
        return nullptr;
    }
    try {
        std::vector<fd::Detection> dets;
        {
            fd::ScopedBackendBinding bind(detector_ctx->backend);
            dets = detector_ctx->model->detect(img);
        }
        if (min_score > 0.0f) {
            dets.erase(std::remove_if(dets.begin(), dets.end(),
                                      [min_score](const fd::Detection& d) {
                                          return d.score < min_score;
                                      }),
                       dets.end());
        }
        if (dets.empty()) {
            return dup_cstr("{\"faces\":[]}");
        }
        std::vector<fd::DenseLandmarkFace> faces;
        {
            fd::ScopedBackendBinding bind(landmark_ctx->backend);
            faces = landmark_ctx->model->dense_landmarks(img, dets);
        }
        return dup_cstr(dense_landmarks_to_json(faces));
    } catch (const std::exception& e) {
        detector_ctx->last_error = e.what();
        return nullptr;
    }
}

AICORE_CAPI int aicore_facedetect_embed_path(aicore_facedetect_ctx* ctx,
                                             const char* image_path,
                                             float min_detection_score,
                                             float** out_vec,
                                             int* out_dim) {
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
    try {
        fd::ScopedBackendBinding bind(ctx->backend);
        const float thr = threshold > 0.0f
                                  ? threshold
                                  : ctx->model->config().verify_threshold;
        fd::Image ia, ib;
        if (!fd::load_image_rgb(a, ia)) {
            ctx->last_error = std::string("failed to load image: ") + a;
            return -1;
        }
        if (!fd::load_image_rgb(b, ib)) {
            ctx->last_error = std::string("failed to load image: ") + b;
            return -1;
        }
        const std::vector<float> ea = ctx->model->embed(ia);
        const std::vector<float> eb = ctx->model->embed(ib);
        double dot = 0.0;
        const size_t n = std::min(ea.size(), eb.size());
        for (size_t i = 0; i < n; ++i)
            dot += static_cast<double>(ea[i]) * eb[i];
        const float dist = static_cast<float>(1.0 - dot);
        *out_distance = dist;
        int verified = dist <= thr ? 1 : 0;
        if (verified && anti_spoof != 0 &&
            ctx->model->config().antispoof_present) {
            auto live = [&](const fd::Image& im) -> bool {
                std::vector<fd::Detection> d = ctx->model->detect(im);
                if (d.empty()) return false;
                const fd::Detection& primary = *std::max_element(
                        d.begin(), d.end(),
                        [](const fd::Detection& x, const fd::Detection& y) {
                            return (x.x2 - x.x1) * (x.y2 - x.y1) <
                                   (y.x2 - y.x1) * (y.y2 - y.y1);
                        });
                return ctx->model->is_real(im, primary);
            };
            if (!live(ia) || !live(ib)) verified = 0;
        }
        *out_verified = verified;
        return 0;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return -1;
    }
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

AICORE_CAPI void aicore_facedetect_shutdown(void) { fd::shutdown_backend(); }

AICORE_CAPI char* aicore_facedetect_model_cache_dir(void) {
    return dup_cstr(aicore::facedetect_model_cache_dir());
}

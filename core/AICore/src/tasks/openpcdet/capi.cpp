// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// C ABI for the OpenPCDet LiDAR 3D detection task. Wraps the ported
// OpenPCDet-GGML engine (openpcdet::model subclasses) with the standard
// AICore task surface: opaque context, explicit options, typed results,
// common timings, last_error, readiness, cache dir and shutdown.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include "aicore/backend_capi.h"
#include "aicore/openpcdet_capi.h"
#include "aicore/runtime_capi.h"
#include "common/capi_utils.hpp"
#include "common/ggml_backend_utils.hpp"
#include "common/model_cache.hpp"
#include "gguf_loader.h"
#include "models/bevfusion.h"
#include "models/caddn.h"
#include "models/parta2.h"
#include "models/pointpillar.h"
#include "models/pointrcnn.h"
#include "models/pvrcnn.h"
#include "models/second.h"
#include "models/transfusion.h"
#include "models/voxelnext.h"
#include "models/voxelrcnn.h"

using aicore::capi::dup_cstr;
using aicore::capi::json_escape;

#define AICORE_OPENPCDET_ABI_VERSION 1

namespace {

std::unique_ptr<openpcdet::model> create_engine_model(const std::string& path,
                                                      std::string& error) {
    openpcdet::model_file probe;
    if (!probe.open(path, /*with_data=*/false)) {
        error = probe.error;
        return nullptr;
    }
    switch (probe.hp.arch) {
        case openpcdet::arch_kind::pointpillar:
            return std::make_unique<openpcdet::pointpillar>();
        case openpcdet::arch_kind::second:
        case openpcdet::arch_kind::voxel_detector:
            return std::make_unique<openpcdet::second>();
        case openpcdet::arch_kind::voxelnext:
            return std::make_unique<openpcdet::voxelnext>();
        case openpcdet::arch_kind::pointrcnn:
            return std::make_unique<openpcdet::pointrcnn>();
        case openpcdet::arch_kind::pvrcnn:
            return std::make_unique<openpcdet::pvrcnn>();
        case openpcdet::arch_kind::voxelrcnn:
            return std::make_unique<openpcdet::voxelrcnn>();
        case openpcdet::arch_kind::transfusion:
            return std::make_unique<openpcdet::transfusion>();
        case openpcdet::arch_kind::parta2:
            return std::make_unique<openpcdet::parta2>();
        case openpcdet::arch_kind::caddn:
            return std::make_unique<openpcdet::caddn>();
        case openpcdet::arch_kind::bevfusion:
            return std::make_unique<openpcdet::bevfusion>();
        default:
            error = "unsupported GGUF architecture: the published "
                    "OpenPCDet-GGUF "
                    "release inventory is enabled; focal_former3d/fastpillars "
                    "have no published assets";
            return nullptr;
    }
}

const char* arch_string(const openpcdet::hparams& hp) {
    switch (hp.arch) {
        case openpcdet::arch_kind::pointpillar:
            return "openpcdet.pointpillar";
        case openpcdet::arch_kind::second:
            return "openpcdet.second";
        case openpcdet::arch_kind::voxel_detector:
            return "openpcdet.voxel";
        case openpcdet::arch_kind::voxelnext:
            return "openpcdet.voxelnext";
        case openpcdet::arch_kind::pointrcnn:
            return "openpcdet.pointrcnn";
        case openpcdet::arch_kind::pvrcnn:
            return "openpcdet.pvrcnn";
        case openpcdet::arch_kind::voxelrcnn:
            return "openpcdet.voxelrcnn";
        case openpcdet::arch_kind::transfusion:
            return "openpcdet.transfusion";
        case openpcdet::arch_kind::parta2:
            return "openpcdet.parta2";
        case openpcdet::arch_kind::caddn:
            return "openpcdet.caddn";
        case openpcdet::arch_kind::bevfusion:
            return "openpcdet.bevfusion";
        default:
            return "openpcdet.unknown";
    }
}

}  // namespace

namespace {

// Adapter: turn a tap-dump directory into a tap_sink writing <name>.f32 plus
// <name>.meta.json, matching the upstream OpenPCDet-GGML compare_taps.py
// format. Diagnostics only; empty dir disables dumping.
struct dir_tap_sink {
    std::string dir;
    explicit dir_tap_sink(std::string d) : dir(std::move(d)) {}
    void operator()(const std::string& name,
                    const float* data,
                    int64_t n_rows,
                    int64_t n_cols) const {
        if (dir.empty()) return;
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) return;
        std::string fname = name;
        for (char& c : fname) {
            if (c == '/' || c == '.') c = '_';
        }
        FILE* f = std::fopen((dir + "/" + fname + ".f32").c_str(), "wb");
        if (f == nullptr) return;
        std::fwrite(data, sizeof(float), static_cast<size_t>(n_rows * n_cols),
                    f);
        std::fclose(f);
        FILE* m = std::fopen((dir + "/" + fname + ".meta.json").c_str(), "w");
        if (m != nullptr) {
            std::fprintf(m,
                         "{\"name\":\"%s\",\"n_rows\":%lld,\"n_cols\":%lld}\n",
                         name.c_str(), static_cast<long long>(n_rows),
                         static_cast<long long>(n_cols));
            std::fclose(m);
        }
    }
};

}  // namespace

struct aicore_openpcdet_options {
    aicore::capi::CommonOptions common;
    std::string dump_taps_dir;  // empty => tap dumping disabled
    bool score_thresh_set = false;
    float score_thresh = 0.0f;
    bool nms_thresh_set = false;
    float nms_thresh = 0.0f;
    bool max_boxes_set = false;
    int max_boxes = 0;
};

struct aicore_openpcdet_ctx {
    std::unique_ptr<openpcdet::model> engine;
    std::string model_path;
    std::string device;  // backend-resolved device name
    int32_t threads = 0;
    std::string last_error;

    // Runtime threshold overrides captured from the options (NaN/0 = keep
    // GGUF values; applied to the engine hparams right after load).
    float score_thresh = 0.0f;
    float nms_thresh = 0.0f;
    int max_boxes = 0;

    // Optional tap-dump directory (diagnostics; empty = disabled).
    std::string dump_taps_dir;

    // Borrowed detection store from the most recent detect call.
    std::vector<aicore_openpcdet_detection> detections;
    aicore_pipeline_timings timings{};
};

// ---------------------------------------------------------------------------
// ABI / options
// ---------------------------------------------------------------------------

AICORE_CAPI int aicore_openpcdet_abi_version(void) {
    return AICORE_OPENPCDET_ABI_VERSION;
}

AICORE_CAPI aicore_openpcdet_options* aicore_openpcdet_options_new(void) {
    return new (std::nothrow) aicore_openpcdet_options{};
}

AICORE_CAPI void aicore_openpcdet_options_free(aicore_openpcdet_options* opts) {
    delete opts;
}

AICORE_CAPI void aicore_openpcdet_options_set_device(
        aicore_openpcdet_options* opts, const char* device) {
    if (opts != nullptr) aicore::capi::set_device(opts->common, device);
}

AICORE_CAPI void aicore_openpcdet_options_set_threads(
        aicore_openpcdet_options* opts, int n_threads) {
    if (opts != nullptr) aicore::capi::set_threads(opts->common, n_threads);
}

AICORE_CAPI void aicore_openpcdet_options_set_score_thresh(
        aicore_openpcdet_options* opts, float t) {
    if (opts == nullptr || std::isnan(t)) return;
    opts->score_thresh = t;
    opts->score_thresh_set = true;
}

AICORE_CAPI void aicore_openpcdet_options_set_nms_thresh(
        aicore_openpcdet_options* opts, float t) {
    if (opts == nullptr || std::isnan(t)) return;
    opts->nms_thresh = t;
    opts->nms_thresh_set = true;
}

AICORE_CAPI void aicore_openpcdet_options_set_max_boxes(
        aicore_openpcdet_options* opts, int n) {
    if (opts == nullptr || n <= 0) return;
    opts->max_boxes = n;
    opts->max_boxes_set = true;
}

AICORE_CAPI void aicore_openpcdet_options_set_dump_taps_dir(
        aicore_openpcdet_options* opts, const char* dir) {
    if (opts == nullptr) return;
    opts->dump_taps_dir = dir != nullptr ? dir : "";
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

AICORE_CAPI aicore_openpcdet_ctx* aicore_openpcdet_load_opts(
        const char* gguf_path, const aicore_openpcdet_options* opts) {
    if (gguf_path == nullptr || gguf_path[0] == '\0') return nullptr;
    auto* ctx = new (std::nothrow) aicore_openpcdet_ctx{};
    if (ctx == nullptr) return nullptr;
    ctx->model_path = gguf_path;
    ctx->threads = opts ? opts->common.threads : 0;
    ctx->score_thresh =
            opts && opts->score_thresh_set ? opts->score_thresh : 0.0f;
    ctx->nms_thresh = opts && opts->nms_thresh_set ? opts->nms_thresh : 0.0f;
    ctx->max_boxes = opts && opts->max_boxes_set ? opts->max_boxes : 0;
    ctx->dump_taps_dir = opts ? opts->dump_taps_dir : std::string();
    try {
        std::string factory_error;
        ctx->engine = create_engine_model(ctx->model_path, factory_error);
        if (!ctx->engine) {
            ctx->last_error = factory_error;
            return ctx;
        }
        const std::string device_req = opts ? opts->common.device : "auto";
        if (!ctx->engine->load(ctx->model_path, device_req, ctx->threads)) {
            ctx->last_error = ctx->engine->error;
            return ctx;
        }
        ctx->device = ctx->engine->be.device;
        // Apply runtime threshold overrides after the GGUF hparams exist.
        if (ctx->score_thresh > 0.0f) {
            ctx->engine->file.hp.score_thresh = ctx->score_thresh;
        }
        if (ctx->nms_thresh > 0.0f) {
            ctx->engine->file.hp.nms_thresh = ctx->nms_thresh;
        }
        if (ctx->max_boxes > 0) {
            ctx->engine->file.hp.max_boxes = ctx->max_boxes;
        }
    } catch (const std::exception& e) {
        ctx->last_error = std::string("openpcdet load failed: ") + e.what();
    } catch (...) {
        ctx->last_error = "openpcdet load failed: unknown exception";
    }
    return ctx;
}

AICORE_CAPI void aicore_openpcdet_free(aicore_openpcdet_ctx* ctx) {
    delete ctx;
}

AICORE_CAPI int aicore_openpcdet_is_ready(const aicore_openpcdet_ctx* ctx) {
    return ctx && ctx->engine && ctx->engine->be.be &&
                           ctx->engine->error.empty()
                   ? 1
                   : 0;
}

AICORE_CAPI const char* aicore_openpcdet_last_error(
        const aicore_openpcdet_ctx* ctx) {
    return ctx ? ctx->last_error.c_str() : "";
}

AICORE_CAPI void aicore_openpcdet_free_buffer(void* p) { free(p); }

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

AICORE_CAPI int aicore_openpcdet_geometry_of(const aicore_openpcdet_ctx* ctx,
                                             aicore_openpcdet_geometry* out) {
    if (ctx == nullptr || out == nullptr || !ctx->engine) return -1;
    const openpcdet::hparams& hp = ctx->engine->file.hp;
    if (hp.point_cloud_range.size() < 6 || hp.voxel_size.size() < 3 ||
        hp.grid_size.size() < 3) {
        return -1;
    }
    *out = {};
    out->abi_version = AICORE_OPENPCDET_ABI_VERSION;
    out->num_point_features = hp.num_point_features;
    out->grid_size_x = hp.grid_size[0];
    out->grid_size_y = hp.grid_size[1];
    out->grid_size_z = hp.grid_size[2];
    out->max_points_per_voxel = hp.max_points_per_voxel;
    out->max_voxels = hp.max_voxels;
    out->num_class = hp.num_class;
    out->voxel_size_x = hp.voxel_size[0];
    out->voxel_size_y = hp.voxel_size[1];
    out->voxel_size_z = hp.voxel_size[2];
    for (int i = 0; i < 6; ++i)
        out->point_cloud_range[i] = hp.point_cloud_range[i];
    std::snprintf(out->arch, sizeof(out->arch), "%s", arch_string(hp));
    return 0;
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

AICORE_CAPI int aicore_openpcdet_detect_points(aicore_openpcdet_ctx* ctx,
                                               const float* points,
                                               int32_t n_points,
                                               int32_t point_dim) {
    if (ctx == nullptr) return -1;
    ctx->last_error.clear();
    if (!ctx->engine) {
        ctx->last_error = "no loaded model";
        return -1;
    }
    if (n_points < 0 || (n_points > 0 && points == nullptr)) {
        ctx->last_error = "invalid point buffer";
        return -1;
    }
    const auto start = aicore::capi::PipelineClock::now();
    try {
        std::vector<openpcdet::detection> dets;
        openpcdet::tap_sink sink;
        if (!ctx->dump_taps_dir.empty()) {
            sink = dir_tap_sink(ctx->dump_taps_dir);
        }
        if (!ctx->engine->forward(points, n_points, point_dim, dets, sink)) {
            ctx->last_error = ctx->engine->error;
            return -1;
        }
        ctx->detections.clear();
        ctx->detections.reserve(dets.size());
        for (const openpcdet::detection& d : dets) {
            ctx->detections.push_back({d.x, d.y, d.z, d.dx, d.dy, d.dz,
                                       d.heading, d.score, d.label});
        }
    } catch (const std::exception& e) {
        ctx->last_error =
                std::string("openpcdet inference failed: ") + e.what();
        return -1;
    } catch (...) {
        ctx->last_error = "openpcdet inference failed: unknown exception";
        return -1;
    }
    // Stage truthfulness: the ported forward is monolithic (host voxelization,
    // VFE, graph compute and decode share one call), so the inference stage
    // and the E2E envelope carry the same measured duration; preprocess and
    // postprocess stay zero with their bits clear.
    ctx->timings = {};
    ctx->timings.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    ctx->timings.valid_fields = AICORE_TIMING_INFERENCE | AICORE_TIMING_E2E;
    ctx->timings.inference_ms =
            std::chrono::duration<double, std::milli>(
                    aicore::capi::PipelineClock::now() - start)
                    .count();
    ctx->timings.e2e_ms = ctx->timings.inference_ms;
    return static_cast<int>(ctx->detections.size());
}

AICORE_CAPI int aicore_openpcdet_detect_scene(
        aicore_openpcdet_ctx* ctx, const aicore_openpcdet_scene* scene) {
    if (ctx == nullptr) return -1;
    ctx->last_error.clear();
    if (!ctx->engine) {
        ctx->last_error = "no loaded model";
        return -1;
    }
    if (scene == nullptr ||
        scene->abi_version != AICORE_OPENPCDET_SCENE_ABI_VERSION) {
        ctx->last_error = "invalid or unsupported scene struct";
        return -1;
    }
    if (scene->n_points < 0 ||
        (scene->n_points > 0 && scene->points == nullptr) ||
        scene->num_cameras <= 0 || scene->image_height <= 0 ||
        scene->image_width <= 0 || scene->image_channels <= 0 ||
        scene->images == nullptr || scene->lidar_to_image == nullptr ||
        scene->camera_to_lidar == nullptr ||
        scene->camera_intrinsics == nullptr ||
        scene->image_augmentation == nullptr ||
        scene->lidar_augmentation == nullptr) {
        ctx->last_error = "invalid scene buffers";
        return -1;
    }
    openpcdet::scene_input input;
    input.points = scene->points;
    input.n_points = scene->n_points;
    input.point_dim = scene->point_dim;
    input.images = scene->images;
    input.num_cameras = scene->num_cameras;
    input.image_height = scene->image_height;
    input.image_width = scene->image_width;
    input.image_channels = scene->image_channels;
    input.lidar_to_image = scene->lidar_to_image;
    input.camera_to_lidar = scene->camera_to_lidar;
    input.camera_intrinsics = scene->camera_intrinsics;
    input.image_augmentation = scene->image_augmentation;
    input.lidar_augmentation = scene->lidar_augmentation;

    const auto start = aicore::capi::PipelineClock::now();
    try {
        std::vector<openpcdet::detection> dets;
        openpcdet::tap_sink sink;
        if (!ctx->dump_taps_dir.empty()) {
            sink = dir_tap_sink(ctx->dump_taps_dir);
        }
        if (!ctx->engine->forward_scene(input, dets, sink)) {
            ctx->last_error = ctx->engine->error;
            return -1;
        }
        ctx->detections.clear();
        ctx->detections.reserve(dets.size());
        for (const openpcdet::detection& d : dets) {
            ctx->detections.push_back({d.x, d.y, d.z, d.dx, d.dy, d.dz,
                                       d.heading, d.score, d.label});
        }
    } catch (const std::exception& e) {
        ctx->last_error =
                std::string("openpcdet scene inference failed: ") + e.what();
        return -1;
    } catch (...) {
        ctx->last_error = "openpcdet scene inference failed: unknown exception";
        return -1;
    }
    ctx->timings = {};
    ctx->timings.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    ctx->timings.valid_fields = AICORE_TIMING_INFERENCE | AICORE_TIMING_E2E;
    ctx->timings.inference_ms =
            std::chrono::duration<double, std::milli>(
                    aicore::capi::PipelineClock::now() - start)
                    .count();
    ctx->timings.e2e_ms = ctx->timings.inference_ms;
    return static_cast<int>(ctx->detections.size());
}

namespace {

// Upstream scene-directory loader (scripts/dump_bevfusion_scene.py payload):
// images.bin (NHWC f32), four [num_cameras*16] matrices, lidar_aug.bin [16]
// and a tiny scene.json with the image grid dimensions.
struct SceneFileBuffer {
    std::vector<float> data;
};

bool read_scene_bin(const std::string& path, std::vector<float>& out) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return false;
    const std::streampos size = stream.tellg();
    stream.seekg(0);
    if (size % static_cast<std::streampos>(sizeof(float)) != 0) return false;
    out.resize(static_cast<size_t>(size / sizeof(float)));
    return static_cast<bool>(
            stream.read(reinterpret_cast<char*>(out.data()), size));
}

}  // namespace

AICORE_CAPI int aicore_openpcdet_load_scene_dir(const char* scene_dir,
                                                aicore_openpcdet_scene* out,
                                                char** error) {
    if (scene_dir == nullptr || scene_dir[0] == '\0' || out == nullptr) {
        return -1;
    }
    *out = {};
    out->abi_version = AICORE_OPENPCDET_SCENE_ABI_VERSION;
    std::string failure;
    auto fail = [&](const char* reason) {
        failure = reason;
        return false;
    };
    const std::string prefix = std::string(scene_dir) + "/";
    // The decoded buffers are caller-visible borrowed pointers owned by one
    // heap block; a registry keyed on the scene struct address lets
    // aicore_openpcdet_free_scene release it.
    auto buffers = new SceneFileBuffer[6];
    auto read_all = [&](const char* name, std::vector<float>& dst) {
        return read_scene_bin(prefix + name, dst) ? true : fail(name);
    };
    if (!read_all("images.bin", buffers[0].data) ||
        !read_all("lidar_to_image.bin", buffers[1].data) ||
        !read_all("camera_to_lidar.bin", buffers[2].data) ||
        !read_all("camera_intrinsics.bin", buffers[3].data) ||
        !read_all("image_aug.bin", buffers[4].data) ||
        !read_all("lidar_aug.bin", buffers[5].data)) {
        delete[] buffers;
        if (error != nullptr) *error = dup_cstr(failure);
        return -1;
    }
    // scene.json: num_cameras / image_height / image_width / image_channels.
    std::ifstream json(prefix + "scene.json");
    if (!json) {
        delete[] buffers;
        if (error != nullptr) *error = dup_cstr("scene.json");
        return -1;
    }
    std::string line;
    while (std::getline(json, line)) {
        auto grab = [&](const char* key, int32_t& value) {
            const std::string needle = std::string("\"") + key + "\"";
            const auto pos = line.find(needle);
            if (pos != std::string::npos) {
                value = std::atoi(line.c_str() + line.find(':', pos) + 1);
            }
        };
        grab("num_cameras", out->num_cameras);
        grab("image_height", out->image_height);
        grab("image_width", out->image_width);
        grab("image_channels", out->image_channels);
    }
    if (out->num_cameras <= 0 || out->image_height <= 0 ||
        out->image_width <= 0 || out->image_channels <= 0) {
        delete[] buffers;
        if (error != nullptr) *error = dup_cstr("scene.json dimensions");
        return -1;
    }
    if (static_cast<int>(buffers[1].data.size()) != out->num_cameras * 16 ||
        static_cast<int>(buffers[3].data.size()) != out->num_cameras * 16 ||
        static_cast<size_t>(buffers[0].data.size()) !=
                static_cast<size_t>(out->num_cameras) * out->image_height *
                        out->image_width * out->image_channels) {
        delete[] buffers;
        if (error != nullptr) *error = dup_cstr("scene sizes inconsistent");
        return -1;
    }
    out->images = buffers[0].data.data();
    out->lidar_to_image = buffers[1].data.data();
    out->camera_to_lidar = buffers[2].data.data();
    out->camera_intrinsics = buffers[3].data.data();
    out->image_augmentation = buffers[4].data.data();
    out->lidar_augmentation = buffers[5].data.data();
    // Keep the storage alive: stash the heap block pointer in reserved and
    // let free_scene recover it. reserved is int32 x2, so store a compressed
    // owner handle instead — a static registry is the portable option.
    static std::mutex scene_registry_mutex;
    static std::map<const aicore_openpcdet_scene*, SceneFileBuffer*>
            scene_registry;
    std::lock_guard<std::mutex> lock(scene_registry_mutex);
    scene_registry[out] = buffers;
    return 0;
}

AICORE_CAPI void aicore_openpcdet_free_scene(aicore_openpcdet_scene* scene) {
    if (scene == nullptr) return;
    static std::mutex scene_registry_mutex;
    static std::map<const aicore_openpcdet_scene*, SceneFileBuffer*>
            scene_registry;
    std::lock_guard<std::mutex> lock(scene_registry_mutex);
    const auto it = scene_registry.find(scene);
    if (it != scene_registry.end()) {
        delete[] it->second;
        scene_registry.erase(it);
    }
    *scene = {};
}

AICORE_CAPI int aicore_openpcdet_detection_count(
        const aicore_openpcdet_ctx* ctx) {
    if (ctx == nullptr) return -1;
    return static_cast<int>(ctx->detections.size());
}

AICORE_CAPI int aicore_openpcdet_detection_at(const aicore_openpcdet_ctx* ctx,
                                              int index,
                                              aicore_openpcdet_detection* out) {
    if (ctx == nullptr || out == nullptr || index < 0 ||
        index >= static_cast<int>(ctx->detections.size())) {
        return -1;
    }
    *out = ctx->detections[static_cast<size_t>(index)];
    return 0;
}

AICORE_CAPI int aicore_openpcdet_last_pipeline_timings(
        const aicore_openpcdet_ctx* ctx, aicore_pipeline_timings* out) {
    if (ctx == nullptr || out == nullptr) return -1;
    *out = ctx->timings;
    return 0;
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

AICORE_CAPI const char* aicore_openpcdet_context_device(
        aicore_openpcdet_ctx* ctx) {
    static const char* kEmpty = "";
    return ctx ? ctx->device.c_str() : kEmpty;
}

AICORE_CAPI int aicore_openpcdet_context_threads(aicore_openpcdet_ctx* ctx) {
    return ctx ? ctx->threads : 0;
}

AICORE_CAPI char* aicore_openpcdet_info_json(aicore_openpcdet_ctx* ctx) {
    if (ctx == nullptr || !ctx->engine) return nullptr;
    const openpcdet::hparams& hp = ctx->engine->file.hp;
    std::ostringstream json;
    json << "{";
    json << "\"abi_version\":" << AICORE_OPENPCDET_ABI_VERSION << ",";
    json << "\"architecture\":\"" << arch_string(hp) << "\",";
    json << "\"num_point_features\":" << hp.num_point_features << ",";
    json << "\"num_class\":" << hp.num_class << ",";
    json << "\"grid_size\":[";
    if (hp.grid_size.size() >= 3) {
        json << hp.grid_size[0] << "," << hp.grid_size[1] << ","
             << hp.grid_size[2];
    }
    json << "],";
    json << "\"score_thresh\":" << hp.score_thresh << ",";
    json << "\"nms_thresh\":" << hp.nms_thresh << ",";
    json << "\"max_boxes\":" << hp.max_boxes << ",";
    json << "\"device\":\"" << json_escape(ctx->device) << "\",";
    json << "\"threads\":" << ctx->threads;
    json << "}";
    return dup_cstr(json.str());
}

AICORE_CAPI int aicore_openpcdet_warmup_backend(const char* device) {
    return aicore_warmup_backend(device != nullptr ? device : "auto");
}

AICORE_CAPI void aicore_openpcdet_shutdown(void) { aicore_runtime_shutdown(); }

AICORE_CAPI char* aicore_openpcdet_model_cache_dir(void) {
    return dup_cstr(aicore::openpcdet_model_cache_dir());
}

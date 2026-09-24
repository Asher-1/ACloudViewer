// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// C ABI implementation for the SAM 3D Objects task. Bridges the borrowed
// aicore_image_view input contract to the native pipeline, owns typed result
// storage, and reports the common pipeline timing contract. Exceptions from
// the C++ session are fenced; they never cross this boundary.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "aicore/runtime_capi.h"
#include "aicore/sam3d_capi.h"

namespace {

constexpr int kSam3dAbiVersion = AICORE_SAM3D_ABI_VERSION;

}  // namespace

#include "asset_io.hpp"
#include "common.hpp"
#include "common/capi_utils.hpp"
#include "common/ggml_backend_registry.hpp"
#include "common/runtime_cleanup.hpp"
#include "model_catalog.hpp"
#include "sam3dggml.h"
#include "scene_assemble.hpp"

// Opaque struct definitions live at global scope so they complete the
// typedefs declared in the public header (an anonymous-namespace definition
// would be a DIFFERENT type and make every use ambiguous).
struct aicore_sam3d_options {
    std::string models_dir;
    std::string moge_gguf;
    std::string device = "auto";
    aicore_sam3d_dtype dtype = AICORE_SAM3D_DTYPE_F16;
    int n_threads = 0;  // 0 -> hardware concurrency
    int seed = 42;
    int ss_steps = 0;    // 0 -> official 25
    int slat_steps = 0;  // 0 -> official 25
    // 0 = F16-KV flash throughput path — the CLI production default since
    // 2026-09-18 (~2.9x SS speed, 27-scene validated). Strict F32 score
    // materialization stays available as the parity diagnostic path.
    int strict_ss_attention = 0;
    int gs_portable_attention = 0;
    int disable_moge_cache = 0;
    uint32_t philox_blocks = 0;
    std::string noise_dir;          // accuracy-diagnostic fixed noise input
    std::string conditions_out;     // caller-owned condition dump/replay dir
    bool scene_attributes = false;  // capture composer interchange attrs
};

struct aicore_sam3d_ctx {
    bool ready = false;
    std::string last_error;
    std::string backend = "none";
    std::string backend_note;
    int caps = AICORE_SAM3D_CAP_GAUSSIAN;
    aicore_pipeline_timings timings{};
    std::unique_ptr<sam3d::ImageTo3DSession> session;
    sam3d::ImageTo3DOptions request;
};

struct aicore_sam3d_result {
    bool has_mesh = false;
    std::vector<float> mesh_vertices;  // xyz per vertex
    std::vector<float> mesh_normals;   // xyz per vertex (optional)
    std::vector<uint32_t> mesh_triangles;
    int64_t gaussian_count = 0;
    int voxel_count = 0;
    // Display-ready splat artifacts (in-memory result path).
    std::vector<float> splat_centers;  // 3 * gaussian_count, world domain
    std::vector<float> splat_rgb;      // 3 * gaussian_count, 0..1
    // Scene-composer interchange attributes (scene_attributes contexts).
    bool has_pose = false;
    std::array<float, 10> pose{};        // wxyz(4) + translation(3) + scale(3)
    std::vector<float> splat_sh0;        // 3 * N, raw f_dc
    std::vector<float> splat_log_scale;  // 3 * N, PLY scale_N
    std::vector<float> splat_opacity_logit;  // N, PLY opacity
    std::vector<float> splat_rot_ply;        // 4 * N, PLY rot (unnormalized)
};

namespace {

const char* dtype_string(aicore_sam3d_dtype dtype) {
    switch (dtype) {
        case AICORE_SAM3D_DTYPE_F16:
            return "f16";
        case AICORE_SAM3D_DTYPE_Q8_0:
            return "q8_0";
        case AICORE_SAM3D_DTYPE_Q4_K:
            return "q4_k";
    }
    return nullptr;
}

bool view_pixels_per_pixel(const aicore_image_view* view, int& channels) {
    switch (view->format) {
        case AICORE_IMAGE_GRAY8:
            channels = 1;
            return true;
        case AICORE_IMAGE_RGB8:
        case AICORE_IMAGE_BGR8:
            channels = 3;
            return true;
        case AICORE_IMAGE_RGBA8:
        case AICORE_IMAGE_BGRA8:
            channels = 4;
            return true;
        default:
            return false;
    }
}

// Converts any supported view to the pipeline's RGBA layout. The mask
// semantics (alpha > 0 keeps the pixel) mirror the official `mask > 0`
// binary-alpha contract: RGB/gray inputs become fully opaque.
bool rgba_from_views(const aicore_image_view* image,
                     const aicore_image_view* mask,
                     sam3d::RgbaImage& out,
                     std::string& error) {
    if (image == nullptr || image->data == nullptr || image->width <= 0 ||
        image->height <= 0) {
        error = "image view must reference decoded pixels";
        return false;
    }
    int image_channels = 0;
    if (!view_pixels_per_pixel(image, image_channels)) {
        error = "unsupported image view format";
        return false;
    }
    const size_t min_stride = static_cast<size_t>(image->width) *
                              static_cast<size_t>(image_channels);
    if (image->row_stride_bytes < min_stride) {
        error = "image view row stride is smaller than width * bytes_per_pixel";
        return false;
    }
    // Overflow-guarded allocation bound for the borrowed view.
    if (static_cast<size_t>(image->height) >
        SIZE_MAX / image->row_stride_bytes) {
        error = "image view dimensions overflow";
        return false;
    }

    out.width = image->width;
    out.height = image->height;
    out.rgba.assign(static_cast<size_t>(image->width) * image->height * 4, 255);
    const uint8_t* rows = image->data;
    for (int y = 0; y < image->height; ++y) {
        const uint8_t* row =
                rows + static_cast<size_t>(y) * image->row_stride_bytes;
        uint8_t* dst =
                out.rgba.data() + static_cast<size_t>(y) * image->width * 4;
        for (int x = 0; x < image->width; ++x) {
            const uint8_t* px = row + static_cast<size_t>(x) * image_channels;
            uint8_t r = 0, g = 0, b = 0, a = 255;
            switch (image->format) {
                case AICORE_IMAGE_GRAY8:
                    r = g = b = px[0];
                    break;
                case AICORE_IMAGE_RGB8:
                    r = px[0];
                    g = px[1];
                    b = px[2];
                    break;
                case AICORE_IMAGE_BGR8:
                    b = px[0];
                    g = px[1];
                    r = px[2];
                    break;
                case AICORE_IMAGE_RGBA8:
                    r = px[0];
                    g = px[1];
                    b = px[2];
                    a = px[3];
                    break;
                case AICORE_IMAGE_BGRA8:
                    b = px[0];
                    g = px[1];
                    r = px[2];
                    a = px[3];
                    break;
                default:
                    break;
            }
            dst[x * 4 + 0] = r;
            dst[x * 4 + 1] = g;
            dst[x * 4 + 2] = b;
            dst[x * 4 + 3] = a;
        }
    }

    if (mask != nullptr && mask->data != nullptr) {
        if (mask->width != image->width || mask->height != image->height) {
            error = "mask view dimensions must match the image view";
            return false;
        }
        int mask_channels = 0;
        if (!view_pixels_per_pixel(mask, mask_channels)) {
            error = "unsupported mask view format";
            return false;
        }
        if (mask->row_stride_bytes <
            static_cast<size_t>(mask->width) *
                    static_cast<size_t>(mask_channels)) {
            error = "mask view row stride is smaller than width * "
                    "bytes_per_pixel";
            return false;
        }
        for (int y = 0; y < mask->height; ++y) {
            const uint8_t* row = mask->data + static_cast<size_t>(y) *
                                                      mask->row_stride_bytes;
            uint8_t* dst =
                    out.rgba.data() + static_cast<size_t>(y) * mask->width * 4;
            for (int x = 0; x < mask->width; ++x) {
                const uint8_t* px =
                        row + static_cast<size_t>(x) * mask_channels;
                // Last decoded channel carries the binary mask (official
                // `mask > 0` semantics): GRAY8 uses the value itself.
                const uint8_t value = mask->format == AICORE_IMAGE_GRAY8
                                              ? px[0]
                                              : px[mask_channels - 1];
                dst[x * 4 + 3] = value == 0 ? 0 : 255;
            }
        }
    }
    return true;
}

void set_timings(aicore_sam3d_ctx* ctx, double e2e_ms) {
    ctx->timings.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    ctx->timings.valid_fields = AICORE_TIMING_E2E;
    ctx->timings.preprocess_ms = 0.0;
    ctx->timings.inference_ms = 0.0;
    ctx->timings.postprocess_ms = 0.0;
    ctx->timings.serialization_ms = 0.0;
    ctx->timings.e2e_ms = e2e_ms;
}

}  // namespace

extern "C" {

int aicore_sam3d_abi_version(void) { return kSam3dAbiVersion; }

aicore_sam3d_options* aicore_sam3d_options_new(void) {
    return new (std::nothrow) aicore_sam3d_options();
}

void aicore_sam3d_options_free(aicore_sam3d_options* options) {
    delete options;
}

void aicore_sam3d_options_set_models_dir(aicore_sam3d_options* options,
                                         const char* models_dir) {
    if (options && models_dir) options->models_dir = models_dir;
}

void aicore_sam3d_options_set_moge_gguf(aicore_sam3d_options* options,
                                        const char* moge_gguf) {
    if (options && moge_gguf) options->moge_gguf = moge_gguf;
}

void aicore_sam3d_options_set_dtype(aicore_sam3d_options* options,
                                    aicore_sam3d_dtype dtype) {
    if (options) options->dtype = dtype;
}

void aicore_sam3d_options_set_device(aicore_sam3d_options* options,
                                     const char* device) {
    if (options && device) options->device = device;
}

void aicore_sam3d_options_set_threads(aicore_sam3d_options* options,
                                      int n_threads) {
    if (options) options->n_threads = n_threads;
}

void aicore_sam3d_options_set_seed(aicore_sam3d_options* options, int seed) {
    if (options) options->seed = seed;
}

void aicore_sam3d_options_set_steps(aicore_sam3d_options* options,
                                    int ss_steps,
                                    int slat_steps) {
    if (options) {
        options->ss_steps = ss_steps;
        options->slat_steps = slat_steps;
    }
}

void aicore_sam3d_options_set_strict_ss_attention(aicore_sam3d_options* options,
                                                  int strict) {
    if (options) options->strict_ss_attention = strict;
}

void aicore_sam3d_options_set_gs_portable_attention(
        aicore_sam3d_options* options, int portable) {
    if (options) options->gs_portable_attention = portable;
}

void aicore_sam3d_options_set_philox_blocks(aicore_sam3d_options* options,
                                            uint32_t blocks) {
    if (options) options->philox_blocks = blocks;
}

void aicore_sam3d_options_set_disable_moge_cache(aicore_sam3d_options* options,
                                                 int disable) {
    if (options) options->disable_moge_cache = disable;
}

void aicore_sam3d_options_set_noise_dir(aicore_sam3d_options* options,
                                        const char* noise_dir) {
    if (options && noise_dir) options->noise_dir = noise_dir;
}

void aicore_sam3d_options_set_conditions_out(aicore_sam3d_options* options,
                                             const char* conditions_out) {
    if (options && conditions_out) options->conditions_out = conditions_out;
}

void aicore_sam3d_options_set_scene_attributes(aicore_sam3d_options* options,
                                               int scene_attributes) {
    if (options) options->scene_attributes = scene_attributes != 0;
}

aicore_sam3d_ctx* aicore_sam3d_load_opts(const aicore_sam3d_options* options,
                                         char* err,
                                         size_t err_size) {
    const auto fail = [&](const std::string& message) -> aicore_sam3d_ctx* {
        if (err && err_size > 0) {
            std::snprintf(err, err_size, "%s", message.c_str());
        }
        return nullptr;
    };
    if (!options) return fail("options must not be NULL");
    if (options->models_dir.empty()) return fail("models_dir is required");
    const char* dtype = dtype_string(options->dtype);
    if (dtype == nullptr) return fail("unsupported dtype enum value");

    auto ctx = std::make_unique<aicore_sam3d_ctx>();
    sam3d::ImageTo3DOptions& request = ctx->request;
    request.models_dir = options->models_dir;
    request.moge_model = options->moge_gguf.empty()
                                 ? options->models_dir + "/moge_vitl-f16.gguf"
                                 : options->moge_gguf;
    request.backend = options->device.empty() ? "auto" : options->device;
    request.dtype = dtype;
    request.n_threads =
            options->n_threads > 0
                    ? options->n_threads
                    : static_cast<int>(std::thread::hardware_concurrency());
    if (request.n_threads <= 0) request.n_threads = 8;
    request.seed = options->seed;
    request.ss_steps = options->ss_steps > 0 ? options->ss_steps : 25;
    request.slat_steps = options->slat_steps > 0 ? options->slat_steps : 25;
    request.strict_ss_attention = options->strict_ss_attention != 0;
    request.gs_portable_attention = options->gs_portable_attention != 0;
    request.philox_blocks = options->philox_blocks;
    request.disable_moge_pointmap_cache = options->disable_moge_cache != 0;
    request.noise_dir = options->noise_dir;
    request.conditions_out = options->conditions_out;
    request.scene_attributes = options->scene_attributes;

    ctx->caps = AICORE_SAM3D_CAP_GAUSSIAN;
    {
        const std::string mesh_model = options->models_dir +
                                       "/slat_decoder_mesh-" +
                                       std::string(dtype) + ".gguf";
        std::error_code ec;
        if (std::filesystem::exists(mesh_model, ec)) {
            ctx->caps |= AICORE_SAM3D_CAP_MESH;
        }
    }

    ctx->session = std::make_unique<sam3d::ImageTo3DSession>();
    std::string error;
    if (!ctx->session->init(request, error)) {
        return fail(error.empty() ? "session init failed" : error);
    }
    ctx->backend = ctx->session->backend_name();
    ctx->ready = true;
    return ctx.release();
}

void aicore_sam3d_free(aicore_sam3d_ctx* ctx) { delete ctx; }

int aicore_sam3d_is_ready(const aicore_sam3d_ctx* ctx) {
    return ctx && ctx->ready && ctx->session ? 1 : 0;
}

const char* aicore_sam3d_last_error(const aicore_sam3d_ctx* ctx) {
    return ctx && !ctx->last_error.empty() ? ctx->last_error.c_str() : NULL;
}

const char* aicore_sam3d_backend(const aicore_sam3d_ctx* ctx) {
    return ctx ? ctx->backend.c_str() : "none";
}

const char* aicore_sam3d_backend_note(const aicore_sam3d_ctx* ctx) {
    return ctx && !ctx->backend_note.empty() ? ctx->backend_note.c_str() : NULL;
}

int aicore_sam3d_caps(const aicore_sam3d_ctx* ctx) {
    return ctx ? ctx->caps : 0;
}

aicore_sam3d_result* aicore_sam3d_generate(aicore_sam3d_ctx* ctx,
                                           const aicore_image_view* image,
                                           const aicore_image_view* mask,
                                           const char* out_ply,
                                           int decode_mesh,
                                           aicore_sam3d_progress_fn progress,
                                           void* user) {
    if (!ctx || !ctx->ready || !ctx->session) {
        if (ctx) ctx->last_error = "context is not ready";
        return nullptr;
    }
    ctx->last_error.clear();
    // out_ply is optional: the in-memory artifact sink below delivers the
    // gaussian splats and the mesh without any file round-trip. A non-empty
    // out_ply keeps the CLI-compatible PLY export (and is still required by
    // the PBR branch inside the session).
    if ((decode_mesh != 0) && !(ctx->caps & AICORE_SAM3D_CAP_MESH)) {
        ctx->last_error =
                "mesh decoding requested but slat_decoder_mesh-<dtype>.gguf is "
                "not present in models_dir";
        return nullptr;
    }

    auto rgba = std::make_shared<sam3d::RgbaImage>();
    std::string error;
    if (!rgba_from_views(image, mask, *rgba, error)) {
        ctx->last_error = error;
        return nullptr;
    }

    sam3d::ImageTo3DOptions request = ctx->request;
    request.image_override = rgba;
    request.mask_path.clear();
    request.image_path.clear();
    request.out_ply = out_ply ? out_ply : "";
    request.decode_mesh = decode_mesh != 0;

    // In-memory artifact sink: no scratch directories, no PLY header re-scan,
    // no SAMT file round-trip for the mesh.
    sam3d::Sam3dArtifacts artifacts;
    request.artifacts = &artifacts;

    if (progress != nullptr) {
        request.progress = [progress, user](int stage, int step, int total) {
            progress(user, stage, step, total);
        };
    } else {
        request.progress = nullptr;
    }

    const auto started = std::chrono::steady_clock::now();
    sam3d::RunResult run;
    {
        // Serialize every graph-consuming call on the process-shared handles.
        aicore::runtime::BackendLeaseLock lease_lock =
                aicore::runtime::lock_backend_leases(ctx->session->leases());
        run = ctx->session->run(request);
    }
    const double e2e_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - started)
                                  .count();
    set_timings(ctx, e2e_ms);

    if (!run.ok) {
        ctx->last_error = run.error.empty()
                                  ? "native image-to-3D pipeline failed"
                                  : run.error;
        return nullptr;
    }
    if (artifacts.gaussian_count <= 0 || artifacts.splat_centers.empty()) {
        ctx->last_error = "the pipeline delivered no gaussian artifacts";
        return nullptr;
    }

    auto result = std::make_unique<aicore_sam3d_result>();
    result->gaussian_count = artifacts.gaussian_count;
    result->splat_centers = std::move(artifacts.splat_centers);
    result->splat_rgb = std::move(artifacts.splat_rgb);
    if (request.scene_attributes) {
        result->has_pose = artifacts.has_pose;
        if (artifacts.has_pose) {
            std::copy(artifacts.pose.rotation_wxyz.begin(),
                      artifacts.pose.rotation_wxyz.end(), result->pose.begin());
            std::copy(artifacts.pose.translation.begin(),
                      artifacts.pose.translation.end(),
                      result->pose.begin() + 4);
            // Official receipt semantics (write_native_pose_json top level /
            // the upstream scene-assemble input): the per-axis native scale
            // is collapsed to its uniform mean, exactly like the official
            // pose_decoder wrapper in inference_utils.py. The composer and
            // every consumer of this receipt apply the uniform value, so a
            // composed scene stays comparable with the upstream
            // scene-assemble flow.
            const float uniform_scale =
                    (artifacts.pose.scale[0] + artifacts.pose.scale[1] +
                     artifacts.pose.scale[2]) /
                    3.0f;
            result->pose[7] = uniform_scale;
            result->pose[8] = uniform_scale;
            result->pose[9] = uniform_scale;
        }
        result->splat_sh0 = std::move(artifacts.splat_sh0);
        result->splat_log_scale = std::move(artifacts.splat_log_scale);
        result->splat_opacity_logit = std::move(artifacts.splat_opacity_logit);
        result->splat_rot_ply = std::move(artifacts.splat_rot_ply);
    }

    if (decode_mesh != 0) {
        if (artifacts.mesh_vertices.empty() ||
            artifacts.mesh_triangles.empty()) {
            ctx->last_error =
                    "mesh decoding was requested but the pipeline delivered no "
                    "mesh artifacts";
            return nullptr;
        }
        result->has_mesh = true;
        result->mesh_vertices = std::move(artifacts.mesh_vertices);
        result->mesh_triangles = std::move(artifacts.mesh_triangles);
    }

    return result.release();
}

int aicore_sam3d_result_has_mesh(const aicore_sam3d_result* result) {
    return result && result->has_mesh ? 1 : 0;
}

int aicore_sam3d_result_mesh_vertex_count(const aicore_sam3d_result* result) {
    return result && result->has_mesh
                   ? static_cast<int>(result->mesh_vertices.size() / 3)
                   : 0;
}

int aicore_sam3d_result_mesh_triangle_count(const aicore_sam3d_result* result) {
    return result && result->has_mesh
                   ? static_cast<int>(result->mesh_triangles.size() / 3)
                   : 0;
}

const float* aicore_sam3d_result_mesh_vertices(
        const aicore_sam3d_result* result) {
    return result && result->has_mesh ? result->mesh_vertices.data() : NULL;
}

const float* aicore_sam3d_result_mesh_normals(
        const aicore_sam3d_result* result) {
    return NULL;  // the FlexiCubes export does not carry geometric normals
}

const uint32_t* aicore_sam3d_result_mesh_triangles(
        const aicore_sam3d_result* result) {
    return result && result->has_mesh ? result->mesh_triangles.data() : NULL;
}

int aicore_sam3d_result_voxel_count(const aicore_sam3d_result* result) {
    return result ? result->voxel_count : 0;
}

int64_t aicore_sam3d_result_gaussian_count(const aicore_sam3d_result* result) {
    return result ? result->gaussian_count : 0;
}

const float* aicore_sam3d_result_splat_centers(
        const aicore_sam3d_result* result) {
    return result && !result->splat_centers.empty()
                   ? result->splat_centers.data()
                   : NULL;
}

const float* aicore_sam3d_result_splat_rgb(const aicore_sam3d_result* result) {
    return result && !result->splat_rgb.empty() ? result->splat_rgb.data()
                                                : NULL;
}

int aicore_sam3d_result_has_pose(const aicore_sam3d_result* result) {
    return result && result->has_pose ? 1 : 0;
}

const float* aicore_sam3d_result_pose(const aicore_sam3d_result* result) {
    return result && result->has_pose ? result->pose.data() : NULL;
}

const float* aicore_sam3d_result_splat_sh0(const aicore_sam3d_result* result) {
    return result && !result->splat_sh0.empty() ? result->splat_sh0.data()
                                                : NULL;
}

const float* aicore_sam3d_result_splat_log_scale(
        const aicore_sam3d_result* result) {
    return result && !result->splat_log_scale.empty()
                   ? result->splat_log_scale.data()
                   : NULL;
}

const float* aicore_sam3d_result_splat_opacity_logit(
        const aicore_sam3d_result* result) {
    return result && !result->splat_opacity_logit.empty()
                   ? result->splat_opacity_logit.data()
                   : NULL;
}

const float* aicore_sam3d_result_splat_rot_ply(
        const aicore_sam3d_result* result) {
    return result && !result->splat_rot_ply.empty()
                   ? result->splat_rot_ply.data()
                   : NULL;
}

void aicore_sam3d_result_free(aicore_sam3d_result* result) { delete result; }

// ---- Multi-object scene assembly -------------------------------------------

struct aicore_sam3d_scene_result {
    sam3d::SceneSplatSet splats;
};

aicore_sam3d_scene_result* aicore_sam3d_scene_assemble(
        const aicore_sam3d_scene_object* objects,
        int object_count,
        int normalize,
        char* err,
        size_t err_size) {
    const auto fail = [&](const std::string& message) {
        if (err && err_size > 0) {
            std::snprintf(err, err_size, "%s", message.c_str());
        }
        return static_cast<aicore_sam3d_scene_result*>(nullptr);
    };
    if (object_count <= 0 || !objects) {
        return fail("scene assembly requires at least one object");
    }
    // Exceptions are fenced here: the composition is pure host math, but the
    // boundary contract never lets one cross.
    std::unique_ptr<aicore_sam3d_scene_result> result;
    try {
        std::vector<sam3d::SceneObjectInput> inputs(object_count);
        for (int index = 0; index < object_count; ++index) {
            const aicore_sam3d_scene_object& src = objects[index];
            if (src.splat_count <= 0) {
                return fail("scene object " + std::to_string(index) +
                            " has an empty splat set");
            }
            inputs[index].splat_count = static_cast<size_t>(src.splat_count);
            inputs[index].centers = src.centers;
            inputs[index].sh0 = src.sh0;
            inputs[index].opacity_logit = src.opacity_logit;
            inputs[index].log_scale = src.log_scale;
            inputs[index].rot_ply = src.rot_ply;
            if (!src.centers || !src.sh0 || !src.opacity_logit ||
                !src.log_scale || !src.rot_ply || !src.pose) {
                return fail("scene object " + std::to_string(index) +
                            " is missing a required array or pose");
            }
        }
        std::vector<sam3d::NativeInstancePose> poses(
                static_cast<size_t>(object_count));
        // decode the packed 10-float receipts into the task pose struct
        for (int index = 0; index < object_count; ++index) {
            sam3d::NativeInstancePose& pose = poses[index];
            const float* packed = objects[index].pose;
            for (int c = 0; c < 4; ++c) pose.rotation_wxyz[c] = packed[c];
            for (int c = 0; c < 3; ++c) {
                pose.translation[c] = packed[4 + c];
                pose.scale[c] = packed[7 + c];
            }
            inputs[index].pose = &pose;
        }
        result = std::make_unique<aicore_sam3d_scene_result>();
        std::string error;
        if (!compose_scene(inputs.data(), static_cast<size_t>(object_count),
                           normalize != 0, result->splats, error) ||
            !result->splats.valid()) {
            return fail(error.empty() ? "scene composition failed" : error);
        }
    } catch (const std::exception& exception) {
        return fail(std::string("scene assembly failed: ") + exception.what());
    } catch (...) {
        return fail("scene assembly failed: unknown exception");
    }
    return result.release();
}

int64_t aicore_sam3d_scene_result_splat_count(
        const aicore_sam3d_scene_result* result) {
    return result ? static_cast<int64_t>(result->splats.size()) : 0;
}

const float* aicore_sam3d_scene_result_positions(
        const aicore_sam3d_scene_result* result) {
    return result && !result->splats.positions.empty()
                   ? result->splats.positions.data()
                   : NULL;
}

const float* aicore_sam3d_scene_result_sh0(
        const aicore_sam3d_scene_result* result) {
    return result && !result->splats.sh0.empty() ? result->splats.sh0.data()
                                                 : NULL;
}

const float* aicore_sam3d_scene_result_opacities(
        const aicore_sam3d_scene_result* result) {
    return result && !result->splats.opacities.empty()
                   ? result->splats.opacities.data()
                   : NULL;
}

const float* aicore_sam3d_scene_result_scales(
        const aicore_sam3d_scene_result* result) {
    return result && !result->splats.scales.empty()
                   ? result->splats.scales.data()
                   : NULL;
}

const float* aicore_sam3d_scene_result_rotations(
        const aicore_sam3d_scene_result* result) {
    return result && !result->splats.rotations.empty()
                   ? result->splats.rotations.data()
                   : NULL;
}

void aicore_sam3d_scene_result_free(aicore_sam3d_scene_result* result) {
    delete result;
}

int aicore_sam3d_last_pipeline_timings(const aicore_sam3d_ctx* ctx,
                                       aicore_pipeline_timings* timings) {
    if (!ctx || !timings) return -1;
    *timings = ctx->timings;
    return 0;
}

void aicore_sam3d_free_buffer(void* buffer) { free(buffer); }

void aicore_sam3d_shutdown(void) { aicore_runtime_shutdown(); }

int aicore_sam3d_model_count(void) { return aicore::sam3d_model_count(); }

const aicore_sam3d_model_entry* aicore_sam3d_model_at(int index) {
    return aicore::sam3d_model_at(index);
}

const aicore_sam3d_model_entry* aicore_sam3d_model_by_filename(
        const char* filename) {
    return aicore::sam3d_model_by_filename(filename);
}

const char* aicore_sam3d_model_download_base(void) {
    return aicore::sam3d_model_download_base();
}

const char* aicore_sam3d_model_cache_dir(void) {
    static const std::string cache_dir = aicore::sam3d_model_cache_dir();
    return cache_dir.c_str();
}

const char* aicore_sam3d_info_json(void) {
    static const std::string info = [] {
        std::string json = "{\"abi_version\":";
        json += std::to_string(kSam3dAbiVersion);
        json += ",\"task\":\"sam3d\"";
        json += ",\"dtypes\":[\"f16\",\"q8_0\",\"q4_k\"]";
        json += ",\"model_count\":";
        json += std::to_string(aicore::sam3d_model_count());
        json += ",\"download_base\":";
        json += "\"";
        json += aicore::sam3d_model_download_base();
        json += "\"}";
        return json;
    }();
    return info.c_str();
}

}  // extern "C"

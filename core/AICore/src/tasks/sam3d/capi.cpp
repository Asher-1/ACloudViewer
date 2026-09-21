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

#include <chrono>
#include <cmath>
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

#if defined(_WIN32)
#include <process.h>
#define SAM3D_GETPID _getpid
#else
#include <unistd.h>
#define SAM3D_GETPID getpid
#endif

#include "asset_io.hpp"
#include "common.hpp"
#include "common/capi_utils.hpp"
#include "common/ggml_backend_registry.hpp"
#include "common/runtime_cleanup.hpp"
#include "model_catalog.hpp"
#include "sam3dggml.h"

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
    std::string noise_dir;       // accuracy-diagnostic fixed noise input
    std::string conditions_out;  // caller-owned condition dump/replay dir
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

// Light PLY header scan for the Gaussian count export artifact.
int64_t gaussian_count_from_ply_header(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return 0;
    char buffer[4096] = {0};
    const size_t n = std::fread(buffer, 1, sizeof(buffer) - 1, f);
    std::fclose(f);
    const char* element = std::strstr(buffer, "element vertex ");
    if (element == nullptr) return 0;
    const long long parsed = std::strtoll(element + 15, nullptr, 10);
    return parsed > 0 ? static_cast<int64_t>(parsed) : 0;
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

std::string mesh_temp_dir() {
    std::error_code ec;
    std::filesystem::path base = std::filesystem::temp_directory_path(ec);
    if (ec) base = std::filesystem::path(".");
    return (base / ("aicore_sam3d_" + std::to_string(SAM3D_GETPID()))).string();
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
    if (!out_ply || *out_ply == '\0') {
        ctx->last_error = "out_ply is required";
        return nullptr;
    }
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
    request.out_ply = out_ply;

    std::string mesh_dir;
    if (decode_mesh != 0) {
        mesh_dir = mesh_temp_dir();
        std::error_code ec;
        std::filesystem::create_directories(mesh_dir, ec);
        if (ec) {
            ctx->last_error = "failed to create the mesh scratch directory";
            return nullptr;
        }
        request.out_mesh_vertices =
                (std::filesystem::path(mesh_dir) / "vertices.samt").string();
        request.out_mesh_faces =
                (std::filesystem::path(mesh_dir) / "faces.samt").string();
    }

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
        if (decode_mesh != 0) {
            std::error_code ec;
            std::filesystem::remove_all(mesh_dir, ec);
        }
        return nullptr;
    }

    auto result = std::make_unique<aicore_sam3d_result>();
    result->gaussian_count = gaussian_count_from_ply_header(out_ply);

    if (decode_mesh != 0) {
        sam3d::RawTensor vertices;
        sam3d::RawTensor faces;
        const bool loaded =
                sam3d::load_raw_tensor(request.out_mesh_vertices, vertices) &&
                sam3d::load_raw_tensor(request.out_mesh_faces, faces) &&
                vertices.ne.size() == 2 && vertices.ne[0] == 3 &&
                vertices.type == GGML_TYPE_F32 && faces.ne.size() == 2 &&
                faces.ne[0] == 3 && faces.type == GGML_TYPE_I32;
        if (!loaded) {
            ctx->last_error = "mesh decoder artifacts are missing or malformed";
        } else {
            const int64_t n_vertices = vertices.ne[1];
            const int64_t n_faces = faces.ne[1];
            const auto* vertex_data =
                    reinterpret_cast<const float*>(vertices.data.data());
            const auto* face_data =
                    reinterpret_cast<const int32_t*>(faces.data.data());
            result->has_mesh = true;
            result->mesh_vertices.assign(vertex_data,
                                         vertex_data + n_vertices * 3);
            result->mesh_triangles.reserve(static_cast<size_t>(n_faces) * 3);
            for (int64_t i = 0; i < n_faces * 3; ++i) {
                result->mesh_triangles.push_back(
                        static_cast<uint32_t>(face_data[i]));
            }
        }
        std::error_code ec;
        std::filesystem::remove_all(mesh_dir, ec);
        if (!ctx->last_error.empty()) return nullptr;
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

void aicore_sam3d_result_free(aicore_sam3d_result* result) { delete result; }

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

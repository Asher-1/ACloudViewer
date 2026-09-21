// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SAM 3D Objects acceptance probe: full native pipeline on the real GGUF
// assets, reporting accuracy/stability/timing metrics for the validation
// runner. Missing assets exit 77 (skip) — the runner downloads them first.
//
// usage: bench_sam3d_backend_acceptance <models_dir> <image> <inference_runs>
//            <backend> [dtype] [steps] [seed]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <QtCore/QString>
#include <QtGui/QImage>

#include "aicore/pipeline_timing.h"
#include "aicore/sam3d_capi.h"

namespace {

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 77;

uint64_t fnv1a(const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

const char* sha12(uint64_t hash, char out[13]) {
    std::snprintf(out, 13, "%012llx", static_cast<unsigned long long>(hash));
    return out;
}

bool models_available(const std::string& dir, aicore_sam3d_dtype dtype) {
    const char* suffix =
            dtype == AICORE_SAM3D_DTYPE_Q8_0
                    ? "q8_0"
                    : (dtype == AICORE_SAM3D_DTYPE_Q4_K ? "q4_k" : "f16");
    static const char* kStages[] = {"ss_generator", "ss_decoder",
                                    "slat_generator", "slat_decoder_gs"};
    for (const char* stage : kStages) {
        std::filesystem::path p = std::filesystem::path(dir) /
                                  (std::string(stage) + "-" + suffix + ".gguf");
        std::error_code ec;
        if (!std::filesystem::exists(p, ec)) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(
                stderr,
                "usage: %s <models_dir> <image> <inference_runs> <backend> "
                "[dtype] [steps] [seed]\n",
                argv[0]);
        return kExitFail;
    }
    const std::string models_dir = argv[1];
    const std::string image_path = argv[2];
    const int inference_runs = std::atoi(argv[3]);
    const std::string backend = argv[4];
    const std::string dtype_name = argc > 5 ? argv[5] : "q4_k";
    const int steps = argc > 6 ? std::atoi(argv[6]) : 25;
    const int seed = argc > 7 ? std::atoi(argv[7]) : 42;

    aicore_sam3d_dtype dtype = AICORE_SAM3D_DTYPE_Q4_K;
    if (dtype_name == "f16")
        dtype = AICORE_SAM3D_DTYPE_F16;
    else if (dtype_name == "q8_0")
        dtype = AICORE_SAM3D_DTYPE_Q8_0;
    else if (dtype_name == "q4_k")
        dtype = AICORE_SAM3D_DTYPE_Q4_K;
    else {
        std::fprintf(stderr, "unsupported dtype %s\n", dtype_name.c_str());
        return kExitFail;
    }

    if (inference_runs < 1 || inference_runs > 8) {
        std::fprintf(stderr, "inference_runs must be in [1, 8]\n");
        return kExitFail;
    }

    if (!models_available(models_dir, dtype)) {
        std::fprintf(stderr, "sam3d models not present under %s\n",
                     models_dir.c_str());
        return kExitSkip;
    }

    const QImage decoded(QString::fromStdString(image_path));
    if (decoded.isNull()) {
        std::fprintf(stderr, "cannot decode image %s\n", image_path.c_str());
        return kExitFail;
    }
    // Non-premultiplied RGBA matches the AICORE_IMAGE_RGBA8 view contract.
    const QImage image = decoded.convertToFormat(QImage::Format_RGBA8888);
    const int width = image.width();
    const int height = image.height();

    aicore_sam3d_options* options = aicore_sam3d_options_new();
    aicore_sam3d_options_set_models_dir(options, models_dir.c_str());
    aicore_sam3d_options_set_dtype(options, dtype);
    aicore_sam3d_options_set_device(options, backend.c_str());
    aicore_sam3d_options_set_threads(options, 8);
    aicore_sam3d_options_set_seed(options, seed);
    aicore_sam3d_options_set_steps(options, steps, steps);
    aicore_sam3d_options_set_disable_moge_cache(options, 1);

    char err[512] = {0};
    aicore_sam3d_ctx* ctx = aicore_sam3d_load_opts(options, err, sizeof(err));
    aicore_sam3d_options_free(options);
    if (ctx == nullptr) {
        std::fprintf(stderr, "load_opts failed: %s\n", err);
        return kExitFail;
    }

    aicore_image_view view{};
    view.data = const_cast<uchar*>(image.constBits());
    view.width = width;
    view.height = height;
    view.row_stride_bytes = static_cast<size_t>(image.bytesPerLine());
    view.format = AICORE_IMAGE_RGBA8;

    const std::string ply_path =
            (std::filesystem::temp_directory_path() / "aicore_sam3d_probe.ply")
                    .string();

    double p50 = 0.0;
    std::vector<double> e2e;
    std::string first_hash;
    int64_t gaussian_count = 0;
    int64_t first_gaussian_count = 0;
    int mesh_vertices = 0;
    int mesh_triangles = 0;
    bool stable = true;
    // The mesh decoder is cross-run nondeterministic by design (CUDA atomics
    // perturb the FlexiCubes vertex extraction by ~0.01%), so the vertex
    // hash is informational only. The deterministic stability gate is the
    // Gaussian count, which is bit-identical across runs and processes.
    bool mesh_hash_stable = true;
    bool finite = true;

    for (int run = 0; run < inference_runs; ++run) {
        aicore_sam3d_result* result = aicore_sam3d_generate(
                ctx, &view, nullptr, ply_path.c_str(), 1, nullptr, nullptr);
        if (result == nullptr) {
            std::fprintf(stderr, "generate failed: %s\n",
                         aicore_sam3d_last_error(ctx)
                                 ? aicore_sam3d_last_error(ctx)
                                 : "?");
            aicore_sam3d_free(ctx);
            return kExitFail;
        }
        const float* verts = aicore_sam3d_result_mesh_vertices(result);
        const int n_vertices = aicore_sam3d_result_mesh_vertex_count(result);
        mesh_vertices = n_vertices;
        mesh_triangles = aicore_sam3d_result_mesh_triangle_count(result);
        gaussian_count = aicore_sam3d_result_gaussian_count(result);
        if (n_vertices <= 0 || mesh_triangles <= 0 || gaussian_count <= 0) {
            std::fprintf(stderr, "degenerate output (v=%d f=%d gs=%lld)\n",
                         n_vertices, mesh_triangles,
                         static_cast<long long>(gaussian_count));
            finite = false;
        }
        for (int i = 0; finite && i < n_vertices * 3; ++i) {
            if (!std::isfinite(verts[i])) {
                std::fprintf(stderr, "non-finite vertex coordinate at %d\n", i);
                finite = false;
                break;
            }
        }
        char hash[13];
        const std::string current =
                sha12(fnv1a(verts, static_cast<size_t>(n_vertices) * 3 *
                                           sizeof(float)),
                      hash);
        if (run == 0) {
            first_hash = current;
            first_gaussian_count = gaussian_count;
        } else {
            if (gaussian_count != first_gaussian_count) stable = false;
            if (current != first_hash) mesh_hash_stable = false;
        }
        aicore_pipeline_timings timings{};
        if (aicore_sam3d_last_pipeline_timings(ctx, &timings) == 0 &&
            (timings.valid_fields & AICORE_TIMING_E2E) != 0) {
            e2e.push_back(timings.e2e_ms);
        }
        aicore_sam3d_result_free(result);
        std::remove(ply_path.c_str());
    }

    if (!e2e.empty()) {
        std::vector<double> sorted = e2e;
        std::sort(sorted.begin(), sorted.end());
        p50 = sorted[sorted.size() / 2];
    }

    const char* resolved_backend = aicore_sam3d_backend(ctx);
    aicore_pipeline_timings timings{};
    double last_e2e = 0.0;
    if (aicore_sam3d_last_pipeline_timings(ctx, &timings) == 0)
        last_e2e = timings.e2e_ms;
    aicore_sam3d_free(ctx);

    std::printf(
            "{\"task\":\"sam3d\",\"device\":\"%s\",\"dtype\":\"%s\",\"steps\":%"
            "d,"
            "\"seed\":%d,\"gaussian_count\":%lld,\"mesh_vertices\":%d,"
            "\"mesh_triangles\":%d,\"geometry_sha12\":\"%s\",\"stable\":%s,"
            "\"mesh_hash_stable\":%s,\"finite\":%s,\"e2e_ms\":%.1f,\"p50_ms\":%"
            ".1f}\n",
            resolved_backend, dtype_name.c_str(), steps, seed,
            static_cast<long long>(gaussian_count), mesh_vertices,
            mesh_triangles, first_hash.c_str(), stable ? "true" : "false",
            mesh_hash_stable ? "true" : "false", finite ? "true" : "false",
            last_e2e, p50);

    if (!finite || !stable) return kExitFail;
    return kExitPass;
}

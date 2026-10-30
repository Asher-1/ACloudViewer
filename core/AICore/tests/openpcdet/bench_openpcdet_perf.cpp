// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Manual perf/accuracy probe for the OpenPCDet LiDAR detection path — NOT a
// ctest (needs a local GGUF + a raw float .bin point cloud: KITTI 4-dim
// x/y/z/intensity or nuScenes 5-dim x/y/z/intensity/ring).
//
// Usage:
//   bench_openpcdet_perf <model.gguf> <points.bin> [device] [threads]
//                        [warmups] [iters] [point_dim] [scene_dir]
//
// Pass <scene_dir> (upstream dump_bevfusion_scene.py layout) for multi-modal
// models (BEVFusion, CaDDN): the timed call is aicore_openpcdet_detect_scene
// instead of aicore_openpcdet_detect_points.
//
// Loads the model through the public C API, runs a warmup + timed loop on a
// fixed real sweep, checks run-to-run output stability, and emits the common
// validation JSON record on stdout:
//   {"suite":"aicore-validation","task":"openpcdet",...}
// Accuracy reference: detections are matched against the upstream
// OpenPCDet-GGML C++ engine baselines by the validation runner; this probe
// reports the detection count, per-class counts, first-box summary and the
// FNV-1a output hash for cross-backend/cross-build comparison.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/openpcdet_capi.h"
#include "common/validation_probe.hpp"

namespace {

uint64_t fnv1a(const void* data, size_t n) {
    uint64_t hash = 1469598103934665603ull;
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

double median_of(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <points.bin> [device] [threads] "
                     "[warmups] [iters] [point_dim=4|5] [scene_dir]\n",
                     argv[0]);
        return 2;
    }
    const char* device = argc > 3 ? argv[3] : "cpu";
    const int threads = argc > 4 ? std::atoi(argv[4]) : 0;
    const int warmups = argc > 5 ? std::atoi(argv[5]) : 2;
    const int iters = argc > 6 ? std::atoi(argv[6]) : 8;
    const int point_dim = argc > 7 ? std::atoi(argv[7]) : 4;
    const char* scene_dir = argc > 8 ? argv[8] : nullptr;
    if (point_dim != 4 && point_dim != 5) {
        std::fprintf(stderr, "unsupported point_dim %d (need 4 or 5)\n",
                     point_dim);
        return 2;
    }

    FILE* pf = fopen(argv[2], "rb");
    if (!pf) {
        std::fprintf(stderr, "cannot open points file %s\n", argv[2]);
        return 2;
    }
    fseek(pf, 0, SEEK_END);
    const long bytes = ftell(pf);
    fseek(pf, 0, SEEK_SET);
    std::vector<float> points(static_cast<size_t>(bytes) / sizeof(float));
    const size_t got = fread(points.data(), sizeof(float), points.size(), pf);
    fclose(pf);
    if (got != points.size() ||
        points.size() % static_cast<size_t>(point_dim) != 0) {
        std::fprintf(stderr, "truncated or misaligned points file\n");
        return 2;
    }
    const int32_t n_points =
            static_cast<int32_t>(points.size() / point_dim);

    aicore_openpcdet_options* opts = aicore_openpcdet_options_new();
    aicore_openpcdet_options_set_device(opts, device);
    aicore_openpcdet_options_set_threads(opts, threads);
    aicore_openpcdet_ctx* ctx = aicore_openpcdet_load_opts(argv[1], opts);
    aicore_openpcdet_options_free(opts);
    if (ctx == nullptr || !aicore_openpcdet_is_ready(ctx)) {
        std::fprintf(stderr, "load failed: %s\n",
                     aicore_openpcdet_last_error(ctx));
        aicore_openpcdet_free(ctx);
        return 1;
    }

    // Optional multi-modal scene input (BEVFusion / CaDDN models). Owned
    // buffers come from aicore_openpcdet_load_scene_dir; the point cloud
    // stays caller-owned.
    aicore_openpcdet_scene scene{};
    const bool use_scene = scene_dir != nullptr;
    if (use_scene) {
        char* scene_error = nullptr;
        if (aicore_openpcdet_load_scene_dir(scene_dir, &scene,
                                            &scene_error) != 0) {
            std::fprintf(stderr, "scene load failed: %s\n",
                         scene_error != nullptr ? scene_error : "?");
            aicore_openpcdet_free_buffer(scene_error);
            aicore_openpcdet_free(ctx);
            return 2;
        }
        scene.points = points.data();
        scene.n_points = n_points;
        scene.point_dim = point_dim;
    }

    for (int i = 0; i < warmups; ++i) {
        const int warmed = use_scene
                ? aicore_openpcdet_detect_scene(ctx, &scene)
                : aicore_openpcdet_detect_points(ctx, points.data(), n_points,
                                                 point_dim);
        if (warmed < 0) {
            std::fprintf(stderr, "warmup detect failed: %s\n",
                         aicore_openpcdet_last_error(ctx));
            if (use_scene) aicore_openpcdet_free_scene(&scene);
            aicore_openpcdet_free(ctx);
            return 1;
        }
    }

    std::vector<double> e2e_ms;
    std::vector<uint64_t> hashes;
    int last_count = -1;
    bool stable = true;
    for (int i = 0; i < iters; ++i) {
        const auto start = std::chrono::steady_clock::now();
        const int count = use_scene
                ? aicore_openpcdet_detect_scene(ctx, &scene)
                : aicore_openpcdet_detect_points(ctx, points.data(), n_points,
                                                 point_dim);
        const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - start)
                                  .count();
        if (count < 0) {
            std::fprintf(stderr, "detect failed: %s\n",
                         aicore_openpcdet_last_error(ctx));
            if (use_scene) aicore_openpcdet_free_scene(&scene);
            aicore_openpcdet_free(ctx);
            return 1;
        }
        e2e_ms.push_back(ms);
        if (last_count >= 0 && count != last_count) stable = false;
        last_count = count;
        uint64_t hash = 1469598103934665603ull;
        for (int d = 0; d < count; ++d) {
            aicore_openpcdet_detection det;
            aicore_openpcdet_detection_at(ctx, d, &det);
            hash = fnv1a(&det, sizeof(det));
        }
        hashes.push_back(hash);
    }
    const bool hash_stable =
            std::all_of(hashes.begin(), hashes.end(),
                        [&](uint64_t h) { return h == hashes.front(); });

    aicore_openpcdet_geometry geo;
    aicore_openpcdet_geometry_of(ctx, &geo);
    std::vector<int> per_class;
    per_class.resize(geo.num_class > 0 ? geo.num_class : 0, 0);
    float first_box[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int d = 0; d < last_count; ++d) {
        aicore_openpcdet_detection det;
        aicore_openpcdet_detection_at(ctx, d, &det);
        if (det.label >= 1 && det.label <= static_cast<int>(per_class.size())) {
            ++per_class[static_cast<size_t>(det.label - 1)];
        }
        if (d == 0) {
            first_box[0] = det.x;
            first_box[1] = det.y;
            first_box[2] = det.z;
            first_box[3] = det.dx;
            first_box[4] = det.dy;
            first_box[5] = det.dz;
            first_box[6] = det.heading;
            first_box[7] = det.score;
        }
    }

    std::printf(
            "{\"suite\":\"aicore-validation\",\"task\":\"openpcdet\","
            "\"device\":\"%s\",\"model\":\"%s\",\"arch\":\"%s\","
            "\"scene\":%s,"
            "\"num_points\":%d,\"inference_ms\":%.6f,\"e2e_ms\":%.6f,"
            "\"detection_count\":%d,\"per_class\":[",
            aicore_openpcdet_context_device(ctx), argv[1], geo.arch,
            use_scene ? "true" : "false", n_points,
            median_of(e2e_ms), median_of(e2e_ms), last_count);
    for (size_t i = 0; i < per_class.size(); ++i) {
        std::printf("%s%d", i ? "," : "", per_class[i]);
    }
    std::printf("],\"first_box\":[");
    for (int i = 0; i < 8; ++i) std::printf("%s%.7g", i ? "," : "", first_box[i]);
    std::printf("],\"count_stable\":%s,\"hash_stable\":%s,"
                "\"output_hash\":\"%016llx\"}\n",
                stable ? "true" : "false", hash_stable ? "true" : "false",
                static_cast<unsigned long long>(hashes.empty()
                                                        ? 0
                                                        : hashes.front()));

    if (use_scene) aicore_openpcdet_free_scene(&scene);
    aicore_openpcdet_free(ctx);
    aicore_openpcdet_shutdown();
    return stable ? 0 : 1;
}

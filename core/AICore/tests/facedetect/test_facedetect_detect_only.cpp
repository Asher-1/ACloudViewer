// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Detector-only validation for pure-detector SCRFD packs (e.g. the
// scrfd_10g_bnkps GGUF) that carry no recognizer weights. The embed/match
// tests require an ArcFace/SFace head, so this probe exercises the
// detection contract alone: ready load, faces found on a real image, all
// 5 landmarks finite, and run-to-run stability with timing. SKIP (77)
// without the shared model/image environment.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "aicore/facedetect_capi.h"
#include "tests/common/validation_probe.hpp"

namespace {

struct RunResult {
    int count = 0;
    std::vector<float> flat;  // score,x1,y1,x2,y2 + 10 landmarks per face
    bool finite = true;
};

int positiveEnvOr(const char* name, int fallback) {
    const char* value = std::getenv(name);
    const int parsed = value ? std::atoi(value) : 0;
    return parsed > 0 ? parsed : fallback;
}

double maxAbsDiff(const std::vector<float>& reference,
                  const std::vector<float>& candidate) {
    if (reference.size() != candidate.size()) return INFINITY;
    double maximum = 0.0;
    for (size_t i = 0; i < reference.size(); ++i)
        maximum = std::max(maximum, std::abs(static_cast<double>(reference[i]) -
                                             candidate[i]));
    return maximum;
}

RunResult detect(aicore_facedetect_ctx* ctx,
                 const char* image,
                 double* elapsed_ms) {
    RunResult run;
    // The image decodes through the task's Qt loader; call twice through
    // the typed path to compare the context-owned result stores.
    const auto start = std::chrono::steady_clock::now();
    aicore_image_view view{};
    uint8_t* rgb = nullptr;
    int32_t w = 0, h = 0;
    if (aicore_facedetect_load_path_rgb(image, &rgb, &w, &h) != 0 || !rgb) {
        std::fprintf(stderr, "failed to decode %s\n", image);
        return run;
    }
    view.format = AICORE_IMAGE_RGB8;
    view.data = rgb;
    view.width = w;
    view.height = h;
    view.row_stride_bytes = static_cast<size_t>(w) * 3;
    if (aicore_facedetect_detect_image(ctx, &view) != 0) {
        std::fprintf(stderr, "detect_image failed: %s\n",
                     aicore_facedetect_last_error(ctx)
                             ? aicore_facedetect_last_error(ctx)
                             : "(null)");
        aicore_facedetect_free_buffer(rgb);
        return run;
    }
    const auto stop = std::chrono::steady_clock::now();
    *elapsed_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
    run.count = static_cast<int>(aicore_facedetect_detection_count(ctx));
    for (int i = 0; i < run.count; ++i) {
        aicore_facedetect_detection d{};
        if (aicore_facedetect_detection_at(ctx, i, &d) != 0) continue;
        const float values[15] = {d.score,
                                  d.x1,
                                  d.y1,
                                  d.x2,
                                  d.y2,
                                  d.landmarks_xy10[0],
                                  d.landmarks_xy10[1],
                                  d.landmarks_xy10[2],
                                  d.landmarks_xy10[3],
                                  d.landmarks_xy10[4],
                                  d.landmarks_xy10[5],
                                  d.landmarks_xy10[6],
                                  d.landmarks_xy10[7],
                                  d.landmarks_xy10[8],
                                  d.landmarks_xy10[9]};
        for (float v : values) {
            run.flat.push_back(v);
            if (!std::isfinite(v)) run.finite = false;
        }
    }
    aicore_facedetect_free_buffer(rgb);
    return run;
}

}  // namespace

int main() {
    const char* model = std::getenv("AICORE_TEST_FACEDETECT_GGUF");
    const char* image = std::getenv("AICORE_TEST_FACEDETECT_IMAGE");
    if (!model || !image) return 77;

    aicore_facedetect_options* options = aicore_facedetect_options_new();
    if (!options) return 1;
    if (const char* device = std::getenv("AICORE_TEST_FACEDETECT_DEVICE")) {
        aicore_facedetect_options_set_device(options, device);
    }
    aicore_facedetect_ctx* ctx = aicore_facedetect_load_opts(model, options);
    aicore_facedetect_options_free(options);
    if (!ctx || !aicore_facedetect_is_ready(ctx)) {
        std::fprintf(stderr, "failed to load detector-only pack %s: %s\n",
                     model,
                     ctx && aicore_facedetect_last_error(ctx)
                             ? aicore_facedetect_last_error(ctx)
                             : "(no error)");
        aicore_facedetect_free(ctx);
        return 1;
    }

    const int warmups = positiveEnvOr("AICORE_TEST_WARMUP_RUNS", 2);
    const int runs = positiveEnvOr("AICORE_TEST_INFERENCE_RUNS", 7);
    for (int i = 0; i < warmups; ++i) {
        double ignored = 0.0;
        RunResult warmup = detect(ctx, image, &ignored);
        if (warmup.count < 1 || !warmup.finite) {
            std::fprintf(stderr, "invalid warmup output\n");
            aicore_facedetect_free(ctx);
            return 1;
        }
    }

    std::vector<double> elapsed;
    elapsed.reserve(runs);
    RunResult reference;
    double worst = 0.0;
    for (int i = 0; i < runs; ++i) {
        double ms = 0.0;
        RunResult run = detect(ctx, image, &ms);
        if (i == 0) reference = run;
        elapsed.push_back(ms);
        if (run.count != reference.count || !run.finite) {
            std::fprintf(stderr, "unstable or non-finite detection output\n");
            aicore_facedetect_free(ctx);
            return 1;
        }
        worst = std::max(worst, maxAbsDiff(reference.flat, run.flat));
    }
    aicore_facedetect_free(ctx);

    if (reference.count < 1) {
        std::fprintf(stderr, "detector found no faces on %s\n", image);
        return 1;
    }
    // Same context and input must keep typed detections numerically stable.
    if (worst > 1.0e-4) {
        std::fprintf(stderr, "detection drift %.7f exceeds 1e-4\n", worst);
        return 1;
    }
    double sum = 0.0;
    for (float v : reference.flat) sum += v;
    std::sort(elapsed.begin(), elapsed.end());
    const double p50 = elapsed[elapsed.size() / 2];
    const double p95 =
            elapsed[static_cast<size_t>(std::ceil(
                            0.95 * static_cast<double>(elapsed.size()))) -
                    1];
    const char* device = std::getenv("AICORE_TEST_FACEDETECT_DEVICE");
    // output_hash: byte fingerprint of the typed detections, so the runner's
    // exact policy also proves cross-process determinism and baseline drift.
    const uint64_t output_hash = aicore::test::fnv1a(
            reference.flat.data(), reference.flat.size() * sizeof(float));
    std::printf(
            "{\"suite\":\"aicore\",\"task\":\"facedetect\","
            "\"device\":\"%s\",\"faces\":%d,\"sum\":%.4f,"
            "\"p50_ms\":%.2f,\"p95_ms\":%.2f,\"warmups\":%d,"
            "\"iterations\":%d,\"max_abs_diff\":%.7f,"
            "\"output_hash\":\"%016llx\"}\n",
            device ? device : "auto", reference.count, sum, p50, p95, warmups,
            runs, worst, static_cast<unsigned long long>(output_hash));
    return 0;
}

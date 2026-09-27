// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Upstream numeric-parity probe for the LingBot-Map integration.
//
// Streams the pinned courthouse fixture frames through the official
// preprocessing (aicore_lingbot_preprocess_image) and the persistent-KV
// stream, then compares the engine's pose_enc / intrinsics / depth /
// depth-conf against GOLDENS captured from the upstream lingbot-map-cli
// (lingbot-map-ggml@156117f, cpp_ggml) on the byte-identical preprocessed
// stream. This is the numeric gate that catches semantic/convention drift
// (e.g. the pose w2c/c2w inversion bug) that finite/deterministic gates
// cannot see. The capture procedure lives in
// tests/lingbot/gen_stream_parity_golden.py; per-element A/B evidence comes
// from tests/lingbot/compare_lingbot_parity.py.
//
// Env:
//   AICORE_TEST_LINGBOT_GGUF      required map GGUF (missing -> exit 77 skip)
//   AICORE_TEST_LINGBOT_INPUT_DIR required dir of 000000.png.. frame fixtures
//   AICORE_TEST_DEVICE            backend (default "cpu")
//   AICORE_TEST_LINGBOT_KV        "scale,window" (default "2,8" = the golden
//                                 capture profile)
//   AICORE_TEST_LINGBOT_DUMP_STREAM  capture mode: write the preprocessed
//                                 [N,3,H,W] f32 stream to this path (plus
//                                 "<path>.meta.json") and exit 0 without
//                                 inferring (golden-capture helper).
//
// Metrics are exported as a JSON line (flattened by validate_all.py into
// json0/* and enforced by the manifest's structured accuracy checks) and
// enforced locally with the same thresholds so the probe also gates under
// plain ctest.

#include <QImage>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/lingbot_capi.h"
#include "lingbot_stream_parity_golden.h"
#include "tests/common/test_macros.hpp"
#include "tests/common/validation_probe.hpp"

namespace {

// Thresholds calibrated per backend against the same-backend upstream
// goldens on the capture host (RTX 4090). Measured A/C deltas: CUDA
// pose t ~3e-5 / intr ~1.2e-2; Vulkan ~3e-4 / ~1.5e-1; CPU ~3e-4 /
// ~2.2e-1 (fx/fy carry the fov rounding through (W/2)/tan(fov/2),
// which amplifies per-backend kernel fp differences; pose/depth/confs
// stay tight). The failure modes this gate exists for
// (pose-convention/scale drift) produce O(0.1)..O(10) errors — orders
// of magnitude above every band.
struct ParityTols {
    double poseT;      // scene units
    double quat;       // elementwise |dq|
    double intr;       // elementwise |d fx fy cx cy|
    double depthPool;  // rel to golden pool mean
    double conf;       // per-frame conf mean
};

constexpr ParityTols kTolsCuda{0.02, 0.005, 0.05, 0.02, 0.05};
constexpr ParityTols kTolsVulkan{0.02, 0.005, 0.5, 0.02, 0.05};
constexpr ParityTols kTolsCpu{0.02, 0.005, 0.5, 0.02, 0.05};

std::vector<std::string> listPngFrames(const std::string& dir) {
    std::vector<std::string> files;
    for (int i = 0;; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "%08d.png", i);
        const std::string path = dir + "/" + name;
        std::FILE* probe = std::fopen(path.c_str(), "rb");
        if (!probe) break;
        std::fclose(probe);
        files.push_back(path);
    }
    return files;
}

// Uniform 16x16 block-mean grid over one depth plane (fractional block
// edges; MUST stay in sync with gen_stream_parity_golden.py's poolDepth).
std::vector<float> poolDepth(const float* depth, int w, int h, int grid) {
    std::vector<float> pool(static_cast<size_t>(grid) * grid, 0.f);
    for (int gy = 0; gy < grid; ++gy) {
        const double y0 = static_cast<double>(gy) * h / grid;
        const double y1 = static_cast<double>(gy + 1) * h / grid;
        for (int gx = 0; gx < grid; ++gx) {
            const double x0 = static_cast<double>(gx) * w / grid;
            const double x1 = static_cast<double>(gx + 1) * w / grid;
            double sum = 0.0;
            int count = 0;
            for (int y = static_cast<int>(y0);
                 y < static_cast<int>(std::ceil(y1)) && y < h; ++y) {
                for (int x = static_cast<int>(x0);
                     x < static_cast<int>(std::ceil(x1)) && x < w; ++x) {
                    sum += depth[static_cast<size_t>(y) * w + x];
                    ++count;
                }
            }
            pool[static_cast<size_t>(gy) * grid + gx] =
                    count > 0 ? static_cast<float>(sum / count) : 0.f;
        }
    }
    return pool;
}

}  // namespace

// Stream state shared with the C callback (runs on the inference thread
// while infer_stream is on the stack, so plain members are safe).
namespace {
struct ParityState {
    int frames = 0;
    int delivered = 0;
    std::vector<std::array<float, 9>> pose;
    std::vector<std::array<float, 4>> intr;
    std::vector<std::vector<float>> depth;
    std::vector<float> confMean;
};

int parityCb(void* user, const aicore_lingbot_result* r) {
    auto* state = static_cast<ParityState*>(user);
    if (!r || !r->depth || !r->depth_conf || !r->pose_enc || !r->intrinsics ||
        state->delivered >= state->frames) {
        return 1;
    }
    const size_t plane =
            static_cast<size_t>(r->width) * static_cast<size_t>(r->height);
    const int f = state->delivered;
    std::memcpy(state->pose[f].data(), r->pose_enc, 9 * sizeof(float));
    std::memcpy(state->intr[f].data(), r->intrinsics, 4 * sizeof(float));
    state->depth[f].assign(r->depth, r->depth + plane);
    double confSum = 0.0;
    for (size_t i = 0; i < plane; ++i) confSum += r->depth_conf[i];
    state->confMean[f] = static_cast<float>(confSum / plane);
    ++state->delivered;
    return 0;
}
// Per-backend goldens: each backend compares against the UPSTREAM CLI's
// outputs on that same backend (cross-backend fp divergence is real —
// CPU vs CUDA conf means differ by O(1) — so a single golden set cannot
// gate all backends). Unknown backends (e.g. Metal) have no captured
// goldens and skip (exit 77).
struct ParityGoldens {
    const float* data;
    const ParityTols* tols;
    const char* label;
};

static const ParityGoldens selectGoldens(const char* device) {
    const std::string dev = device ? device : "";
    if (dev.rfind("cuda", 0) == 0)
        return {kLingbotParityGoldenCuda, &kTolsCuda, "cuda"};
    if (dev.rfind("vulkan", 0) == 0)
        return {kLingbotParityGoldenVulkan, &kTolsVulkan, "vulkan"};
    if (dev.rfind("cpu", 0) == 0)
        return {kLingbotParityGoldenCpu, &kTolsCpu, "cpu"};
    return {nullptr, nullptr, dev.c_str()};
}

}  // namespace

static int failures = 0;

int main() {
    const char* gguf = std::getenv("AICORE_TEST_LINGBOT_GGUF");
    const char* inputDir = std::getenv("AICORE_TEST_LINGBOT_INPUT_DIR");
    if (!gguf || gguf[0] == '\0' || !inputDir || inputDir[0] == '\0') {
        return 77;  // required assets missing: a skip, never a pass
    }
    const char* device = std::getenv("AICORE_TEST_DEVICE");
    if (!device || device[0] == '\0') device = "cpu";
    const char* dumpStream = std::getenv("AICORE_TEST_LINGBOT_DUMP_STREAM");

    const std::vector<std::string> files = listPngFrames(inputDir);
    if (files.size() < 2) {
        std::fprintf(stderr, "no frame fixtures under %s\n", inputDir);
        return 77;
    }

    // Decode + official preprocess into one contiguous [N,3,H,W] stream.
    std::vector<float> stream;
    int procW = 0, procH = 0;
    for (const std::string& path : files) {
        // QImage ships the same PNG decode the plugin uses; RGB888 keeps a
        // tightly packed 3-bytes-per-pixel row layout for the image view.
        QImage decoded(QString::fromStdString(path));
        if (decoded.isNull()) {
            std::fprintf(stderr, "cannot decode fixture: %s\n", path.c_str());
            return 1;
        }
        QImage rgb = decoded.convertToFormat(QImage::Format_RGB888);
        aicore_image_view view{};
        view.data = rgb.bits();
        view.width = rgb.width();
        view.height = rgb.height();
        view.row_stride_bytes = rgb.bytesPerLine();
        view.format = AICORE_IMAGE_RGB8;
        int32_t pw = 0, ph = 0;
        const int needed = aicore_lingbot_preprocess_image(&view, 518, nullptr,
                                                           0, &pw, &ph);
        if (needed <= 0) {
            std::fprintf(stderr, "preprocess sizing failed: %s\n",
                         path.c_str());
            return 1;
        }
        if (stream.empty()) {
            procW = pw;
            procH = ph;
            stream.reserve(static_cast<size_t>(files.size()) *
                           static_cast<size_t>(needed));
        } else if (pw != procW || ph != procH) {
            std::fprintf(stderr, "mismatched processed size: %s\n",
                         path.c_str());
            return 1;
        }
        const size_t offset = stream.size();
        stream.resize(offset + static_cast<size_t>(needed));
        if (aicore_lingbot_preprocess_image(&view, 518, stream.data() + offset,
                                            needed, nullptr,
                                            nullptr) != needed) {
            std::fprintf(stderr, "preprocess failed: %s\n", path.c_str());
            return 1;
        }
    }
    const int n = static_cast<int>(files.size());

    // Golden-capture helper: hand the preprocessed stream to the upstream
    // reference (gen_stream_parity_golden.py drives the rest).
    if (dumpStream && dumpStream[0]) {
        std::FILE* out = std::fopen(dumpStream, "wb");
        if (!out) {
            std::fprintf(stderr, "cannot write %s\n", dumpStream);
            return 2;
        }
        std::fwrite(stream.data(), sizeof(float), stream.size(), out);
        std::fclose(out);
        const std::string metaPath = std::string(dumpStream) + ".meta.json";
        std::FILE* meta = std::fopen(metaPath.c_str(), "w");
        if (meta) {
            std::fprintf(
                    meta,
                    "{\"n\":%d,\"width\":%d,\"height\":%d,"
                    "\"kv\":\"%s\"}\n",
                    n, procW, procH,
                    std::getenv("AICORE_TEST_LINGBOT_KV") &&
                                    std::getenv("AICORE_TEST_LINGBOT_KV")[0]
                            ? std::getenv("AICORE_TEST_LINGBOT_KV")
                            : "2,8");
            std::fclose(meta);
        }
        std::printf("dumped stream: n=%d %dx%d -> %s\n", n, procW, procH,
                    dumpStream);
        return 0;
    }

    if (n != kLingbotParityFrames || procW != kLingbotParityWidth ||
        procH != kLingbotParityHeight) {
        std::fprintf(stderr,
                     "fixture layout mismatch: n=%d %dx%d, golden n=%d %dx%d\n",
                     n, procW, procH, kLingbotParityFrames, kLingbotParityWidth,
                     kLingbotParityHeight);
        return 1;
    }
    const ParityGoldens goldens = selectGoldens(device);
    if (!goldens.data) {
        std::fprintf(stderr,
                     "SKIP: no captured upstream goldens for backend '%s' "
                     "(capture with gen_stream_parity_golden.py)\n",
                     device);
        return 77;
    }

    aicore_lingbot_options* opts = aicore_lingbot_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_lingbot_options_set_device(opts, device);
    aicore_lingbot_options_set_threads(opts, 4);
    aicore_lingbot_options_set_stream_capacity(opts, n);
    // The golden capture profile (2,8): a 2-frame scale pass plus window
    // eviction inside the 12-frame stream — the same semantics the upstream
    // goldens were captured with (--kv-f16 strict --kv-scale 2 --kv-window 8).
    const char* kvEnv = std::getenv("AICORE_TEST_LINGBOT_KV");
    const char* kv = kvEnv && kvEnv[0] ? kvEnv : "2,8";
    int kvScale = 0, kvWindow = 0;
    AICORE_CHECK(std::sscanf(kv, "%d,%d", &kvScale, &kvWindow) == 2);
    aicore_lingbot_options_set_kv_profile(opts, kvScale, kvWindow);

    aicore_lingbot_ctx* ctx = aicore_lingbot_load_opts(gguf, opts);
    aicore_lingbot_options_free(opts);
    if (!ctx || aicore_lingbot_is_ready(ctx) != 1) {
        std::fprintf(stderr, "lingbot load failed: %s (path=%s)\n",
                     aicore_lingbot_last_error(ctx), gguf);
        aicore_lingbot_free(ctx);
        return 1;
    }

    // Collected per-frame engine outputs.
    ParityState state;
    state.frames = n;
    state.pose.resize(n);
    state.intr.resize(n);
    state.depth.resize(n);
    state.confMean.assign(n, 0.f);
    if (aicore_lingbot_infer_stream(ctx, stream.data(), n, procW, procH,
                                    &parityCb, &state) != 0) {
        std::fprintf(stderr, "infer_stream failed: %s (delivered %d)\n",
                     aicore_lingbot_last_error(ctx), state.delivered);
        aicore_lingbot_free(ctx);
        return 1;
    }
    AICORE_CHECK(state.delivered == n);
    aicore_pipeline_timings timings{};
    AICORE_CHECK(aicore_lingbot_last_pipeline_timings(ctx, &timings) == 0);

    // ---- reductions vs the upstream goldens ----
    const size_t plane = static_cast<size_t>(procW) * procH;
    const int grid = kLingbotParityPool;
    double tDiffSq = 0.0;
    double quatMax = 0.0, intrMax = 0.0, confMax = 0.0;
    double poolDiffSq = 0.0, poolAbsSum = 0.0;
    size_t poolCount = 0;
    for (int f = 0; f < n; ++f) {
        const float* g = goldens.data +
                         static_cast<size_t>(f) * kLingbotParityFrameStride;
        for (int i = 0; i < 3; ++i) {
            const double d = static_cast<double>(state.pose[f][i]) - g[i];
            tDiffSq += d * d;
        }
        for (int i = 3; i < 7; ++i) {
            quatMax = std::max(
                    quatMax,
                    std::abs(static_cast<double>(state.pose[f][i]) - g[i]));
        }
        for (int i = 0; i < 4; ++i) {
            intrMax = std::max(
                    intrMax,
                    std::abs(static_cast<double>(state.intr[f][i]) - g[9 + i]));
        }
        confMax = std::max(
                confMax,
                std::abs(static_cast<double>(state.confMean[f]) - g[13]));
        const std::vector<float> pool =
                poolDepth(state.depth[f].data(), procW, procH, grid);
        for (int i = 0; i < grid * grid; ++i) {
            const double golden = g[14 + i];
            const double d = static_cast<double>(pool[i]) - golden;
            poolDiffSq += d * d;
            poolAbsSum += std::abs(golden);
            ++poolCount;
        }
    }
    const double poseTRmse = std::sqrt(tDiffSq / n);
    const double depthPoolRelRmse =
            poolAbsSum > 0.0
                    ? std::sqrt(poolDiffSq / poolCount) * poolCount / poolAbsSum
                    : 0.0;

    aicore_lingbot_shutdown();
    aicore_lingbot_free(ctx);

    // Metric export (flattened to json0/* and gated by the manifest's
    // structured accuracy checks; the same bands are enforced below so the
    // probe also gates under plain ctest).
    std::printf(
            "{\"suite\":\"aicore-validation\",\"task\":\"lingbot\","
            "\"scenario\":\"lingbot-stream-parity\",\"device\":\"%s\","
            "\"frames\":%d,\"width\":%d,\"height\":%d,"
            "\"pose_t_rmse\":%.9f,\"pose_quat_maxdiff\":%.9f,"
            "\"intr_maxdiff\":%.9f,\"depth_pool_rel_rmse\":%.9f,"
            "\"conf_mean_maxdiff\":%.9f}\n",
            device, n, procW, procH, poseTRmse, quatMax, intrMax,
            depthPoolRelRmse, confMax);

    int accuracyFailures = 0;
    const ParityTols& tols = *goldens.tols;
    const auto expect = [&](double actual, double tol, const char* what) {
        if (!(actual <= tol)) {
            std::fprintf(stderr, "FAIL %s: %.6g > %.6g\n", what, actual, tol);
            ++accuracyFailures;
        }
    };
    expect(poseTRmse, tols.poseT, "pose_t_rmse");
    expect(quatMax, tols.quat, "pose_quat_maxdiff");
    expect(intrMax, tols.intr, "intr_maxdiff");
    expect(depthPoolRelRmse, tols.depthPool, "depth_pool_rel_rmse");
    expect(confMax, tols.conf, "conf_mean_maxdiff");

    if (failures == 0 && accuracyFailures == 0) {
        aicore::test::printValidationResult("lingbot", device, 0, &timings);
        std::printf("test_lingbot_stream_parity: PASS\n");
        return 0;
    }
    std::printf("test_lingbot_stream_parity: %d FAILURES\n",
                failures + accuracyFailures);
    return 1;
}

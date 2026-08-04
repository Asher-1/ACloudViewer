// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// ALIKED GPU parity + latency gate: CPU F32 reference vs CUDA/Vulkan @1024.
//
// Env (optional if argv provided):
//   AICORE_TEST_ALIKED_GGUF   — aliked-n16rot-f32.gguf
//   AICORE_TEST_ALIKED_IMAGE  — sacre_coeur1.jpg
//
// Usage:
//   test_aliked_capi_parity [gguf] [image] [max_kpts] [resize_long_edge]
//
// Gates (vs CPU ref): kpt median <= 0.005 px, desc cosine median >= 0.9996
// Skip (77): missing assets or backend unavailable

#include <QImage>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "aicore/aliked_capi.h"
#include "aicore/backend_capi.h"
#include "aicore/lightglue_capi.h"

namespace {

constexpr float kKptMedianTolPx = 0.005f;
constexpr float kDescCosMedianTol = 0.9996f;

struct RgbImage {
    std::vector<uint8_t> rgb;
    int32_t width = 0;
    int32_t height = 0;
};

struct ExtractResult {
    std::vector<float> keypoints;  // x,y interleaved
    std::vector<float> descriptors;
    int32_t count = 0;
    int32_t dim = 0;
    double median_ms = 0.0;
    bool ok = false;
};

bool LoadRgbImage(const char* path, RgbImage* out) {
    if (path == nullptr || out == nullptr) {
        return false;
    }
    QImage img(QString::fromUtf8(path));
    if (img.isNull()) {
        return false;
    }
    QImage rgb = img.convertToFormat(QImage::Format_RGB888);
    out->width = rgb.width();
    out->height = rgb.height();
    out->rgb.resize(static_cast<size_t>(out->width) * out->height * 3);
    for (int y = 0; y < rgb.height(); ++y) {
        const uchar* row = rgb.constScanLine(y);
        std::memcpy(out->rgb.data() + static_cast<size_t>(y) * out->width * 3,
                    row, static_cast<size_t>(out->width) * 3);
    }
    return true;
}

ExtractResult ExtractTimed(const char* gguf,
                           const char* device,
                           const RgbImage& image,
                           int32_t max_kpts,
                           int32_t resize_long_edge,
                           int runs) {
    ExtractResult result;
    if (gguf == nullptr || device == nullptr || runs <= 0) {
        return result;
    }

    aicore_aliked_options* opts = aicore_aliked_options_new();
    aicore_aliked_options_set_device(opts, device);
    aicore_aliked_options_set_threads(opts, 0);
    aicore_aliked_options_set_max_keypoints(opts, max_kpts);
    aicore_aliked_options_set_resize_long_edge(opts, resize_long_edge);
    aicore_aliked_ctx* ctx = aicore_aliked_load_opts(gguf, opts);
    aicore_aliked_options_free(opts);
    if (ctx == nullptr) {
        return result;
    }

    aicore_lightglue_features features{};
    auto once = [&]() -> bool {
        return aicore_aliked_extract_rgb(ctx, image.rgb.data(), image.width,
                                         image.height, image.width * 3,
                                         &features) == 0;
    };

    const bool skip_warmup =
            std::getenv("AICORE_ALIKED_SKIP_WARMUP") != nullptr;
    if (!skip_warmup && !once()) {
        std::fprintf(stderr, "extract failed device=%s: %s\n", device,
                     aicore_aliked_last_error(ctx));
        aicore_lightglue_free_features(&features);
        aicore_aliked_free(ctx);
        return result;
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(runs));
    for (int i = 0; i < runs; ++i) {
        aicore_lightglue_free_features(&features);
        features = {};
        const auto t0 = std::chrono::steady_clock::now();
        if (!once()) {
            break;
        }
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    if (samples.empty()) {
        aicore_lightglue_free_features(&features);
        aicore_aliked_free(ctx);
        return result;
    }

    std::sort(samples.begin(), samples.end());
    result.median_ms = samples[samples.size() / 2];
    result.count = features.n_keypoints;
    result.dim = features.descriptor_dim;
    result.keypoints.reserve(static_cast<size_t>(result.count) * 2);
    for (int32_t i = 0; i < result.count; ++i) {
        result.keypoints.push_back(features.keypoints[i].x);
        result.keypoints.push_back(features.keypoints[i].y);
    }
    result.descriptors.assign(
            features.descriptors,
            features.descriptors +
                    static_cast<size_t>(result.count) *
                            static_cast<size_t>(std::max(1, result.dim)));
    result.ok = true;

    aicore_lightglue_free_features(&features);
    aicore_aliked_free(ctx);
    return result;
}

float KptDistance(float x0, float y0, float x1, float y1) {
    const float dx = x0 - x1;
    const float dy = y0 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

float DescCosine(const float* a, const float* b, int32_t dim) {
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (int32_t i = 0; i < dim; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        norm_a += static_cast<double>(a[i]) * a[i];
        norm_b += static_cast<double>(b[i]) * b[i];
    }
    if (norm_a <= 0.0 || norm_b <= 0.0) {
        return -1.0f;
    }
    return static_cast<float>(dot / std::sqrt(norm_a * norm_b));
}

bool FeaturesLookValid(const ExtractResult& result, const char* device) {
    const size_t count = static_cast<size_t>(std::max(0, result.count));
    const size_t dim = static_cast<size_t>(std::max(0, result.dim));
    if (!result.ok || count == 0 || dim == 0 ||
        result.keypoints.size() != count * 2 ||
        result.descriptors.size() != count * dim) {
        std::fprintf(stderr, "FAIL: %s returned malformed features\n", device);
        return false;
    }

    for (float value : result.keypoints) {
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "FAIL: %s returned non-finite keypoints\n",
                         device);
            return false;
        }
    }
    for (float value : result.descriptors) {
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "FAIL: %s returned non-finite descriptors\n",
                         device);
            return false;
        }
    }

    constexpr float kDuplicateTolPx = 1.0e-4f;
    for (size_t i = 0; i < count; ++i) {
        const float* desc = result.descriptors.data() + i * dim;
        double norm2 = 0.0;
        for (size_t d = 0; d < dim; ++d) {
            norm2 += static_cast<double>(desc[d]) * desc[d];
        }
        if (norm2 <= 0.0) {
            std::fprintf(stderr, "FAIL: %s returned a zero descriptor\n",
                         device);
            return false;
        }
        for (size_t j = 0; j < i; ++j) {
            if (KptDistance(result.keypoints[i * 2],
                            result.keypoints[i * 2 + 1],
                            result.keypoints[j * 2],
                            result.keypoints[j * 2 + 1]) <= kDuplicateTolPx) {
                std::fprintf(stderr,
                             "FAIL: %s returned duplicate keypoints at %zu and "
                             "%zu\n",
                             device, j, i);
                return false;
            }
        }
    }
    return true;
}

bool CompareParity(const ExtractResult& ref,
                   const ExtractResult& test,
                   float* out_kpt_median,
                   float* out_desc_median) {
    if (!ref.ok || !test.ok || ref.count <= 0 || test.count <= 0 ||
        ref.count != test.count || ref.dim != test.dim) {
        return false;
    }

    std::vector<float> kpt_errors;
    std::vector<float> desc_cos;
    kpt_errors.reserve(static_cast<size_t>(test.count));
    desc_cos.reserve(static_cast<size_t>(test.count));

    std::vector<bool> matched(static_cast<size_t>(ref.count), false);
    for (int32_t i = 0; i < test.count; ++i) {
        const float tx = test.keypoints[static_cast<size_t>(i) * 2];
        const float ty = test.keypoints[static_cast<size_t>(i) * 2 + 1];
        float best = std::numeric_limits<float>::infinity();
        int32_t best_j = 0;
        for (int32_t j = 0; j < ref.count; ++j) {
            if (matched[static_cast<size_t>(j)]) {
                continue;
            }
            const float rx = ref.keypoints[static_cast<size_t>(j) * 2];
            const float ry = ref.keypoints[static_cast<size_t>(j) * 2 + 1];
            const float d = KptDistance(tx, ty, rx, ry);
            if (d < best) {
                best = d;
                best_j = j;
            }
        }
        matched[static_cast<size_t>(best_j)] = true;
        kpt_errors.push_back(best);
        const float* tdesc =
                test.descriptors.data() +
                static_cast<size_t>(i) * static_cast<size_t>(test.dim);
        const float* rdesc =
                ref.descriptors.data() +
                static_cast<size_t>(best_j) * static_cast<size_t>(ref.dim);
        desc_cos.push_back(DescCosine(tdesc, rdesc, test.dim));
    }

    std::sort(kpt_errors.begin(), kpt_errors.end());
    std::sort(desc_cos.begin(), desc_cos.end());
    *out_kpt_median = kpt_errors[kpt_errors.size() / 2];
    *out_desc_median = desc_cos[desc_cos.size() / 2];
    return true;
}

bool WriteAkout(const char* path,
                const ExtractResult& result,
                int32_t width,
                int32_t height) {
    if (path == nullptr || path[0] == '\0' || !result.ok) {
        return false;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out.write("AKOUT01\0", 8);
    const uint32_t count = static_cast<uint32_t>(result.count);
    const uint32_t dim = static_cast<uint32_t>(result.dim);
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&width), sizeof(width));
    out.write(reinterpret_cast<const char*>(&height), sizeof(height));
    out.write(reinterpret_cast<const char*>(result.keypoints.data()),
              result.keypoints.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(result.descriptors.data()),
              result.descriptors.size() * sizeof(float));
    return static_cast<bool>(out);
}

}  // namespace

int BenchRuns(int default_runs) {
    const char* env = std::getenv("AICORE_ALIKED_BENCH_RUNS");
    if (env == nullptr || env[0] == '\0') {
        return default_runs;
    }
    const int n = std::atoi(env);
    return n > 0 ? n : default_runs;
}

int main(int argc, char** argv) {
    const char* gguf = std::getenv("AICORE_TEST_ALIKED_GGUF");
    const char* image_path = std::getenv("AICORE_TEST_ALIKED_IMAGE");
    if (argc >= 2) {
        gguf = argv[1];
    }
    if (argc >= 3) {
        image_path = argv[2];
    }
    if (gguf == nullptr || gguf[0] == '\0' || image_path == nullptr ||
        image_path[0] == '\0') {
        std::fprintf(
                stderr,
                "SKIP: set AICORE_TEST_ALIKED_GGUF + AICORE_TEST_ALIKED_IMAGE "
                "or pass gguf image args\n");
        return 77;
    }

    FILE* f = std::fopen(gguf, "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "SKIP: missing GGUF: %s\n", gguf);
        return 77;
    }
    std::fclose(f);

    RgbImage image;
    if (!LoadRgbImage(image_path, &image)) {
        std::fprintf(stderr, "SKIP: failed to load image: %s\n", image_path);
        return 77;
    }

    const int32_t max_kpts = (argc >= 4) ? std::atoi(argv[3]) : 1024;
    const int32_t resize = (argc >= 5) ? std::atoi(argv[4]) : 1024;

    std::fprintf(stderr, "ALIKED parity: gguf=%s image=%s kpts=%d resize=%d\n",
                 gguf, image_path, max_kpts, resize);

    const char* only = std::getenv("AICORE_TEST_DEVICE");
    const char* bench_only = std::getenv("AICORE_ALIKED_BENCH_ONLY");
    if (only != nullptr && only[0] != '\0' && std::strcmp(only, "cpu") == 0 &&
        bench_only == nullptr) {
        std::fprintf(stderr,
                     "SKIP: set AICORE_TEST_DEVICE to cuda or vulkan\n");
        return 77;
    }

    if (bench_only != nullptr && only != nullptr && only[0] != '\0') {
        const ExtractResult gpu =
                ExtractTimed(gguf, only, image, max_kpts, resize, BenchRuns(3));
        if (!gpu.ok) {
            std::fprintf(stderr, "FAIL: %s extract failed\n", only);
            return 1;
        }
        if (!FeaturesLookValid(gpu, only)) {
            return 1;
        }
        std::printf("%s: kpts=%d median_ms=%.2f (bench-only)\n", only,
                    gpu.count, gpu.median_ms);
        return 0;
    }

    const ExtractResult cpu =
            ExtractTimed(gguf, "cpu", image, max_kpts, resize, 1);
    if (!cpu.ok) {
        std::fprintf(stderr, "FAIL: CPU reference extract failed\n");
        return 1;
    }
    if (!FeaturesLookValid(cpu, "cpu")) {
        return 1;
    }
    std::printf("cpu: kpts=%d median_ms=%.2f\n", cpu.count, cpu.median_ms);

    if (const char* dump = std::getenv("AICORE_DUMP_AKOUT_CPU")) {
        if (WriteAkout(dump, cpu, resize, resize)) {
            std::printf("wrote cpu akout: %s\n", dump);
        }
    }

    int failures = 0;
    const char* backends[] = {"cuda", "vulkan"};
    for (const char* backend : backends) {
        if (only != nullptr && only[0] != '\0' &&
            std::strcmp(only, backend) != 0) {
            continue;
        }
        const ExtractResult gpu = ExtractTimed(gguf, backend, image, max_kpts,
                                               resize, BenchRuns(3));
        if (!gpu.ok) {
            if (!aicore_device_available(backend)) {
                std::printf("SKIP backend=%s (backend unavailable)\n", backend);
                continue;
            }
            std::printf("FAIL backend=%s (load or extract failed)\n", backend);
            ++failures;
            continue;
        }
        if (!FeaturesLookValid(gpu, backend)) {
            ++failures;
            continue;
        }

        float kpt_med = 0.0f;
        float cos_med = 0.0f;
        if (!CompareParity(cpu, gpu, &kpt_med, &cos_med)) {
            std::printf("FAIL backend=%s parity compare failed\n", backend);
            ++failures;
            continue;
        }

        std::printf(
                "%s: kpts=%d median_ms=%.2f kpt_med=%.4fpx desc_cos_med=%.4f\n",
                backend, gpu.count, gpu.median_ms, kpt_med, cos_med);

        if (const char* dump_env = std::getenv("AICORE_DUMP_AKOUT_GPU")) {
            if (WriteAkout(dump_env, gpu, resize, resize)) {
                std::printf("wrote gpu akout: %s\n", dump_env);
            }
        }

        if (kpt_med > kKptMedianTolPx || cos_med < kDescCosMedianTol) {
            std::printf("FAIL backend=%s parity gate (kpt<=%.4f cos>=%.4f)\n",
                        backend, kKptMedianTolPx, kDescCosMedianTol);
            ++failures;
        }
    }

    return failures > 0 ? 1 : 0;
}

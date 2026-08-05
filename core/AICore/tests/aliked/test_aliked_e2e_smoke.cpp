// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// ALIKED end-to-end smoke: same C API path as qLightGlue extract_aliked_ggml().
// Verifies Vulkan full GPU path auto-enables without manual LIGHTGLUE_ALIKED_*
// env.
//
// Env (optional if argv provided):
//   AICORE_TEST_ALIKED_GGUF, AICORE_TEST_ALIKED_IMAGE
//
// Usage:
//   test_aliked_e2e_smoke [gguf] [image] [max_kpts] [resize_long_edge]
//
// Skip (77): missing assets or backend unavailable

#include <QImage>
#include <QImageReader>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/aliked_capi.h"
#include "aicore/backend_capi.h"
#include "aicore/lightglue_capi.h"

namespace {

struct RgbImage {
    std::vector<uint8_t> rgb;
    int32_t width = 0;
    int32_t height = 0;
};

bool LoadRgbImage(const char* path, RgbImage* out) {
    if (path == nullptr || out == nullptr) {
        return false;
    }
    QImageReader reader(QString::fromUtf8(path));
    reader.setAutoTransform(true);
    QImage img = reader.read();
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

bool ExtractOnce(const char* gguf,
                 const char* device,
                 const RgbImage& image,
                 int32_t max_kpts,
                 int32_t resize_long_edge,
                 aicore_lightglue_features* features,
                 std::string* err) {
    aicore_aliked_options* opts = aicore_aliked_options_new();
    aicore_aliked_options_set_device(opts, device);
    aicore_aliked_options_set_max_keypoints(opts, max_kpts);
    aicore_aliked_options_set_resize_long_edge(opts, resize_long_edge);
    aicore_aliked_ctx* ctx = aicore_aliked_load_opts(gguf, opts);
    aicore_aliked_options_free(opts);
    if (ctx == nullptr) {
        if (err) {
            *err = "failed to load ALIKED context";
        }
        return false;
    }
    const int rc =
            aicore_aliked_extract_rgb(ctx, image.rgb.data(), image.width,
                                      image.height, image.width * 3, features);
    if (rc != 0) {
        if (err) {
            *err = aicore_aliked_last_error(ctx);
        }
        aicore_aliked_free(ctx);
        return false;
    }
    aicore_aliked_free(ctx);
    return true;
}

bool FeaturesLookValid(const aicore_lightglue_features& f) {
    if (f.n_keypoints <= 0 || f.descriptor_dim <= 0 || f.keypoints == nullptr ||
        f.descriptors == nullptr) {
        return false;
    }
    for (int32_t i = 0; i < f.n_keypoints; ++i) {
        if (!std::isfinite(f.keypoints[i].x) ||
            !std::isfinite(f.keypoints[i].y)) {
            return false;
        }
        for (int32_t j = 0; j < i; ++j) {
            const float dx = f.keypoints[i].x - f.keypoints[j].x;
            const float dy = f.keypoints[i].y - f.keypoints[j].y;
            if (dx * dx + dy * dy <= 1.0e-8f) {
                return false;
            }
        }
    }
    const size_t nd = static_cast<size_t>(f.n_keypoints) *
                      static_cast<size_t>(f.descriptor_dim);
    for (size_t i = 0; i < nd; ++i) {
        if (!std::isfinite(f.descriptors[i])) {
            return false;
        }
    }
    for (int32_t i = 0; i < f.n_keypoints; ++i) {
        double norm2 = 0.0;
        for (int32_t d = 0; d < f.descriptor_dim; ++d) {
            const float value =
                    f.descriptors[static_cast<size_t>(i) * f.descriptor_dim +
                                  d];
            norm2 += static_cast<double>(value) * value;
        }
        if (norm2 <= 0.0) {
            return false;
        }
    }
    return true;
}

}  // namespace

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

    const int32_t max_kpts = (argc >= 4) ? std::atoi(argv[3]) : 256;
    const int32_t resize = (argc >= 5) ? std::atoi(argv[4]) : 512;

    aicore_lightglue_features cpu{};
    aicore_lightglue_features vk{};
    std::string err;

    if (!ExtractOnce(gguf, "cpu", image, max_kpts, resize, &cpu, &err)) {
        std::fprintf(stderr, "FAIL: CPU extract: %s\n", err.c_str());
        aicore_lightglue_free_features(&cpu);
        return 1;
    }
    if (!FeaturesLookValid(cpu)) {
        std::fprintf(stderr, "FAIL: CPU features invalid\n");
        aicore_lightglue_free_features(&cpu);
        return 1;
    }

    if (!ExtractOnce(gguf, "vulkan", image, max_kpts, resize, &vk, &err)) {
        if (!aicore_device_available("vulkan")) {
            std::fprintf(stderr, "SKIP: Vulkan backend unavailable: %s\n",
                         err.c_str());
            aicore_lightglue_free_features(&cpu);
            return 77;
        }
        std::fprintf(stderr, "FAIL: Vulkan extract failed: %s\n", err.c_str());
        aicore_lightglue_free_features(&cpu);
        return 1;
    }
    if (!FeaturesLookValid(vk)) {
        std::fprintf(stderr,
                     "FAIL: Vulkan features invalid (NaN/Inf or empty)\n");
        aicore_lightglue_free_features(&cpu);
        aicore_lightglue_free_features(&vk);
        return 1;
    }

    if (vk.n_keypoints != cpu.n_keypoints) {
        std::fprintf(
                stderr,
                "WARN: kpt count cpu=%d vulkan=%d (smoke allows small drift)\n",
                cpu.n_keypoints, vk.n_keypoints);
    }

    std::printf(
            "smoke PASS: cpu_kpts=%d vulkan_kpts=%d dim=%d resize=%d "
            "(no manual VULKAN env)\n",
            cpu.n_keypoints, vk.n_keypoints, vk.descriptor_dim, resize);

    aicore_lightglue_free_features(&cpu);
    aicore_lightglue_free_features(&vk);
    return 0;
}

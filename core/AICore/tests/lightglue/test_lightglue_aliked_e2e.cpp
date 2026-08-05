// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Model-backed qLightGlue core path: two ALIKED extracts followed by one
// LightGlue match. The default "auto" device reproduces the GUI contract.

#include <QImage>
#include <QImageReader>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/aliked_capi.h"
#include "aicore/lightglue_capi.h"

namespace {

struct RgbImage {
    std::vector<uint8_t> pixels;
    int32_t width = 0;
    int32_t height = 0;
};

bool LoadRgb(const char* path, int max_resize, RgbImage* out) {
    if (path == nullptr || out == nullptr) return false;
    QImageReader reader(QString::fromUtf8(path));
    reader.setAutoTransform(true);
    const QImage input = reader.read();
    if (input.isNull()) return false;
    QImage rgb = input.convertToFormat(QImage::Format_RGB888);
    // Match qLightGlue's production path exactly.  It applies EXIF rotation
    // and Qt's smooth resize before passing bytes to the ALIKED C API.
    if (max_resize > 0) {
        const int max_dim = std::max(rgb.width(), rgb.height());
        if (max_dim > max_resize) {
            rgb = rgb.scaled(max_resize, max_resize, Qt::KeepAspectRatio,
                             Qt::SmoothTransformation);
        }
    }
    out->width = rgb.width();
    out->height = rgb.height();
    out->pixels.resize(static_cast<size_t>(out->width) * out->height * 3);
    for (int y = 0; y < rgb.height(); ++y) {
        std::memcpy(
                out->pixels.data() + static_cast<size_t>(y) * out->width * 3,
                rgb.constScanLine(y), static_cast<size_t>(out->width) * 3);
    }
    return true;
}

bool Extract(const char* model,
             const char* device,
             const RgbImage& image,
             int max_keypoints,
             int resize,
             aicore_lightglue_features* out,
             std::string* error) {
    aicore_aliked_options* options = aicore_aliked_options_new();
    aicore_aliked_options_set_device(options, device);
    aicore_aliked_options_set_max_keypoints(options, max_keypoints);
    aicore_aliked_options_set_resize_long_edge(options, resize);
    aicore_aliked_ctx* ctx = aicore_aliked_load_opts(model, options);
    aicore_aliked_options_free(options);
    if (ctx == nullptr) {
        *error = "failed to allocate ALIKED context";
        return false;
    }
    const int rc =
            aicore_aliked_extract_rgb(ctx, image.pixels.data(), image.width,
                                      image.height, image.width * 3, out);
    if (rc != 0) *error = aicore_aliked_last_error(ctx);
    aicore_aliked_free(ctx);
    return rc == 0 && out->n_keypoints > 0 && out->descriptor_dim > 0;
}

}  // namespace

int main(int argc, char** argv) {
    const char* matcher = std::getenv("AICORE_TEST_LIGHTGLUE_GGUF");
    const char* extractor = std::getenv("AICORE_TEST_ALIKED_GGUF");
    const char* image0 = std::getenv("AICORE_TEST_LIGHTGLUE_IMAGE0");
    const char* image1 = std::getenv("AICORE_TEST_LIGHTGLUE_IMAGE1");
    const char* device = std::getenv("AICORE_TEST_DEVICE");
    if (argc >= 5) {
        matcher = argv[1];
        extractor = argv[2];
        image0 = argv[3];
        image1 = argv[4];
    }
    if (argc >= 6) device = argv[5];
    if (device == nullptr || device[0] == '\0') device = "auto";
    const int max_keypoints = argc >= 7 ? std::atoi(argv[6]) : 256;
    const int resize = argc >= 8 ? std::atoi(argv[7]) : 512;
    if (matcher == nullptr || extractor == nullptr || image0 == nullptr ||
        image1 == nullptr) {
        std::fprintf(stderr, "SKIP: matcher/extractor/two images required\n");
        return 77;
    }

    RgbImage rgb0;
    RgbImage rgb1;
    if (!LoadRgb(image0, resize, &rgb0) || !LoadRgb(image1, resize, &rgb1)) {
        std::fprintf(stderr, "FAIL: unable to load input images\n");
        return 1;
    }

    aicore_lightglue_features features0{};
    aicore_lightglue_features features1{};

    // qLightGlue creates the matcher first to resolve the selected device,
    // then creates one ALIKED context per image. Keep this order here: a
    // reverse-order test misses shared-backend lifetime regressions.
    aicore_lightglue_options* options = aicore_lightglue_options_new();
    aicore_lightglue_options_set_device(options, device);
    aicore_lightglue_options_set_matcher_type(options, 2);
    aicore_lightglue_ctx* ctx = aicore_lightglue_load_opts(matcher, options);
    aicore_lightglue_options_free(options);
    const char* load_error = aicore_lightglue_last_error(ctx);
    aicore_lightglue_geometry geometry{};
    if (ctx == nullptr || load_error != nullptr ||
        aicore_lightglue_geometry_of(ctx, &geometry) != 0) {
        std::fprintf(stderr, "FAIL: LightGlue load (%s): %s\n", device,
                     load_error ? load_error : "matcher is not ready");
        aicore_lightglue_free_features(&features0);
        aicore_lightglue_free_features(&features1);
        aicore_lightglue_free(ctx);
        return 1;
    }

    std::string error;
    if (!Extract(extractor, device, rgb0, max_keypoints, resize, &features0,
                 &error) ||
        !Extract(extractor, device, rgb1, max_keypoints, resize, &features1,
                 &error)) {
        std::fprintf(stderr, "FAIL: ALIKED extraction (%s): %s\n", device,
                     error.c_str());
        aicore_lightglue_free_features(&features0);
        aicore_lightglue_free_features(&features1);
        aicore_lightglue_free(ctx);
        return 1;
    }
    std::printf("ALIKED extracted: device=%s features=%d/%d dim=%d/%d\n",
                device, features0.n_keypoints, features1.n_keypoints,
                features0.descriptor_dim, features1.descriptor_dim);

    aicore_lightglue_match* matches = nullptr;
    int32_t match_count = 0;
    const int rc = aicore_lightglue_run_match(ctx, &features0, &features1,
                                              &matches, &match_count);
    if (rc != 0 || match_count <= 0) {
        std::fprintf(stderr, "FAIL: LightGlue match (%s): count=%d error=%s\n",
                     device, match_count, aicore_lightglue_last_error(ctx));
        aicore_lightglue_free_matches(matches);
        aicore_lightglue_free_features(&features0);
        aicore_lightglue_free_features(&features1);
        aicore_lightglue_free(ctx);
        return 1;
    }

    std::printf(
            "qLightGlue core e2e PASS: device=%s features=%d/%d matches=%d "
            "resize=%d\n",
            device, features0.n_keypoints, features1.n_keypoints, match_count,
            resize);
    aicore_lightglue_free_matches(matches);
    aicore_lightglue_free_features(&features0);
    aicore_lightglue_free_features(&features1);
    aicore_lightglue_free(ctx);
    return 0;
}

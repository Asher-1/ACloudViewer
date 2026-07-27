// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QImage>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "aicore/deeplsd_capi.h"
#include "common/test_macros.hpp"

static int failures = 0;

int main() {
    const char* gguf = std::getenv("AICORE_TEST_DEEPLSD_GGUF");
    if (!gguf || gguf[0] == '\0') return 77;
    const char* image_path = std::getenv("AICORE_TEST_DEEPLSD_IMAGE");
    if (!image_path || image_path[0] == '\0') return 77;
    const char* device = std::getenv("AICORE_TEST_DEVICE");
    if (!device || device[0] == '\0') device = "cpu";

    QImage img(QString::fromUtf8(image_path));
    if (img.isNull()) {
        std::fprintf(stderr, "failed to load image: %s\n", image_path);
        return 1;
    }
    QImage gray = img.convertToFormat(QImage::Format_Grayscale8);

    aicore_deeplsd_options* opts = aicore_deeplsd_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_deeplsd_options_set_device(opts, device);
    aicore_deeplsd_options_set_threads(opts, 0);
    aicore_deeplsd_ctx* ctx = aicore_deeplsd_load_opts(gguf, opts);
    aicore_deeplsd_options_free(opts);
    AICORE_CHECK(ctx != nullptr);
    if (!ctx) return 1;

    float* df = nullptr;
    float* ang = nullptr;
    int32_t ow = 0;
    int32_t oh = 0;
    AICORE_CHECK(aicore_deeplsd_extract_gray(
                         ctx, gray.constBits(), gray.width(), gray.height(),
                         static_cast<int32_t>(gray.bytesPerLine()), &df, &ang,
                         &ow, &oh) == 0);
    AICORE_CHECK(df != nullptr && ang != nullptr && ow > 0 && oh > 0);

    char* json = aicore_deeplsd_info_json(ctx);
    AICORE_CHECK(json != nullptr && std::strstr(json, "deeplsd") != nullptr);
    aicore_deeplsd_free_string(json);

    std::free(df);
    std::free(ang);
    aicore_deeplsd_free(ctx);
    std::fprintf(stderr, "deeplsd load ok: %s device=%s %dx%d\n", gguf, device,
                 ow, oh);
    return failures == 0 ? 0 : 1;
}

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

#include "aicore/eloftr_capi.h"
#include "common/test_macros.hpp"

static int failures = 0;

int main() {
    const char* gguf = std::getenv("AICORE_TEST_ELOFTR_GGUF");
    if (!gguf || gguf[0] == '\0') return 77;
    const char* image_path = std::getenv("AICORE_TEST_ELOFTR_IMAGE");
    if (!image_path || image_path[0] == '\0') return 77;
    const char* device = std::getenv("AICORE_TEST_DEVICE");
    if (!device || device[0] == '\0') device = "cpu";

    QImage img(QString::fromUtf8(image_path));
    if (img.isNull()) {
        std::fprintf(stderr, "failed to load image: %s\n", image_path);
        return 1;
    }
    QImage gray = img.convertToFormat(QImage::Format_Grayscale8);
    gray = gray.scaled(640, 640, Qt::IgnoreAspectRatio,
                       Qt::SmoothTransformation);

    aicore_eloftr_options* opts = aicore_eloftr_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_eloftr_options_set_device(opts, device);
    aicore_eloftr_options_set_threads(opts, 0);
    aicore_eloftr_ctx* ctx = aicore_eloftr_load_opts(gguf, opts);
    aicore_eloftr_options_free(opts);
    AICORE_CHECK(ctx != nullptr);
    if (!ctx) return 1;

    aicore_eloftr_match* matches = nullptr;
    int32_t count = 0;
    AICORE_CHECK(
            aicore_eloftr_match_gray(ctx, gray.constBits(), gray.constBits(),
                                     gray.width(), gray.height(),
                                     static_cast<int32_t>(gray.bytesPerLine()),
                                     &matches, &count) == 0);

    char* json = aicore_eloftr_info_json(ctx);
    AICORE_CHECK(json != nullptr && std::strstr(json, "eloftr") != nullptr);
    aicore_eloftr_free_string(json);

    aicore_eloftr_free_matches(matches);
    aicore_eloftr_free(ctx);
    std::fprintf(stderr, "eloftr load ok: %s device=%s matches=%d\n", gguf,
                 device, count);
    return failures == 0 ? 0 : 1;
}

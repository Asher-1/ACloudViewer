// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// C API plumbing of aicore_yolo_options_{set,get}_mask_upsample. No GGUF
// assets needed. The compose_masks behaviour itself is covered by
// test_yolo_mask_upsample.cpp (whitebox).

#include <cstdio>

#include "aicore/yolo_capi.h"

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, \
                         #cond);                                         \
            return 1;                                                    \
        }                                                                \
    } while (0)

int main() {
    CHECK(aicore_yolo_options_get_mask_upsample(nullptr) == 0);
    aicore_yolo_options_set_mask_upsample(nullptr, 1);  // must not crash
    aicore_yolo_options* opts = aicore_yolo_options_new();
    CHECK(opts != nullptr);
    CHECK(aicore_yolo_options_get_mask_upsample(opts) == 0);  // default
    aicore_yolo_options_set_mask_upsample(opts, 1);
    CHECK(aicore_yolo_options_get_mask_upsample(opts) == 1);
    aicore_yolo_options_set_mask_upsample(opts, 7);  // any non-zero
    CHECK(aicore_yolo_options_get_mask_upsample(opts) == 1);
    aicore_yolo_options_set_mask_upsample(opts, 0);
    CHECK(aicore_yolo_options_get_mask_upsample(opts) == 0);
    aicore_yolo_options_free(opts);
    std::printf("PASS test_yolo_mask_upsample_capi\n");
    return 0;
}

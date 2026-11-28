// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// C API plumbing of aicore_yolo_options_{set,get}_mask_upsample. No GGUF
// assets are needed. compose_masks behavior is covered by the white-box test.

#include <gtest/gtest.h>

#include "aicore/yolo_capi.h"

TEST(YoloMaskUpsampleCApi, NullOptionsUseDefaultAndSettersAreSafe) {
    EXPECT_EQ(aicore_yolo_options_get_mask_upsample(nullptr), 0);
    aicore_yolo_options_set_mask_upsample(nullptr, 1);
}

TEST(YoloMaskUpsampleCApi, NewOptionsDefaultToDisabled) {
    aicore_yolo_options* opts = aicore_yolo_options_new();
    ASSERT_NE(opts, nullptr);
    EXPECT_EQ(aicore_yolo_options_get_mask_upsample(opts), 0);
    aicore_yolo_options_free(opts);
}

TEST(YoloMaskUpsampleCApi, AnyNonzeroValueEnablesAndZeroDisables) {
    aicore_yolo_options* opts = aicore_yolo_options_new();
    ASSERT_NE(opts, nullptr);
    aicore_yolo_options_set_mask_upsample(opts, 1);
    EXPECT_EQ(aicore_yolo_options_get_mask_upsample(opts), 1);
    aicore_yolo_options_set_mask_upsample(opts, 7);
    EXPECT_EQ(aicore_yolo_options_get_mask_upsample(opts), 1);
    aicore_yolo_options_set_mask_upsample(opts, 0);
    EXPECT_EQ(aicore_yolo_options_get_mask_upsample(opts), 0);
    aicore_yolo_options_free(opts);
}

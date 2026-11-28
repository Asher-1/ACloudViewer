// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstdint>

#include "aicore/image_view.h"
#include "common/image_utils.hpp"

static_assert(AICORE_IMAGE_RGB8 == 1);
static_assert(AICORE_IMAGE_RGBA8 == 2);
static_assert(AICORE_IMAGE_GRAY8 == 3);
static_assert(AICORE_IMAGE_BGR8 == 4);
static_assert(AICORE_IMAGE_BGRA8 == 5);

TEST(ImageViewContract, BorrowsPaddedRgbRowsWithoutCopying) {
    const uint8_t pixels[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  99, 99, 99,
                              10, 11, 12, 13, 14, 15, 16, 17, 18, 88, 88, 88};
    aicore_image_view view{pixels, 3, 2, 12, AICORE_IMAGE_RGB8};
    aicore::img::Image out;
    ASSERT_TRUE(aicore::img::image_from_view(view, out));
    EXPECT_EQ(out.width, 3);
    EXPECT_EQ(out.height, 2);
    EXPECT_TRUE(out.rgb.empty());
    EXPECT_EQ(out.borrowed_data, pixels);
    EXPECT_EQ(out.stride(), 12);
    EXPECT_EQ(out.channels, 3);
    EXPECT_FALSE(out.bgr);
    EXPECT_EQ(out.channel(0, 1, 0), 10);
    EXPECT_EQ(out.channel(2, 1, 2), 18);
}

TEST(ImageViewContract, BorrowsBgrAndExposesRgbChannelOrder) {
    const uint8_t pixels[] = {3, 2, 1, 6, 5, 4};
    aicore_image_view view{pixels, 2, 1, 6, AICORE_IMAGE_BGR8};
    aicore::img::Image out;
    ASSERT_TRUE(aicore::img::image_from_view(view, out));
    EXPECT_TRUE(out.rgb.empty());
    EXPECT_EQ(out.borrowed_data, pixels);
    EXPECT_EQ(out.channels, 3);
    EXPECT_TRUE(out.bgr);
    EXPECT_EQ(out.channel(0, 0, 0), 1);
    EXPECT_EQ(out.channel(1, 0, 2), 6);
}

TEST(ImageViewContract, RejectsShortRowsAndUnknownFormats) {
    const uint8_t pixels[24] = {};
    aicore_image_view view{pixels, 3, 2, 8, AICORE_IMAGE_RGB8};
    aicore::img::Image out;
    EXPECT_FALSE(aicore::img::image_from_view(view, out));
    view.row_stride_bytes = 12;
    view.format = static_cast<aicore_image_format>(99);
    EXPECT_FALSE(aicore::img::image_from_view(view, out));
}

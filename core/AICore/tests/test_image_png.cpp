// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QImage>
#include <limits>

#include "common/image_utils.hpp"

namespace {
// Rows are top-first and padded. Transparent RGB must survive encoding.
constexpr uint8_t kRgba[] = {10, 20, 30, 0,   40,  50,  60,  128, 9, 9, 9, 9,
                             70, 80, 90, 255, 100, 110, 120, 1,   9, 9, 9, 9};
}  // namespace

TEST(AicorePng, PreservesRgbaAlphaPaddedStrideAndTopFirstRows) {
    const aicore_image_view view{kRgba, 2, 2, 12, AICORE_IMAGE_RGBA8};
    std::vector<uint8_t> png;
    ASSERT_TRUE(aicore::img::encode_png(view, png));
    const QImage decoded = QImage::fromData(png.data(), int(png.size()), "PNG");
    ASSERT_EQ(decoded.width(), 2);
    ASSERT_EQ(decoded.height(), 2);
    const QColor expected[] = {QColor(10, 20, 30, 0), QColor(40, 50, 60, 128),
                               QColor(70, 80, 90, 255),
                               QColor(100, 110, 120, 1)};
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            SCOPED_TRACE(::testing::Message() << "pixel " << x << "," << y);
            EXPECT_EQ(decoded.pixelColor(x, y), expected[y * 2 + x]);
        }
    }
}

TEST(AicorePng, RejectsInvalidViewsWithoutChangingOutput) {
    const aicore_image_view view{kRgba, 2, 2, 12, AICORE_IMAGE_RGBA8};
    std::vector<uint8_t> png;
    ASSERT_TRUE(aicore::img::encode_png(view, png));
    const auto unchanged = png;
    int index = 0;
    for (const aicore_image_view invalid :
         {aicore_image_view{nullptr, 2, 2, 12, AICORE_IMAGE_RGBA8},
          aicore_image_view{kRgba, 0, 2, 12, AICORE_IMAGE_RGBA8},
          aicore_image_view{kRgba, 2, 2, 7, AICORE_IMAGE_RGBA8},
          aicore_image_view{kRgba, 2, 2, std::numeric_limits<size_t>::max(),
                            AICORE_IMAGE_RGBA8}}) {
        SCOPED_TRACE(::testing::Message() << "invalid view " << index++);
        EXPECT_FALSE(aicore::img::encode_png(invalid, png));
        EXPECT_EQ(png, unchanged);
    }
}

TEST(AicorePng, ConvertsBgrToRgb) {
    const uint8_t bgr[] = {30, 20, 10, 60, 50, 40};
    const aicore_image_view view{bgr, 2, 1, 6, AICORE_IMAGE_BGR8};
    std::vector<uint8_t> png;
    ASSERT_TRUE(aicore::img::encode_png(view, png));
    const QImage decoded = QImage::fromData(png.data(), int(png.size()), "PNG");
    ASSERT_EQ(decoded.width(), 2);
    ASSERT_EQ(decoded.height(), 1);
    EXPECT_EQ(decoded.pixelColor(0, 0), QColor(10, 20, 30));
    EXPECT_EQ(decoded.pixelColor(1, 0), QColor(40, 50, 60));
}

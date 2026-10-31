// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QImage>
#include <cstdio>
#include <limits>

#include "common/image_utils.hpp"

int main() {
    // Rows are top-first and padded. Transparent RGB must survive encoding.
    const uint8_t rgba[] = {10, 20, 30, 0,   40,  50,  60,  128, 9, 9, 9, 9,
                            70, 80, 90, 255, 100, 110, 120, 1,   9, 9, 9, 9};
    aicore_image_view view{rgba, 2, 2, 12, AICORE_IMAGE_RGBA8};
    std::vector<uint8_t> png;
    if (!aicore::img::encode_png(view, png)) return 1;
    const QImage decoded = QImage::fromData(png.data(), int(png.size()), "PNG");
    if (decoded.size() != QSize(2, 2)) return 1;
    const QColor expected[] = {QColor(10, 20, 30, 0), QColor(40, 50, 60, 128),
                               QColor(70, 80, 90, 255),
                               QColor(100, 110, 120, 1)};
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x)
            if (decoded.pixelColor(x, y) != expected[y * 2 + x]) return 1;
    const auto unchanged = png;
    for (const aicore_image_view invalid :
         {aicore_image_view{nullptr, 2, 2, 12, AICORE_IMAGE_RGBA8},
          aicore_image_view{rgba, 0, 2, 12, AICORE_IMAGE_RGBA8},
          aicore_image_view{rgba, 2, 2, 7, AICORE_IMAGE_RGBA8},
          aicore_image_view{rgba, 2, 2, std::numeric_limits<size_t>::max(),
                            AICORE_IMAGE_RGBA8}}) {
        if (aicore::img::encode_png(invalid, png) || png != unchanged) return 1;
    }
    const uint8_t bgr[] = {30, 20, 10, 60, 50, 40};
    view = {bgr, 2, 1, 6, AICORE_IMAGE_BGR8};
    if (!aicore::img::encode_png(view, png)) return 1;
    const QImage rgb = QImage::fromData(png.data(), int(png.size()), "PNG");
    if (rgb.pixelColor(0, 0) != QColor(10, 20, 30) ||
        rgb.pixelColor(1, 0) != QColor(40, 50, 60))
        return 1;
    std::puts(
            "PASS: shared PNG preserves RGBA, alpha, top-first rows, stride "
            "and rejects invalid views");
    return 0;
}

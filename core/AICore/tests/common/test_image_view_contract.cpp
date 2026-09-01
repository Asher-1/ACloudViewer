// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdint>

#include "aicore/image_view.h"
#include "tasks/facedetect/image_io.hpp"

#define CHECK(expr)          \
    do {                     \
        if (!(expr)) {       \
            return __LINE__; \
        }                    \
    } while (false)

int main() {
    static_assert(AICORE_IMAGE_RGB8 == 1);
    static_assert(AICORE_IMAGE_RGBA8 == 2);
    static_assert(AICORE_IMAGE_GRAY8 == 3);
    static_assert(AICORE_IMAGE_BGR8 == 4);
    static_assert(AICORE_IMAGE_BGRA8 == 5);

    // Odd width plus explicit row padding exercises the Qt QImage case.
    const uint8_t pixels[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  99, 99, 99,
                              10, 11, 12, 13, 14, 15, 16, 17, 18, 88, 88, 88};
    aicore_image_view view{pixels, 3, 2, 12, AICORE_IMAGE_RGB8};
    fd::Image out;
    CHECK(fd::image_from_view(view, out));
    CHECK(out.width == 3 && out.height == 2);
    CHECK(out.rgb.empty());
    CHECK(out.borrowed_data == pixels);
    CHECK(out.stride() == 12);
    CHECK(out.channels == 3 && !out.bgr);
    CHECK(out.channel(0, 1, 0) == 10);
    CHECK(out.channel(2, 1, 2) == 18);

    const uint8_t bgr[] = {3, 2, 1, 6, 5, 4};
    aicore_image_view bgr_view{bgr, 2, 1, 6, AICORE_IMAGE_BGR8};
    CHECK(fd::image_from_view(bgr_view, out));
    CHECK(out.rgb.empty() && out.borrowed_data == bgr);
    CHECK(out.channels == 3 && out.bgr);
    CHECK(out.channel(0, 0, 0) == 1);
    CHECK(out.channel(1, 0, 2) == 6);

    aicore_image_view bad = view;
    bad.row_stride_bytes = 8;
    CHECK(!fd::image_from_view(bad, out));
    bad = view;
    bad.format = static_cast<aicore_image_format>(99);
    CHECK(!fd::image_from_view(bad, out));
    return 0;
}

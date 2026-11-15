// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/facedetect/preprocess.hpp"

namespace fd {

std::vector<float> to_blob(
        const Image& img, int size, float mean, float std, bool swap_rb) {
    // The caller resizes/aligns the image to `size`x`size` before calling, so
    // in production img.width == img.height == size. We index by the image's
    // own dimensions so the NCHW plane stride is exactly width*height; this
    // keeps the layout correct even for non-square inputs (e.g. the unit test's
    // 2x1).
    (void)size;
    const int w = img.width, h = img.height;
    const size_t hw = (size_t)w * h;
    std::vector<float> blob(3 * hw);
    const float inv = 1.0f / std;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            // swap_rb on an RGB image -> output plane order B,G,R.
            const float r = static_cast<float>(img.channel(x, y, 0));
            const float g = static_cast<float>(img.channel(x, y, 1));
            const float b = static_cast<float>(img.channel(x, y, 2));
            float c0 = swap_rb ? b : r, c1 = g, c2 = swap_rb ? r : b;
            const size_t idx = (size_t)y * w + x;
            blob[0 * hw + idx] = (c0 - mean) * inv;
            blob[1 * hw + idx] = (c1 - mean) * inv;
            blob[2 * hw + idx] = (c2 - mean) * inv;
        }
    return blob;
}

}  // namespace fd

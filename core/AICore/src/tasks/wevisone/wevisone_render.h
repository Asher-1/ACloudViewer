// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "aicore/image_view.h"

namespace wwv {

// Draws red width-3 rectangles for the boxes and, when `matte` is non-null
// (row-major w*h floats in [0, 1]), composites the matte in green, then
// writes a PNG through the shared AICore Qt encoder.
bool render_result_png(const aicore_image_view& img,
                       const float* boxes,
                       int32_t n_boxes,
                       const float* matte,
                       int32_t matte_w,
                       int32_t matte_h,
                       const std::string& out_png_path);

}  // namespace wwv

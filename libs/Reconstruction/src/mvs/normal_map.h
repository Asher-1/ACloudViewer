// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

#include "mvs/mat.h"
#include "util/bitmap.h"

namespace colmap {
namespace mvs {

// Normal map class that stores per-pixel normals as a MxNx3 image.
class NormalMap : public Mat<float> {
public:
    NormalMap();
    NormalMap(const size_t width, const size_t height);
    explicit NormalMap(const Mat<float>& mat);

    void Rescale(const float factor);
    void Downsize(const size_t max_width, const size_t max_height);

    Bitmap ToBitmap() const;
};

}  // namespace mvs
}  // namespace colmap

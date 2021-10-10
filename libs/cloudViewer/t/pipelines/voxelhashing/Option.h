// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include "core/Tensor.h"
#include "t/geometry/Image.h"
#include "t/geometry/RGBDImage.h"
#include "t/geometry/TSDFVoxelGrid.h"
#include "t/pipelines/odometry/RGBDOdometry.h"
#include "t/pipelines/voxelhashing/Frame.h"

namespace cloudViewer {
namespace t {
namespace pipelines {
namespace voxelhashing {

struct Option {
    Option() {}

    /// TSDF VoxelBlock options
    float voxel_size = 3.0 / 512.0;
    int est_block_count = 40000;

    /// Input options
    float depth_scale = 1000.0f;
    float depth_max = 3.0f;
    float depth_diff = 0.07f;
};

}  // namespace voxelhashing
}  // namespace pipelines
}  // namespace t
}  // namespace cloudViewer

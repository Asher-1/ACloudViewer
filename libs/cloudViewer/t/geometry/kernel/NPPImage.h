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

#ifdef BUILD_CUDA_MODULE
#include "core/Dtype.h"
#include "core/Tensor.h"
#include "t/geometry/Image.h"

namespace cloudViewer {
namespace t {
namespace geometry {
namespace npp {

void RGBToGray(const core::Tensor &src_im, core::Tensor &dst_im);

void Dilate(const cloudViewer::core::Tensor &srcim,
            cloudViewer::core::Tensor &dstim,
            int kernel_size);

void Resize(const cloudViewer::core::Tensor &srcim,
            cloudViewer::core::Tensor &dstim,
            t::geometry::Image::InterpType interp_type);

void Filter(const cloudViewer::core::Tensor &srcim,
            cloudViewer::core::Tensor &dstim,
            const cloudViewer::core::Tensor &kernel);

void FilterBilateral(const cloudViewer::core::Tensor &srcim,
                     cloudViewer::core::Tensor &dstim,
                     int kernel_size,
                     float value_sigma,
                     float distance_sigma);

void FilterGaussian(const cloudViewer::core::Tensor &srcim,
                    cloudViewer::core::Tensor &dstim,
                    int kernel_size,
                    float sigma);

void FilterSobel(const cloudViewer::core::Tensor &srcim,
                 cloudViewer::core::Tensor &dstim_dx,
                 cloudViewer::core::Tensor &dstim_dy,
                 int kernel_size);
}  // namespace npp
}  // namespace geometry
}  // namespace t
}  // namespace cloudViewer

#endif  // BUILD_CUDA_MODULE

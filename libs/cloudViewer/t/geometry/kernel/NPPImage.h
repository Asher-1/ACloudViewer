// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#pragma once

#ifdef BUILD_CUDA_MODULE
#include "cloudViewer/core/Dtype.h"
#include "cloudViewer/core/Tensor.h"
#include "cloudViewer/t/geometry/Image.h"

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

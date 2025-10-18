// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RGBDImage.h"

#include "ecvBBox.h"

namespace cloudViewer {
namespace geometry {

RGBDImage &RGBDImage::Clear() {
    color_.Clear();
    depth_.Clear();
    return *this;
}

ccBBox RGBDImage::getOwnBB(bool withGLFeatures) {
    return GetAxisAlignedBoundingBox();
}

Eigen::Vector2d RGBDImage::GetMin2DBound() const {
    return Eigen::Vector2d(0.0, 0.0);
}

Eigen::Vector2d RGBDImage::GetMax2DBound() const {
    return Eigen::Vector2d(color_.width_ + depth_.width_, color_.height_);
}

RGBDImagePyramid RGBDImage::FilterPyramid(
        const RGBDImagePyramid &rgbd_image_pyramid, Image::FilterType type) {
    RGBDImagePyramid rgbd_image_pyramid_filtered;
    rgbd_image_pyramid_filtered.clear();
    int num_of_levels = (int)rgbd_image_pyramid.size();
    for (int level = 0; level < num_of_levels; level++) {
        auto color_level = rgbd_image_pyramid[level]->color_;
        auto depth_level = rgbd_image_pyramid[level]->depth_;
        auto color_level_filtered = color_level.Filter(type);
        auto depth_level_filtered = depth_level.Filter(type);
        auto rgbd_image_level_filtered = cloudViewer::make_shared<RGBDImage>(
                RGBDImage(*color_level_filtered, *depth_level_filtered));
        rgbd_image_pyramid_filtered.push_back(rgbd_image_level_filtered);
    }
    return rgbd_image_pyramid_filtered;
}

RGBDImagePyramid RGBDImage::CreatePyramid(
        size_t num_of_levels,
        bool with_gaussian_filter_for_color /* = true */,
        bool with_gaussian_filter_for_depth /* = false */) const {
    ImagePyramid color_pyramid =
            color_.CreatePyramid(num_of_levels, with_gaussian_filter_for_color);
    ImagePyramid depth_pyramid =
            depth_.CreatePyramid(num_of_levels, with_gaussian_filter_for_depth);
    RGBDImagePyramid rgbd_image_pyramid;
    rgbd_image_pyramid.clear();
    for (size_t level = 0; level < num_of_levels; level++) {
        auto rgbd_image_level = cloudViewer::make_shared<RGBDImage>(
                RGBDImage(*color_pyramid[level], *depth_pyramid[level]));
        rgbd_image_pyramid.push_back(rgbd_image_level);
    }
    return rgbd_image_pyramid;
}

}  // namespace geometry
}  // namespace cloudViewer

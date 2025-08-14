// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <vector>

#include "camera/PinholeCameraTrajectory.h"
#include <Image.h>
#include <ecvMesh.h>
#include <RGBDImage.h>
#include "pipelines/color_map/ImageWarpingField.h"
#include <Eigen.h>
#include <Optional.h>

namespace cloudViewer {
namespace pipelines {
namespace color_map {

std::tuple<std::vector<geometry::Image>,
           std::vector<geometry::Image>,
           std::vector<geometry::Image>,
           std::vector<geometry::Image>,
           std::vector<geometry::Image>>
CreateUtilImagesFromRGBD(const std::vector<geometry::RGBDImage>& images_rgbd);

std::vector<geometry::Image> CreateDepthBoundaryMasks(
        const std::vector<geometry::Image>& images_depth,
        double depth_threshold_for_discontinuity_check,
        int half_dilation_kernel_size_for_discontinuity_map);

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
CreateVertexAndImageVisibility(
        const ccMesh& mesh,
        const std::vector<geometry::Image>& images_depth,
        const std::vector<geometry::Image>& images_mask,
        const camera::PinholeCameraTrajectory& camera_trajectory,
        double maximum_allowable_depth,
        double depth_threshold_for_visibility_check);

void SetProxyIntensityForVertex(
        const ccMesh& mesh,
        const std::vector<geometry::Image>& images_gray,
        const cloudViewer::utility::optional<std::vector<ImageWarpingField>>& warping_fields,
        const camera::PinholeCameraTrajectory& camera_trajectory,
        const std::vector<std::vector<int>>& visibility_vertex_to_image,
        std::vector<double>& proxy_intensity,
        int image_boundary_margin);

void SetGeometryColorAverage(
        ccMesh& mesh,
        const std::vector<geometry::Image>& images_color,
        const cloudViewer::utility::optional<std::vector<ImageWarpingField>>& warping_fields,
        const camera::PinholeCameraTrajectory& camera_trajectory,
        const std::vector<std::vector<int>>& visibility_vertex_to_image,
        int image_boundary_margin = 10,
        int invisible_vertex_color_knn = 3);

}  // namespace color_map
}  // namespace pipelines
}  // namespace cloudViewer

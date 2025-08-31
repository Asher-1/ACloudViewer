// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <iostream>
#include <tuple>
#include <vector>

#include "camera/PinholeCameraIntrinsic.h"
#include "pipelines/odometry/OdometryOption.h"
#include "pipelines/odometry/RGBDOdometryJacobian.h"

#include <Eigen.h>
#include <Logging.h>

namespace cloudViewer {

namespace geometry {
class RGBDImage;
}

namespace pipelines {
namespace odometry {

/// \brief Function to estimate 6D rigid motion from two RGBD image pairs.
///
/// \param source Source RGBD image.
/// \param target Target RGBD image.
/// \param pinhole_camera_intrinsic Camera intrinsic parameters.
/// \param odo_init Initial 4x4 motion matrix estimation.
/// \param jacobin_method The odometry Jacobian method to use.
/// \param option Odometry hyper parameteres.
/// \return is_success, 4x4 motion matrix, 6x6 information matrix.
std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> 
ComputeRGBDOdometry(
        const geometry::RGBDImage &source,
        const geometry::RGBDImage &target,
        const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic =
                camera::PinholeCameraIntrinsic(),
        const Eigen::Matrix4d &odo_init = Eigen::Matrix4d::Identity(),
        const RGBDOdometryJacobian &jacobian_method =
                RGBDOdometryJacobianFromHybridTerm(),
        const OdometryOption &option = OdometryOption());

}  // namespace odometry
}  // namespace pipelines
}  // namespace cloudViewer

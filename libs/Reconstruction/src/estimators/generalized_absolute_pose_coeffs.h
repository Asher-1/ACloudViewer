// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_ESTIMATORS_GENERALIZED_ABSOLUTE_POSE_COEFFS_H_
#define COLMAP_SRC_ESTIMATORS_GENERALIZED_ABSOLUTE_POSE_COEFFS_H_

#include <Eigen/Core>

namespace colmap {

Eigen::Matrix<double, 9, 1> ComputeDepthsSylvesterCoeffs(
        const Eigen::Matrix<double, 3, 6>& K);

}  // namespace colmap

#endif  // COLMAP_SRC_ESTIMATORS_GENERALIZED_ABSOLUTE_POSE_COEFFS_H_

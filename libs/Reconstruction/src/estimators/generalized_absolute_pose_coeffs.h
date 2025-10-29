// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>

namespace colmap {

Eigen::Matrix<double, 9, 1> ComputeDepthsSylvesterCoeffs(
        const Eigen::Matrix<double, 3, 6>& K);

}  // namespace colmap

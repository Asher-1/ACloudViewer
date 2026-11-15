// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Geometry>
#include <vector>

namespace colmap {

// Divide a bounding box into equal-sized sub-boxes.
//
// @param bbox    The bounding box to divide.
// @param split   Number of splits along each axis (x, y, z).
//
// @return        Vector of sub-boxes covering the original box.
std::vector<Eigen::AlignedBox3d> ComputeEqualPartsBboxes(
        const Eigen::AlignedBox3d& bbox, const Eigen::Vector3i& split);

}  // namespace colmap

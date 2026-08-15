// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <vector>

#include "BevRemapBackend.h"

namespace mcalib {

struct LidarProjResult {
    std::vector<cv::Point2f> image_points;
    std::vector<float> depths;
};

struct KannalaBrandtCoeffs {
    double k1 = 0;
    double k2 = 0;
    double k3 = 0;
    double k4 = 0;
};

/// GPU-accelerated point transform + camera projection (pinhole or
/// Kannala-Brandt).
class LidarProjBackend {
public:
    static bool projectPoints(
            BevRemapMode mode,
            const std::vector<Eigen::Vector3f>& points_sensing,
            const Eigen::Matrix3d& rotation,
            const Eigen::Vector3d& translation,
            double fx,
            double fy,
            double cx,
            double cy,
            LidarProjResult& out);

    static bool projectPointsKb(
            BevRemapMode mode,
            const std::vector<Eigen::Vector3f>& points_sensing,
            const Eigen::Matrix3d& rotation,
            const Eigen::Vector3d& translation,
            double fx,
            double fy,
            double cx,
            double cy,
            const KannalaBrandtCoeffs& kb,
            LidarProjResult& out);
};

}  // namespace mcalib

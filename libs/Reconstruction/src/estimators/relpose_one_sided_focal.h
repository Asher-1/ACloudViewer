// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <vector>

#include "base/pose.h"

namespace colmap {

// Minimal six-point relative pose estimator with an unknown focal length in
// the first (pixel) view and calibrated rays in the second view. PoseLib is
// optional; this header is only compiled into the reconstruction target when
// RECONSTRUCTION_FETCH_POSELIB is enabled.
class RelativePoseOneSidedFocalEstimator {
public:
    using X_t = Eigen::Vector2d;
    // Keep the calibrated bearing and d(bearing)/d(pixel) together so the
    // RANSAC residual is measured in pixels for distorted central cameras.
    using Y_t = CamRayWithJac;
    struct M_t {
        Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
        double focal = 0.0;
    };
    static const int kMinNumSamples = 6;

    static std::vector<M_t> Estimate(const std::vector<X_t>& points1,
                                     const std::vector<Y_t>& points2);
    // Refine an initial six-point hypothesis with the same pixel-unit tangent
    // Sampson residual used by RANSAC. Returns false without modifying model
    // when the hypothesis is degenerate.
    static bool Refine(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       M_t* model);
    static void Residuals(const std::vector<X_t>& points1,
                          const std::vector<Y_t>& points2,
                          const M_t& model,
                          std::vector<double>* residuals);
};

}  // namespace colmap

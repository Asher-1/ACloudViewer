// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <vector>

#include "optim/ransac.h"
#include "util/eigen_alignment.h"
#include "util/types.h"

namespace colmap {

class AffineTransformEstimator {
public:
    using X_t = Eigen::Vector2d;
    using Y_t = Eigen::Vector2d;
    using M_t = Eigen::Matrix2x3d;

    // The minimum number of samples needed to estimate a model.
    static const int kMinNumSamples = 3;

    // Estimate the affine transformation from at least 3 correspondences.
    static void Estimate(const std::vector<X_t>& src,
                         const std::vector<Y_t>& tgt,
                         std::vector<M_t>* tgt_from_src);

    // Compute the squared transformation error.
    static void Residuals(const std::vector<X_t>& src,
                          const std::vector<Y_t>& tgt,
                          const M_t& tgt_from_src,
                          std::vector<double>* residuals);
};

bool EstimateAffine2d(const std::vector<Eigen::Vector2d>& src,
                      const std::vector<Eigen::Vector2d>& tgt,
                      Eigen::Matrix2x3d& tgt_from_src);

typename RANSAC<AffineTransformEstimator>::Report EstimateAffine2dRobust(
        const std::vector<Eigen::Vector2d>& src,
        const std::vector<Eigen::Vector2d>& tgt,
        const RANSACOptions& options,
        Eigen::Matrix2x3d& tgt_from_src);

}  // namespace colmap

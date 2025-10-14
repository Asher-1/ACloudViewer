// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <vector>

#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

// Direct linear transformation algorithm to compute the homography between
// point pairs. This algorithm computes the least squares estimate for
// the homography from at least 4 correspondences.
class HomographyMatrixEstimator {
public:
    typedef Eigen::Vector2d X_t;
    typedef Eigen::Vector2d Y_t;
    typedef Eigen::Matrix3d M_t;

    // The minimum number of samples needed to estimate a model.
    static const int kMinNumSamples = 4;

    // Estimate the projective transformation (homography).
    //
    // The number of corresponding points must be at least 4.
    //
    // @param points1    First set of corresponding points.
    // @param points2    Second set of corresponding points.
    //
    // @return         3x3 homogeneous transformation matrix.
    static std::vector<M_t> Estimate(const std::vector<X_t>& points1,
                                     const std::vector<Y_t>& points2);

    // Calculate the transformation error for each corresponding point pair.
    //
    // Residuals are defined as the squared transformation error when
    // transforming the source to the destination coordinates.
    //
    // @param points1    First set of corresponding points.
    // @param points2    Second set of corresponding points.
    // @param H          3x3 projective matrix.
    // @param residuals  Output vector of residuals.
    static void Residuals(const std::vector<X_t>& points1,
                          const std::vector<Y_t>& points2,
                          const M_t& H,
                          std::vector<double>* residuals);
};

}  // namespace colmap

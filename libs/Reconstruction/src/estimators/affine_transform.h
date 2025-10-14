// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_ESTIMATORS_AFFINE_TRANSFORM_H_
#define COLMAP_SRC_ESTIMATORS_AFFINE_TRANSFORM_H_

#include <Eigen/Core>
#include <vector>

#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

class AffineTransformEstimator {
public:
    typedef Eigen::Vector2d X_t;
    typedef Eigen::Vector2d Y_t;
    typedef Eigen::Matrix<double, 2, 3> M_t;

    // The minimum number of samples needed to estimate a model.
    static const int kMinNumSamples = 3;

    // Estimate the affine transformation from at least 3 correspondences.
    static std::vector<M_t> Estimate(const std::vector<X_t>& points1,
                                     const std::vector<Y_t>& points2);

    // Compute the squared transformation error.
    static void Residuals(const std::vector<X_t>& points1,
                          const std::vector<Y_t>& points2,
                          const M_t& E,
                          std::vector<double>* residuals);
};

}  // namespace colmap

#endif  // COLMAP_SRC_ESTIMATORS_AFFINE_TRANSFORM_H_

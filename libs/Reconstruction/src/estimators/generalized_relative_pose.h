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

// Solver for the Generalized Relative Pose problem using a minimal of 8 2D-2D
// correspondences. This implementation is based on:
//
//    "Efficient Computation of Relative Pose for Multi-Camera Systems",
//    Kneip and Li. CVPR 2014.
//
// Note that the solution to this problem is degenerate in the case of pure
// translation and when all correspondences are observed from the same cameras.
//
// The implementation is a modified and improved version of Kneip's original
// implementation in OpenGV licensed under the BSD license.
class GR6PEstimator {
public:
    // The generalized image observations of the left camera, which is composed
    // of the relative pose of the specific camera in the generalized camera and
    // its image observation.
    struct X_t {
        CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW
        // The relative transformation from the generalized camera to the camera
        // frame of the observation.
        Eigen::Matrix3x4d rel_tform;
        // The 2D image feature observation.
        Eigen::Vector2d xy;
    };

    // The normalized image feature points in the left camera.
    typedef X_t Y_t;
    // The relative transformation between the two generalized cameras.
    typedef Eigen::Matrix3x4d M_t;

    // The minimum number of samples needed to estimate a model. Note that in
    // theory the minimum required number of samples is 6 but Laurent Kneip
    // showed in his paper that using 8 samples is more stable.
    static const int kMinNumSamples = 8;

    // Estimate the most probable solution of the GR6P problem from a set of
    // six 2D-2D point correspondences.
    static std::vector<M_t> Estimate(const std::vector<X_t>& points1,
                                     const std::vector<Y_t>& points2);

    // Calculate the squared Sampson error between corresponding points.
    static void Residuals(const std::vector<X_t>& points1,
                          const std::vector<Y_t>& points2,
                          const M_t& proj_matrix,
                          std::vector<double>* residuals);
};

}  // namespace colmap

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(colmap::GR6PEstimator::X_t)

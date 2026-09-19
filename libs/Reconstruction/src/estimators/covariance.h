// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Upstream port (COLMAP d3ccaf35 estimators/covariance.{h,cc}). Fork
// adaptation: the fork's bundle adjustment registers the pose as separate
// qvec (4) and tvec (3) parameter blocks instead of a single contiguous
// Rigid3d block, so `internal::PoseParam` carries two pointers and the
// Schur assembly pushes both blocks per pose. The tangent-space results are
// identical: [rotation(3), translation(3)] per pose under the quaternion
// manifold.

#include <ceres/problem.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <optional>
#include <vector>

#include "estimators/bundle_adjustment_ceres.h"
#include "geometry/rigid3.h"
#include "scene/reconstruction.h"
#include "util/hash_containers.h"
#include "util/types.h"

namespace colmap {
namespace internal {
struct PoseParam;
}

struct BACovariance {
    explicit BACovariance(
            FlatHashMap<point3D_t, Eigen::MatrixXd> point_covs,
            NodeHashMap<image_t, std::pair<int, int>> pose_L_start_size,
            NodeHashMap<const double*, std::pair<int, int>> other_L_start_size,
            Eigen::MatrixXd L_inv);

    // Covariance for 3D points, conditioned on all other variables set
    // constant. If some dimensions are kept constant, the respective
    // rows/columns are omitted. Returns null if 3D point not a variable in
    // the problem.
    std::optional<Eigen::MatrixXd> GetPointCov(point3D_t point3D_id) const;

    // Tangent space covariance in the order [rotation, translation]. If some
    // dimensions are kept constant, the respective rows/columns are omitted.
    // Returns null if image is not a variable in the problem.
    std::optional<Eigen::MatrixXd> GetCamCovFromWorld(image_t image_id) const;
    std::optional<Eigen::MatrixXd> GetCamCrossCovFromWorld(
            image_t image_id1, image_t image_id2) const;
    // Get relative pose covariance in the order [rotation, translation].
    // Returns null if some dimensions are kept constant for either of the two
    // poses. This does not mean that one cannot get relative pose covariance
    // for such case, but requires custom logic to fill in zero block in the
    // covariance matrix.
    std::optional<Eigen::MatrixXd> GetCam2CovFromCam1(
            image_t image_id1,
            const Rigid3d& cam1_from_world,
            image_t image_id2,
            const Rigid3d& cam2_from_world) const;

    // Tangent space covariance for any other variable parameter block in the
    // problem. If some dimensions are kept constant, the respective
    // rows/columns are omitted. Returns null if parameter block is not a
    // variable in the problem.
    std::optional<Eigen::MatrixXd> GetOtherParamsCov(
            const double* params) const;

private:
    const FlatHashMap<point3D_t, Eigen::MatrixXd> point_covs_;
    const NodeHashMap<image_t, std::pair<int, int>> pose_L_start_size_;
    const NodeHashMap<const double*, std::pair<int, int>> other_L_start_size_;
    const Eigen::MatrixXd L_inv_;
};

struct BACovarianceOptions {
    enum class Params {
        POSES,
        POINTS,
        POSES_AND_POINTS,
        ALL,  // + Others
    };

    // For which parameters to compute the covariance.
    Params params = Params::ALL;

    // Damping factor for the Hessian in the Schur complement solver.
    // Enables to robustly deal with poorly conditioned parameters.
    double damping = 1e-8;

    // WARNING: This option will be removed in a future release, use at your
    // own risk. For custom bundle adjustment problems, this enables to
    // specify a custom set of pose parameter blocks to consider. Note that
    // these pose blocks must not necessarily be part of the reconstruction
    // but they must follow the standard requirement for applying the Schur
    // complement trick.
    std::vector<internal::PoseParam> experimental_custom_poses;
};

// Computes covariances for the parameters in a bundle adjustment problem.
// It is important that the problem has a structure suitable for solving
// using the Schur complement trick. This is the case for the standard
// configuration of bundle adjustment problems, but be careful if you modify
// the underlying problem with custom residuals.
// Returns null if the estimation was not successful.
std::optional<BACovariance> EstimateBACovariance(
        const BACovarianceOptions& options,
        const Reconstruction& reconstruction,
        CeresBundleAdjuster& bundle_adjuster);
std::optional<BACovariance> EstimateBACovarianceFromProblem(
        const BACovarianceOptions& options,
        const Reconstruction& reconstruction,
        ceres::Problem& problem);

namespace internal {

// Fork adaptation: the pose is a (qvec, tvec) pair of parameter blocks
// instead of a single Rigid3d block. Either pointer may be null when that
// part of the pose is not a variable in the problem.
struct PoseParam {
    image_t image_id = kInvalidImageId;
    const double* qvec = nullptr;
    const double* tvec = nullptr;
};

std::vector<PoseParam> GetPoseParams(const Reconstruction& reconstruction,
                                     const ceres::Problem& problem);

struct PointParam {
    point3D_t point3D_id = kInvalidPoint3DId;
    const double* xyz = nullptr;
};

std::vector<PointParam> GetPointParams(const Reconstruction& reconstruction,
                                       const ceres::Problem& problem);

std::vector<const double*> GetOtherParams(
        const ceres::Problem& problem,
        const std::vector<PoseParam>& poses,
        const std::vector<PointParam>& points);

}  // namespace internal
}  // namespace colmap

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <vector>

#include "geometry/pose.h"
#include "math/math.h"
#include "optim/ransac.h"
#include "scene/camera.h"
#include "util/eigen_alignment.h"
#include "util/types.h"

namespace colmap {

// Triangulation estimator to estimate 3D point from multiple observations.
// The triangulation must satisfy the following constraints:
//    - Sufficient triangulation angle between observation pairs.
//    - All observations must satisfy cheirality constraint.
//
// An observation is composed of an image measurement and the corresponding
// camera pose and calibration.
//
// Upstream parity (d3ccaf35 estimators/triangulation.h): observations are
// carried as camera-frame unit bearing vectors (Camera::CamRayFromImg), the
// canonical representation for all camera models, including omnidirectional
// (EQUIRECTANGULAR) back-hemisphere rays that the 2D normalized-plane
// representation cannot encode. The fork keeps the default constructor plus
// SetMinTriAngle/SetResidualType mutators because the fork LORANSAC engine
// default-constructs its estimators.
class TriangulationEstimator {
public:
    enum class ResidualType {
        ANGULAR_ERROR,
        REPROJECTION_ERROR,
    };

    struct PointData {
        PointData() = default;
        PointData(const Eigen::Vector2d& img_point,
                  const Eigen::Vector3d& cam_ray)
            : img_point(img_point), cam_ray(cam_ray) {}
        // Image observation in pixels. Only needs to be set for
        // REPROJECTION_ERROR.
        Eigen::Vector2d img_point = Eigen::Vector2d::Zero();
        // Unit bearing vector in the camera frame (Camera::CamRayFromImg).
        Eigen::Vector3d cam_ray = Eigen::Vector3d::Zero();
    };

    struct PoseData {
        PoseData() : camera(nullptr) {}
        PoseData(const Eigen::Matrix3x4d& cam_from_world,
                 const Eigen::Vector3d& proj_center,
                 const Camera* camera)
            : cam_from_world(cam_from_world),
              proj_center(proj_center),
              camera(camera) {}
        // The pose of the camera of the observation as 3x4 matrix.
        Eigen::Matrix3x4d cam_from_world = Eigen::Matrix3x4d::Zero();
        // The projection center for the image of the observation.
        Eigen::Vector3d proj_center = Eigen::Vector3d::Zero();
        // The camera for the image of the observation.
        const Camera* camera = nullptr;
    };

    using X_t = PointData;
    using Y_t = PoseData;
    using M_t = Eigen::Vector3d;

    TriangulationEstimator() = default;
    TriangulationEstimator(double min_tri_angle, ResidualType residual_type);

    // The minimum number of samples needed to estimate a model.
    static const int kMinNumSamples = 2;

    // Estimate a 3D point from a two-view observation.
    //
    // @param point_data        Image measurements.
    // @param pose_data         Camera poses.
    //
    // @return                  Triangulated point if successful, otherwise
    // none.
    std::vector<M_t> Estimate(const std::vector<X_t>& point_data,
                              const std::vector<Y_t>& pose_data) const;

    // Calculate residuals in terms of squared reprojection or angular error.
    //
    // @param point_data        Image measurements.
    // @param pose_data         Camera poses.
    // @param xyz               3D point.
    //
    // @return                  Residual for each observation.
    void Residuals(const std::vector<X_t>& point_data,
                   const std::vector<Y_t>& pose_data,
                   const M_t& xyz,
                   std::vector<double>* residuals) const;

    // Fork LORANSAC engine requires default construction + mutation.
    void SetMinTriAngle(const double min_tri_angle);
    void SetResidualType(const ResidualType residual_type);

private:
    double min_tri_angle_ = 0.0;
    ResidualType residual_type_ = ResidualType::ANGULAR_ERROR;
};

struct EstimateTriangulationOptions {
    // Minimum triangulation angle in radians.
    double min_tri_angle = 0.0;

    // The employed residual type.
    TriangulationEstimator::ResidualType residual_type =
            TriangulationEstimator::ResidualType::ANGULAR_ERROR;

    // RANSAC options for TriangulationEstimator.
    RANSACOptions ransac_options;

    EstimateTriangulationOptions() {
        ransac_options.max_error = DegToRad(2.0);
        ransac_options.confidence = 0.9999;
        ransac_options.min_inlier_ratio = 0.02;
        ransac_options.max_num_trials = 10000;
    }

    void Check() const {
        THROW_CHECK_GE(min_tri_angle, 0.0);
        ransac_options.Check();
    }
};

// Robustly estimate 3D point from observations in multiple views using RANSAC
// and a subsequent non-linear refinement using all inliers. Returns true if
// the estimated number of inliers has more than two views.
//
// Upstream parity (d3ccaf35): the observation-to-bearing conversion happens
// here, so callers only provide pixel observations and camera poses.
bool EstimateTriangulation(const EstimateTriangulationOptions& options,
                           const std::vector<Eigen::Vector2d>& points,
                           const std::vector<Rigid3d>& cams_from_world,
                           const std::vector<Camera const*>& cameras,
                           std::vector<char>* inlier_mask,
                           Eigen::Vector3d* xyz);

}  // namespace colmap

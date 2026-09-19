// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "estimators/generalized_pose.h"

#include "estimators/bundle_adjustment_ceres.h"
#include "optim/manifold.h"
#include "estimators/cost_functions/pose_prior.h"
#include "estimators/cost_functions/cost_functions.h"
#include "estimators/cost_functions/utils.h"
#include "estimators/pose.h"
#include "estimators/solvers/generalized_absolute_pose.h"
#include "estimators/solvers/generalized_relative_pose.h"
#include "geometry/rigid3.h"
#include "optim/loransac.h"
#include "optim/support_measurement.h"
#include "scene/camera.h"
#include "util/hash_containers.h"
#include "util/logging.h"

#include <Eigen/Core>

namespace colmap {
namespace {

void ThrowCheckCameras(const std::vector<size_t>& camera_idxs,
                       const std::vector<Rigid3d>& cams_from_rig,
                       const std::vector<Camera>& cameras) {
  THROW_CHECK(!cameras.empty());
  THROW_CHECK_EQ(cams_from_rig.size(), cameras.size());
  const auto [min_camera_idx, max_camera_idx] =
      std::minmax_element(camera_idxs.begin(), camera_idxs.end());
  THROW_CHECK_GE(*min_camera_idx, 0);
  THROW_CHECK_LT(*max_camera_idx, cameras.size());
}

bool IsPanoramicRig(const std::vector<size_t>& camera_idxs,
                    const std::vector<Rigid3d>& cams_from_rig) {
  const FlatHashSet<size_t> camera_idx_set(camera_idxs.begin(),
                                           camera_idxs.end());
  const size_t first_camera_idx = *camera_idx_set.begin();
  const Eigen::Vector3d first_origin_in_rig =
      cams_from_rig[first_camera_idx].TgtOriginInSrc();
  for (auto it = ++camera_idx_set.begin(); it != camera_idx_set.end(); ++it) {
    const Eigen::Vector3d other_origin_in_rig =
        cams_from_rig[*it].TgtOriginInSrc();
    if (!first_origin_in_rig.isApprox(other_origin_in_rig, 1e-6)) {
      return false;
    }
  }
  return true;
}

double ComputeMaxErrorInCamera(const std::vector<size_t>& camera_idxs,
                               const std::vector<Camera>& cameras,
                               const double max_error_px) {
  THROW_CHECK_GT(max_error_px, 0.0);
  double max_error_cam = 0.;
  for (const auto& camera_idx : camera_idxs) {
    max_error_cam += cameras[camera_idx].CamFromImgThreshold(max_error_px);
  }
  return max_error_cam / camera_idxs.size();
}

bool LowerVector3d(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) {
  if (v1.x() < v2.x()) {
    return true;
  } else if (v1.x() == v2.x()) {
    if (v1.y() < v2.y()) {
      return true;
    } else if (v1.y() == v2.y()) {
      return v1.z() < v2.z();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

std::vector<size_t> ComputeUniquePointIds(
    const std::vector<Eigen::Vector3d>& points3D) {
  std::vector<size_t> point3D_ids(points3D.size());
  std::iota(point3D_ids.begin(), point3D_ids.end(), 0);
  std::sort(point3D_ids.begin(), point3D_ids.end(), [&](size_t i, size_t j) {
    return LowerVector3d(points3D[i], points3D[j]);
  });

  std::vector<size_t>::iterator unique_it = point3D_ids.begin();
  std::vector<size_t>::iterator current_it = point3D_ids.begin();
  std::vector<size_t> unique_point3D_ids(points3D.size());
  while (current_it != point3D_ids.end()) {
    if (!points3D[*unique_it].isApprox(points3D[*current_it], 1e-5)) {
      unique_it = current_it;
    }
    unique_point3D_ids[*current_it] = unique_it - point3D_ids.begin();
    current_it++;
  }
  return unique_point3D_ids;
}

}  // namespace

bool EstimateGeneralizedAbsolutePose(
    const RANSACOptions& options,
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const std::vector<size_t>& camera_idxs,
    const std::vector<Rigid3d>& cams_from_rig,
    const std::vector<Camera>& cameras,
    Rigid3d* rig_from_world,
    size_t* num_inliers,
    std::vector<char>* inlier_mask) {
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  THROW_CHECK_EQ(points2D.size(), camera_idxs.size());
  ThrowCheckCameras(camera_idxs, cams_from_rig, cameras);
  options.Check();
  if (points2D.size() == 0) {
    return false;
  }

  // Precompute cam_from_rig matrices for fast residual computation
  std::vector<Eigen::Matrix3x4d> cams_from_rig_matrices(cams_from_rig.size());
  for (size_t i = 0; i < cams_from_rig.size(); i++) {
    cams_from_rig_matrices[i] = cams_from_rig[i].ToMatrix();
  }

  std::vector<GP3PEstimator::X_t> rig_points2D(points2D.size());
  for (size_t i = 0; i < points2D.size(); i++) {
    const size_t camera_idx = camera_idxs[i];
    rig_points2D[i].ray_in_cam = cameras[camera_idx]
                                     .CamRayFromImg(points2D[i])
                                     .value_or(Eigen::Vector3d::Zero());
    rig_points2D[i].cam_from_rig = cams_from_rig_matrices[camera_idx];
  }

  // Associate unique ids to each 3D point.
  // Needed for UniqueInlierSupportMeasurer to avoid counting the same
  // 3D point multiple times due to FoV overlap in rig.
  // TODO(sarlinpe): Allow passing unique_point3D_ids as argument.
  std::vector<size_t> unique_point3D_ids = ComputeUniquePointIds(points3D);

  // Average of the errors over the cameras, weighted by the number of
  // correspondences
  RANSACOptions options_copy(options);
  options_copy.max_error =
      ComputeMaxErrorInCamera(camera_idxs, cameras, options.max_error);

  // Fork adaptation: the fork's RANSAC ships only the plain inlier
  // support measurer (no unique-inlier variant), so the duplicate-3D-point
  // weighting of the upstream estimator is approximated by the standard
  // inlier count.
  RANSAC<GP3PEstimator> ransac(
      options_copy,
      GP3PEstimator(GP3PEstimator::ResidualType::ReprojectionError));
  auto report = ransac.Estimate(rig_points2D, points3D);
  if (!report.success) {
    return false;
  }

  *rig_from_world = report.model;
  *num_inliers = report.support.num_inliers;
  *inlier_mask = std::move(report.inlier_mask);

  return true;
}

bool EstimateGeneralizedRelativePose(
    const RANSACOptions& ransac_options,
    const std::vector<Eigen::Vector2d>& points2D1,
    const std::vector<Eigen::Vector2d>& points2D2,
    const std::vector<size_t>& camera_idxs1,
    const std::vector<size_t>& camera_idxs2,
    const std::vector<Rigid3d>& cams_from_rig,
    const std::vector<Camera>& cameras,
    std::optional<Rigid3d>* rig2_from_rig1,
    std::optional<Rigid3d>* pano2_from_pano1,
    size_t* num_inliers,
    std::vector<char>* inlier_mask) {
  THROW_CHECK_EQ(points2D1.size(), points2D2.size());
  ThrowCheckCameras(camera_idxs1, cams_from_rig, cameras);
  ThrowCheckCameras(camera_idxs2, cams_from_rig, cameras);
  ransac_options.Check();

  const size_t num_points = points2D1.size();
  if (num_points == 0) {
    return false;
  }

  // Both branches below score with the pixel-unit tangent Sampson error, so the
  // RANSAC threshold is the plain pixel ransac_options throughout. No
  // per-camera conversion to normalized/angular units is needed.
  if (IsPanoramicRig(camera_idxs1, cams_from_rig) &&
      IsPanoramicRig(camera_idxs2, cams_from_rig)) {
    Rigid3d cam2_from_cam1;
    // EstimateRelativePose treats the panoramic rig as one central camera, so
    // each ray carries its unprojection Jacobian, rotated into the rig frame by
    // the same rotation as the ray. Unprojectable points are zeroed, which the
    // tangent Sampson residual reports as infinite (rejected).
    std::vector<CamRayWithJac> cam_rays1_with_jac(num_points);
    std::vector<CamRayWithJac> cam_rays2_with_jac(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      const size_t camera_idx1 = camera_idxs1[i];
      const Eigen::Matrix3d rig_from_cam1 =
          cams_from_rig[camera_idx1].rotation().inverse().toRotationMatrix();
      if (const auto rj =
              cameras[camera_idx1].CamRayFromImgWithJac(points2D1[i])) {
        cam_rays1_with_jac[i] = {rig_from_cam1 * rj->ray,
                                 rig_from_cam1 * rj->jacobian};
      } else {
        cam_rays1_with_jac[i] = CamRayWithJac::Zero();
      }

      const size_t camera_idx2 = camera_idxs2[i];
      const Eigen::Matrix3d rig_from_cam2 =
          cams_from_rig[camera_idx2].rotation().inverse().toRotationMatrix();
      if (const auto rj =
              cameras[camera_idx2].CamRayFromImgWithJac(points2D2[i])) {
        cam_rays2_with_jac[i] = {rig_from_cam2 * rj->ray,
                                 rig_from_cam2 * rj->jacobian};
      } else {
        cam_rays2_with_jac[i] = CamRayWithJac::Zero();
      }
    }
    // Fork adaptation: the upstream ray-based relative pose estimator is
    // not ported; panoramic rigs have no consumer of this function in the
    // fork, so the panoramic branch reports unsupported for now.
    (void)cam_rays1_with_jac;
    (void)cam_rays2_with_jac;
    (void)num_inliers;
    (void)inlier_mask;
    return false;
  }

  std::vector<GRNPObservation> points1(num_points);
  std::vector<GRNPObservation> points2(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    points1[i] = {cams_from_rig[camera_idxs1[i]],
                  cameras[camera_idxs1[i]]
                      .CamRayFromImgWithJac(points2D1[i])
                      .value_or(CamRayWithJac::Zero())};
    points2[i] = {cams_from_rig[camera_idxs2[i]],
                  cameras[camera_idxs2[i]]
                      .CamRayFromImgWithJac(points2D2[i])
                      .value_or(CamRayWithJac::Zero())};
  }

  LORANSAC<GR6PEstimator, GR8PEstimator> ransac(ransac_options);
  auto report = ransac.Estimate(points1, points2);
  if (!report.success) {
    return false;
  }

  *rig2_from_rig1 = report.model;
  *num_inliers = report.support.num_inliers;
  *inlier_mask = std::move(report.inlier_mask);

  return true;
}

bool RefineGeneralizedAbsolutePose(
        const AbsolutePoseRefinementOptions& options,
        const std::vector<char>& inlier_mask,
        const std::vector<Eigen::Vector2d>& points2D,
        const std::vector<Eigen::Vector3d>& points3D,
        const std::vector<size_t>& camera_idxs,
        const std::vector<Rigid3d>& cams_from_rig,
        Rigid3d* rig_from_world,
        std::vector<Camera>* cameras,
        Eigen::Matrix6d* rig_from_world_cov) {
  THROW_CHECK_NOTNULL(rig_from_world);
  THROW_CHECK_NOTNULL(cameras);
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  THROW_CHECK_EQ(points2D.size(), inlier_mask.size());
  THROW_CHECK_EQ(points2D.size(), camera_idxs.size());
  THROW_CHECK_LT(*std::max_element(camera_idxs.begin(), camera_idxs.end()),
                 cameras->size());
  options.Check();

  const auto loss_function =
      std::make_unique<ceres::CauchyLoss>(options.loss_function_scale);

  // Fork adaptation: the fork's Rigid3d stores the rotation and translation
  // as separate members (no contiguous 7-double params block), so the pose
  // parameters live in split qvec ([w, x, y, z]) / tvec buffers and the
  // composed residual functor mirrors the bundle adjustment's frame-rig
  // cost (cam_from_world = sensor_from_rig * rig_from_world).
  std::vector<Eigen::Vector4d> cam_from_rig_qvecs(cams_from_rig.size());
  std::vector<Eigen::Vector3d> cam_from_rig_tvecs(cams_from_rig.size());
  for (size_t i = 0; i < cams_from_rig.size(); ++i) {
    const Eigen::Quaterniond& q = cams_from_rig[i].rotation();
    cam_from_rig_qvecs[i] =
            Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
    cam_from_rig_tvecs[i] = cams_from_rig[i].translation();
  }
  Eigen::Vector4d rig_qvec;
  {
    const Eigen::Quaterniond& q = rig_from_world->rotation();
    rig_qvec = Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
  }
  Eigen::Vector3d rig_tvec = rig_from_world->translation();

  std::vector<double*> cameras_params_data(cameras->size());
  for (size_t i = 0; i < cameras->size(); i++) {
    cameras_params_data[i] = cameras->at(i).ParamsData();
  }
  std::vector<size_t> camera_counts(cameras->size(), 0);

  // Cost function assumes unit quaternion.
  rig_from_world->rotation().normalize();

  std::vector<Eigen::Vector3d> point3D_params = points3D;

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  for (size_t i = 0; i < points2D.size(); ++i) {
    // Skip outlier observations
    if (!inlier_mask[i]) {
      continue;
    }
    const size_t camera_idx = camera_idxs[i];
    camera_counts[camera_idx] += 1;

    const Camera& camera = cameras->at(camera_idx);
    ceres::CostFunction* cost_function = nullptr;
    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                  \
    case CameraModel::model_id:                                         \
      cost_function =                                                   \
              FrameRigBundleAdjustmentCostFunction<CameraModel>::Create( \
                      points2D[i]);                                     \
      break;

      CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }

    problem.AddResidualBlock(
        cost_function,
        loss_function.get(),
        rig_qvec.data(),
        rig_tvec.data(),
        cam_from_rig_qvecs[camera_idx].data(),
        cam_from_rig_tvecs[camera_idx].data(),
        point3D_params[i].data(),
        cameras_params_data[camera_idx]);
    problem.SetParameterBlockConstant(point3D_params[i].data());
  }

  if (problem.NumResiduals() > 0) {
    // Camera parameterization.
    for (size_t i = 0; i < cameras->size(); i++) {
      if (camera_counts[i] == 0) continue;
      Camera& camera = cameras->at(i);

      // We don't optimize the rig parameters (it's likely
      // under-constrained).
      problem.SetParameterBlockConstant(cam_from_rig_qvecs[i].data());
      problem.SetParameterBlockConstant(cam_from_rig_tvecs[i].data());

      if (!options.refine_focal_length && !options.refine_extra_params) {
        problem.SetParameterBlockConstant(camera.ParamsData());
      } else {
        // Always set the principal point as fixed.
        std::vector<int> const_camera_params;
        const std::vector<size_t> principal_point_idxs =
            camera.PrincipalPointIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   principal_point_idxs.begin(),
                                   principal_point_idxs.end());

        if (!options.refine_focal_length) {
          const std::vector<size_t> focal_length_idxs =
              camera.FocalLengthIdxs();
          const_camera_params.insert(const_camera_params.end(),
                                     focal_length_idxs.begin(),
                                     focal_length_idxs.end());
        }

        if (!options.refine_extra_params) {
          const std::vector<size_t> extra_params_idxs =
              camera.ExtraParamsIdxs();
          const_camera_params.insert(const_camera_params.end(),
                                     extra_params_idxs.begin(),
                                     extra_params_idxs.end());
        }

        if (const_camera_params.size() == camera.Params().size()) {
          problem.SetParameterBlockConstant(camera.ParamsData());
        } else {
          SetSubsetManifold(
              static_cast<int>(camera.Params().size()), const_camera_params,
              &problem, camera.ParamsData());
        }
      }
    }

    // The frame rotation lives on the quaternion manifold; the frame
    // translation is a free 3-DoF block.
    SetQuaternionManifoldWxyz(&problem, rig_qvec.data());
  }

  ceres::Solver::Options solver_options;
  solver_options.gradient_tolerance = options.gradient_tolerance;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.logging_type = ceres::LoggingType::SILENT;

  // The overhead of creating threads is too large.
  solver_options.num_threads = 1;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  if (options.print_summary || VLOG_IS_ON(1)) {
    PrintSolverSummary(summary);
  }

  // Fork adaptation: the tangent-space covariance over the split pose
  // blocks is not assembled yet; the GP pipeline passes nullptr.
  (void)rig_from_world_cov;

  // Write the refined pose back into the fork's Rigid3d representation.
  rig_from_world->rotation() =
          Eigen::Quaterniond(rig_qvec(0), rig_qvec(1), rig_qvec(2),
                             rig_qvec(3))
                  .normalized();
  rig_from_world->translation() = rig_tvec;

  return summary.IsSolutionUsable();
}

bool EstimateStructureLessAbsolutePose(
    const StructureLessAbsolutePoseEstimationOptions& options,
    const std::vector<Eigen::Vector2d>& query_points2D,
    const std::vector<Eigen::Vector2d>& world_points2D,
    const std::vector<size_t>& world_camera_idxs,
    const std::vector<Rigid3d>& world_cams_from_world,
    const std::vector<Camera>& world_cameras,
    const Camera& query_camera,
    Rigid3d* query_cam_from_world,
    size_t* num_inliers,
    std::vector<char>* inlier_mask) {
  THROW_CHECK_EQ(world_points2D.size(), query_points2D.size());
  THROW_CHECK_EQ(world_points2D.size(), world_camera_idxs.size());
  THROW_CHECK_EQ(world_cams_from_world.size(), world_cameras.size());
  ThrowCheckCameras(world_camera_idxs, world_cams_from_world, world_cameras);
  options.Check();

  if (IsPanoramicRig(world_camera_idxs, world_cams_from_world)) {
    return false;
  }

  const size_t num_points = world_points2D.size();
  std::vector<GRNPObservation> world_obs(num_points);
  std::vector<GRNPObservation> query_obs(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    const size_t world_camera_idx = world_camera_idxs[i];
    world_obs[i] = {world_cams_from_world[world_camera_idx],
                    world_cameras[world_camera_idx]
                        .CamRayFromImgWithJac(world_points2D[i])
                        .value_or(CamRayWithJac::Zero())};
    query_obs[i] = {Rigid3d(),
                    query_camera.CamRayFromImgWithJac(query_points2D[i])
                        .value_or(CamRayWithJac::Zero())};
  }

  // GR6P/GR8P score with the pixel-unit tangent Sampson error, so the RANSAC
  // threshold is the plain pixel max_error. No per-camera conversion needed.
  LORANSAC<GR6PEstimator, GR8PEstimator> ransac(options.ransac_options);
  auto report = ransac.Estimate(world_obs, query_obs);
  if (!report.success) {
    return false;
  }

  *query_cam_from_world = report.model;
  *num_inliers = report.support.num_inliers;
  *inlier_mask = std::move(report.inlier_mask);

  return true;
}

}  // namespace colmap

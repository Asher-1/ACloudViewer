// SPDX-License-Identifier: BSD-3-Clause
//
// Ported from upstream MVS/colmap d3ccaf35 (batch 24 test alignment);
// include paths mapped to the fork layout.

#include "estimators/cost_functions/reprojection_error.h"

#include "geometry/rigid3.h"
#include "math/random.h"
#include "sensor/models.h"
#include "util/eigen_matchers.h"

#include <algorithm>
#include <cmath>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(ReprojErrorCostFunctor, Nominal) {
  using CostFunctor = ReprojErrorCostFunctor<SimplePinholeCameraModel>;
  std::unique_ptr<ceres::CostFunction> cost_function(
      CostFunctor::Create(Eigen::Vector2d::Zero()));
  double cam_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[3] = {point3D, cam_from_world, camera_params};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);

  point3D[1] = 1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 1);

  camera_params[0] = 2;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 2);

  point3D[0] = -1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], -2);
  EXPECT_EQ(residuals[1], 2);

  point3D[2] = -1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
}

TEST(AnalyticalReprojError, BehindCameraZeroesResidualsAndJacobians) {
  using CameraModel = SimplePinholeCameraModel;
  constexpr int kNumParams = CameraModel::num_params;
  const Eigen::Vector2d kPoint2D = Eigen::Vector2d::Zero();

  double cam_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double point3D[3] = {0, 0, -1};
  double camera_params[kNumParams] = {1, 0, 0};
  const double* parameters[3] = {point3D, cam_from_world, camera_params};

  double residuals[2] = {1, 1};
  double j_point[6], j_pose[14], j_params[2 * kNumParams];
  for (double& v : j_point) v = 1;
  for (double& v : j_pose) v = 1;
  for (double& v : j_params) v = 1;
  double* jacobians[3] = {j_point, j_pose, j_params};

  AnalyticalReprojErrorCostFunction<CameraModel> cost_function(kPoint2D);
  EXPECT_TRUE(cost_function.Evaluate(parameters, residuals, jacobians));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  for (double v : j_point) EXPECT_EQ(v, 0);
  for (double v : j_pose) EXPECT_EQ(v, 0);
  for (double v : j_params) EXPECT_EQ(v, 0);

  for (double& v : residuals) v = 1;
  for (double& v : j_point) v = 1;
  for (double& v : j_params) v = 1;
  const double* const_pose_parameters[2] = {point3D, camera_params};
  double* const_pose_jacobians[2] = {j_point, j_params};
  AnalyticalReprojErrorConstantPoseCostFunction<CameraModel>
      const_pose_cost_function(kPoint2D, Rigid3d());
  EXPECT_TRUE(const_pose_cost_function.Evaluate(
      const_pose_parameters, residuals, const_pose_jacobians));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  for (double v : j_point) EXPECT_EQ(v, 0);
  for (double v : j_params) EXPECT_EQ(v, 0);
}

// Fork adaptation: ext_ceres is built without -march while this tree uses
// -march=x86-64-v3, so Eigen's heap allocator differs across the libceres
// boundary (EIGEN_MAX_ALIGN_BYTES 16 vs 32). A GradientChecker::ProbeResults
// filled inside libceres and destroyed here frees a misaligned pointer
// (SIGSEGV; upstream a2bd15c2 built with the same flags crashes identically).
// The analytical Jacobians are therefore compared against the autodiff
// Jacobians, both evaluated in this translation unit.
void ExpectAnalyticalMatchesAutoDiff(ceres::CostFunction* analytical,
                                     ceres::CostFunction* auto_diff,
                                     double* const* parameter_blocks,
                                     const double res_eps,
                                     const double jac_eps) {
  const std::vector<int32_t>& block_sizes = analytical->parameter_block_sizes();
  ASSERT_EQ(block_sizes, auto_diff->parameter_block_sizes());
  const int num_residuals = analytical->num_residuals();
  ASSERT_EQ(num_residuals, auto_diff->num_residuals());

  std::vector<std::vector<double>> analytical_jacobians;
  std::vector<std::vector<double>> auto_diff_jacobians;
  std::vector<double*> analytical_jacobian_ptrs;
  std::vector<double*> auto_diff_jacobian_ptrs;
  for (const int32_t block_size : block_sizes) {
    analytical_jacobians.emplace_back(num_residuals * block_size);
    auto_diff_jacobians.emplace_back(num_residuals * block_size);
  }
  for (size_t i = 0; i < block_sizes.size(); ++i) {
    analytical_jacobian_ptrs.push_back(analytical_jacobians[i].data());
    auto_diff_jacobian_ptrs.push_back(auto_diff_jacobians[i].data());
  }

  std::vector<double> analytical_residuals(num_residuals);
  std::vector<double> auto_diff_residuals(num_residuals);
  ASSERT_TRUE(analytical->Evaluate(parameter_blocks,
                                   analytical_residuals.data(),
                                   analytical_jacobian_ptrs.data()));
  ASSERT_TRUE(auto_diff->Evaluate(parameter_blocks,
                                  auto_diff_residuals.data(),
                                  auto_diff_jacobian_ptrs.data()));
  for (int r = 0; r < num_residuals; ++r) {
    EXPECT_NEAR(analytical_residuals[r], auto_diff_residuals[r], res_eps);
  }
  for (size_t i = 0; i < block_sizes.size(); ++i) {
    for (size_t k = 0; k < analytical_jacobians[i].size(); ++k) {
      const double a = analytical_jacobians[i][k];
      const double b = auto_diff_jacobians[i][k];
      EXPECT_LE(std::abs(a - b),
                jac_eps * std::max({1.0, std::abs(a), std::abs(b)}))
          << "block " << i << " entry " << k;
    }
  }
}

// Validates the fully-variable analytical reprojection error cost function
// against the autodiff residual and Jacobians. Both Jacobians are exact, so
// the relative tolerance is far tighter than upstream's numeric (Ridders)
// 1e-4 (see ExpectAnalyticalMatchesAutoDiff for the fork adaptation).
template <typename CameraModel>
void TestAnalyticalReprojError(const std::vector<double>& camera_params) {
  SetPRNGSeed(42);
  constexpr double kJacEps = 1e-9;
  constexpr double kResEps = 1e-9;

  const Eigen::Vector2d kPoint2D(200, 300);
  auto analytical_cost_function =
      std::make_unique<AnalyticalReprojErrorCostFunction<CameraModel>>(
          kPoint2D);
  std::unique_ptr<ceres::CostFunction> auto_diff_cost_function(
      ReprojErrorCostFunctor<CameraModel>::Create(kPoint2D));

  for (const double x : {-1, 0, 1}) {
    for (const double y : {-1, 0, 1}) {
      for (const double z : {0, 1, 2, 3}) {
        Rigid3d cam_from_world(Eigen::Quaterniond(Eigen::AngleAxisd(
                                   RandomUniformReal<double>(0, 2 * EIGEN_PI),
                                   Eigen::Vector3d(0.1, -0.1, 1).normalized())),
                               Eigen::Vector3d(1, 2, 3));
        Eigen::Vector3d point3D(x, y, z);
        std::vector<double> params = camera_params;

        // Restrict to a realistic field of view. The OpenCV/rational models are
        // only well-conditioned for moderate normalized radii, where the
        // finite-difference (Ridders) gradient check is reliable; the exact
        // analytical Jacobians are validated over the full range against Ceres
        // Jets in sensor/models_jacobian_test.
        const Eigen::Vector3d point3D_in_cam = cam_from_world * point3D;
        if (point3D_in_cam.z() < 0.5 ||
            point3D_in_cam.hnormalized().norm() > 0.8) {
          continue;
        }

        std::vector<double*> parameter_blocks{
            point3D.data(), cam_from_world.params.data(), params.data()};

        ExpectAnalyticalMatchesAutoDiff(analytical_cost_function.get(),
                                        auto_diff_cost_function.get(),
                                        parameter_blocks.data(),
                                        kResEps,
                                        kJacEps);
      }
    }
  }
}

// Validates the fixed-pose analytical reprojection error cost function against
// the autodiff residual and Jacobians. See TestAnalyticalReprojError for the
// tolerance rationale.
template <typename CameraModel>
void TestAnalyticalReprojErrorConstantPose(
    const std::vector<double>& camera_params) {
  SetPRNGSeed(42);
  constexpr double kJacEps = 1e-9;
  constexpr double kResEps = 1e-9;

  const Eigen::Vector2d kPoint2D(200, 300);
  for (const double x : {-1, 0, 1}) {
    for (const double y : {-1, 0, 1}) {
      for (const double z : {0, 1, 2, 3}) {
        Rigid3d cam_from_world(Eigen::Quaterniond(Eigen::AngleAxisd(
                                   RandomUniformReal<double>(0, 2 * EIGEN_PI),
                                   Eigen::Vector3d(0.1, -0.1, 1).normalized())),
                               Eigen::Vector3d(1, 2, 3));
        Eigen::Vector3d point3D(x, y, z);
        std::vector<double> params = camera_params;

        // Restrict to a realistic field of view. The OpenCV/rational models are
        // only well-conditioned for moderate normalized radii, where the
        // finite-difference (Ridders) gradient check is reliable; the exact
        // analytical Jacobians are validated over the full range against Ceres
        // Jets in sensor/models_jacobian_test.
        const Eigen::Vector3d point3D_in_cam = cam_from_world * point3D;
        if (point3D_in_cam.z() < 0.5 ||
            point3D_in_cam.hnormalized().norm() > 0.8) {
          continue;
        }

        auto analytical_cost_function = std::make_unique<
            AnalyticalReprojErrorConstantPoseCostFunction<CameraModel>>(
            kPoint2D, cam_from_world);
        std::unique_ptr<ceres::CostFunction> auto_diff_cost_function(
            ReprojErrorConstantPoseCostFunctor<CameraModel>::Create(
                kPoint2D, cam_from_world));

        std::vector<double*> parameter_blocks{point3D.data(), params.data()};

        ExpectAnalyticalMatchesAutoDiff(analytical_cost_function.get(),
                                        auto_diff_cost_function.get(),
                                        parameter_blocks.data(),
                                        kResEps,
                                        kJacEps);
      }
    }
  }
}

TEST(ReprojErrorCostFunctor, AnalyticalVersusAutoDiff) {
  TestAnalyticalReprojError<SimplePinholeCameraModel>({200, 100, 120});
  TestAnalyticalReprojError<PinholeCameraModel>({200, 210, 100, 120});
  TestAnalyticalReprojError<SimpleRadialCameraModel>({200, 100, 120, 0.1});
  TestAnalyticalReprojError<RadialCameraModel>({200, 100, 120, 0.1, 0.05});
  TestAnalyticalReprojError<OpenCVCameraModel>(
      {200, 210, 100, 120, -0.1, 0.05, -0.001, 0.002});
  TestAnalyticalReprojError<FullOpenCVCameraModel>(
      {200, 210, 100, 120, -0.1, 0.05, -0.001, 0.002, 0.01, 0.02, -0.02, 0.01});
  TestAnalyticalReprojError<FOVCameraModel>({200, 210, 100, 120, 0.9});
  TestAnalyticalReprojError<SimpleRadialFisheyeCameraModel>(
      {200, 100, 120, 0.1});
  TestAnalyticalReprojError<RadialFisheyeCameraModel>(
      {200, 100, 120, 0.1, 0.02});
  TestAnalyticalReprojError<OpenCVFisheyeCameraModel>(
      {200, 210, 100, 120, -0.05, 0.02, -0.001, 0.001});
  TestAnalyticalReprojError<ThinPrismFisheyeCameraModel>({200,
                                                          210,
                                                          100,
                                                          120,
                                                          -0.05,
                                                          0.02,
                                                          -0.001,
                                                          0.001,
                                                          0.001,
                                                          0.002,
                                                          0.001,
                                                          -0.001});
  TestAnalyticalReprojError<RadTanThinPrismFisheyeModel>({200,
                                                          210,
                                                          100,
                                                          120,
                                                          -0.05,
                                                          0.02,
                                                          -0.005,
                                                          0.001,
                                                          0.0005,
                                                          0.0002,
                                                          -0.001,
                                                          0.001,
                                                          0.001,
                                                          -0.001,
                                                          0.0005,
                                                          -0.0005});
  TestAnalyticalReprojError<SimpleFisheyeCameraModel>({200, 100, 120});
  TestAnalyticalReprojError<FisheyeCameraModel>({200, 210, 100, 120});
  TestAnalyticalReprojError<SimpleDivisionCameraModel>({200, 100, 120, 0.1});
  TestAnalyticalReprojError<DivisionCameraModel>({200, 210, 100, 120, 0.1});
  TestAnalyticalReprojError<EUCMCameraModel>({200, 210, 100, 120, 0.6, 1.2});
  TestAnalyticalReprojError<EquirectangularCameraModel>({1000, 500});
}

TEST(ReprojErrorConstantPoseCostFunctor, AnalyticalVersusAutoDiff) {
  TestAnalyticalReprojErrorConstantPose<SimplePinholeCameraModel>(
      {200, 100, 120});
  TestAnalyticalReprojErrorConstantPose<PinholeCameraModel>(
      {200, 210, 100, 120});
  TestAnalyticalReprojErrorConstantPose<SimpleRadialCameraModel>(
      {200, 100, 120, 0.1});
  TestAnalyticalReprojErrorConstantPose<RadialCameraModel>(
      {200, 100, 120, 0.1, 0.05});
  TestAnalyticalReprojErrorConstantPose<OpenCVCameraModel>(
      {200, 210, 100, 120, -0.1, 0.05, -0.001, 0.002});
  TestAnalyticalReprojErrorConstantPose<FullOpenCVCameraModel>(
      {200, 210, 100, 120, -0.1, 0.05, -0.001, 0.002, 0.01, 0.02, -0.02, 0.01});
  TestAnalyticalReprojErrorConstantPose<FOVCameraModel>(
      {200, 210, 100, 120, 0.9});
  TestAnalyticalReprojErrorConstantPose<SimpleRadialFisheyeCameraModel>(
      {200, 100, 120, 0.1});
  TestAnalyticalReprojErrorConstantPose<RadialFisheyeCameraModel>(
      {200, 100, 120, 0.1, 0.02});
  TestAnalyticalReprojErrorConstantPose<OpenCVFisheyeCameraModel>(
      {200, 210, 100, 120, -0.05, 0.02, -0.001, 0.001});
  TestAnalyticalReprojErrorConstantPose<ThinPrismFisheyeCameraModel>({200,
                                                                      210,
                                                                      100,
                                                                      120,
                                                                      -0.05,
                                                                      0.02,
                                                                      -0.001,
                                                                      0.001,
                                                                      0.001,
                                                                      0.002,
                                                                      0.001,
                                                                      -0.001});
  TestAnalyticalReprojErrorConstantPose<RadTanThinPrismFisheyeModel>({200,
                                                                      210,
                                                                      100,
                                                                      120,
                                                                      -0.05,
                                                                      0.02,
                                                                      -0.005,
                                                                      0.001,
                                                                      0.0005,
                                                                      0.0002,
                                                                      -0.001,
                                                                      0.001,
                                                                      0.001,
                                                                      -0.001,
                                                                      0.0005,
                                                                      -0.0005});
  TestAnalyticalReprojErrorConstantPose<SimpleFisheyeCameraModel>(
      {200, 100, 120});
  TestAnalyticalReprojErrorConstantPose<FisheyeCameraModel>(
      {200, 210, 100, 120});
  TestAnalyticalReprojErrorConstantPose<SimpleDivisionCameraModel>(
      {200, 100, 120, 0.1});
  TestAnalyticalReprojErrorConstantPose<DivisionCameraModel>(
      {200, 210, 100, 120, 0.1});
  TestAnalyticalReprojErrorConstantPose<EUCMCameraModel>(
      {200, 210, 100, 120, 0.6, 1.2});
  TestAnalyticalReprojErrorConstantPose<EquirectangularCameraModel>(
      {1000, 500});
}

TEST(ReprojErrorConstantPoseCostFunctor, Nominal) {
  Rigid3d cam_from_world;
  std::unique_ptr<ceres::CostFunction> cost_function(
      ReprojErrorConstantPoseCostFunctor<SimplePinholeCameraModel>::Create(
          Eigen::Vector2d::Zero(), cam_from_world));
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[2] = {point3D, camera_params};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);

  point3D[1] = 1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 1);

  camera_params[0] = 2;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 2);

  point3D[0] = -1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], -2);
  EXPECT_EQ(residuals[1], 2);
}

TEST(ReprojErrorConstantPoint3DCostFunctor, Nominal) {
  Eigen::Vector2d point2D = Eigen::Vector2d::Zero();
  Eigen::Vector3d point3D;
  point3D << 0, 0, 1;

  double cam_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[2] = {cam_from_world, camera_params};

  {
    std::unique_ptr<ceres::CostFunction> cost_function(
        ReprojErrorConstantPoint3DCostFunctor<SimplePinholeCameraModel>::Create(
            point2D, point3D));
    EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
    EXPECT_EQ(residuals[0], 0);
    EXPECT_EQ(residuals[1], 0);
  }

  {
    point3D[1] = 1;
    std::unique_ptr<ceres::CostFunction> cost_function(
        ReprojErrorConstantPoint3DCostFunctor<SimplePinholeCameraModel>::Create(
            point2D, point3D));
    EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
    EXPECT_EQ(residuals[0], 0);
    EXPECT_EQ(residuals[1], 1);
  }

  {
    camera_params[0] = 2;
    std::unique_ptr<ceres::CostFunction> cost_function(
        ReprojErrorConstantPoint3DCostFunctor<SimplePinholeCameraModel>::Create(
            point2D, point3D));
    EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
    EXPECT_EQ(residuals[0], 0);
    EXPECT_EQ(residuals[1], 2);
  }

  {
    point3D[0] = -1;
    std::unique_ptr<ceres::CostFunction> cost_function(
        ReprojErrorConstantPoint3DCostFunctor<SimplePinholeCameraModel>::Create(
            point2D, point3D));
    EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
    EXPECT_EQ(residuals[0], -2);
    EXPECT_EQ(residuals[1], 2);
  }
}

TEST(CovarianceWeightedCostFunctor, ReprojErrorCostFunctor) {
  using CostFunctor = ReprojErrorCostFunctor<SimplePinholeCameraModel>;
  double cam_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double point3D[3] = {-1, 1, 1};
  double camera_params[3] = {2, 0, 0};
  double residuals[2];
  const double* parameters[3] = {point3D, cam_from_world, camera_params};

  std::unique_ptr<ceres::CostFunction> cost_function1(
      CovarianceWeightedCostFunctor<CostFunctor>::Create(
          Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero()));
  EXPECT_TRUE(cost_function1->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], -2);
  EXPECT_EQ(residuals[1], 2);

  std::unique_ptr<ceres::CostFunction> cost_function2(
      CovarianceWeightedCostFunctor<CostFunctor>::Create(
          4 * Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero()));
  EXPECT_TRUE(cost_function2->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], -1);
  EXPECT_EQ(residuals[1], 1);
}

// The weighted dispatch must agree with whitening the dispatched cost function
// by hand, for every camera model. This is what lets the inner cost function
// keep its analytical jacobian.
template <typename CameraModel>
void TestWeightedCameraCostFunction() {
  const Eigen::Vector2d point2D(0.7, -0.3);
  const double stddev = 2.0;
  const Eigen::Matrix2d covariance =
      stddev * stddev * Eigen::Matrix2d::Identity();

  double cam_from_world[7] = {0, 0, 0, 1, 0.1, 0.2, 3.0};
  double point3D[3] = {0.3, -0.2, 5.0};
  std::vector<double> camera_params(CameraModel::num_params, 0.0);
  for (size_t i = 0; i < camera_params.size(); ++i) {
    camera_params[i] = 0.1 * (i + 1);
  }
  camera_params[0] = 100.0;
  const double* parameters[3] = {point3D, cam_from_world, camera_params.data()};

  std::unique_ptr<ceres::CostFunction> unweighted(
      CreateCameraCostFunction<ReprojErrorCostFunctor>(CameraModel::model_id,
                                                       point2D));
  std::unique_ptr<ceres::CostFunction> cov_weighted(
      CreateCovarianceWeightedCameraCostFunction<ReprojErrorCostFunctor>(
          CameraModel::model_id, covariance, point2D));
  std::unique_ptr<ceres::CostFunction> scale_weighted(
      CreateScaleWeightedCameraCostFunction<ReprojErrorCostFunctor>(
          CameraModel::model_id, stddev, point2D));

  const int num_params = CameraModel::num_params;
  Eigen::Vector2d unweighted_residuals;
  Eigen::Vector2d cov_residuals;
  Eigen::Vector2d scale_residuals;
  Eigen::Matrix<double, 2, 3, Eigen::RowMajor> unweighted_jacobian_point;
  Eigen::Matrix<double, 2, 7, Eigen::RowMajor> unweighted_jacobian_pose;
  Eigen::MatrixXd unweighted_jacobian_params(2, num_params);
  Eigen::Matrix<double, 2, 3, Eigen::RowMajor> cov_jacobian_point;
  Eigen::Matrix<double, 2, 7, Eigen::RowMajor> cov_jacobian_pose;
  Eigen::MatrixXd cov_jacobian_params(2, num_params);
  Eigen::Matrix<double, 2, 3, Eigen::RowMajor> scale_jacobian_point;
  Eigen::Matrix<double, 2, 7, Eigen::RowMajor> scale_jacobian_pose;
  Eigen::MatrixXd scale_jacobian_params(2, num_params);
  double* unweighted_jacobians[3] = {unweighted_jacobian_point.data(),
                                     unweighted_jacobian_pose.data(),
                                     unweighted_jacobian_params.data()};
  double* cov_jacobians[3] = {cov_jacobian_point.data(),
                              cov_jacobian_pose.data(),
                              cov_jacobian_params.data()};
  double* scale_jacobians[3] = {scale_jacobian_point.data(),
                                scale_jacobian_pose.data(),
                                scale_jacobian_params.data()};

  ASSERT_TRUE(unweighted->Evaluate(
      parameters, unweighted_residuals.data(), unweighted_jacobians));
  ASSERT_TRUE(
      cov_weighted->Evaluate(parameters, cov_residuals.data(), cov_jacobians));
  ASSERT_TRUE(scale_weighted->Evaluate(
      parameters, scale_residuals.data(), scale_jacobians));

  // Whitening by an isotropic covariance is a division by stddev.
  EXPECT_THAT(
      cov_residuals,
      EigenMatrixNear(Eigen::Vector2d(unweighted_residuals / stddev), 1e-10));
  EXPECT_THAT(cov_jacobian_point,
              EigenMatrixNear(Eigen::Matrix<double, 2, 3, Eigen::RowMajor>(
                                  unweighted_jacobian_point / stddev),
                              1e-10));
  EXPECT_THAT(cov_jacobian_pose,
              EigenMatrixNear(Eigen::Matrix<double, 2, 7, Eigen::RowMajor>(
                                  unweighted_jacobian_pose / stddev),
                              1e-10));
  EXPECT_THAT(cov_jacobian_params,
              EigenMatrixNear(
                  Eigen::MatrixXd(unweighted_jacobian_params / stddev), 1e-10));

  // The scale form must match the equivalent isotropic covariance.
  EXPECT_THAT(scale_residuals, EigenMatrixNear(cov_residuals, 1e-10));
  EXPECT_THAT(scale_jacobian_point, EigenMatrixNear(cov_jacobian_point, 1e-10));
  EXPECT_THAT(scale_jacobian_pose, EigenMatrixNear(cov_jacobian_pose, 1e-10));
  EXPECT_THAT(scale_jacobian_params,
              EigenMatrixNear(cov_jacobian_params, 1e-10));
}

TEST(WeightedCameraCostFunction, MatchesDispatchedCostFunction) {
  TestWeightedCameraCostFunction<SimplePinholeCameraModel>();
  TestWeightedCameraCostFunction<PinholeCameraModel>();
  TestWeightedCameraCostFunction<SimpleRadialCameraModel>();
  TestWeightedCameraCostFunction<RadialCameraModel>();
  TestWeightedCameraCostFunction<OpenCVCameraModel>();
  TestWeightedCameraCostFunction<FullOpenCVCameraModel>();
}

TEST(RigReprojErrorCostFunctor, Nominal) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      RigReprojErrorCostFunctor<SimplePinholeCameraModel>::Create(
          Eigen::Vector2d::Zero()));
  double cam_from_rig[7] = {0, 0, 0, 1, 0, 0, -1};
  double rig_from_world[7] = {0, 0, 0, 1, 0, 0, 1};
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[4] = {
      point3D, cam_from_rig, rig_from_world, camera_params};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);

  point3D[1] = 1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 1);

  camera_params[0] = 2;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 2);

  point3D[0] = -1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], -2);
  EXPECT_EQ(residuals[1], 2);
}

TEST(ScaledRigReprojErrorCostFunctor, Nominal) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      ScaledRigReprojErrorCostFunctor<SimplePinholeCameraModel>::Create(
          Eigen::Vector2d::Zero()));
  double cam_from_rig[7] = {0, 0, 0, 1, 0, 0, -1};
  double rig_from_world[8] = {0, 0, 0, 1, 0, 0, 1, 0};
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[4] = {
      point3D, cam_from_rig, rig_from_world, camera_params};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);

  point3D[1] = 1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 1);

  camera_params[0] = 2;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 2);

  point3D[0] = -1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], -2);
  EXPECT_EQ(residuals[1], 2);

  // Observations behind the camera contribute a zero residual.
  point3D[2] = -3;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
}

TEST(ScaledRigReprojErrorCostFunctor, Scale) {
  double cam_from_rig[7] = {0, 0, 0, 1, 0, 0, 0};
  double rig_from_world[8] = {0, 0, 0, 1, 0, 0, 1, std::log(2.)};
  double point3D[3] = {0, 1, 0};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[4] = {
      point3D, cam_from_rig, rig_from_world, camera_params};

  std::unique_ptr<ceres::CostFunction> log_scale_cost_function(
      ScaledRigReprojErrorCostFunctor<SimplePinholeCameraModel>::Create(
          Eigen::Vector2d::Zero(), /*use_log_scale=*/true));
  EXPECT_TRUE(
      log_scale_cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 2);

  // The same scale, stored directly instead of in log-space.
  std::unique_ptr<ceres::CostFunction> cost_function(
      ScaledRigReprojErrorCostFunctor<SimplePinholeCameraModel>::Create(
          Eigen::Vector2d::Zero(), /*use_log_scale=*/false));
  rig_from_world[7] = 2;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 2);
}

TEST(RigReprojErrorConstantRigCostFunctor, Nominal) {
  Rigid3d cam_from_rig;
  cam_from_rig.translation() << 0, 0, -1;
  std::unique_ptr<ceres::CostFunction> cost_function(
      RigReprojErrorConstantRigCostFunctor<SimplePinholeCameraModel>::Create(
          Eigen::Vector2d::Zero(), cam_from_rig));

  double rig_from_world[7] = {0, 0, 0, 1, 0, 0, 1};
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[3] = {point3D, rig_from_world, camera_params};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);

  point3D[1] = 1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 1);

  camera_params[0] = 2;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 2);

  point3D[0] = -1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], -2);
  EXPECT_EQ(residuals[1], 2);
}

constexpr double kEquirectangularCameraWidth = 1000;
constexpr double kEquirectangularCameraHeight = 500;

TEST(WrapEquirectangularHorizontalSeam, Nominal) {
  const double camera_params[2] = {kEquirectangularCameraWidth,
                                   kEquirectangularCameraHeight};

  // No-op for non-periodic models, regardless of residual magnitude.
  {
    double residuals[2] = {993.0, -7.0};
    WrapEquirectangularHorizontalSeam<SimplePinholeCameraModel>(camera_params,
                                                                residuals);
    EXPECT_EQ(residuals[0], 993.0);
    EXPECT_EQ(residuals[1], -7.0);
  }

  // Equirectangular folds the x-residual into [-w/2, w/2); y is untouched.
  {
    double residuals[2] = {1000.0, 5.0};  // Exactly one period -> 0.
    WrapEquirectangularHorizontalSeam<EquirectangularCameraModel>(camera_params,
                                                                  residuals);
    EXPECT_EQ(residuals[0], 0.0);
    EXPECT_EQ(residuals[1], 5.0);
  }
  {
    double residuals[2] = {998.0, 0.0};  // Just under a period -> -2.
    WrapEquirectangularHorizontalSeam<EquirectangularCameraModel>(camera_params,
                                                                  residuals);
    EXPECT_EQ(residuals[0], -2.0);
  }
  {
    double residuals[2] = {-998.0, 0.0};
    WrapEquirectangularHorizontalSeam<EquirectangularCameraModel>(camera_params,
                                                                  residuals);
    EXPECT_EQ(residuals[0], 2.0);
  }
  {
    double residuals[2] = {-3.0, 0.0};  // Already minimal -> unchanged.
    WrapEquirectangularHorizontalSeam<EquirectangularCameraModel>(camera_params,
                                                                  residuals);
    EXPECT_EQ(residuals[0], -3.0);
  }
  {
    double residuals[2] = {500.0, 0.0};  // +w/2 boundary folds to -w/2.
    WrapEquirectangularHorizontalSeam<EquirectangularCameraModel>(camera_params,
                                                                  residuals);
    EXPECT_EQ(residuals[0], -500.0);
  }
}

template <typename CreateCostFunction>
void ExpectEquirectangularSeamWrap(
    const CreateCostFunction& create,
    const std::vector<const double*>& parameters) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      create(Eigen::Vector2d(2, kEquirectangularCameraHeight / 2)));
  double residuals[2];
  EXPECT_TRUE(cost_function->Evaluate(parameters.data(), residuals, nullptr));
  EXPECT_EQ(residuals[0], -2);  // Raw ~w folded into [-w/2, w/2).
  EXPECT_EQ(residuals[1], 0);
}

TEST(ReprojErrorCostFunctor, EquirectangularSeamWrap) {
  double cam_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double point3D[3] = {0, 0, -1};  // Azimuth = ±π -> projects to x = w.
  double camera_params[2] = {kEquirectangularCameraWidth,
                             kEquirectangularCameraHeight};
  ExpectEquirectangularSeamWrap(
      [](const Eigen::Vector2d& point2D) {
        return ReprojErrorCostFunctor<EquirectangularCameraModel>::Create(
            point2D);
      },
      {point3D, cam_from_world, camera_params});
}

TEST(ReprojErrorConstantPoseCostFunctor, EquirectangularSeamWrap) {
  const Rigid3d cam_from_world;
  double point3D[3] = {0, 0, -1};
  double camera_params[2] = {kEquirectangularCameraWidth,
                             kEquirectangularCameraHeight};
  ExpectEquirectangularSeamWrap(
      [&cam_from_world](const Eigen::Vector2d& point2D) {
        return ReprojErrorConstantPoseCostFunctor<
            EquirectangularCameraModel>::Create(point2D, cam_from_world);
      },
      {point3D, camera_params});
}

TEST(ReprojErrorConstantPoint3DCostFunctor, EquirectangularSeamWrap) {
  const Eigen::Vector3d point3D(0, 0, -1);
  double cam_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double camera_params[2] = {kEquirectangularCameraWidth,
                             kEquirectangularCameraHeight};
  ExpectEquirectangularSeamWrap(
      [&point3D](const Eigen::Vector2d& point2D) {
        return ReprojErrorConstantPoint3DCostFunctor<
            EquirectangularCameraModel>::Create(point2D, point3D);
      },
      {cam_from_world, camera_params});
}

TEST(RigReprojErrorCostFunctor, EquirectangularSeamWrap) {
  double cam_from_rig[7] = {0, 0, 0, 1, 0, 0, 0};
  double rig_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double point3D[3] = {0, 0, -1};
  double camera_params[2] = {kEquirectangularCameraWidth,
                             kEquirectangularCameraHeight};
  ExpectEquirectangularSeamWrap(
      [](const Eigen::Vector2d& point2D) {
        return RigReprojErrorCostFunctor<EquirectangularCameraModel>::Create(
            point2D);
      },
      {point3D, cam_from_rig, rig_from_world, camera_params});
}

TEST(ScaledRigReprojErrorCostFunctor, EquirectangularSeamWrap) {
  double cam_from_rig[7] = {0, 0, 0, 1, 0, 0, 0};
  double rig_from_world[8] = {0, 0, 0, 1, 0, 0, 0, 0};
  double point3D[3] = {0, 0, -1};
  double camera_params[2] = {kEquirectangularCameraWidth,
                             kEquirectangularCameraHeight};
  ExpectEquirectangularSeamWrap(
      [](const Eigen::Vector2d& point2D) {
        return ScaledRigReprojErrorCostFunctor<
            EquirectangularCameraModel>::Create(point2D);
      },
      {point3D, cam_from_rig, rig_from_world, camera_params});
}

TEST(RigReprojErrorConstantRigCostFunctor, EquirectangularSeamWrap) {
  const Rigid3d cam_from_rig;
  double rig_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double point3D[3] = {0, 0, -1};
  double camera_params[2] = {kEquirectangularCameraWidth,
                             kEquirectangularCameraHeight};
  ExpectEquirectangularSeamWrap(
      [&cam_from_rig](const Eigen::Vector2d& point2D) {
        return RigReprojErrorConstantRigCostFunctor<
            EquirectangularCameraModel>::Create(point2D, cam_from_rig);
      },
      {point3D, rig_from_world, camera_params});
}

}  // namespace
}  // namespace colmap

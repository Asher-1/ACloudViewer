// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#define TEST_NAME "base/cost_functions"
#include "util/testing.h"

#include <cmath>
#include <memory>
#include <vector>

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "base/pose.h"

using namespace colmap;

namespace {

void CheckCostFunctionJacobians(ceres::CostFunction* cost_function,
                                const std::vector<double*>& parameters) {
  constexpr double kStep = 1e-6;
  constexpr double kTolerance = 2e-4;
  const std::vector<int32_t>& block_sizes =
      cost_function->parameter_block_sizes();
  ASSERT_EQ(block_sizes.size(), parameters.size());

  std::vector<const double*> const_parameters(parameters.begin(),
                                              parameters.end());
  std::vector<std::vector<double>> analytic_jacobians;
  std::vector<double*> jacobian_ptrs;
  for (const int32_t block_size : block_sizes) {
    analytic_jacobians.emplace_back(2 * block_size);
    jacobian_ptrs.push_back(analytic_jacobians.back().data());
  }
  double residuals[2];
  ASSERT_TRUE(cost_function->Evaluate(const_parameters.data(), residuals,
                                       jacobian_ptrs.data()));

  for (size_t block = 0; block < parameters.size(); ++block) {
    for (int32_t column = 0; column < block_sizes[block]; ++column) {
      double& parameter = parameters[block][column];
      parameter += kStep;
      double plus[2];
      ASSERT_TRUE(cost_function->Evaluate(const_parameters.data(), plus,
                                           nullptr));
      parameter -= 2.0 * kStep;
      double minus[2];
      ASSERT_TRUE(cost_function->Evaluate(const_parameters.data(), minus,
                                           nullptr));
      parameter += kStep;
      for (int residual = 0; residual < 2; ++residual) {
        const double numeric = (plus[residual] - minus[residual]) /
                               (2.0 * kStep);
        const double analytic = analytic_jacobians[block]
                                [residual * block_sizes[block] + column];
        ASSERT_LE(std::abs(analytic - numeric), kTolerance);
      }
    }
  }
}

}  // namespace

TEST(base_cost_functions, TestBundleAdjustmentCostFunction) {
  ceres::CostFunction* cost_function =
      BundleAdjustmentCostFunction<SimplePinholeCameraModel>::Create(
          Eigen::Vector2d::Zero());
  double qvec[4] = {1, 0, 0, 0};
  double tvec[3] = {0, 0, 0};
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[4] = {qvec, tvec, point3D, camera_params};
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

TEST(base_cost_functions, TestBundleAdjustmentConstantPoseCostFunction) {
  ceres::CostFunction* cost_function = BundleAdjustmentConstantPoseCostFunction<
      SimplePinholeCameraModel>::Create(ComposeIdentityQuaternion(),
                                        Eigen::Vector3d::Zero(),
                                        Eigen::Vector2d::Zero());
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

TEST(base_cost_functions, TestRigBundleAdjustmentCostFunction) {
  ceres::CostFunction* cost_function =
      RigBundleAdjustmentCostFunction<SimplePinholeCameraModel>::Create(
          Eigen::Vector2d::Zero());
  double rig_qvec[4] = {1, 0, 0, 0};
  double rig_tvec[3] = {0, 0, -1};
  double rel_qvec[4] = {1, 0, 0, 0};
  double rel_tvec[3] = {0, 0, 1};
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[6] = {rig_qvec, rig_tvec, rel_qvec,
                                 rel_tvec, point3D,  camera_params};
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

TEST(base_cost_functions, TestEquirectangularBundleAdjustmentCostFunctions) {
  const double pi = EIGEN_PI;
  const double observed_near_left[2] = {0.5, 250.0};
  const double camera_params[2] = {1000.0, 500.0};
  double qvec[4] = {1, 0, 0, 0};
  double tvec[3] = {0, 0, 0};
  // A bearing just across the +/- pi seam from the observation. A perspective
  // residual would also reject this back-hemisphere point after division by Z.
  double point3D[3] = {-std::sin(0.001 * pi), 0, -std::cos(0.001 * pi)};
  double residuals[2];
  const double* parameters[4] = {qvec, tvec, point3D, camera_params};
  std::unique_ptr<ceres::CostFunction> variable_cost(
      EquirectangularBundleAdjustmentCostFunction::Create(
          Eigen::Vector2d(observed_near_left[0], observed_near_left[1])));
  ASSERT_TRUE(variable_cost->Evaluate(parameters, residuals, nullptr));
  ASSERT_LE(std::abs(residuals[0]), 1e-10);
  ASSERT_LE(std::abs(residuals[1]), 1e-10);

  std::unique_ptr<ceres::CostFunction> constant_cost(
      EquirectangularBundleAdjustmentConstantPoseCostFunction::Create(
          ComposeIdentityQuaternion(), Eigen::Vector3d::Zero(),
          Eigen::Vector2d(observed_near_left[0], observed_near_left[1])));
  const double* constant_parameters[2] = {point3D, camera_params};
  ASSERT_TRUE(constant_cost->Evaluate(constant_parameters, residuals, nullptr));
  ASSERT_LE(std::abs(residuals[0]), 1e-10);
  ASSERT_LE(std::abs(residuals[1]), 1e-10);

  std::unique_ptr<ceres::CostFunction> rig_cost(
      EquirectangularRigBundleAdjustmentCostFunction::Create(
          Eigen::Vector2d(observed_near_left[0], observed_near_left[1])));
  double rel_qvec[4] = {1, 0, 0, 0};
  double rel_tvec[3] = {0, 0, 0};
  const double* rig_parameters[6] = {qvec, tvec, rel_qvec,
                                     rel_tvec, point3D, camera_params};
  ASSERT_TRUE(rig_cost->Evaluate(rig_parameters, residuals, nullptr));
  ASSERT_LE(std::abs(residuals[0]), 1e-10);
  ASSERT_LE(std::abs(residuals[1]), 1e-10);
}

TEST(base_cost_functions, TestEquirectangularBundleAdjustmentJacobians) {
  // This generic bearing stays away from the longitude seam and both poles,
  // so central differences test the smooth residual branch used by BA.
  const Eigen::Vector2d observation(571.0, 227.0);
  double camera_params[2] = {1000.0, 500.0};
  double point3D[3] = {1.2, -0.4, 2.5};
  double qvec[4] = {std::cos(0.1), 0.0, std::sin(0.1), 0.0};
  double tvec[3] = {0.1, -0.2, 0.3};

  std::unique_ptr<ceres::CostFunction> variable_cost(
      EquirectangularBundleAdjustmentCostFunction::Create(observation));
  CheckCostFunctionJacobians(variable_cost.get(),
                             {qvec, tvec, point3D, camera_params});

  std::unique_ptr<ceres::CostFunction> constant_cost(
      EquirectangularBundleAdjustmentConstantPoseCostFunction::Create(
          Eigen::Map<Eigen::Vector4d>(qvec), Eigen::Map<Eigen::Vector3d>(tvec),
          observation));
  CheckCostFunctionJacobians(constant_cost.get(), {point3D, camera_params});

  double rig_qvec[4] = {std::cos(0.15), std::sin(0.15), 0.0, 0.0};
  double rig_tvec[3] = {-0.2, 0.1, 0.4};
  double rel_qvec[4] = {std::cos(0.05), 0.0, 0.0, std::sin(0.05)};
  double rel_tvec[3] = {0.05, -0.1, 0.02};
  std::unique_ptr<ceres::CostFunction> rig_cost(
      EquirectangularRigBundleAdjustmentCostFunction::Create(observation));
  CheckCostFunctionJacobians(rig_cost.get(),
                             {rig_qvec, rig_tvec, rel_qvec, rel_tvec,
                              point3D, camera_params});
}

TEST(base_cost_functions, TestRelativePoseCostFunction) {
  ceres::CostFunction* cost_function = RelativePoseCostFunction::Create(
      Eigen::Vector2d(0, 0), Eigen::Vector2d(0, 0));
  double qvec[4] = {1, 0, 0, 0};
  double tvec[3] = {0, 1, 0};
  double residuals[1];
  const double* parameters[2] = {qvec, tvec};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);

  cost_function = RelativePoseCostFunction::Create(Eigen::Vector2d(0, 0),
                                                   Eigen::Vector2d(1, 0));
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0.5);

  cost_function = RelativePoseCostFunction::Create(Eigen::Vector2d(0, 0),
                                                   Eigen::Vector2d(1, 1));
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0.5);
}

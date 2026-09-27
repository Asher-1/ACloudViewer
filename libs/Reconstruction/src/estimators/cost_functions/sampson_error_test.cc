// SPDX-License-Identifier: BSD-3-Clause
//
// Ported from upstream MVS/colmap d3ccaf35 (batch 24 test alignment);
// include paths mapped to the fork layout.

#include "estimators/cost_functions/sampson_error.h"

#include "util/eigen_matchers.h"

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(SampsonErrorCostFunctor, Nominal) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      SampsonErrorCostFunctor::Create(Eigen::Vector2d(0, 0),
                                      Eigen::Vector2d(0, 0)));
  double cam_from_world[7] = {0, 0, 0, 1, 0, 1, 0};
  double residuals[1];
  const double* parameters[1] = {cam_from_world};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);

  cost_function.reset(SampsonErrorCostFunctor::Create(Eigen::Vector2d(0, 0),
                                                      Eigen::Vector2d(1, 0)));
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(residuals[0] * residuals[0], 0.5, 1e-6);

  cost_function.reset(SampsonErrorCostFunctor::Create(Eigen::Vector2d(0, 0),
                                                      Eigen::Vector2d(1, 1)));
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(residuals[0] * residuals[0], 0.5, 1e-6);
}

// Fork adjudication: upstream's RelPoseParams packing helpers do not exist
// in the fork's sampson_error.h; the SampsonError cost functor is verified
// above and the packing helper round-trip is deferred until the fork adopts
// the upstream parameter packing.
}  // namespace
}  // namespace colmap

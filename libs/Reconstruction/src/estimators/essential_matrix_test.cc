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

#define TEST_NAME "estimators/essential_matrix"
#include "util/testing.h"

#include <cmath>

#include <Eigen/Core>

#include "base/camera_models.h"
#include "base/camera.h"
#include "base/essential_matrix.h"
#include "base/pose.h"
#include "base/projection.h"
#include "estimators/essential_matrix.h"
#include "optim/ransac.h"
#include "util/random.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestCamRayWithJacPinholeNumericalGate) {
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::kModelId, 800.0, 1600,
                          1200);
  const Eigen::Vector2d pixel(960.0, 480.0);
  const auto ray_with_jac = camera.CamRayFromImgWithJac(pixel);
  BOOST_REQUIRE(ray_with_jac.has_value());

  const Eigen::Vector3d unnormalized(0.2, -0.15, 1.0);
  const Eigen::Vector3d expected_ray = unnormalized.normalized();
  const Eigen::Matrix3d normalizer =
          (Eigen::Matrix3d::Identity() - expected_ray * expected_ray.transpose()) /
          unnormalized.norm();
  Eigen::Matrix<double, 3, 2> expected_jac;
  expected_jac.col(0) = normalizer * Eigen::Vector3d(1.0 / 800.0, 0, 0);
  expected_jac.col(1) = normalizer * Eigen::Vector3d(0, 1.0 / 800.0, 0);

  BOOST_CHECK_SMALL((ray_with_jac->ray - expected_ray).norm(), 1e-12);
  BOOST_CHECK_SMALL((ray_with_jac->jacobian - expected_jac).norm(), 1e-9);
}

BOOST_AUTO_TEST_CASE(TestTangentSampsonNumericalGate) {
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::kModelId, 900.0, 1800,
                          1200);
  const Eigen::Vector2d point1(950, 640);
  const Eigen::Vector2d point2(1000, 640);
  const auto ray1 = camera.CamRayFromImgWithJac(point1);
  const auto ray2 = camera.CamRayFromImgWithJac(point2);
  BOOST_REQUIRE(ray1.has_value());
  BOOST_REQUIRE(ray2.has_value());

  Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
  E(1, 2) = -1.0;
  E(2, 1) = 1.0;
  const double error = ComputeSquaredTangentSampsonError(*ray1, *ray2, E);
  BOOST_CHECK(std::isfinite(error));
  BOOST_CHECK_LT(error, 1e-12);

  // Compare the tangent denominator with an independently evaluated centered
  // finite-difference pixel gradient. This is the numerical acceptance gate
  // used by ray-based solvers, not merely a zero-residual smoke test.
  const Eigen::Vector2d perturbation(0.35, -0.45);
  const auto perturbed_ray2 =
          camera.CamRayFromImgWithJac(point2 + perturbation);
  BOOST_REQUIRE(perturbed_ray2.has_value());
  const double tangent_error =
          ComputeSquaredTangentSampsonError(*ray1, *perturbed_ray2, E);
  const double residual = perturbed_ray2->ray.dot(E * ray1->ray);
  constexpr double kStep = 1e-4;
  Eigen::Vector2d numerical_gradient;
  for (int axis = 0; axis < 2; ++axis) {
    Eigen::Vector2d backward = point2 + perturbation;
    Eigen::Vector2d forward = point2 + perturbation;
    backward[axis] -= kStep;
    forward[axis] += kStep;
    const auto ray_backward = camera.CamRayFromImgWithJac(backward);
    const auto ray_forward = camera.CamRayFromImgWithJac(forward);
    BOOST_REQUIRE(ray_backward.has_value());
    BOOST_REQUIRE(ray_forward.has_value());
    numerical_gradient[axis] =
            (ray_forward->ray.dot(E * ray1->ray) -
             ray_backward->ray.dot(E * ray1->ray)) /
            (2.0 * kStep);
  }
  const Eigen::Vector2d gradient1 =
          ray1->jacobian.transpose() * E.transpose() * perturbed_ray2->ray;
  const double expected = residual * residual /
                          (gradient1.squaredNorm() +
                           numerical_gradient.squaredNorm());
  BOOST_CHECK_SMALL(tangent_error - expected, 1e-9);
}

BOOST_AUTO_TEST_CASE(TestFivePoint) {
  const double points1_raw[] = {
      0.4964, 1.0577, 0.3650,  -0.0919, -0.5412, 0.0159, -0.5239, 0.9467,
      0.3467, 0.5301, 0.2797,  0.0012,  -0.1986, 0.0460, -0.1622, 0.5347,
      0.0796, 0.2379, -0.3946, 0.7969,  0.2,     0.7,    0.6,     0.3};

  const double points2_raw[] = {
      0.7570, 2.7340, 0.3961,  0.6981, -0.6014, 0.7110, -0.7385, 2.2712,
      0.4177, 1.2132, 0.3052,  0.4835, -0.2171, 0.5057, -0.2059, 1.1583,
      0.0946, 0.7013, -0.6236, 3.0253, 0.5,     0.9,    0.9,     0.2};

  const size_t num_points = 12;

  std::vector<Eigen::Vector2d> points1(num_points);
  std::vector<Eigen::Vector2d> points2(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    points1[i] = Eigen::Vector2d(points1_raw[2 * i], points1_raw[2 * i + 1]);
    points2[i] = Eigen::Vector2d(points2_raw[2 * i], points2_raw[2 * i + 1]);
  }

  // Enforce repeatable tests
  SetPRNGSeed(0);

  RANSACOptions options;
  options.max_error = 0.02;
  options.confidence = 0.9999;
  options.min_inlier_ratio = 0.1;

  RANSAC<EssentialMatrixFivePointEstimator> ransac(options);

  const auto report = ransac.Estimate(points1, points2);

  std::vector<double> residuals;
  EssentialMatrixFivePointEstimator::Residuals(points1, points2, report.model,
                                               &residuals);

  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_LE(residuals[i], options.max_error * options.max_error);
  }

  BOOST_CHECK(!report.inlier_mask[10]);
  BOOST_CHECK(!report.inlier_mask[11]);
}

BOOST_AUTO_TEST_CASE(TestEightPoint) {
  const double points1_raw[] = {1.839035, 1.924743, 0.543582,  0.375221,
                                0.473240, 0.142522, 0.964910,  0.598376,
                                0.102388, 0.140092, 15.994343, 9.622164,
                                0.285901, 0.430055, 0.091150,  0.254594};

  const double points2_raw[] = {
      1.002114, 1.129644, 1.521742, 1.846002, 1.084332, 0.275134,
      0.293328, 0.588992, 0.839509, 0.087290, 1.779735, 1.116857,
      0.878616, 0.602447, 0.642616, 1.028681,
  };

  const size_t kNumPoints = 8;
  std::vector<Eigen::Vector2d> points1(kNumPoints);
  std::vector<Eigen::Vector2d> points2(kNumPoints);
  for (size_t i = 0; i < kNumPoints; ++i) {
    points1[i] = Eigen::Vector2d(points1_raw[2 * i], points1_raw[2 * i + 1]);
    points2[i] = Eigen::Vector2d(points2_raw[2 * i], points2_raw[2 * i + 1]);
  }

  EssentialMatrixEightPointEstimator estimator;
  const auto E = estimator.Estimate(points1, points2)[0];

  // Reference values.
  BOOST_CHECK(std::abs(E(0, 0) - -0.0368602) < 1e-5);
  BOOST_CHECK(std::abs(E(0, 1) - 0.265019) < 1e-5);
  BOOST_CHECK(std::abs(E(0, 2) - -0.0625948) < 1e-5);
  BOOST_CHECK(std::abs(E(1, 0) - -0.299679) < 1e-5);
  BOOST_CHECK(std::abs(E(1, 1) - -0.110667) < 1e-5);
  BOOST_CHECK(std::abs(E(1, 2) - 0.147114) < 1e-5);
  BOOST_CHECK(std::abs(E(2, 0) - 0.169381) < 1e-5);
  BOOST_CHECK(std::abs(E(2, 1) - -0.21072) < 1e-5);
  BOOST_CHECK(std::abs(E(2, 2) - -0.00401306) < 1e-5);

  // Check that the internal constraint is satisfied (two singular values equal
  // and one zero).
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(E);
  Eigen::Vector3d s = svd.singularValues();
  BOOST_CHECK(std::abs(s(0) - s(1)) < 1e-5);
  BOOST_CHECK(std::abs(s(2)) < 1e-5);
}

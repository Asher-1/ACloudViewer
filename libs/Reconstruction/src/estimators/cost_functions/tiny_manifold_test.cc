// SPDX-License-Identifier: BSD-3-Clause
//
// Ported from upstream MVS/colmap d3ccaf35 (batch 24 test alignment);
// include paths mapped to the fork layout.

#include "estimators/cost_functions/tiny_manifold.h"

#include "optim/tiny_solver.h"
#include "util/eigen_matchers.h"

#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Central finite-difference approximation of the (ambient x tangent) Jacobian
// of Manifold::Plus at delta = 0.
template <typename Manifold>
Eigen::MatrixXd NumericPlusJacobian(const Manifold& manifold, const double* x) {
  constexpr int kAmbient = Manifold::kAmbientSize;
  constexpr int kTangent = Manifold::kTangentSize;
  constexpr double kEps = 1e-6;
  Eigen::MatrixXd J(kAmbient, kTangent);
  for (int j = 0; j < kTangent; ++j) {
    std::vector<double> delta_plus(kTangent, 0.0);
    std::vector<double> delta_minus(kTangent, 0.0);
    delta_plus[j] = kEps;
    delta_minus[j] = -kEps;
    std::vector<double> x_plus(kAmbient);
    std::vector<double> x_minus(kAmbient);
    manifold.Plus(x, delta_plus.data(), x_plus.data());
    manifold.Plus(x, delta_minus.data(), x_minus.data());
    for (int r = 0; r < kAmbient; ++r) {
      J(r, j) = (x_plus[r] - x_minus[r]) / (2 * kEps);
    }
  }
  return J;
}

// Row-major analytic Jacobian returned by Manifold::PlusJacobian, as a matrix.
template <typename Manifold>
Eigen::MatrixXd AnalyticPlusJacobian(const Manifold& manifold,
                                     const double* x) {
  constexpr int kAmbient = Manifold::kAmbientSize;
  constexpr int kTangent = Manifold::kTangentSize;
  std::vector<double> data(kAmbient * kTangent);
  manifold.PlusJacobian(x, data.data());
  Eigen::MatrixXd J(kAmbient, kTangent);
  for (int r = 0; r < kAmbient; ++r) {
    for (int c = 0; c < kTangent; ++c) {
      J(r, c) = data[r * kTangent + c];
    }
  }
  return J;
}

TEST(EigenQuaternionManifold, PlusAtZeroIsIdentity) {
  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()));
  const EigenQuaternionManifold manifold;
  const double delta[3] = {0, 0, 0};
  double x_plus[4];
  manifold.Plus(q.coeffs().data(), delta, x_plus);
  EXPECT_THAT(Eigen::Map<const Eigen::Vector4d>(x_plus),
              EigenMatrixNear(Eigen::Vector4d(q.coeffs()), 1e-12));
}

TEST(EigenQuaternionManifold, PlusComposesRotationOnTheRight) {
  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()));
  const Eigen::Vector3d delta(0.05, -0.1, 0.08);
  const EigenQuaternionManifold manifold;
  double x_plus[4];
  manifold.Plus(q.coeffs().data(), delta.data(), x_plus);

  const Eigen::Map<const Eigen::Quaterniond> q_plus_map(x_plus);
  const Eigen::Quaterniond q_plus(q_plus_map);
  const Eigen::Matrix3d expected =
      q.toRotationMatrix() *
      Eigen::AngleAxisd(delta.norm(), delta.normalized()).toRotationMatrix();
  EXPECT_THAT(q_plus.toRotationMatrix(), EigenMatrixNear(expected, 1e-10));
}

TEST(EigenQuaternionManifold, PlusJacobianMatchesFiniteDiff) {
  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()));
  const EigenQuaternionManifold manifold;
  EXPECT_THAT(
      AnalyticPlusJacobian(manifold, q.coeffs().data()),
      EigenMatrixNear(NumericPlusJacobian(manifold, q.coeffs().data()), 1e-6));
}

TEST(SphereManifold, PlusStaysOnUnitSphere) {
  const Eigen::Vector3d x = Eigen::Vector3d(0.5, -1.0, 2.0).normalized();
  const SphereManifold<3> manifold;
  const Eigen::Vector2d delta(0.15, -0.2);
  double x_plus[3];
  manifold.Plus(x.data(), delta.data(), x_plus);
  EXPECT_NEAR(Eigen::Map<const Eigen::Vector3d>(x_plus).norm(), 1.0, 1e-12);

  const double zero[2] = {0, 0};
  double x_plus_zero[3];
  manifold.Plus(x.data(), zero, x_plus_zero);
  EXPECT_THAT(Eigen::Map<const Eigen::Vector3d>(x_plus_zero),
              EigenMatrixNear(x, 1e-12));
}

TEST(SphereManifold, PlusJacobianMatchesFiniteDiff) {
  const Eigen::Vector3d x = Eigen::Vector3d(0.5, -1.0, 2.0).normalized();
  const SphereManifold<3> manifold;
  // The columns span the tangent plane, i.e. are orthogonal to x.
  const Eigen::MatrixXd J = AnalyticPlusJacobian(manifold, x.data());
  EXPECT_LT((x.transpose() * J).norm(), 1e-12);
  EXPECT_THAT(J,
              EigenMatrixNear(NumericPlusJacobian(manifold, x.data()), 1e-6));
}

TEST(EigenQuaternionManifold, PlusStaysNormalized) {
  const Eigen::Quaterniond q(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()));
  const EigenQuaternionManifold manifold;
  // A large step, unlike the small deltas of the rotation-composition test.
  const Eigen::Vector3d delta(0.5, -1.2, 2.0);
  double x_plus[4];
  manifold.Plus(q.coeffs().data(), delta.data(), x_plus);
  EXPECT_NEAR(Eigen::Map<const Eigen::Vector4d>(x_plus).norm(), 1.0, 1e-12);
}

// Fork adjudication: the fork's SphereManifold is only implemented for the
// 2-sphere (kAmbientSize = 3); upstream's 9-dimensional instantiation is
// deferred until the fork generalizes the manifold.

}  // namespace
}  // namespace colmap

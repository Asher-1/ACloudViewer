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

#define TEST_NAME "base/pose"
#include "util/testing.h"

#include <Eigen/Core>

#include "geometry/pose.h"

#include "math/random_eigen.h"
#include "util/eigen_matchers.h"
#include "scene/projection.h"
#include "math/math.h"

using namespace colmap;

TEST(base_pose, TestCrossProductMatrix) {
  EXPECT_EQ(CrossProductMatrix(Eigen::Vector3d(0, 0, 0)),
                    Eigen::Matrix3d::Zero());
  Eigen::Matrix3d ref_matrix;
  ref_matrix << 0, -3, 2, 3, 0, -1, -2, 1, 0;
  EXPECT_EQ(CrossProductMatrix(Eigen::Vector3d(1, 2, 3)), ref_matrix);
}

TEST(base_pose, TestEulerAnglesX) {
  const double rx = 0.3;
  const double ry = 0;
  const double rz = 0;
  double rxx, ryy, rzz;

  RotationMatrixToEulerAngles(EulerAnglesToRotationMatrix(rx, ry, rz), &rxx,
                              &ryy, &rzz);

  EXPECT_NEAR(rx, rxx, std::abs(rxx) * (1e-6) / 100.0);
  EXPECT_NEAR(ry, ryy, std::abs(ryy) * (1e-6) / 100.0);
  EXPECT_NEAR(rz, rzz, std::abs(rzz) * (1e-6) / 100.0);
}

TEST(base_pose, TestEulerAnglesY) {
  const double rx = 0;
  const double ry = 0.3;
  const double rz = 0;
  double rxx, ryy, rzz;

  RotationMatrixToEulerAngles(EulerAnglesToRotationMatrix(rx, ry, rz), &rxx,
                              &ryy, &rzz);

  EXPECT_NEAR(rx, rxx, std::abs(rxx) * (1e-6) / 100.0);
  EXPECT_NEAR(ry, ryy, std::abs(ryy) * (1e-6) / 100.0);
  EXPECT_NEAR(rz, rzz, std::abs(rzz) * (1e-6) / 100.0);
}

TEST(base_pose, TestEulerAnglesZ) {
  const double rx = 0;
  const double ry = 0;
  const double rz = 0.3;
  double rxx, ryy, rzz;

  RotationMatrixToEulerAngles(EulerAnglesToRotationMatrix(rx, ry, rz), &rxx,
                              &ryy, &rzz);

  EXPECT_NEAR(rx, rxx, std::abs(rxx) * (1e-6) / 100.0);
  EXPECT_NEAR(ry, ryy, std::abs(ryy) * (1e-6) / 100.0);
  EXPECT_NEAR(rz, rzz, std::abs(rzz) * (1e-6) / 100.0);
}

TEST(base_pose, TestEulerAnglesXYZ) {
  const double rx = 0.1;
  const double ry = 0.2;
  const double rz = 0.3;
  double rxx, ryy, rzz;

  RotationMatrixToEulerAngles(EulerAnglesToRotationMatrix(rx, ry, rz), &rxx,
                              &ryy, &rzz);

  EXPECT_NEAR(rx, rxx, std::abs(rxx) * (1e-6) / 100.0);
  EXPECT_NEAR(ry, ryy, std::abs(ryy) * (1e-6) / 100.0);
  EXPECT_NEAR(rz, rzz, std::abs(rzz) * (1e-6) / 100.0);
}

TEST(base_pose, TestQuaternionToRotationMatrix) {
  const double rx = 0;
  const double ry = 0;
  const double rz = 0.3;
  const Eigen::Matrix3d rot_mat0 = EulerAnglesToRotationMatrix(rx, ry, rz);
  const Eigen::Matrix3d rot_mat1 =
      QuaternionToRotationMatrix(RotationMatrixToQuaternion(rot_mat0));
  EXPECT_TRUE(rot_mat0.isApprox(rot_mat1));
}

TEST(base_pose, TestComposeIdentityQuaternion) {
  EXPECT_EQ(ComposeIdentityQuaternion(), Eigen::Vector4d(1, 0, 0, 0));
}

TEST(base_pose, TestNormalizeQuaternion) {
  EXPECT_EQ(NormalizeQuaternion(ComposeIdentityQuaternion()),
                    ComposeIdentityQuaternion());
  EXPECT_EQ(NormalizeQuaternion(Eigen::Vector4d(2, 0, 0, 0)),
                    ComposeIdentityQuaternion());
  EXPECT_EQ(NormalizeQuaternion(Eigen::Vector4d(0.5, 0, 0, 0)),
                    ComposeIdentityQuaternion());
  EXPECT_EQ(NormalizeQuaternion(Eigen::Vector4d(0, 0, 0, 0)),
                    ComposeIdentityQuaternion());
  EXPECT_TRUE(
      NormalizeQuaternion(Eigen::Vector4d(1, 1, 0, 0))
          .isApprox(Eigen::Vector4d(std::sqrt(2) / 2, std::sqrt(2) / 2, 0, 0)));
  EXPECT_TRUE(
      NormalizeQuaternion(Eigen::Vector4d(0.5, 0.5, 0, 0))
          .isApprox(Eigen::Vector4d(std::sqrt(2) / 2, std::sqrt(2) / 2, 0, 0)));
}

TEST(base_pose, TestInvertQuaternion) {
  EXPECT_EQ(InvertQuaternion(ComposeIdentityQuaternion()),
                    Eigen::Vector4d(1, -0, -0, -0));
  EXPECT_EQ(InvertQuaternion(Eigen::Vector4d(2, 0, 0, 0)),
                    Eigen::Vector4d(2, -0, -0, -0));
  EXPECT_EQ(
      InvertQuaternion(InvertQuaternion(Eigen::Vector4d(0.1, 0.2, 0.3, 0.4))),
      Eigen::Vector4d(0.1, 0.2, 0.3, 0.4));
}

TEST(base_pose, TestConcatenateQuaternions) {
  EXPECT_EQ(ConcatenateQuaternions(ComposeIdentityQuaternion(),
                                           ComposeIdentityQuaternion()),
                    ComposeIdentityQuaternion());
  EXPECT_EQ(ConcatenateQuaternions(Eigen::Vector4d(2, 0, 0, 0),
                                           ComposeIdentityQuaternion()),
                    ComposeIdentityQuaternion());
  EXPECT_EQ(ConcatenateQuaternions(ComposeIdentityQuaternion(),
                                           Eigen::Vector4d(2, 0, 0, 0)),
                    ComposeIdentityQuaternion());
  EXPECT_TRUE(ConcatenateQuaternions(
                  Eigen::Vector4d(0.1, 0.2, 0.3, 0.4),
                  InvertQuaternion(Eigen::Vector4d(0.1, 0.2, 0.3, 0.4)))
                  .isApprox(ComposeIdentityQuaternion()));
  EXPECT_TRUE(ConcatenateQuaternions(
                  InvertQuaternion(Eigen::Vector4d(0.1, 0.2, 0.3, 0.4)),
                  Eigen::Vector4d(0.1, 0.2, 0.3, 0.4))
                  .isApprox(ComposeIdentityQuaternion()));
}

TEST(base_pose, TestQuaternionRotatePoint) {
  EXPECT_EQ(QuaternionRotatePoint(ComposeIdentityQuaternion(),
                                          Eigen::Vector3d(0, 0, 0)),
                    Eigen::Vector3d(0, 0, 0));
  EXPECT_EQ(QuaternionRotatePoint(Eigen::Vector4d(0.1, 0, 0, 0),
                                          Eigen::Vector3d(0, 0, 0)),
                    Eigen::Vector3d(0, 0, 0));
  EXPECT_EQ(QuaternionRotatePoint(ComposeIdentityQuaternion(),
                                          Eigen::Vector3d(1, 1, 0)),
                    Eigen::Vector3d(1, 1, 0));
  EXPECT_EQ(QuaternionRotatePoint(Eigen::Vector4d(0.1, 0, 0, 0),
                                          Eigen::Vector3d(1, 1, 0)),
                    Eigen::Vector3d(1, 1, 0));
  EXPECT_TRUE(
      QuaternionRotatePoint(
          RotationMatrixToQuaternion(EulerAnglesToRotationMatrix(M_PI, 0, 0)),
          Eigen::Vector3d(1, 1, 0))
          .isApprox(Eigen::Vector3d(1, -1, 0)));
}

TEST(base_pose, TestAverageQuaternions) {
  std::vector<Eigen::Vector4d> qvecs;
  std::vector<double> weights;

  qvecs = {{ComposeIdentityQuaternion()}};
  weights = {1.0};
  EXPECT_EQ(AverageQuaternions(qvecs, weights),
                    ComposeIdentityQuaternion());

  qvecs = {ComposeIdentityQuaternion()};
  weights = {2.0};
  EXPECT_EQ(AverageQuaternions(qvecs, weights),
                    ComposeIdentityQuaternion());

  qvecs = {ComposeIdentityQuaternion(), ComposeIdentityQuaternion()};
  weights = {1.0, 1.0};
  EXPECT_EQ(AverageQuaternions(qvecs, weights),
                    ComposeIdentityQuaternion());

  qvecs = {ComposeIdentityQuaternion(), ComposeIdentityQuaternion()};
  weights = {1.0, 2.0};
  EXPECT_EQ(AverageQuaternions(qvecs, weights),
                    ComposeIdentityQuaternion());

  qvecs = {ComposeIdentityQuaternion(), Eigen::Vector4d(2, 0, 0, 0)};
  weights = {1.0, 2.0};
  EXPECT_EQ(AverageQuaternions(qvecs, weights),
                    ComposeIdentityQuaternion());

  qvecs = {ComposeIdentityQuaternion(), Eigen::Vector4d(1, 1, 0, 0)};
  weights = {1.0, 1.0};
  EXPECT_TRUE(AverageQuaternions(qvecs, weights)
                  .isApprox(Eigen::Vector4d(0.92388, 0.382683, 0, 0), 1e-6));

  qvecs = {ComposeIdentityQuaternion(), Eigen::Vector4d(1, 1, 0, 0)};
  weights = {1.0, 2.0};
  EXPECT_TRUE(AverageQuaternions(qvecs, weights)
                  .isApprox(Eigen::Vector4d(0.850651, 0.525731, 0, 0), 1e-6));
}

TEST(base_pose, TestRotationFromUnitVectors) {
  EXPECT_EQ(RotationFromUnitVectors(Eigen::Vector3d(0, 0, 1),
                                            Eigen::Vector3d(0, 0, 1)),
                    Eigen::Matrix3d::Identity());
  EXPECT_EQ(RotationFromUnitVectors(Eigen::Vector3d(0, 0, 2),
                                            Eigen::Vector3d(0, 0, 2)),
                    Eigen::Matrix3d::Identity());

  Eigen::Matrix3d ref_matrix1;
  ref_matrix1 << 1, 0, 0, 0, 0, 1, 0, -1, 0;
  EXPECT_EQ(RotationFromUnitVectors(Eigen::Vector3d(0, 0, 1),
                                            Eigen::Vector3d(0, 1, 0)),
                    ref_matrix1);
  EXPECT_EQ(ref_matrix1 * Eigen::Vector3d(0, 0, 1),
                    Eigen::Vector3d(0, 1, 0));
  EXPECT_EQ(RotationFromUnitVectors(Eigen::Vector3d(0, 0, 2),
                                            Eigen::Vector3d(0, 2, 0)),
                    ref_matrix1);
  EXPECT_EQ(ref_matrix1 * Eigen::Vector3d(0, 0, 2),
                    Eigen::Vector3d(0, 2, 0));

  EXPECT_EQ(RotationFromUnitVectors(Eigen::Vector3d(0, 0, 1),
                                            Eigen::Vector3d(0, 0, -1)),
                    Eigen::Matrix3d::Identity());
}

TEST(base_pose, TestPoseFromProjectionMatrix) {
  const Eigen::Vector4d qvec = Eigen::Vector4d::Random().normalized();
  const Eigen::Vector3d tvec(3, 4, 5);
  const Eigen::Matrix3x4d proj_matrix = ComposeProjectionMatrix(qvec, tvec);
  const Eigen::Matrix3x4d inv_proj_matrix = InvertProjectionMatrix(proj_matrix);
  const Eigen::Vector3d pose = ProjectionCenterFromMatrix(proj_matrix);
  EXPECT_NEAR((inv_proj_matrix.rightCols<1>() - pose).norm(), 0, std::abs(0) * (1e-6) / 100.0);
}

TEST(base_pose, TestPoseFromProjectionParameters) {
  const Eigen::Vector4d qvec = Eigen::Vector4d::Random().normalized();
  const Eigen::Vector3d tvec(3, 4, 5);
  const Eigen::Matrix3x4d proj_matrix = ComposeProjectionMatrix(qvec, tvec);
  const Eigen::Matrix3x4d inv_proj_matrix = InvertProjectionMatrix(proj_matrix);
  const Eigen::Vector3d pose = ProjectionCenterFromPose(qvec, tvec);
  EXPECT_TRUE((inv_proj_matrix.rightCols<1>() - pose).norm() < 1e-6);
}

TEST(base_pose, TestComputeRelativePose) {
  Eigen::Vector4d qvec12;
  Eigen::Vector3d tvec12;

  ComputeRelativePose(ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0),
                      ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0),
                      &qvec12, &tvec12);
  EXPECT_EQ(qvec12, ComposeIdentityQuaternion());
  EXPECT_EQ(tvec12, Eigen::Vector3d(0, 0, 0));

  ComputeRelativePose(ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0),
                      ComposeIdentityQuaternion(), Eigen::Vector3d(1, 0, 0),
                      &qvec12, &tvec12);
  EXPECT_EQ(qvec12, ComposeIdentityQuaternion());
  EXPECT_EQ(tvec12, Eigen::Vector3d(1, 0, 0));

  ComputeRelativePose(ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0),
                      Eigen::Vector4d(1, 1, 0, 0), Eigen::Vector3d(0, 0, 0),
                      &qvec12, &tvec12);
  EXPECT_TRUE(qvec12.isApprox(Eigen::Vector4d(0.707107, 0.707107, 0, 0), 1e-6));
  EXPECT_EQ(tvec12, Eigen::Vector3d(0, 0, 0));

  ComputeRelativePose(ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0),
                      Eigen::Vector4d(1, 1, 0, 0), Eigen::Vector3d(1, 0, 0),
                      &qvec12, &tvec12);
  EXPECT_TRUE(qvec12.isApprox(Eigen::Vector4d(0.707107, 0.707107, 0, 0), 1e-6));
  EXPECT_EQ(tvec12, Eigen::Vector3d(1, 0, 0));

  ComputeRelativePose(Eigen::Vector4d(1, 1, 0, 0), Eigen::Vector3d(0, 0, 0),
                      Eigen::Vector4d(1, 1, 0, 0), Eigen::Vector3d(1, 0, 0),
                      &qvec12, &tvec12);
  EXPECT_TRUE(qvec12.isApprox(ComposeIdentityQuaternion()));
  EXPECT_EQ(tvec12, Eigen::Vector3d(1, 0, 0));

  ComputeRelativePose(ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 1),
                      Eigen::Vector4d(1, 1, 0, 0), Eigen::Vector3d(0, 0, 0),
                      &qvec12, &tvec12);
  EXPECT_TRUE(qvec12.isApprox(Eigen::Vector4d(0.707107, 0.707107, 0, 0), 1e-6));
  EXPECT_TRUE(tvec12.isApprox(Eigen::Vector3d(0, 1, 0)));
}

TEST(base_pose, TestConcatenatePoses) {
  Eigen::Vector4d qvec12;
  Eigen::Vector3d tvec12;

  ConcatenatePoses(ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0),
                   ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0),
                   &qvec12, &tvec12);
  EXPECT_EQ(qvec12, ComposeIdentityQuaternion());
  EXPECT_EQ(tvec12, Eigen::Vector3d(0, 0, 0));

  ConcatenatePoses(ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0),
                   ComposeIdentityQuaternion(), Eigen::Vector3d(0, 1, 2),
                   &qvec12, &tvec12);
  EXPECT_EQ(qvec12, ComposeIdentityQuaternion());
  EXPECT_EQ(tvec12, Eigen::Vector3d(0, 1, 2));

  ConcatenatePoses(ComposeIdentityQuaternion(), Eigen::Vector3d(0, 1, 2),
                   ComposeIdentityQuaternion(), Eigen::Vector3d(3, 4, 5),
                   &qvec12, &tvec12);
  EXPECT_EQ(qvec12, ComposeIdentityQuaternion());
  EXPECT_EQ(tvec12, Eigen::Vector3d(3, 5, 7));

  Eigen::Vector4d rel_qvec12;
  Eigen::Vector3d rel_tvec12;
  ComputeRelativePose(Eigen::Vector4d(1, 1, 0, 0), Eigen::Vector3d(0, 1, 2),
                      Eigen::Vector4d(1, 3, 0, 0), Eigen::Vector3d(3, 4, 5),
                      &rel_qvec12, &rel_tvec12);
  ConcatenatePoses(Eigen::Vector4d(1, 1, 0, 0), Eigen::Vector3d(0, 1, 2),
                   rel_qvec12, rel_tvec12, &qvec12, &tvec12);
  EXPECT_TRUE(
      qvec12.isApprox(NormalizeQuaternion(Eigen::Vector4d(1, 3, 0, 0))));
  EXPECT_TRUE(tvec12.isApprox(Eigen::Vector3d(3, 4, 5)));
}

TEST(base_pose, TestInvertPose) {
  Eigen::Vector4d inv_qvec;
  Eigen::Vector3d inv_tvec;
  InvertPose(ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0), &inv_qvec,
             &inv_tvec);
  EXPECT_EQ(inv_qvec, ComposeIdentityQuaternion());
  EXPECT_EQ(inv_tvec, Eigen::Vector3d(0, 0, 0));
  InvertPose(Eigen::Vector4d(0, 1, 2, 3), Eigen::Vector3d(0, 1, 2), &inv_qvec,
             &inv_tvec);
  Eigen::Vector4d inv_inv_qvec;
  Eigen::Vector3d inv_inv_tvec;
  InvertPose(inv_qvec, inv_tvec, &inv_inv_qvec, &inv_inv_tvec);
  EXPECT_EQ(inv_inv_qvec, Eigen::Vector4d(0, 1, 2, 3));
  EXPECT_TRUE(inv_inv_tvec.isApprox(Eigen::Vector3d(0, 1, 2)));
}

TEST(base_pose, TestInterpolatePose) {
  const Eigen::Vector4d qvec1 = Eigen::Vector4d::Random().normalized();
  const Eigen::Vector3d tvec1 = Eigen::Vector3d::Random();
  const Eigen::Vector4d qvec2 = Eigen::Vector4d::Random().normalized();
  const Eigen::Vector3d tvec2 = Eigen::Vector3d::Random();

  Eigen::Vector4d qveci;
  Eigen::Vector3d tveci;

  InterpolatePose(qvec1, tvec1, qvec2, tvec2, 0, &qveci, &tveci);
  EXPECT_TRUE(tvec1.isApprox(tveci));

  InterpolatePose(qvec1, tvec1, qvec2, tvec2, 1, &qveci, &tveci);
  EXPECT_TRUE(tvec2.isApprox(tveci));

  InterpolatePose(qvec1, tvec1, qvec2, tvec2, 0.5, &qveci, &tveci);
  EXPECT_TRUE(((tvec1 + tvec2) / 2).isApprox(tveci));
}

TEST(base_pose, TestCalculateBaseline) {
  Eigen::Vector4d qvec1(1, 0, 0, 0);
  Eigen::Vector4d qvec2(1, 0, 0, 0);

  Eigen::Vector3d tvec1(0, 0, 0);
  Eigen::Vector3d tvec2(0, 0, 1);

  const double baseline1 = CalculateBaseline(qvec1, tvec1, qvec2, tvec2).norm();
  EXPECT_NEAR(baseline1, 1, std::abs(1) * (1e-10) / 100.0);

  tvec2(2) = 2;

  const double baseline2 = CalculateBaseline(qvec1, tvec1, qvec2, tvec2).norm();
  EXPECT_NEAR(baseline2, 2, std::abs(2) * (1e-10) / 100.0);
}

TEST(base_pose, TestCheckCheirality) {
  const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t(1, 0, 0);

  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  std::vector<Eigen::Vector3d> points3D;

  points1.emplace_back(0, 0);
  points2.emplace_back(0.1, 0);
  EXPECT_TRUE(CheckCheirality(R, t, points1, points2, &points3D));
  EXPECT_EQ(points3D.size(), 1);

  points1.emplace_back(0, 0);
  points2.emplace_back(-0.1, 0);
  EXPECT_TRUE(CheckCheirality(R, t, points1, points2, &points3D));
  EXPECT_EQ(points3D.size(), 1);

  points2[1][0] = 0.2;
  EXPECT_TRUE(CheckCheirality(R, t, points1, points2, &points3D));
  EXPECT_EQ(points3D.size(), 2);

  points2[0][0] = -0.2;
  points2[1][0] = -0.2;
  EXPECT_FALSE(CheckCheirality(R, t, points1, points2, &points3D));
  EXPECT_EQ(points3D.size(), 0);
}

TEST(RotationMatrixToAngleAxis, Roundtrip) {
  const Eigen::Matrix3d R = RandomEigenQuaterniond().toRotationMatrix();
  EXPECT_THAT(AngleAxisToRotationMatrix(RotationMatrixToAngleAxis(R)),
              EigenMatrixNear(R, 1e-6));
}

TEST(AngleAxisToRotationMatrix, Roundtrip) {
  const Eigen::AngleAxisd aa(RandomEigenQuaterniond());
  const Eigen::Vector3d w = aa.angle() * aa.axis();
  EXPECT_THAT(RotationMatrixToAngleAxis(AngleAxisToRotationMatrix(w)),
              EigenMatrixNear(w, 1e-6));
}

TEST(YAxisAngleFromRotation, Roundtrip) {
  for (int i = 0; i < 100; ++i) {
    const double angle = RandomUniformReal<double>(-M_PI, M_PI);
    const Eigen::Matrix3d R = RotationFromYAxisAngle(angle);
    const double recovered_angle = YAxisAngleFromRotation(R);
    EXPECT_THAT(RotationFromYAxisAngle(recovered_angle),
                EigenMatrixNear(R, 1e-6));
  }
}

TEST(QuaternionFromAngleAxis, Zero) {
  const Eigen::Quaterniond q = QuaternionFromAngleAxis(Eigen::Vector3d::Zero());
  EXPECT_NEAR(q.w(), 1.0, 1e-12);
  EXPECT_NEAR(q.vec().norm(), 0.0, 1e-12);
}

TEST(QuaternionFromAngleAxis, SmallAngle) {
  // Just above and below the threshold — results should be continuous.
  const Eigen::Vector3d axis = Eigen::Vector3d(1, 2, 3).normalized();
  const double theta_small = 1e-11;
  const double theta_medium = 1e-9;
  const Eigen::Quaterniond q_small =
      QuaternionFromAngleAxis(axis * theta_small);
  const Eigen::Quaterniond q_medium =
      QuaternionFromAngleAxis(axis * theta_medium);
  // Both should be near identity but preserve direction.
  EXPECT_NEAR(q_small.angularDistance(q_medium), 0.0, 1e-8);
  // Small angle should NOT snap to identity — it should preserve direction.
  const Eigen::Vector3d recovered =
      Eigen::AngleAxisd(q_small).axis().normalized();
  EXPECT_NEAR(std::abs(recovered.dot(axis)), 1.0, 1e-6);
}

TEST(QuaternionFromAngleAxis, Roundtrip) {
  for (int i = 0; i < 100; ++i) {
    const Eigen::AngleAxisd aa(RandomEigenQuaterniond());
    const Eigen::Vector3d omega = aa.angle() * aa.axis();
    const Eigen::Quaterniond q = QuaternionFromAngleAxis(omega);
    const Eigen::Matrix3d R_expected = aa.toRotationMatrix();
    const Eigen::Matrix3d R_actual = q.toRotationMatrix();
    EXPECT_THAT(R_actual, EigenMatrixNear(R_expected, 1e-10));
  }
}

TEST(LeftJacobianFromAngleAxis, IdentityAtZero) {
  const Eigen::Matrix3d Jl = LeftJacobianFromAngleAxis(Eigen::Vector3d::Zero());
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  EXPECT_THAT(Jl, EigenMatrixNear(I, 1e-10));
}

TEST(LeftJacobianFromAngleAxis, RelationToRight) {
  // Jr(w) = Jl(-w) for all w.
  for (int i = 0; i < 100; ++i) {
    const Eigen::AngleAxisd aa(RandomEigenQuaterniond());
    const Eigen::Vector3d omega = aa.angle() * aa.axis();
    const Eigen::Matrix3d Jr = RightJacobianFromAngleAxis(omega);
    const Eigen::Matrix3d Jl_neg = LeftJacobianFromAngleAxis(-omega);
    EXPECT_THAT(Jr, EigenMatrixNear(Jl_neg, 1e-10));
  }
}

TEST(RightJacobianFromAngleAxis, SmallAngle) {
  // Near zero, Jr ≈ I - 0.5 * [w]_x.
  const Eigen::Vector3d omega(1e-12, 2e-12, 3e-12);
  const Eigen::Matrix3d Jr = RightJacobianFromAngleAxis(omega);
  Eigen::Matrix3d expected = Eigen::Matrix3d::Identity();
  expected -= 0.5 * CrossProductMatrix(omega);
  EXPECT_THAT(Jr, EigenMatrixNear(expected, 1e-10));
}

TEST(RightJacobianFromAngleAxis, NumericDerivative) {
  // Verify Jr by numeric differentiation of Exp(w + dw) ≈ Exp(w) * Exp(Jr*dw).
  const double eps = 1e-7;
  for (int i = 0; i < 50; ++i) {
    const Eigen::AngleAxisd aa(RandomEigenQuaterniond());
    const Eigen::Vector3d omega = aa.angle() * aa.axis();
    const Eigen::Matrix3d Jr = RightJacobianFromAngleAxis(omega);
    const Eigen::Matrix3d R = AngleAxisToRotationMatrix(omega);
    // Numeric: d/dw_k Exp(w) ≈ (Exp(w + eps*e_k) - Exp(w)) / eps
    Eigen::Matrix3d Jr_numeric;
    for (int k = 0; k < 3; ++k) {
      Eigen::Vector3d dw = Eigen::Vector3d::Zero();
      dw(k) = eps;
      const Eigen::Matrix3d R_perturbed = AngleAxisToRotationMatrix(omega + dw);
      // R_perturbed ≈ R * Exp(Jr * dw), so Exp(Jr*dw) ≈ R^T * R_perturbed
      const Eigen::Matrix3d dR = R.transpose() * R_perturbed;
      const Eigen::Vector3d log_dR = RotationMatrixToAngleAxis(dR);
      Jr_numeric.col(k) = log_dR / eps;
    }
    EXPECT_THAT(Jr, EigenMatrixNear(Jr_numeric, 1e-5));
  }
}

TEST(LeftJacobianFromAngleAxis, NumericDerivative) {
  // Verify Jl by numeric differentiation of Exp(w + dw) ≈ Exp(Jl*dw) * Exp(w).
  const double eps = 1e-7;
  for (int i = 0; i < 50; ++i) {
    const Eigen::AngleAxisd aa(RandomEigenQuaterniond());
    const Eigen::Vector3d omega = aa.angle() * aa.axis();
    const Eigen::Matrix3d Jl = LeftJacobianFromAngleAxis(omega);
    const Eigen::Matrix3d R = AngleAxisToRotationMatrix(omega);
    Eigen::Matrix3d Jl_numeric;
    for (int k = 0; k < 3; ++k) {
      Eigen::Vector3d dw = Eigen::Vector3d::Zero();
      dw(k) = eps;
      const Eigen::Matrix3d R_perturbed = AngleAxisToRotationMatrix(omega + dw);
      // R_perturbed ≈ Exp(Jl * dw) * R, so Exp(Jl*dw) ≈ R_perturbed * R^T.
      const Eigen::Matrix3d dR = R_perturbed * R.transpose();
      const Eigen::Vector3d log_dR = RotationMatrixToAngleAxis(dR);
      Jl_numeric.col(k) = log_dR / eps;
    }
    EXPECT_THAT(Jl, EigenMatrixNear(Jl_numeric, 1e-5));
  }
}

// Upstream parity (c60355a1 #4837).
TEST(LeftJacobianFromAngleAxis, SmallAngleSeries) {
  const Eigen::Vector3d axis = Eigen::Vector3d(1, -2, 3).normalized();
  for (const double theta : {1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-4}) {
    const Eigen::Vector3d omega = theta * axis;
    const Eigen::Matrix3d omega_x = CrossProductMatrix(omega);
    // J_l = integral_0^1 Exp(t [omega]_x) dt = sum [omega]_x^k / (k+1)!.
    Eigen::Matrix3d term = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d expected = term;
    for (int k = 1; k <= 8; ++k) {
      term = (term * omega_x / (k + 1)).eval();
      expected += term;
    }
    EXPECT_THAT(LeftJacobianFromAngleAxis(omega),
                EigenMatrixNear(expected, 2e-16))
        << "theta=" << theta;
    EXPECT_THAT(RightJacobianFromAngleAxis(omega),
                EigenMatrixNear(expected.transpose().eval(), 2e-16))
        << "theta=" << theta;
  }
}

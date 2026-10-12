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

#define TEST_NAME "base/triangulation"
#include "geometry/triangulation.h"

#include <gmock/gmock.h>

#include <Eigen/Core>

#include "geometry/rigid3.h"
#include "geometry/similarity_transform.h"
#include "math/random_eigen.h"
#include "util/eigen_matchers.h"
#include "util/testing.h"

using namespace colmap;

TEST(base_triangulation, TestTriangulatePoint) {
  std::vector<Eigen::Vector3d> points3D(6);
  points3D[0] = Eigen::Vector3d(0, 0.1, 0.1);
  points3D[1] = Eigen::Vector3d(0, 1, 3);
  points3D[2] = Eigen::Vector3d(0, 1, 2);
  points3D[3] = Eigen::Vector3d(0.01, 0.2, 3);
  points3D[4] = Eigen::Vector3d(-1, 0.1, 1);
  points3D[5] = Eigen::Vector3d(0.1, 0.1, 0.2);

  Eigen::Matrix3x4d proj_matrix1 = Eigen::MatrixXd::Identity(3, 4);

  for (double qz = 0; qz < 1; qz += 0.2) {
    for (double tx = 0; tx < 10; tx += 2) {
      SimilarityTransform3 tform(1, Eigen::Vector4d(0.2, 0.3, 0.4, qz),
                                 Eigen::Vector3d(tx, 2, 3));

      Eigen::Matrix3x4d proj_matrix2 = tform.Matrix().topLeftCorner<3, 4>();

      for (size_t i = 0; i < points3D.size(); ++i) {
        const Eigen::Vector3d& point3D = points3D[i];
        const Eigen::Vector4d point3D1(point3D(0), point3D(1), point3D(2), 1);
        Eigen::Vector3d point2D1 = proj_matrix1 * point3D1;
        Eigen::Vector3d point2D2 = proj_matrix2 * point3D1;
        point2D1 /= point2D1(2);
        point2D2 /= point2D2(2);

        const Eigen::Vector2d point2D1_N(point2D1(0), point2D1(1));
        const Eigen::Vector2d point2D2_N(point2D2(0), point2D2(1));

        const Eigen::Vector3d tri_point3D = TriangulatePoint(
            proj_matrix1, proj_matrix2, point2D1_N, point2D2_N);

        EXPECT_TRUE((point3D - tri_point3D).norm() < 1e-10);
      }
    }
  }
}

TEST(base_triangulation, TestCalculateTriangulationAngle) {
  const Eigen::Vector3d tvec1(0, 0, 0);
  const Eigen::Vector3d tvec2(0, 1, 0);

  EXPECT_NEAR(
      CalculateTriangulationAngle(tvec1, tvec2, Eigen::Vector3d(0, 0, 100)),
      0.009999666687, std::abs(0.009999666687) * (1e-8) / 100.0);
  EXPECT_NEAR(
      CalculateTriangulationAngle(tvec1, tvec2, Eigen::Vector3d(0, 0, 50)),
      0.019997333973, std::abs(0.019997333973) * (1e-8) / 100.0);
  EXPECT_NEAR(CalculateTriangulationAngles(tvec1, tvec2,
                                           {Eigen::Vector3d(0, 0, 50)})[0],
              0.019997333973, std::abs(0.019997333973) * (1e-8) / 100.0);
}

// Upstream parity (a2bd15c2 geometry/triangulation_test.cc).
TEST(TriangulatePoint, Nominal) {
  const std::vector<Eigen::Vector3d> points3D = {
      Eigen::Vector3d(0, 0.1, 0.1), Eigen::Vector3d(0, 1, 3),
      Eigen::Vector3d(0, 1, 2),     Eigen::Vector3d(0.01, 0.2, 3),
      Eigen::Vector3d(-1, 0.1, 1),  Eigen::Vector3d(0.1, 0.1, 0.2),
  };

  const Rigid3d cam1_from_world;

  for (int z = 0; z < 5; ++z) {
    const double qz = z / 5.0;
    for (int tx = 0; tx < 10; tx += 2) {
      const Rigid3d cam2_from_world(Eigen::Quaterniond(0.2, 0.3, 0.4, qz),
                                    Eigen::Vector3d(tx, 2, 3));
      for (size_t i = 0; i < points3D.size(); ++i) {
        const Eigen::Vector3d& point3D = points3D[i];
        const Eigen::Vector2d point1 =
            (cam1_from_world * point3D).hnormalized();
        const Eigen::Vector2d point2 =
            (cam2_from_world * point3D).hnormalized();

        Eigen::Vector3d tri_point3D;
        EXPECT_TRUE(TriangulatePoint(cam1_from_world.ToMatrix(),
                                     cam2_from_world.ToMatrix(), point1, point2,
                                     &tri_point3D));

        EXPECT_THAT(point3D, EigenMatrixNear(tri_point3D, 1e-10));
      }
    }
  }
}

TEST(TriangulatePoint, ParallelRays) {
  Eigen::Vector3d point3D;
  EXPECT_FALSE(TriangulatePoint(
      Rigid3d().ToMatrix(),
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0))
          .ToMatrix(),
      Eigen::Vector2d(0, 0), Eigen::Vector2d(0, 0), &point3D));
}

TEST(TriangulatePoint, Bearings) {
  const std::vector<Eigen::Vector3d> points3D = {
      Eigen::Vector3d(0, 0.1, 0.1),
      Eigen::Vector3d(0, 1, 3),
      Eigen::Vector3d(-1, 0.1, 1),
      Eigen::Vector3d(0.1, 0.1, -0.5),  // behind cam1 (negative Z)
      Eigen::Vector3d(0.2, -0.3, -2),   // behind cam1
  };

  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(Eigen::Quaterniond(0.21, 0.31, 0.41, 0.1),
                                Eigen::Vector3d(1, 2, 3));

  for (const Eigen::Vector3d& point3D : points3D) {
    const Eigen::Vector3d cam_ray1 = (cam1_from_world * point3D).normalized();
    const Eigen::Vector3d cam_ray2 = (cam2_from_world * point3D).normalized();

    Eigen::Vector3d tri_point3D;
    EXPECT_TRUE(TriangulatePoint(cam1_from_world.ToMatrix(),
                                 cam2_from_world.ToMatrix(), cam_ray1, cam_ray2,
                                 &tri_point3D));
    EXPECT_THAT(point3D, EigenMatrixNear(tri_point3D, 1e-9));
  }
}

TEST(TriangulatePoint, BearingsParallelRays) {
  Eigen::Vector3d point3D;
  EXPECT_FALSE(TriangulatePoint(
      Rigid3d().ToMatrix(),
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0))
          .ToMatrix(),
      Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, 1), &point3D));
}

TEST(TriangulateMidPoint, Nominal) {
  constexpr int kNumTrials = 10;
  for (int i = 0; i < kNumTrials; ++i) {
    const Rigid3d cam1_from_world(RandomEigenQuaterniond(),
                                  RandomEigenVectord<3>());
    const Rigid3d cam2_from_world(RandomEigenQuaterniond(),
                                  RandomEigenVectord<3>());
    const Eigen::Vector3d point3D = RandomEigenVectord<3>();
    const Eigen::Vector3d cam_ray1 = (cam1_from_world * point3D).normalized();
    const Eigen::Vector3d cam_ray2 = (cam2_from_world * point3D).normalized();

    Eigen::Vector3d point3D_in_cam1;
    ASSERT_TRUE(TriangulateMidPoint(cam2_from_world * Inverse(cam1_from_world),
                                    cam_ray1, cam_ray2, &point3D_in_cam1));
    const Eigen::Vector3d point3D_in_world =
        Inverse(cam1_from_world) * point3D_in_cam1;
    EXPECT_THAT(point3D, EigenMatrixNear(point3D_in_world, 1e-10));
  }
}

TEST(TriangulateMidPoint, NonPerfectIntersection) {
  const Rigid3d cam1_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(-1, 0, 0));
  const Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(1, 0, 0));
  const Eigen::Vector3d expected_point3D(0, 0, 5);
  Eigen::Vector3d point3D_in_cam1;
  EXPECT_TRUE(TriangulateMidPoint(
      cam2_from_world * Inverse(cam1_from_world),
      (cam1_from_world * (expected_point3D + Eigen::Vector3d(0, 0.1, 0)))
          .normalized(),
      (cam2_from_world * (expected_point3D + Eigen::Vector3d(0, -0.1, 0)))
          .normalized(),
      &point3D_in_cam1));
  EXPECT_THAT(point3D_in_cam1,
              EigenMatrixNear(cam1_from_world * expected_point3D, 1e-3));
}

TEST(TriangulateMidPoint, ParallelRays) {
  Eigen::Vector3d point3D_in_cam1;
  EXPECT_FALSE(TriangulateMidPoint(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)),
      Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, 1), &point3D_in_cam1));
}

TEST(TriangulateMidPoint, BehindCameras) {
  Eigen::Vector3d point3D;
  EXPECT_FALSE(TriangulateMidPoint(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)),
      Eigen::Vector3d(0.1, 0, 1).normalized(),
      Eigen::Vector3d(-0.1, 0, 1).normalized(), &point3D));
}

TEST(TriangulateMultiViewPoint, Nominal) {
  const std::vector<Eigen::Vector3d> points3D = {
      Eigen::Vector3d(0, 0.1, 0.1), Eigen::Vector3d(0, 1, 3),
      Eigen::Vector3d(0, 1, 2),     Eigen::Vector3d(0.01, 0.2, 3),
      Eigen::Vector3d(-1, 0.1, 1),  Eigen::Vector3d(0.1, 0.1, 0.2),
  };

  const Rigid3d cam1_from_world;

  for (int z = 0; z < 5; ++z) {
    const double qz = z / 5.0;
    for (int tx = 0; tx < 10; tx += 2) {
      const Rigid3d cam2_from_world(Eigen::Quaterniond(0.21, 0.31, 0.41, qz),
                                    Eigen::Vector3d(tx, 2, 3));
      const Rigid3d cam3_from_world(Eigen::Quaterniond(0.2, 0.3, 0.4, qz),
                                    Eigen::Vector3d(tx, 2.1, 3.1));
      for (size_t i = 0; i < points3D.size(); ++i) {
        const Eigen::Vector3d& point3D = points3D[i];
        const Eigen::Vector2d point1 =
            (cam1_from_world * point3D).hnormalized();
        const Eigen::Vector2d point2 =
            (cam2_from_world * point3D).hnormalized();
        const Eigen::Vector2d point3 =
            (cam3_from_world * point3D).hnormalized();

        const std::vector<Eigen::Matrix3x4d> cams_from_world = {
            cam1_from_world.ToMatrix(), cam2_from_world.ToMatrix(),
            cam3_from_world.ToMatrix()};
        const std::vector<Eigen::Vector2d> points = {point1, point2, point3};

        Eigen::Vector3d tri_point3D;
        EXPECT_TRUE(
            TriangulateMultiViewPoint(cams_from_world, points, &tri_point3D));

        EXPECT_THAT(point3D, EigenMatrixNear(tri_point3D, 1e-10));
      }
    }
  }
}

TEST(TriangulateMultiViewPoint, Bearings) {
  // The 3D bearing overload recovers the same points as the 2D overload, and
  // additionally handles back-hemisphere rays (negative Z in the camera
  // frame) that the 2D (u, v, 1) representation cannot encode -- as produced
  // by omnidirectional (e.g. EQUIRECTANGULAR) cameras.
  const std::vector<Eigen::Vector3d> points3D = {
      Eigen::Vector3d(0, 0.1, 0.1),
      Eigen::Vector3d(0, 1, 3),
      Eigen::Vector3d(-1, 0.1, 1),
      Eigen::Vector3d(0.1, 0.1, -0.5),  // behind cam1 (negative Z)
      Eigen::Vector3d(0.2, -0.3, -2),   // behind cam1
  };

  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(Eigen::Quaterniond(0.21, 0.31, 0.41, 0.1),
                                Eigen::Vector3d(1, 2, 3));
  const Rigid3d cam3_from_world(Eigen::Quaterniond(0.2, 0.3, 0.4, 0.05),
                                Eigen::Vector3d(2, 2.1, 3.1));
  const std::vector<Eigen::Matrix3x4d> cams_from_world = {
      cam1_from_world.ToMatrix(), cam2_from_world.ToMatrix(),
      cam3_from_world.ToMatrix()};

  for (const Eigen::Vector3d& point3D : points3D) {
    const std::vector<Eigen::Vector3d> cam_rays = {
        (cam1_from_world * point3D).normalized(),
        (cam2_from_world * point3D).normalized(),
        (cam3_from_world * point3D).normalized()};

    Eigen::Vector3d tri_point3D;
    EXPECT_TRUE(
        TriangulateMultiViewPoint(cams_from_world, cam_rays, &tri_point3D));
    EXPECT_THAT(point3D, EigenMatrixNear(tri_point3D, 1e-9));
  }
}

TEST(CalculateTriangulationAngle, Nominal) {
  const Eigen::Vector3d tvec1(0, 0, 0);
  const Eigen::Vector3d tvec2(0, 1, 0);

  EXPECT_NEAR(
      CalculateTriangulationAngle(tvec1, tvec2, Eigen::Vector3d(0, 0, 100)),
      0.009999666687, 1e-8);
  EXPECT_NEAR(
      CalculateTriangulationAngle(tvec1, tvec2, Eigen::Vector3d(0, 0, 50)),
      0.019997333973, 1e-8);
  EXPECT_NEAR(CalculateTriangulationAngles(tvec1, tvec2,
                                           {Eigen::Vector3d(0, 0, 100)})[0],
              0.009999666687, 1e-8);
  EXPECT_NEAR(CalculateTriangulationAngles(tvec1, tvec2,
                                           {Eigen::Vector3d(0, 0, 50)})[0],
              0.019997333973, 1e-8);
  // Parallel rays.
  EXPECT_THAT(CalculateTriangulationAngles(
                  Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                  {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(50, 0, 0),
                   Eigen::Vector3d(0, 50, 0), Eigen::Vector3d(0, 0, 50)}),
              testing::Each(testing::DoubleNear(0, 1e-6)));
  // Orthogonal rays.
  EXPECT_THAT(CalculateTriangulationAngles(
                  Eigen::Vector3d::Zero(), Eigen::Vector3d(50, 0, 50),
                  {Eigen::Vector3d(50, 0, 0), Eigen::Vector3d(0, 0, 50)}),
              testing::Each(testing::DoubleNear(EIGEN_PI / 2, 1e-6)));
  // Opposing rays.
  EXPECT_THAT(CalculateTriangulationAngles(
                  Eigen::Vector3d::Zero(), Eigen::Vector3d(0, 0, 50),
                  {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 50),
                   Eigen::Vector3d(0, 0, 25), Eigen::Vector3d(0, 0, -25),
                   Eigen::Vector3d(0, 0, 75)}),
              testing::Each(testing::DoubleNear(0, 1e-6)));
}

// Fork addition (predates upstream 70dea881): the optimal (Lindstrom)
// triangulation recovers exact correspondences under a general (rotated)
// relative pose, and the batch form uses the optimal observations.
TEST(TriangulateOptimalPoint, Nominal) {
  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(
      Eigen::Quaterniond(0.2, 0.3, 0.4, 0.1).normalized(),
      Eigen::Vector3d(1, 2, 3));
  const std::vector<Eigen::Vector3d> points3D = {
      Eigen::Vector3d(0, 0.1, 0.1),
      Eigen::Vector3d(0, 1, 3),
      Eigen::Vector3d(-1, 0.1, 1),
  };
  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  for (const Eigen::Vector3d& point3D : points3D) {
    points1.push_back((cam1_from_world * point3D).hnormalized());
    points2.push_back((cam2_from_world * point3D).hnormalized());
    Eigen::Vector3d tri_point3D;
    EXPECT_TRUE(TriangulateOptimalPoint(
        cam1_from_world.ToMatrix(), cam2_from_world.ToMatrix(), points1.back(),
        points2.back(), &tri_point3D));
    EXPECT_THAT(point3D, EigenMatrixNear(tri_point3D, 1e-8));
  }
  const std::vector<Eigen::Vector3d> tri_points3D = TriangulateOptimalPoints(
      cam1_from_world.ToMatrix(), cam2_from_world.ToMatrix(), points1, points2);
  ASSERT_EQ(tri_points3D.size(), points3D.size());
  for (size_t i = 0; i < points3D.size(); ++i) {
    EXPECT_THAT(points3D[i], EigenMatrixNear(tri_points3D[i], 1e-8));
  }
}

TEST(TriangulateOptimalPoint, OptimalObservations) {
  // With noisy observations the batch form must equal the per-point optimal
  // triangulation (not the plain DLT of the raw observations).
  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(
      Eigen::Quaterniond(0.2, 0.3, 0.4, 0.1).normalized(),
      Eigen::Vector3d(1, 2, 3));
  const Eigen::Vector3d point3D(0, 1, 3);
  const std::vector<Eigen::Vector2d> points1 = {
      (cam1_from_world * point3D).hnormalized() + Eigen::Vector2d(1e-3, 0)};
  const std::vector<Eigen::Vector2d> points2 = {
      (cam2_from_world * point3D).hnormalized() + Eigen::Vector2d(0, -1e-3)};
  Eigen::Vector3d optimal_point3D;
  ASSERT_TRUE(TriangulateOptimalPoint(cam1_from_world.ToMatrix(),
                                      cam2_from_world.ToMatrix(), points1[0],
                                      points2[0], &optimal_point3D));
  const std::vector<Eigen::Vector3d> tri_points3D = TriangulateOptimalPoints(
      cam1_from_world.ToMatrix(), cam2_from_world.ToMatrix(), points1, points2);
  EXPECT_EQ(tri_points3D[0], optimal_point3D);
  Eigen::Vector3d dlt_point3D;
  ASSERT_TRUE(TriangulatePoint(cam1_from_world.ToMatrix(),
                               cam2_from_world.ToMatrix(), points1[0],
                               points2[0], &dlt_point3D));
  EXPECT_NE(tri_points3D[0], dlt_point3D);
}

// Upstream parity (70dea881, #4835).
TEST(TriangulateOptimalPoint, RotatedCameras) {
  const std::vector<Eigen::Vector3d> points3D = {Eigen::Vector3d(-0.5, 0.2, 3),
                                                 Eigen::Vector3d(0.3, -0.4, 5),
                                                 Eigen::Vector3d(0.7, 0.6, 8)};
  for (const Rigid3d& cam1_from_world :
       {Rigid3d(),
        Rigid3d(Eigen::Quaterniond(Eigen::AngleAxisd(
                    0.1, Eigen::Vector3d(1, 2, 3).normalized())),
                Eigen::Vector3d(0.2, -0.1, 0.3))}) {
    for (const Eigen::Vector3d& axis : {Eigen::Vector3d::UnitX().eval(),
                                        Eigen::Vector3d::UnitY().eval(),
                                        Eigen::Vector3d::UnitZ().eval()}) {
      const Rigid3d cam2_from_cam1(
          Eigen::Quaterniond(Eigen::AngleAxisd(0.2, axis)),
          Eigen::Vector3d(1, 0.2, 0.1));
      const Rigid3d cam2_from_world = cam2_from_cam1 * cam1_from_world;
      for (const Eigen::Vector3d& point3D : points3D) {
        const Eigen::Vector2d point1 =
            (cam1_from_world * point3D).hnormalized();
        const Eigen::Vector2d point2 =
            (cam2_from_world * point3D).hnormalized();
        Eigen::Vector3d tri_point3D;
        ASSERT_TRUE(TriangulateOptimalPoint(cam1_from_world.ToMatrix(),
                                            cam2_from_world.ToMatrix(),
                                            point1,
                                            point2,
                                            &tri_point3D));
        EXPECT_THAT(tri_point3D, EigenMatrixNear(point3D, 1e-10));
        ASSERT_TRUE(TriangulateOptimalPoint(cam2_from_world.ToMatrix(),
                                            cam1_from_world.ToMatrix(),
                                            point2,
                                            point1,
                                            &tri_point3D));
        EXPECT_THAT(tri_point3D, EigenMatrixNear(point3D, 1e-10));
      }
    }
  }
}

TEST(TriangulateOptimalPoint, NoisyRectifiedObservations) {
  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(1, 0, 0));
  Eigen::Vector3d point3D;
  ASSERT_TRUE(TriangulateOptimalPoint(cam1_from_world.ToMatrix(),
                                      cam2_from_world.ToMatrix(),
                                      Eigen::Vector2d(0.2, 0.1),
                                      Eigen::Vector2d(0.4, 0.12),
                                      &point3D));
  // Rectified stereo keeps x and averages y, giving depth 1 / (0.4 - 0.2).
  EXPECT_THAT(point3D, EigenMatrixNear(Eigen::Vector3d(1, 0.55, 5), 1e-10));
}

TEST(CalculateAngleBetweenVectors, ParallelVectors) {
  const Eigen::Vector3d v1(1, 0, 0);
  const Eigen::Vector3d v2(2, 0, 0);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v1, v2), 0.0, 1e-10);
}

TEST(CalculateAngleBetweenVectors, OppositeVectors) {
  const Eigen::Vector3d v1(1, 0, 0);
  const Eigen::Vector3d v2(-1, 0, 0);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v1, v2), EIGEN_PI, 1e-10);
}

TEST(CalculateAngleBetweenVectors, PerpendicularVectors) {
  const Eigen::Vector3d v1(1, 0, 0);
  const Eigen::Vector3d v2(0, 1, 0);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v1, v2), EIGEN_PI / 2, 1e-10);
}

TEST(CalculateAngleBetweenVectors, PerpendicularVectorsDifferentMagnitudes) {
  const Eigen::Vector3d v1(3, 0, 0);
  const Eigen::Vector3d v2(0, 5, 0);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v1, v2), EIGEN_PI / 2, 1e-10);
}

TEST(CalculateAngleBetweenVectors, ZeroVector) {
  const Eigen::Vector3d v1(1, 0, 0);
  const Eigen::Vector3d v2(0, 0, 0);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v1, v2), 0.0, 1e-10);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v2, v1), 0.0, 1e-10);
}

TEST(CalculateAngleBetweenVectors, BothZeroVectors) {
  const Eigen::Vector3d v1(0, 0, 0);
  const Eigen::Vector3d v2(0, 0, 0);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v1, v2), 0.0, 1e-10);
}

TEST(CalculateAngleBetweenVectors, FortyFiveDegrees) {
  const Eigen::Vector3d v1(1, 0, 0);
  const Eigen::Vector3d v2(1, 1, 0);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v1, v2), EIGEN_PI / 4, 1e-10);
}

TEST(CalculateAngleBetweenVectors, IdenticalVectors) {
  const Eigen::Vector3d v1(1, 2, 3);
  const Eigen::Vector3d v2(1, 2, 3);
  EXPECT_NEAR(CalculateAngleBetweenVectors(v1, v2), 0.0, 1e-10);
}

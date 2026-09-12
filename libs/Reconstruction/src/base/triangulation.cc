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

#include "geometry/rigid3.h"
#include "base/triangulation.h"

#include "base/essential_matrix.h"
#include "base/pose.h"

namespace colmap {

Eigen::Vector3d TriangulatePoint(const Eigen::Matrix3x4d& proj_matrix1,
                                 const Eigen::Matrix3x4d& proj_matrix2,
                                 const Eigen::Vector2d& point1,
                                 const Eigen::Vector2d& point2) {
  Eigen::Matrix4d A;

  A.row(0) = point1(0) * proj_matrix1.row(2) - proj_matrix1.row(0);
  A.row(1) = point1(1) * proj_matrix1.row(2) - proj_matrix1.row(1);
  A.row(2) = point2(0) * proj_matrix2.row(2) - proj_matrix2.row(0);
  A.row(3) = point2(1) * proj_matrix2.row(2) - proj_matrix2.row(1);

  Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);

  return svd.matrixV().col(3).hnormalized();
}

std::vector<Eigen::Vector3d> TriangulatePoints(
    const Eigen::Matrix3x4d& proj_matrix1,
    const Eigen::Matrix3x4d& proj_matrix2,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2) {
  CHECK_EQ(points1.size(), points2.size());

  std::vector<Eigen::Vector3d> points3D(points1.size());

  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D[i] =
        TriangulatePoint(proj_matrix1, proj_matrix2, points1[i], points2[i]);
  }

  return points3D;
}

Eigen::Vector3d TriangulateMultiViewPoint(
    const std::vector<Eigen::Matrix3x4d>& proj_matrices,
    const std::vector<Eigen::Vector2d>& points) {
  CHECK_EQ(proj_matrices.size(), points.size());

  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();

  for (size_t i = 0; i < points.size(); i++) {
    const Eigen::Vector3d point = points[i].homogeneous().normalized();
    const Eigen::Matrix3x4d term =
        proj_matrices[i] - point * point.transpose() * proj_matrices[i];
    A += term.transpose() * term;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);

  return eigen_solver.eigenvectors().col(0).hnormalized();
}

Eigen::Vector3d TriangulateOptimalPoint(const Eigen::Matrix3x4d& proj_matrix1,
                                        const Eigen::Matrix3x4d& proj_matrix2,
                                        const Eigen::Vector2d& point1,
                                        const Eigen::Vector2d& point2) {
  const Eigen::Matrix3d E =
      EssentialMatrixFromAbsolutePoses(proj_matrix1, proj_matrix2);

  Eigen::Vector2d optimal_point1;
  Eigen::Vector2d optimal_point2;
  FindOptimalImageObservations(E, point1, point2, &optimal_point1,
                               &optimal_point2);

  return TriangulatePoint(proj_matrix1, proj_matrix2, optimal_point1,
                          optimal_point2);
}

std::vector<Eigen::Vector3d> TriangulateOptimalPoints(
    const Eigen::Matrix3x4d& proj_matrix1,
    const Eigen::Matrix3x4d& proj_matrix2,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2) {
  std::vector<Eigen::Vector3d> points3D(points1.size());

  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D[i] =
        TriangulatePoint(proj_matrix1, proj_matrix2, points1[i], points2[i]);
  }

  return points3D;
}

double CalculateTriangulationAngle(const Eigen::Vector3d& proj_center1,
                                   const Eigen::Vector3d& proj_center2,
                                   const Eigen::Vector3d& point3D) {
  const double baseline_length_squared =
      (proj_center1 - proj_center2).squaredNorm();

  const double ray_length_squared1 = (point3D - proj_center1).squaredNorm();
  const double ray_length_squared2 = (point3D - proj_center2).squaredNorm();

  // Using "law of cosines" to compute the enclosing angle between rays.
  const double denominator =
      2.0 * std::sqrt(ray_length_squared1 * ray_length_squared2);
  if (denominator == 0.0) {
    return 0.0;
  }
  const double nominator =
      ray_length_squared1 + ray_length_squared2 - baseline_length_squared;
  const double angle = std::abs(std::acos(nominator / denominator));

  // Triangulation is unstable for acute angles (far away points) and
  // obtuse angles (close points), so always compute the minimum angle
  // between the two intersecting rays.
  return std::min(angle, M_PI - angle);
}

std::vector<double> CalculateTriangulationAngles(
    const Eigen::Vector3d& proj_center1, const Eigen::Vector3d& proj_center2,
    const std::vector<Eigen::Vector3d>& points3D) {
  // Baseline length between camera centers.
  const double baseline_length_squared =
      (proj_center1 - proj_center2).squaredNorm();

  std::vector<double> angles(points3D.size());

  for (size_t i = 0; i < points3D.size(); ++i) {
    // Ray lengths from cameras to point.
    const double ray_length_squared1 =
        (points3D[i] - proj_center1).squaredNorm();
    const double ray_length_squared2 =
        (points3D[i] - proj_center2).squaredNorm();

    // Using "law of cosines" to compute the enclosing angle between rays.
    const double denominator =
        2.0 * std::sqrt(ray_length_squared1 * ray_length_squared2);
    if (denominator == 0.0) {
      angles[i] = 0.0;
      continue;
    }
    const double nominator =
        ray_length_squared1 + ray_length_squared2 - baseline_length_squared;
    const double angle = std::abs(std::acos(nominator / denominator));

    // Triangulation is unstable for acute angles (far away points) and
    // obtuse angles (close points), so always compute the minimum angle
    // between the two intersecting rays.
    angles[i] = std::min(angle, M_PI - angle);
  }

  return angles;
}


double CalculateAngleBetweenVectors(const Eigen::Vector3d& v1,
                                    const Eigen::Vector3d& v2) {
    const double squared_norm1 = v1.squaredNorm();
    const double squared_norm2 = v2.squaredNorm();
    if (squared_norm1 == 0.0 || squared_norm2 == 0.0) {
        return 0.0;
    }
    return std::acos(std::clamp(
            v1.dot(v2) / std::sqrt(squared_norm1 * squared_norm2), -1.0,
            1.0));
}

bool TriangulateMidPoint(const Rigid3d& cam2_from_cam1,
                         const Eigen::Vector3d& cam_ray1,
                         const Eigen::Vector3d& cam_ray2,
                         Eigen::Vector3d* point3D_in_cam1) {
  const Eigen::Quaterniond cam1_from_cam2_rotation =
      cam2_from_cam1.rotation().inverse();
  const Eigen::Vector3d cam_ray2_in_cam1 = cam1_from_cam2_rotation * cam_ray2;
  const Eigen::Vector3d cam2_in_cam1 =
      cam1_from_cam2_rotation * -cam2_from_cam1.translation();

  Eigen::Matrix3d A;
  A << cam_ray1(0), -cam_ray2_in_cam1(0), -cam2_in_cam1(0), cam_ray1(1),
      -cam_ray2_in_cam1(1), -cam2_in_cam1(1), cam_ray1(2), -cam_ray2_in_cam1(2),
      -cam2_in_cam1(2);

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullV);
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
  if (svd.info() != Eigen::Success) {
    return false;
  }
#endif

  if (svd.matrixV()(2, 2) == 0) {
    return false;
  }

  const Eigen::Vector2d lambda = svd.matrixV().col(2).hnormalized();

  // Check if point is behind cameras.
  if (lambda(0) <= std::numeric_limits<double>::epsilon() ||
      lambda(1) <= std::numeric_limits<double>::epsilon()) {
    return false;
  }

  *point3D_in_cam1 = 0.5 * (lambda(0) * cam_ray1 + cam2_in_cam1 +
                            lambda(1) * cam_ray2_in_cam1);

  return true;
}

}  // namespace colmap

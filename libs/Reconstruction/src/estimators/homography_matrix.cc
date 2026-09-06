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

#include "estimators/homography_matrix.h"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "base/projection.h"
#include "estimators/utils.h"
#include "util/logging.h"

namespace colmap {

std::vector<HomographyMatrixEstimator::M_t> HomographyMatrixEstimator::Estimate(
    const std::vector<X_t>& points1, const std::vector<Y_t>& points2) {
  CHECK_EQ(points1.size(), points2.size());

  const size_t N = points1.size();

  // Center and normalize image points for better numerical stability.
  std::vector<X_t> normed_points1;
  std::vector<Y_t> normed_points2;
  Eigen::Matrix3d points1_norm_matrix;
  Eigen::Matrix3d points2_norm_matrix;
  CenterAndNormalizeImagePoints(points1, &normed_points1, &points1_norm_matrix);
  CenterAndNormalizeImagePoints(points2, &normed_points2, &points2_norm_matrix);

  // Setup constraint matrix.
  Eigen::Matrix<double, Eigen::Dynamic, 9> A = Eigen::MatrixXd::Zero(2 * N, 9);

  for (size_t i = 0, j = N; i < points1.size(); ++i, ++j) {
    const double s_0 = normed_points1[i](0);
    const double s_1 = normed_points1[i](1);
    const double d_0 = normed_points2[i](0);
    const double d_1 = normed_points2[i](1);

    A(i, 0) = -s_0;
    A(i, 1) = -s_1;
    A(i, 2) = -1;
    A(i, 6) = s_0 * d_0;
    A(i, 7) = s_1 * d_0;
    A(i, 8) = d_0;

    A(j, 3) = -s_0;
    A(j, 4) = -s_1;
    A(j, 5) = -1;
    A(j, 6) = s_0 * d_1;
    A(j, 7) = s_1 * d_1;
    A(j, 8) = d_1;
  }

  // Solve for the nullspace of the constraint matrix.
  Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
      A, Eigen::ComputeFullV);

  const Eigen::VectorXd nullspace = svd.matrixV().col(8);
  Eigen::Map<const Eigen::Matrix3d> H_t(nullspace.data());

  const std::vector<M_t> models = {points2_norm_matrix.inverse() *
                                   H_t.transpose() * points1_norm_matrix};
  return models;
}



double ComputeSquaredHomographyError(const Eigen::Vector2d& point1,
                                     const Eigen::Vector2d& point2,
                                     const Eigen::Matrix3d& H) {
  const Eigen::Vector3d Hp1 = H * point1.homogeneous();
  if (Hp1[2] == 0) {
    return std::numeric_limits<double>::max();
  }
  return (point2 - Hp1.hnormalized()).squaredNorm();
}

}  // namespace colmap

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <vector>

#include "base/camera.h"
#include "util/alignment.h"
#include "util/math.h"
#include "util/types.h"

namespace colmap {

// Triangulate 3D point from corresponding image point observations.
//
// Implementation of the direct linear transform triangulation method in
//   R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision,
//   Cambridge Univ. Press, 2003.
//
// @param proj_matrix1   Projection matrix of the first image as 3x4 matrix.
// @param proj_matrix2   Projection matrix of the second image as 3x4 matrix.
// @param point1         Corresponding 2D point in first image.
// @param point2         Corresponding 2D point in second image.
//
// @return               Triangulated 3D point.
Eigen::Vector3d TriangulatePoint(const Eigen::Matrix3x4d& proj_matrix1,
                                 const Eigen::Matrix3x4d& proj_matrix2,
                                 const Eigen::Vector2d& point1,
                                 const Eigen::Vector2d& point2);

// Triangulate multiple 3D points from multiple image correspondences.
std::vector<Eigen::Vector3d> TriangulatePoints(
        const Eigen::Matrix3x4d& proj_matrix1,
        const Eigen::Matrix3x4d& proj_matrix2,
        const std::vector<Eigen::Vector2d>& points1,
        const std::vector<Eigen::Vector2d>& points2);

// Triangulate point from multiple views minimizing the L2 error.
//
// @param proj_matrices       Projection matrices of multi-view observations.
// @param points              Image observations of multi-view observations.
//
// @return                    Estimated 3D point.
Eigen::Vector3d TriangulateMultiViewPoint(
        const std::vector<Eigen::Matrix3x4d>& proj_matrices,
        const std::vector<Eigen::Vector2d>& points);

// Triangulate optimal 3D point from corresponding image point observations by
// finding the optimal image observations.
//
// Note that camera poses should be very good in order for this method to yield
// good results. Otherwise just use `TriangulatePoint`.
//
// Implementation of the method described in
//   P. Lindstrom, "Triangulation Made Easy," IEEE Computer Vision and Pattern
//   Recognition 2010, pp. 1554-1561, June 2010.
//
// @param proj_matrix1   Projection matrix of the first image as 3x4 matrix.
// @param proj_matrix2   Projection matrix of the second image as 3x4 matrix.
// @param point1         Corresponding 2D point in first image.
// @param point2         Corresponding 2D point in second image.
//
// @return               Triangulated optimal 3D point.
Eigen::Vector3d TriangulateOptimalPoint(const Eigen::Matrix3x4d& proj_matrix1,
                                        const Eigen::Matrix3x4d& proj_matrix2,
                                        const Eigen::Vector2d& point1,
                                        const Eigen::Vector2d& point2);

// Triangulate multiple optimal 3D points from multiple image correspondences.
std::vector<Eigen::Vector3d> TriangulateOptimalPoints(
        const Eigen::Matrix3x4d& proj_matrix1,
        const Eigen::Matrix3x4d& proj_matrix2,
        const std::vector<Eigen::Vector2d>& points1,
        const std::vector<Eigen::Vector2d>& points2);

// Calculate angle in radians between the two rays of a triangulated point.
double CalculateTriangulationAngle(const Eigen::Vector3d& proj_center1,
                                   const Eigen::Vector3d& proj_center2,
                                   const Eigen::Vector3d& point3D);
std::vector<double> CalculateTriangulationAngles(
        const Eigen::Vector3d& proj_center1,
        const Eigen::Vector3d& proj_center2,
        const std::vector<Eigen::Vector3d>& points3D);

}  // namespace colmap

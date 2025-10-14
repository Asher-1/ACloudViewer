// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <vector>

#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

// Decompose an homography matrix into the possible rotations, translations,
// and plane normal vectors, according to:
//
//    Malis, Ezio, and Manuel Vargas. "Deeper understanding of the homography
//    decomposition for vision-based control." (2007): 90.
//
// The first pose is assumed to be P = [I | 0]. Note that the homography is
// plane-induced if `R.size() == t.size() == n.size() == 4`. If `R.size() ==
// t.size() == n.size() == 1` the homography is pure-rotational.
//
// @param H          3x3 homography matrix.
// @param K          3x3 calibration matrix.
// @param R          Possible 3x3 rotation matrices.
// @param t          Possible translation vectors.
// @param n          Possible normal vectors.
void DecomposeHomographyMatrix(const Eigen::Matrix3d& H,
                               const Eigen::Matrix3d& K1,
                               const Eigen::Matrix3d& K2,
                               std::vector<Eigen::Matrix3d>* R,
                               std::vector<Eigen::Vector3d>* t,
                               std::vector<Eigen::Vector3d>* n);

// Recover the most probable pose from the given homography matrix.
//
// The pose of the first image is assumed to be P = [I | 0].
//
// @param H            3x3 homography matrix.
// @param K1           3x3 calibration matrix of first camera.
// @param K2           3x3 calibration matrix of second camera.
// @param points1      First set of corresponding points.
// @param points2      Second set of corresponding points.
// @param inlier_mask  Only points with `true` in the inlier mask are
//                     considered in the cheirality test. Size of the
//                     inlier mask must match the number of points N.
// @param R            Most probable 3x3 rotation matrix.
// @param t            Most probable 3x1 translation vector.
// @param n            Most probable 3x1 normal vector.
// @param points3D     Triangulated 3D points infront of camera
//                     (only if homography is not pure-rotational).
void PoseFromHomographyMatrix(const Eigen::Matrix3d& H,
                              const Eigen::Matrix3d& K1,
                              const Eigen::Matrix3d& K2,
                              const std::vector<Eigen::Vector2d>& points1,
                              const std::vector<Eigen::Vector2d>& points2,
                              Eigen::Matrix3d* R,
                              Eigen::Vector3d* t,
                              Eigen::Vector3d* n,
                              std::vector<Eigen::Vector3d>* points3D);

// Compose homography matrix from relative pose.
//
// @param K1      3x3 calibration matrix of first camera.
// @param K2      3x3 calibration matrix of second camera.
// @param R       Most probable 3x3 rotation matrix.
// @param t       Most probable 3x1 translation vector.
// @param n       Most probable 3x1 normal vector.
// @param d       Orthogonal distance from plane.
//
// @return        3x3 homography matrix.
Eigen::Matrix3d HomographyMatrixFromPose(const Eigen::Matrix3d& K1,
                                         const Eigen::Matrix3d& K2,
                                         const Eigen::Matrix3d& R,
                                         const Eigen::Vector3d& t,
                                         const Eigen::Vector3d& n,
                                         const double d);

}  // namespace colmap

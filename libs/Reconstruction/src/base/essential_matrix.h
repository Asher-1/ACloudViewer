// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ceres/ceres.h>

#include <Eigen/Core>
#include <optional>
#include <vector>

#include "geometry/rigid3.h"
#include "base/pose.h"
#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

// Decompose an essential matrix into the possible rotations and translations.
//
// The first pose is assumed to be P = [I | 0] and the set of four other
// possible second poses are defined as: {[R1 | t], [R2 | t],
//                                        [R1 | -t], [R2 | -t]}
//
// @param E          3x3 essential matrix.
// @param R1         First possible 3x3 rotation matrix.
// @param R2         Second possible 3x3 rotation matrix.
// @param t          3x1 possible translation vector (also -t possible).
void DecomposeEssentialMatrix(const Eigen::Matrix3d& E,
                              Eigen::Matrix3d* R1,
                              Eigen::Matrix3d* R2,
                              Eigen::Vector3d* t);

// Recover the most probable pose from the given essential matrix.
//
// The pose of the first image is assumed to be P = [I | 0].
//
// @param E            3x3 essential matrix.
// @param points1      First set of corresponding points.
// @param points2      Second set of corresponding points.
// @param inlier_mask  Only points with `true` in the inlier mask are
//                     considered in the cheirality test. Size of the
//                     inlier mask must match the number of points N.
// @param R            Most probable 3x3 rotation matrix.
// @param t            Most probable 3x1 translation vector.
// @param points3D     Triangulated 3D points infront of camera.
void PoseFromEssentialMatrix(const Eigen::Matrix3d& E,
                             const std::vector<Eigen::Vector2d>& points1,
                             const std::vector<Eigen::Vector2d>& points2,
                             Eigen::Matrix3d* R,
                             Eigen::Vector3d* t,
                             std::vector<Eigen::Vector3d>* points3D);

// Compose essential matrix from relative camera poses.
//
// Assumes that first camera pose has projection matrix P = [I | 0], and
// pose of second camera is given as transformation from world to camera system.
//
// @param R             3x3 rotation matrix.
// @param t             3x1 translation vector.
//
// @return              3x3 essential matrix.
Eigen::Matrix3d EssentialMatrixFromPose(const Eigen::Matrix3d& R,
                                        const Eigen::Vector3d& t);

// Compose essential matrix from two absolute camera poses.
//
// @param proj_matrix1     3x4 projection matrix.
// @param proj_matrix2     3x4 projection matrix.
//
// @return                 3x3 essential matrix.
Eigen::Matrix3d EssentialMatrixFromAbsolutePoses(
        const Eigen::Matrix3x4d& proj_matrix1,
        const Eigen::Matrix3x4d& proj_matrix2);

// Find optimal image points, such that:
//
//     optimal_point1^t * E * optimal_point2 = 0
//
// as described in:
//
//   Lindstrom, P., "Triangulation made easy",
//   Computer Vision and Pattern Recognition (CVPR),
//   2010 IEEE Conference on , vol., no., pp.1554,1561, 13-18 June 2010
//
// @param E                Essential or fundamental matrix.
// @param point1           Corresponding 2D point in first image.
// @param point2           Corresponding 2D point in second image.
// @param optimal_point1   Estimated optimal image point in the first image.
// @param optimal_point2   Estimated optimal image point in the second image.
void FindOptimalImageObservations(const Eigen::Matrix3d& E,
                                  const Eigen::Vector2d& point1,
                                  const Eigen::Vector2d& point2,
                                  Eigen::Vector2d* optimal_point1,
                                  Eigen::Vector2d* optimal_point2);

// Compute the location of the epipole in homogeneous coordinates.
//
// @param E           3x3 essential matrix.
// @param left_image  If true, epipole in left image is computed,
//                    else in right image.
//
// @return            Epipole in homogeneous coordinates.
Eigen::Vector3d EpipoleFromEssentialMatrix(const Eigen::Matrix3d& E,
                                           const bool left_image);

// Invert the essential matrix, i.e. if the essential matrix E describes the
// transformation from camera A to B, the inverted essential matrix E' describes
// the transformation from camera B to A.
//
// @param E      3x3 essential matrix.
//
// @return       Inverted essential matrix.
Eigen::Matrix3d InvertEssentialMatrix(const Eigen::Matrix3d& matrix);

// Refine essential matrix.
//
// Decomposes the essential matrix into rotation and translation components
// and refines the relative pose using the function `RefineRelativePose`.
//
// @param E                3x3 essential matrix.
// @param points1          First set of corresponding points.
// @param points2          Second set of corresponding points.
// @param inlier_mask      Inlier mask for corresponding points.
// @param options          Solver options.
//
// @return                 Flag indicating if solution is usable.
bool RefineEssentialMatrix(const ceres::Solver::Options& options,
                           const std::vector<Eigen::Vector2d>& points1,
                           const std::vector<Eigen::Vector2d>& points2,
                           const std::vector<char>& inlier_mask,
                           Eigen::Matrix3d* E);

// Squared Sampson error in pixel coordinates for arbitrary central cameras.
// The input rays carry d(ray)/d(pixel), so the denominator is the tangent
// gradient pulled back into the image plane rather than a focal-length proxy.
double ComputeSquaredTangentSampsonError(const CamRayWithJac& cam_ray1,
                                         const CamRayWithJac& cam_ray2,
                                         const Eigen::Matrix3d& E);

void ComputeSquaredTangentSampsonError(
        const std::vector<CamRayWithJac>& cam_rays1,
        const std::vector<CamRayWithJac>& cam_rays2,
        const Eigen::Matrix3d& E,
        std::vector<double>* residuals);

// ---- Upstream-parity additions (COLMAP 4.x geometry/essential_matrix.h) ----
Eigen::Matrix3d FundamentalFromEssentialMatrix(const Eigen::Matrix3d& K2,
                                               const Eigen::Matrix3d& E,
                                               const Eigen::Matrix3d& K1);
Eigen::Matrix3d EssentialFromFundamentalMatrix(const Eigen::Matrix3d& K2,
                                               const Eigen::Matrix3d& F,
                                               const Eigen::Matrix3d& K1);

inline double SquaredPixelGradientNorm(const Eigen::Matrix3x2d& J,
                                       const Eigen::Vector3d& g) {
    const double gx = J(0, 0) * g[0] + J(1, 0) * g[1] + J(2, 0) * g[2];
    const double gy = J(0, 1) * g[0] + J(1, 1) * g[1] + J(2, 1) * g[2];
    return gx * gx + gy * gy;
}

// Calculate the squared tangent Sampson error for a single ray pair and a given
// essential matrix.
//
// The Sampson approximation is C(z)^2 / ||dC/dz||^2 for a constraint C and
// measurements z. Taking z to be the *pixel* coordinates, rather than the rays,
// yields an error in pixel units for any central camera model:
//
//     C            = ray2^T E ray1
//     dC/dpx1      = J1^T (E^T ray2)
//     dC/dpx2      = J2^T (E ray1)
//     error        = C^2 / (||dC/dpx1||^2 + ||dC/dpx2||^2)
//
// where J = d(ray) / d(pixel) is the unprojection Jacobian, obtainable via
// Camera::CamRayFromImgWithJac. This is the tangent Sampson error of Terekhov
// and Larsson, "Tangent Sampson Error: Fast Approximate Two-view Reprojection
// Error for Central Camera Models", ICCV 2023.
//
// Pixels are the space in which feature detection noise is (approximately)
// isotropic and uniform, so a threshold on this residual is meaningful in
// pixels across the whole image and for every camera model - unlike the plain
// Sampson error on unit bearings, whose pixel-equivalent tolerance grows with
// the angle from the principal direction.
//
// Note that the Sampson approximation is not invariant to the choice of
// homogeneous representative when that choice varies with the measurements:
// rescaling the constraint by g(z) perturbs the result by a term proportional
// to the residual. For an undistorted pinhole this reduces *exactly* to f^2
// times the Sampson error on normalized image coordinates when the (u, v, 1)
// representative is used (its Jacobian being the constant 1/f), and to first
// order in the residual when unit bearings are used. The latter is the sense in
// which this is an approximation of the true reprojection error.
//
// @param cam_ray1_with_jac  First bearing with its Jacobian d(ray1) /
// d(pixel1).
// @param cam_ray2_with_jac  Second bearing with its Jacobian d(ray2) /
// d(pixel2).
// @param E           3x3 essential matrix.
// @return            Squared tangent Sampson error, in squared pixels.


// Upstream-parity overload taking a rigid transform (COLMAP 4.x).
Eigen::Matrix3d EssentialMatrixFromPose(const Rigid3d& cam2_from_cam1);


// Upstream-parity: recover the relative pose from an essential matrix and
// camera rays, keeping only the candidates with positive-depth observations.
void PoseFromEssentialMatrix(const Eigen::Matrix3d& E,
                             const std::vector<Eigen::Vector3d>& cam_rays1,
                             const std::vector<Eigen::Vector3d>& cam_rays2,
                             Rigid3d* cam2_from_cam1,
                             std::vector<int>* valid_indices);


double ComputeSquaredSampsonError(const Eigen::Vector3d& point1,
                                  const Eigen::Vector3d& point2,
                                  const Eigen::Matrix3d& E);

void ComputeSquaredSampsonError(const std::vector<Eigen::Vector2d>& points1,
                                const std::vector<Eigen::Vector2d>& points2,
                                const Eigen::Matrix3d& E,
                                std::vector<double>* residuals);

void ComputeSquaredSampsonError(const std::vector<Eigen::Vector3d>& points1,
                                const std::vector<Eigen::Vector3d>& points2,
                                const Eigen::Matrix3d& E,
                                std::vector<double>* residuals);

void ComputeSquaredTangentSampsonErrorWithCheirality(
    const std::vector<CamRayWithJac>& cam_rays1_with_jac,
    const std::vector<CamRayWithJac>& cam_rays2_with_jac,
    const Eigen::Matrix3d& E,
    std::vector<double>* residuals);

}  // namespace colmap

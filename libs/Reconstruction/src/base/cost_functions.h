// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>

namespace colmap {

// EQUIRECTANGULAR is a full-sphere camera. Perspective residuals cannot be
// reused here because normalizing a camera-frame point by Z discards the back
// hemisphere. Measure the horizontal error on the azimuth circle so pixels at
// the two sides of the panorama seam remain adjacent during optimization.
template <typename T>
inline void EquirectangularReprojectionResidual(const T* const camera_params,
                                                const T* const point_in_camera,
                                                const double observed_x,
                                                const double observed_y,
                                                T* residuals) {
    const T width = camera_params[0];
    const T height = camera_params[1];
    const T horizontal = ceres::sqrt(point_in_camera[0] * point_in_camera[0] +
                                     point_in_camera[2] * point_in_camera[2]);
    const T theta = ceres::atan2(point_in_camera[0], point_in_camera[2]);
    const T phi = ceres::atan2(-point_in_camera[1], horizontal);
    const T observed_theta =
            T(2.0 * EIGEN_PI) * (T(observed_x) / width - T(0.5));
    const T wrapped_delta = ceres::atan2(ceres::sin(theta - observed_theta),
                                         ceres::cos(theta - observed_theta));

    residuals[0] = wrapped_delta * width / T(2.0 * EIGEN_PI);
    residuals[1] = (T(0.5) - phi / T(EIGEN_PI)) * height - T(observed_y);
}

class EquirectangularBundleAdjustmentCostFunction {
public:
    explicit EquirectangularBundleAdjustmentCostFunction(
            const Eigen::Vector2d& point2D)
        : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

    static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
        return new ceres::AutoDiffCostFunction<
                EquirectangularBundleAdjustmentCostFunction, 2, 4, 3, 3, 2>(
                new EquirectangularBundleAdjustmentCostFunction(point2D));
    }

    template <typename T>
    bool operator()(const T* const qvec,
                    const T* const tvec,
                    const T* const point3D,
                    const T* const camera_params,
                    T* residuals) const {
        T point_in_camera[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, point_in_camera);
        point_in_camera[0] += tvec[0];
        point_in_camera[1] += tvec[1];
        point_in_camera[2] += tvec[2];
        EquirectangularReprojectionResidual(camera_params, point_in_camera,
                                            observed_x_, observed_y_,
                                            residuals);
        return true;
    }

private:
    const double observed_x_;
    const double observed_y_;
};

class EquirectangularBundleAdjustmentConstantPoseCostFunction {
public:
    EquirectangularBundleAdjustmentConstantPoseCostFunction(
            const Eigen::Vector4d& qvec,
            const Eigen::Vector3d& tvec,
            const Eigen::Vector2d& point2D)
        : qw_(qvec(0)),
          qx_(qvec(1)),
          qy_(qvec(2)),
          qz_(qvec(3)),
          tx_(tvec(0)),
          ty_(tvec(1)),
          tz_(tvec(2)),
          observed_x_(point2D(0)),
          observed_y_(point2D(1)) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& qvec,
                                       const Eigen::Vector3d& tvec,
                                       const Eigen::Vector2d& point2D) {
        return new ceres::AutoDiffCostFunction<
                EquirectangularBundleAdjustmentConstantPoseCostFunction, 2, 3,
                2>(new EquirectangularBundleAdjustmentConstantPoseCostFunction(
                qvec, tvec, point2D));
    }

    template <typename T>
    bool operator()(const T* const point3D,
                    const T* const camera_params,
                    T* residuals) const {
        const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};
        T point_in_camera[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, point_in_camera);
        point_in_camera[0] += T(tx_);
        point_in_camera[1] += T(ty_);
        point_in_camera[2] += T(tz_);
        EquirectangularReprojectionResidual(camera_params, point_in_camera,
                                            observed_x_, observed_y_,
                                            residuals);
        return true;
    }

private:
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    const double observed_x_;
    const double observed_y_;
};

class EquirectangularRigBundleAdjustmentCostFunction {
public:
    explicit EquirectangularRigBundleAdjustmentCostFunction(
            const Eigen::Vector2d& point2D)
        : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

    static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
        return new ceres::AutoDiffCostFunction<
                EquirectangularRigBundleAdjustmentCostFunction, 2, 4, 3, 4, 3,
                3, 2>(
                new EquirectangularRigBundleAdjustmentCostFunction(point2D));
    }

    template <typename T>
    bool operator()(const T* const rig_qvec,
                    const T* const rig_tvec,
                    const T* const rel_qvec,
                    const T* const rel_tvec,
                    const T* const point3D,
                    const T* const camera_params,
                    T* residuals) const {
        T qvec[4];
        ceres::QuaternionProduct(rel_qvec, rig_qvec, qvec);
        T tvec[3];
        ceres::UnitQuaternionRotatePoint(rel_qvec, rig_tvec, tvec);
        tvec[0] += rel_tvec[0];
        tvec[1] += rel_tvec[1];
        tvec[2] += rel_tvec[2];
        T point_in_camera[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, point_in_camera);
        point_in_camera[0] += tvec[0];
        point_in_camera[1] += tvec[1];
        point_in_camera[2] += tvec[2];
        EquirectangularReprojectionResidual(camera_params, point_in_camera,
                                            observed_x_, observed_y_,
                                            residuals);
        return true;
    }

private:
    const double observed_x_;
    const double observed_y_;
};

// Standard bundle adjustment cost function for variable
// camera pose and calibration and point parameters.
template <typename CameraModel>
class BundleAdjustmentCostFunction {
public:
    explicit BundleAdjustmentCostFunction(const Eigen::Vector2d& point2D)
        : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

    static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
        return (new ceres::AutoDiffCostFunction<
                BundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 3,
                CameraModel::kNumParams>(
                new BundleAdjustmentCostFunction(point2D)));
    }

    template <typename T>
    bool operator()(const T* const qvec,
                    const T* const tvec,
                    const T* const point3D,
                    const T* const camera_params,
                    T* residuals) const {
        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        // Project to image plane.
        projection[0] /= projection[2];
        projection[1] /= projection[2];

        // Distort and transform to pixel space.
        CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                                  &residuals[0], &residuals[1]);

        // Re-projection error.
        residuals[0] -= T(observed_x_);
        residuals[1] -= T(observed_y_);

        return true;
    }

private:
    const double observed_x_;
    const double observed_y_;
};

// Bundle adjustment cost function for variable
// camera calibration and point parameters, and fixed camera pose.
template <typename CameraModel>
class BundleAdjustmentConstantPoseCostFunction {
public:
    BundleAdjustmentConstantPoseCostFunction(const Eigen::Vector4d& qvec,
                                             const Eigen::Vector3d& tvec,
                                             const Eigen::Vector2d& point2D)
        : qw_(qvec(0)),
          qx_(qvec(1)),
          qy_(qvec(2)),
          qz_(qvec(3)),
          tx_(tvec(0)),
          ty_(tvec(1)),
          tz_(tvec(2)),
          observed_x_(point2D(0)),
          observed_y_(point2D(1)) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& qvec,
                                       const Eigen::Vector3d& tvec,
                                       const Eigen::Vector2d& point2D) {
        return (new ceres::AutoDiffCostFunction<
                BundleAdjustmentConstantPoseCostFunction<CameraModel>, 2, 3,
                CameraModel::kNumParams>(
                new BundleAdjustmentConstantPoseCostFunction(qvec, tvec,
                                                             point2D)));
    }

    template <typename T>
    bool operator()(const T* const point3D,
                    const T* const camera_params,
                    T* residuals) const {
        const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += T(tx_);
        projection[1] += T(ty_);
        projection[2] += T(tz_);

        // Project to image plane.
        projection[0] /= projection[2];
        projection[1] /= projection[2];

        // Distort and transform to pixel space.
        CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                                  &residuals[0], &residuals[1]);

        // Re-projection error.
        residuals[0] -= T(observed_x_);
        residuals[1] -= T(observed_y_);

        return true;
    }

private:
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    const double observed_x_;
    const double observed_y_;
};

// Rig bundle adjustment cost function for variable camera pose and calibration
// and point parameters. Different from the standard bundle adjustment function,
// this cost function is suitable for camera rigs with consistent relative poses
// of the cameras within the rig. The cost function first projects points into
// the local system of the camera rig and then into the local system of the
// camera within the rig.
template <typename CameraModel>
class RigBundleAdjustmentCostFunction {
public:
    explicit RigBundleAdjustmentCostFunction(const Eigen::Vector2d& point2D)
        : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

    static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
        return (new ceres::AutoDiffCostFunction<
                RigBundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 4, 3, 3,
                CameraModel::kNumParams>(
                new RigBundleAdjustmentCostFunction(point2D)));
    }

    template <typename T>
    bool operator()(const T* const rig_qvec,
                    const T* const rig_tvec,
                    const T* const rel_qvec,
                    const T* const rel_tvec,
                    const T* const point3D,
                    const T* const camera_params,
                    T* residuals) const {
        // Concatenate rotations.
        T qvec[4];
        ceres::QuaternionProduct(rel_qvec, rig_qvec, qvec);

        // Concatenate translations.
        T tvec[3];
        ceres::UnitQuaternionRotatePoint(rel_qvec, rig_tvec, tvec);
        tvec[0] += rel_tvec[0];
        tvec[1] += rel_tvec[1];
        tvec[2] += rel_tvec[2];

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        // Project to image plane.
        projection[0] /= projection[2];
        projection[1] /= projection[2];

        // Distort and transform to pixel space.
        CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                                  &residuals[0], &residuals[1]);

        // Re-projection error.
        residuals[0] -= T(observed_x_);
        residuals[1] -= T(observed_y_);

        return true;
    }

private:
    const double observed_x_;
    const double observed_y_;
};

// Frame-aware rig cost function (W3-2b step 4, upstream COLMAP 4.x
// semantics): the camera pose composes as
// cam_from_world = rig_from_world * sensor_from_rig. The legacy
// RigBundleAdjustmentCostFunction above composes in the opposite
// (CameraRig) order. Parameter blocks: rig qvec/tvec and sensor qvec/tvec
// ([w, x, y, z]), 3D point, camera params.
template <typename CameraModel>
class FrameRigBundleAdjustmentCostFunction {
public:
    explicit FrameRigBundleAdjustmentCostFunction(
            const Eigen::Vector2d& point2D)
        : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

    static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
        return (new ceres::AutoDiffCostFunction<
                FrameRigBundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 4,
                3, 3, CameraModel::kNumParams>(
                new FrameRigBundleAdjustmentCostFunction(point2D)));
    }

    template <typename T>
    bool operator()(const T* const rig_qvec,
                    const T* const rig_tvec,
                    const T* const sensor_qvec,
                    const T* const sensor_tvec,
                    const T* const point3D,
                    const T* const camera_params,
                    T* residuals) const {
        // cam_from_world = rig_from_world * sensor_from_rig
        T qvec[4];
        ceres::QuaternionProduct(rig_qvec, sensor_qvec, qvec);

        T tvec[3];
        ceres::UnitQuaternionRotatePoint(rig_qvec, sensor_tvec, tvec);
        tvec[0] += rig_tvec[0];
        tvec[1] += rig_tvec[1];
        tvec[2] += rig_tvec[2];

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        // Project to image plane.
        projection[0] /= projection[2];
        projection[1] /= projection[2];

        // Distort and transform to pixel space.
        CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                                  &residuals[0], &residuals[1]);

        // Re-projection error.
        residuals[0] -= T(observed_x_);
        residuals[1] -= T(observed_y_);

        return true;
    }

private:
    const double observed_x_;
    const double observed_y_;
};

// Cost function for refining two-view geometry based on the Sampson-Error.
//
// First pose is assumed to be located at the origin with 0 rotation. Second
// pose is assumed to be on the unit sphere around the first pose, i.e. the
// pose of the second camera is parameterized by a 3D rotation and a
// 3D translation with unit norm. `tvec` is therefore over-parameterized as is
// and should be down-projected using `HomogeneousVectorParameterization`.
class RelativePoseCostFunction {
public:
    RelativePoseCostFunction(const Eigen::Vector2d& x1,
                             const Eigen::Vector2d& x2)
        : x1_(x1(0)), y1_(x1(1)), x2_(x2(0)), y2_(x2(1)) {}

    static ceres::CostFunction* Create(const Eigen::Vector2d& x1,
                                       const Eigen::Vector2d& x2) {
        return (new ceres::AutoDiffCostFunction<RelativePoseCostFunction, 1, 4,
                                                3>(
                new RelativePoseCostFunction(x1, x2)));
    }

    template <typename T>
    bool operator()(const T* const qvec,
                    const T* const tvec,
                    T* residuals) const {
        Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R;
        ceres::QuaternionToRotation(qvec, R.data());

        // Matrix representation of the cross product t x R.
        Eigen::Matrix<T, 3, 3> t_x;
        t_x << T(0), -tvec[2], tvec[1], tvec[2], T(0), -tvec[0], -tvec[1],
                tvec[0], T(0);

        // Essential matrix.
        const Eigen::Matrix<T, 3, 3> E = t_x * R;

        // Homogeneous image coordinates.
        const Eigen::Matrix<T, 3, 1> x1_h(T(x1_), T(y1_), T(1));
        const Eigen::Matrix<T, 3, 1> x2_h(T(x2_), T(y2_), T(1));

        // Squared sampson error.
        const Eigen::Matrix<T, 3, 1> Ex1 = E * x1_h;
        const Eigen::Matrix<T, 3, 1> Etx2 = E.transpose() * x2_h;
        const T x2tEx1 = x2_h.transpose() * Ex1;
        residuals[0] = x2tEx1 * x2tEx1 /
                       (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Etx2(0) * Etx2(0) +
                        Etx2(1) * Etx2(1));

        return true;
    }

private:
    const double x1_;
    const double y1_;
    const double x2_;
    const double y2_;
};

}  // namespace colmap

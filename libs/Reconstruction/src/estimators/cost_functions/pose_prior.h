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

#include "estimators/cost_functions/quaternion_utils.h"
#include "estimators/cost_functions/utils.h"
#include "geometry/rigid3.h"

namespace colmap {

// 6-DoF error on the absolute sensor pose. The residual is the log of the error
// pose, splitting SE(3) into SO(3) x R^3. The residual is computed in the
// sensor frame. Its first and last three components correspond to the rotation
// and translation errors, respectively.
struct AbsolutePosePriorCostFunctor
    : public AutoDiffCostFunctor<AbsolutePosePriorCostFunctor, 6, 7> {
public:
    explicit AbsolutePosePriorCostFunctor(
            const Rigid3d& sensor_from_world_prior)
        : world_from_sensor_prior_(Inverse(sensor_from_world_prior)) {}

    template <typename T>
    bool operator()(const T* const sensor_from_world, T* residuals_ptr) const {
        const Eigen::Quaternion<T> param_from_prior_rotation =
                EigenQuaternionMap<T>(sensor_from_world) *
                world_from_sensor_prior_.rotation().cast<T>();
        AngleAxisFromEigenQuaternion(param_from_prior_rotation.coeffs().data(),
                                     residuals_ptr);

        Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_prior_translation(
                residuals_ptr + 3);
        param_from_prior_translation =
                EigenVector3Map<T>(sensor_from_world + 4) +
                EigenQuaternionMap<T>(sensor_from_world) *
                        world_from_sensor_prior_.translation().cast<T>();

        return true;
    }

private:
    const Rigid3d world_from_sensor_prior_;
};

// 3-DoF error on the sensor position in the world coordinate frame.
struct AbsolutePosePositionPriorCostFunctor
    : public AutoDiffCostFunctor<AbsolutePosePositionPriorCostFunctor, 3, 7> {
public:
    explicit AbsolutePosePositionPriorCostFunctor(
            const Eigen::Vector3d& position_in_world_prior)
        : position_in_world_prior_(position_in_world_prior) {}

    template <typename T>
    bool operator()(const T* const sensor_from_world, T* residuals_ptr) const {
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
        residuals = position_in_world_prior_.cast<T>() +
                    EigenQuaternionMap<T>(sensor_from_world).inverse() *
                            EigenVector3Map<T>(sensor_from_world + 4);
        return true;
    }

private:
    const Eigen::Vector3d position_in_world_prior_;
};

// Fork adaptation of AbsolutePosePositionPriorCostFunctor over the fork's
// split pose parameter blocks: a [w, x, y, z] quaternion block and a separate
// translation block (the upstream functor assumes a contiguous 7-scalar
// [x, y, z, w, t] block). The residual is
//   r = position_in_world_prior + R(q)^T * t
// which is the negative position error of the sensor center in the world.
struct AbsolutePosePositionPriorQvecTvecCostFunctor {
public:
    explicit AbsolutePosePositionPriorQvecTvecCostFunctor(
            const Eigen::Vector3d& position_in_world_prior)
        : position_in_world_prior_(position_in_world_prior) {}

    template <typename T>
    bool operator()(const T* const qvec,
                    const T* const tvec,
                    T* residuals_ptr) const {
        // The fork stores quaternions as [w, x, y, z].
        const Eigen::Quaternion<T> q_from_world(qvec[0], qvec[1], qvec[2],
                                                qvec[3]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_from_world(tvec);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
        residuals = position_in_world_prior_.template cast<T>() +
                    q_from_world.conjugate() * t_from_world;
        return true;
    }

private:
    const Eigen::Vector3d position_in_world_prior_;
};

// 3-DoF error on the rig sensor position in the world coordinate frame.
struct AbsoluteRigPosePositionPriorCostFunctor
    : public AutoDiffCostFunctor<AbsoluteRigPosePositionPriorCostFunctor,
                                 3,
                                 7,
                                 7> {
public:
    explicit AbsoluteRigPosePositionPriorCostFunctor(
            const Eigen::Vector3d& position_in_world_prior)
        : position_in_world_prior_(position_in_world_prior) {}

    template <typename T>
    bool operator()(const T* const sensor_from_rig,
                    const T* const rig_from_world,
                    T* residuals_ptr) const {
        const Eigen::Quaternion<T> sensor_from_world_rotation =
                EigenQuaternionMap<T>(sensor_from_rig) *
                EigenQuaternionMap<T>(rig_from_world);
        const Eigen::Matrix<T, 3, 1> sensor_from_world_translation =
                EigenVector3Map<T>(sensor_from_rig + 4) +
                EigenQuaternionMap<T>(sensor_from_rig) *
                        EigenVector3Map<T>(rig_from_world + 4);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
        residuals = position_in_world_prior_.cast<T>() +
                    sensor_from_world_rotation.inverse() *
                            sensor_from_world_translation;
        return true;
    }

private:
    const Eigen::Vector3d position_in_world_prior_;
};

// 6-DoF error between two absolute camera poses based on a prior on their
// relative pose, with identical scale for the translation. The residual is
// computed in the frame of camera i. Its first and last three components
// correspond to the rotation and translation errors, respectively.
struct RelativePosePriorCostFunctor
    : public AutoDiffCostFunctor<RelativePosePriorCostFunctor, 6, 7, 7> {
public:
    explicit RelativePosePriorCostFunctor(const Rigid3d& i_from_j_prior)
        : j_from_i_prior_(Inverse(i_from_j_prior)) {}

    template <typename T>
    bool operator()(const T* const i_from_world,
                    const T* const j_from_world,
                    T* residuals_ptr) const {
        const Eigen::Quaternion<T> i_from_j_rotation =
                EigenQuaternionMap<T>(i_from_world) *
                EigenQuaternionMap<T>(j_from_world).inverse();
        const Eigen::Quaternion<T> param_from_prior_rotation =
                i_from_j_rotation *
                j_from_i_prior_.rotation().template cast<T>();
        AngleAxisFromEigenQuaternion(param_from_prior_rotation.coeffs().data(),
                                     residuals_ptr);

        const Eigen::Matrix<T, 3, 1> j_from_i_prior_translation =
                j_from_i_prior_.translation().cast<T>() -
                EigenVector3Map<T>(j_from_world + 4);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_prior_translation(
                residuals_ptr + 3);
        param_from_prior_translation =
                EigenVector3Map<T>(i_from_world + 4) +
                i_from_j_rotation * j_from_i_prior_translation;

        return true;
    }

private:
    const Rigid3d j_from_i_prior_;
};

}  // namespace colmap

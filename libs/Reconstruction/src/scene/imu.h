// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <ostream>

#include "geometry/pose.h"
#include "sensor/imu.h"
#include "util/types.h"

namespace colmap {

struct Imu {
    ImuCalibration calib;
    imu_t imu_id = kInvalidImuId;
    camera_t camera_id = kInvalidCameraId;
    Rigid3d imu_from_cam;
};

struct ImuState {
    Eigen::Matrix<double, 9, 1> params = Eigen::Matrix<double, 9, 1>::Zero();

    ImuState() = default;
    ImuState(const Eigen::Vector3d& velocity,
             const Eigen::Vector3d& bias_gyro,
             const Eigen::Vector3d& bias_accel) {
        params.head<3>() = velocity;
        params.segment<3>(3) = bias_gyro;
        params.tail<3>() = bias_accel;
    }

    Eigen::Map<Eigen::Vector3d> Velocity() {
        return Eigen::Map<Eigen::Vector3d>(params.data());
    }
    Eigen::Map<const Eigen::Vector3d> Velocity() const {
        return Eigen::Map<const Eigen::Vector3d>(params.data());
    }
    Eigen::Map<Eigen::Vector3d> BiasGyro() {
        return Eigen::Map<Eigen::Vector3d>(params.data() + 3);
    }
    Eigen::Map<const Eigen::Vector3d> BiasGyro() const {
        return Eigen::Map<const Eigen::Vector3d>(params.data() + 3);
    }
    Eigen::Map<Eigen::Vector3d> BiasAccel() {
        return Eigen::Map<Eigen::Vector3d>(params.data() + 6);
    }
    Eigen::Map<const Eigen::Vector3d> BiasAccel() const {
        return Eigen::Map<const Eigen::Vector3d>(params.data() + 6);
    }
};

std::ostream& operator<<(std::ostream& stream, const Imu& imu);
std::ostream& operator<<(std::ostream& stream, const ImuState& state);

}  // namespace colmap

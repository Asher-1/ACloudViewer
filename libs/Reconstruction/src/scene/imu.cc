// SPDX-License-Identifier: BSD-3-Clause

#include "scene/imu.h"

namespace colmap {

std::ostream& operator<<(std::ostream& stream, const Imu& imu) {
  return stream << "Imu(imu_id=" << imu.imu_id << ", camera_id="
                << imu.camera_id << ")";
}

std::ostream& operator<<(std::ostream& stream, const ImuState& state) {
  return stream << "ImuState(vel=[" << state.Velocity().transpose()
                << "], bias_gyro=[" << state.BiasGyro().transpose()
                << "], bias_accel=[" << state.BiasAccel().transpose() << "])";
}

}  // namespace colmap

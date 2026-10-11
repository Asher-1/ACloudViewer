// SPDX-License-Identifier: BSD-3-Clause

#include "scene/imu.h"

#include <sstream>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(Imu, DefaultAndPrint) {
  Imu imu;
  EXPECT_EQ(imu.imu_id, kInvalidImuId);
  EXPECT_EQ(imu.camera_id, kInvalidCameraId);
  imu.imu_id = 1;
  imu.camera_id = 2;
  std::ostringstream stream;
  stream << imu;
  EXPECT_EQ(stream.str(), "Imu(imu_id=1, camera_id=2)");
}

TEST(ImuState, LayoutAccessorsAndPrint) {
  ImuState state(Eigen::Vector3d(1, 2, 3), Eigen::Vector3d(4, 5, 6),
                 Eigen::Vector3d(7, 8, 9));
  EXPECT_EQ(state.Velocity(), Eigen::Vector3d(1, 2, 3));
  EXPECT_EQ(state.BiasGyro(), Eigen::Vector3d(4, 5, 6));
  EXPECT_EQ(state.BiasAccel(), Eigen::Vector3d(7, 8, 9));
  state.Velocity() = Eigen::Vector3d(10, 11, 12);
  EXPECT_EQ(state.params.head<3>(), Eigen::Vector3d(10, 11, 12));
  std::ostringstream stream;
  stream << state;
  EXPECT_EQ(stream.str(),
            "ImuState(vel=[10 11 12], bias_gyro=[4 5 6], bias_accel=[7 8 9])");
}

}  // namespace
}  // namespace colmap

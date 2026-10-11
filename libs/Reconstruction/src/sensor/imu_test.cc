// SPDX-License-Identifier: BSD-3-Clause

#include "sensor/imu.h"

#include <sstream>

#include <gtest/gtest.h>

namespace colmap {
namespace {

ImuMeasurement Measurement(const timestamp_t timestamp) {
  return ImuMeasurement(timestamp, Eigen::Vector3d::Zero(),
                        Eigen::Vector3d::Zero());
}

TEST(ImuCalibration, DefaultAndPrint) {
  ImuCalibration calibration;
  EXPECT_GT(calibration.imu_rate, 0.0);
  EXPECT_GT(calibration.gravity_magnitude, 0.0);
  calibration.gyro_noise_density = 1;
  calibration.accel_noise_density = 2;
  calibration.gravity_magnitude = 3;
  std::ostringstream stream;
  stream << calibration;
  EXPECT_EQ(stream.str(),
            "ImuCalibration(gyro_noise_density=1, accel_noise_density=2, "
            "gravity_magnitude=3)");
}

TEST(ImuMeasurement, DefaultConstructAndPrint) {
  const ImuMeasurement empty;
  EXPECT_EQ(empty.timestamp, kInvalidTimestamp);
  EXPECT_EQ(empty.gyro, Eigen::Vector3d::Zero());
  const ImuMeasurement measurement(
      100, Eigen::Vector3d(1, 2, 3), Eigen::Vector3d(4, 5, 6));
  std::ostringstream stream;
  stream << measurement;
  EXPECT_EQ(stream.str(), "ImuMeasurement(t=100, gyro=[1 2 3], accel=[4 5 6])");
}

TEST(ImuMeasurements, SortedInsertMergeAndRemove) {
  ImuMeasurements measurements;
  measurements.Insert(Measurement(30));
  measurements.Insert(Measurement(10));
  measurements.Insert(Measurement(20));
  ASSERT_EQ(measurements.Size(), 3);
  EXPECT_EQ(measurements[0].timestamp, 10);
  EXPECT_EQ(measurements[2].timestamp, 30);

  measurements.InsertSorted({Measurement(15), Measurement(25)});
  ASSERT_EQ(measurements.Size(), 5);
  EXPECT_EQ(measurements[1].timestamp, 15);
  EXPECT_EQ(measurements[3].timestamp, 25);
  EXPECT_ANY_THROW(measurements.Insert(Measurement(25)));
  measurements.Remove(Measurement(20));
  EXPECT_EQ(measurements.Size(), 4);
  EXPECT_ANY_THROW(measurements.Remove(Measurement(99)));
}

TEST(ImuMeasurements, ExtractBracketingRange) {
  ImuMeasurements measurements;
  for (timestamp_t timestamp = 100; timestamp <= 500; timestamp += 100) {
    measurements.Insert(Measurement(timestamp));
  }
  ImuMeasurements range;
  measurements.ExtractMeasurementsInRange(150, 350, &range);
  ASSERT_EQ(range.Size(), 4);
  EXPECT_EQ(range.front().timestamp, 100);
  EXPECT_EQ(range.back().timestamp, 400);

  measurements.ExtractMeasurementsInRange(50, 300, &range);
  EXPECT_TRUE(range.Empty());
  EXPECT_ANY_THROW(measurements.ExtractMeasurementsInRange(300, 200, &range));
}

}  // namespace
}  // namespace colmap

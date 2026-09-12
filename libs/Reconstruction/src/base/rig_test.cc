// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------

#define TEST_NAME "base/rig"
#include "util/testing.h"

#include <sstream>

#include "base/rig.h"

using namespace colmap;

TEST(base_rig, TestTextAndBinaryRoundTrip) {
  Rig rig;
  rig.SetRigId(7);
  rig.AddRefCamera(1);
  rig.AddCamera(2,
                Eigen::Vector4d(2.0, 0.0, 0.0, 0.0),
                Eigen::Vector3d(0.25, -0.5, 1.0));

  std::stringstream text;
  rig.WriteText(&text);
  Rig from_text;
  ASSERT_TRUE(from_text.ReadText(&text));
  EXPECT_EQ(from_text.RigId(), 7);
  EXPECT_EQ(from_text.RefCameraId(), 1);
  EXPECT_EQ(from_text.NumCameras(), 2);
  EXPECT_TRUE(from_text.CamFromRigQvec(2).isApprox(
      Eigen::Vector4d(1.0, 0.0, 0.0, 0.0)));
  EXPECT_TRUE(from_text.CamFromRigTvec(2).isApprox(
      Eigen::Vector3d(0.25, -0.5, 1.0)));

  std::stringstream binary;
  rig.WriteBinary(&binary);
  Rig from_binary;
  ASSERT_TRUE(from_binary.ReadBinary(&binary));
  EXPECT_EQ(from_binary.RigId(), 7);
  EXPECT_EQ(from_binary.RefCameraId(), 1);
  EXPECT_EQ(from_binary.NumCameras(), 2);
  EXPECT_TRUE(from_binary.CamFromRigTvec(2).isApprox(
      Eigen::Vector3d(0.25, -0.5, 1.0)));
}

TEST(base_rig, TestGenericReferenceSensorRoundTrip) {
  Rig rig;
  rig.SetRigId(8);
  rig.AddRefSensor(sensor_t(SensorType::IMU, 3));
  rig.AddCamera(2, Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
                Eigen::Vector3d(0.25, -0.5, 1.0));

  std::stringstream text;
  rig.WriteText(&text);
  Rig from_text;
  ASSERT_TRUE(from_text.ReadText(&text));
  EXPECT_TRUE(from_text.RefSensorId() == sensor_t(SensorType::IMU, 3));
  EXPECT_TRUE(from_text.HasCamera(2));
  EXPECT_FALSE(from_text.HasSensorFromRig(sensor_t(SensorType::IMU, 3)));

  std::stringstream binary;
  rig.WriteBinary(&binary);
  Rig from_binary;
  ASSERT_TRUE(from_binary.ReadBinary(&binary));
  EXPECT_TRUE(from_binary.RefSensorId() == sensor_t(SensorType::IMU, 3));
  EXPECT_TRUE(from_binary.CamFromRigTvec(2).isApprox(
      Eigen::Vector3d(0.25, -0.5, 1.0)));
}

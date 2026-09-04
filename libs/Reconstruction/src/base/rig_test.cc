// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------

#define TEST_NAME "base/rig"
#include "util/testing.h"

#include <sstream>

#include "base/rig.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestTextAndBinaryRoundTrip) {
  Rig rig;
  rig.SetRigId(7);
  rig.AddRefCamera(1);
  rig.AddCamera(2,
                Eigen::Vector4d(2.0, 0.0, 0.0, 0.0),
                Eigen::Vector3d(0.25, -0.5, 1.0));

  std::stringstream text;
  rig.WriteText(&text);
  Rig from_text;
  BOOST_REQUIRE(from_text.ReadText(&text));
  BOOST_CHECK_EQUAL(from_text.RigId(), 7);
  BOOST_CHECK_EQUAL(from_text.RefCameraId(), 1);
  BOOST_CHECK_EQUAL(from_text.NumCameras(), 2);
  BOOST_CHECK(from_text.CamFromRigQvec(2).isApprox(
      Eigen::Vector4d(1.0, 0.0, 0.0, 0.0)));
  BOOST_CHECK(from_text.CamFromRigTvec(2).isApprox(
      Eigen::Vector3d(0.25, -0.5, 1.0)));

  std::stringstream binary;
  rig.WriteBinary(&binary);
  Rig from_binary;
  BOOST_REQUIRE(from_binary.ReadBinary(&binary));
  BOOST_CHECK_EQUAL(from_binary.RigId(), 7);
  BOOST_CHECK_EQUAL(from_binary.RefCameraId(), 1);
  BOOST_CHECK_EQUAL(from_binary.NumCameras(), 2);
  BOOST_CHECK(from_binary.CamFromRigTvec(2).isApprox(
      Eigen::Vector3d(0.25, -0.5, 1.0)));
}

BOOST_AUTO_TEST_CASE(TestGenericReferenceSensorRoundTrip) {
  Rig rig;
  rig.SetRigId(8);
  rig.AddRefSensor(sensor_t(SensorType::IMU, 3));
  rig.AddCamera(2, Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
                Eigen::Vector3d(0.25, -0.5, 1.0));

  std::stringstream text;
  rig.WriteText(&text);
  Rig from_text;
  BOOST_REQUIRE(from_text.ReadText(&text));
  BOOST_CHECK(from_text.RefSensorId() == sensor_t(SensorType::IMU, 3));
  BOOST_CHECK(from_text.HasCamera(2));
  BOOST_CHECK(!from_text.HasSensorFromRig(sensor_t(SensorType::IMU, 3)));

  std::stringstream binary;
  rig.WriteBinary(&binary);
  Rig from_binary;
  BOOST_REQUIRE(from_binary.ReadBinary(&binary));
  BOOST_CHECK(from_binary.RefSensorId() == sensor_t(SensorType::IMU, 3));
  BOOST_CHECK(from_binary.CamFromRigTvec(2).isApprox(
      Eigen::Vector3d(0.25, -0.5, 1.0)));
}

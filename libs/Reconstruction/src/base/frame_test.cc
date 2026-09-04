// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------

#define TEST_NAME "base/frame"
#include "util/testing.h"

#include <sstream>

#include "base/frame.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestTextAndBinaryRoundTrip) {
  Frame frame;
  frame.SetFrameId(11);
  frame.SetRigId(7);
  frame.AddImageId(2);
  frame.AddImageId(1);
  frame.AddDataId(data_t(sensor_t(SensorType::IMU, 3), 21));
  frame.SetRigFromWorld(Eigen::Vector4d(2.0, 0.0, 0.0, 0.0),
                        Eigen::Vector3d(3.0, 4.0, 5.0));

  std::stringstream text;
  frame.WriteText(&text);
  Frame from_text;
  BOOST_REQUIRE(from_text.ReadText(&text));
  BOOST_CHECK_EQUAL(from_text.FrameId(), 11);
  BOOST_CHECK_EQUAL(from_text.RigId(), 7);
  BOOST_CHECK(from_text.HasImageId(1));
  BOOST_CHECK(from_text.HasImageId(2));
  BOOST_CHECK(from_text.HasDataId(data_t(sensor_t(SensorType::IMU, 3), 21)));
  BOOST_CHECK(from_text.HasPose());
  BOOST_CHECK(from_text.RigFromWorldQvec().isApprox(
      Eigen::Vector4d(1.0, 0.0, 0.0, 0.0)));
  BOOST_CHECK(from_text.RigFromWorldTvec().isApprox(
      Eigen::Vector3d(3.0, 4.0, 5.0)));

  std::stringstream binary;
  frame.WriteBinary(&binary);
  Frame from_binary;
  BOOST_REQUIRE(from_binary.ReadBinary(&binary));
  BOOST_CHECK_EQUAL(from_binary.FrameId(), 11);
  BOOST_CHECK_EQUAL(from_binary.RigId(), 7);
  BOOST_CHECK(from_binary.HasImageId(1));
  BOOST_CHECK(from_binary.HasImageId(2));
  BOOST_CHECK(from_binary.HasDataId(data_t(sensor_t(SensorType::IMU, 3), 21)));
  BOOST_CHECK(from_binary.RigFromWorldTvec().isApprox(
      Eigen::Vector3d(3.0, 4.0, 5.0)));
}

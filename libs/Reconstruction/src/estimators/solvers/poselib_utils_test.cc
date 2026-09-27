// SPDX-License-Identifier: BSD-3-Clause
//
// Port of upstream MVS/colmap d3ccaf35
// src/colmap/estimators/solvers/poselib_utils_test.cc (batch 24 test
// alignment); include paths mapped to the fork layout and the fork Camera
// accessor conventions (setters/getters) applied.

#include "estimators/solvers/poselib_utils.h"

#include "geometry/rigid3_matchers.h"
#include "scene/camera.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(PoseLibUtils, CameraRoundTrip) {
  Camera camera;
  camera.SetModelId(CameraModelId::kSimpleRadial);
  camera.SetWidth(1920);
  camera.SetHeight(1080);
  camera.SetParams({1000.0, 960.0, 540.0, 0.1});

  const poselib::Camera poselib_camera = ConvertCameraToPoseLibCamera(camera);
  const Camera camera_back = ConvertPoseLibCameraToCamera(poselib_camera);

  EXPECT_EQ(camera.ModelId(), camera_back.ModelId());
  EXPECT_EQ(camera.Width(), camera_back.Width());
  EXPECT_EQ(camera.Height(), camera_back.Height());
  EXPECT_EQ(camera.Params(), camera_back.Params());
}

TEST(PoseLibUtils, Rigid3dRoundTrip) {
  const Eigen::Quaterniond rotation =
      Eigen::Quaterniond(0.5, 0.5, 0.5, 0.5).normalized();
  const Eigen::Vector3d translation(1.0, 2.0, 3.0);
  const Rigid3d rigid(rotation, translation);

  const poselib::CameraPose poselib_pose = ConvertRigid3dToPoseLibPose(rigid);
  const Rigid3d rigid_back = ConvertPoseLibPoseToRigid3d(poselib_pose);

  EXPECT_THAT(rigid_back, Rigid3dNear(rigid, 1e-10, 1e-10));
}

}  // namespace
}  // namespace colmap

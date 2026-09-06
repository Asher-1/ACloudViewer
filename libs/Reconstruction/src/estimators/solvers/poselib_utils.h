// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <PoseLib/camera_pose.h>

#include "base/camera.h"
#include "geometry/rigid3.h"

namespace colmap {

// Convert COLMAP Camera to PoseLib Camera.
poselib::Camera ConvertCameraToPoseLibCamera(const Camera& camera);

// Convert PoseLib Camera to COLMAP Camera.
Camera ConvertPoseLibCameraToCamera(const poselib::Camera& camera);

// Convert COLMAP Rigid3d to PoseLib CameraPose.
poselib::CameraPose ConvertRigid3dToPoseLibPose(const Rigid3d& rigid);

// Convert PoseLib CameraPose to COLMAP Rigid3d.
Rigid3d ConvertPoseLibPoseToRigid3d(const poselib::CameraPose& pose);

}  // namespace colmap

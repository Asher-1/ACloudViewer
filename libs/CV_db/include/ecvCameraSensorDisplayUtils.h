// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "CV_db.h"
#include "ecvCameraSensor.h"
#include "ecvGLMatrix.h"

//! Shared helpers for COLMAP / reconstruction camera poses and VTK display.
/** Pose convention (see ccCameraSensor::PoseFrame):
    - VtkColmap: store raw COLMAP/OpenCV cam2world; local frustum uses +Z
      forward and Y down (same as reconstruction world coords).
    - OpenGlLegacy: CloudCompare native pose with local -Z forward.
    Legacy display flip (x,-y,-z) helpers below are only for old OpenGL display
    paths — not for VTK DB rendering. Web .splat export keeps its own flip in
    core/AICore/src/gaussian/splat.h (antimatter15 viewer convention). */
namespace ecvCameraSensorDisplay {

constexpr float kBaseCameraWidth = 1024.0f;

//! Apply legacy display Y/Z flip to a row-major 4x4 cam2world matrix.
CV_DB_LIB_API void ApplyDisplayCoordToCam2world(const float* srcRowMajor4x4,
                                                float* dstRowMajor4x4);

//! Row-major 4x4 COLMAP/OpenCV cam2world -> ccCameraSensor (VtkColmap world).
CV_DB_LIB_API ccGLMatrix
RowMajorCam2worldToVtkCameraSensorMatrix(const float* rowMajorCam2world4x4);

//! Row-major COLMAP cam2world -> display world (x,-y,-z) for VTK scenes whose
//! geometry uses ColmapWorldToDisplay (e.g. qFreeSplatter gaussians).
CV_DB_LIB_API ccGLMatrix
ColmapCam2WorldRowMajorToDisplayWorldMatrix(const float* rowMajorCam2world);

//! Row-major cam2world -> ccGLMatrix with legacy display Y/Z flip.
CV_DB_LIB_API ccGLMatrix
Cam2WorldRowMajorToDisplayMatrix(const float* rowMajorCam2world);

//! Local OpenGL ccCameraSensor geometry (+/-Z frustum) -> VTK/COLMAP camera
//! frame (+Z forward). Right-multiply before applying pose to VTK actors.
CV_DB_LIB_API ccGLMatrix OpenGlLocalFrameFlipZ();

//! Local OpenGL ccCameraSensor geometry (+/-Z frustum) -> VTK/COLMAP camera
//! frame (+Z forward). Only needed for legacy OpenGlLegacy pose misuse.
[[deprecated(
        "Use ccCameraSensor::PoseFrame::VtkColmap local geometry instead")]]
CV_DB_LIB_API ccGLMatrix
OpenGlSensorGeometryToVtkWorldTransform(const ccGLMatrix& vtkCam2World);

//! COLMAP InverseProjectionMatrix -> ccGLMatrix (COLMAP / VTK world coords).
CV_DB_LIB_API ccGLMatrix
ColmapInverseProjectionToVtkCameraSensorMatrix(const float* invProj3x4RowMajor);

//! @deprecated Use ColmapInverseProjectionToVtkCameraSensorMatrix.
CV_DB_LIB_API ccGLMatrix
ColmapInverseProjectionToCameraSensorMatrix(const float* invProj3x4RowMajor);

//! Raw COLMAP cam2world without pose-frame conversion.
CV_DB_LIB_API ccGLMatrix
ColmapInverseProjectionToGLMatrix(const float* invProj3x4RowMajor);

//! COLMAP InverseProjectionMatrix -> ccGLMatrix with legacy display Y/Z flip.
CV_DB_LIB_API ccGLMatrix
ColmapInverseProjectionToDisplayMatrix(const float* invProj3x4RowMajor);

//! COLMAP world XYZ -> legacy CloudViewer display coordinates (Y/Z flip).
CV_DB_LIB_API void ColmapWorldToDisplay(
        double x, double y, double z, double& outX, double& outY, double& outZ);

CV_DB_LIB_API ccGLMatrix
ColmapPoseGLMatrixToDisplayMatrix(const ccGLMatrix& colmapPose);

//! ccCameraSensor cam2world -> matrix for SetupProjectiveViewport / vtkCamera.
CV_DB_LIB_API ccGLMatrixd Cam2WorldToViewportCameraMatrix(
        const ccGLMatrixd& cam2world,
        ccCameraSensor::PoseFrame poseFrame =
                ccCameraSensor::PoseFrame::OpenGlLegacy);

//! Vertical field of view (radians) from focal length in pixels.
CV_DB_LIB_API float ComputeVerticalFovRad(float focalPx, int arrayHeight);

//! Display frustum focal length (mm) from image extent in world units.
CV_DB_LIB_API float ComputeFrustumDisplayFocalMmFromExtent(
        float imageDisplaySize,
        int imageWidth,
        int imageHeight,
        float cameraExtentWorld);

//! Display frustum focal length (mm) from focal length in pixels.
CV_DB_LIB_API float ComputeFrustumDisplayFocalMm(float imageDisplaySize,
                                                 int imageWidth,
                                                 int imageHeight,
                                                 float focalPx);

}  // namespace ecvCameraSensorDisplay

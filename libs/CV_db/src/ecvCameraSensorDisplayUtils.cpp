// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvCameraSensorDisplayUtils.h"

#include <ecvCameraSensor.h>

#include <algorithm>
#include <cmath>

namespace ecvCameraSensorDisplay {

void ApplyDisplayCoordToCam2world(const float* src, float* dst) {
    const float r00 = src[0];
    const float r01 = src[1];
    const float r02 = src[2];
    const float r10 = src[4];
    const float r11 = src[5];
    const float r12 = src[6];
    const float r20 = src[8];
    const float r21 = src[9];
    const float r22 = src[10];
    dst[0] = r00;
    dst[1] = -r01;
    dst[2] = -r02;
    dst[4] = -r10;
    dst[5] = r11;
    dst[6] = -r12;
    dst[8] = -r20;
    dst[9] = -r21;
    dst[10] = r22;
    dst[3] = src[3];
    dst[7] = -src[7];
    dst[11] = -src[11];
    dst[12] = 0.0f;
    dst[13] = 0.0f;
    dst[14] = 0.0f;
    dst[15] = 1.0f;
}

ccGLMatrix RowMajorCam2worldToGLMatrix(const float* rowMajorCam2world) {
    float colMajor[16];
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            colMajor[c * 4 + r] = rowMajorCam2world[r * 4 + c];
        }
    }
    return ccGLMatrix(colMajor);
}

ccGLMatrix RowMajorCam2worldToVtkCameraSensorMatrix(
        const float* rowMajorCam2world4x4) {
    return RowMajorCam2worldToGLMatrix(rowMajorCam2world4x4);
}

ccGLMatrix ColmapCam2WorldRowMajorToDisplayWorldMatrix(
        const float* rowMajorCam2world) {
    return Cam2WorldRowMajorToDisplayMatrix(rowMajorCam2world);
}

ccGLMatrix Cam2WorldRowMajorToDisplayMatrix(const float* rowMajorCam2world) {
    float displayM[16];
    ApplyDisplayCoordToCam2world(rowMajorCam2world, displayM);
    float colMajor[16];
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            colMajor[c * 4 + r] = displayM[r * 4 + c];
        }
    }
    return ccGLMatrix(colMajor);
}

ccGLMatrix OpenGlLocalFrameFlipZ() {
    ccGLMatrix flipZ;
    flipZ.toIdentity();
    flipZ.data()[10] = -1.0f;
    return flipZ;
}

ccGLMatrix OpenGlSensorGeometryToVtkWorldTransform(
        const ccGLMatrix& vtkCam2World) {
    return vtkCam2World * OpenGlLocalFrameFlipZ();
}

ccGLMatrix ColmapInverseProjectionToVtkCameraSensorMatrix(
        const float* invProj3x4RowMajor) {
    return ColmapInverseProjectionToGLMatrix(invProj3x4RowMajor);
}

ccGLMatrix ColmapInverseProjectionToCameraSensorMatrix(
        const float* invProj3x4RowMajor) {
    return ColmapInverseProjectionToVtkCameraSensorMatrix(invProj3x4RowMajor);
}

ccGLMatrixd Cam2WorldToViewportCameraMatrix(
        const ccGLMatrixd& cam2world, ccCameraSensor::PoseFrame poseFrame) {
    ccGLMatrixd rotOnly = cam2world;
    rotOnly.clearTranslation();

    CCVector3d forward;
    CCVector3d up;
    if (poseFrame == ccCameraSensor::PoseFrame::VtkColmap) {
        // COLMAP / VTK camera: +Z forward, Y down in local frame.
        forward = rotOnly.getColumnAsVec3D(2);
        up = -rotOnly.getColumnAsVec3D(1);
    } else {
        // CloudCompare legacy ccCameraSensor: -Z forward in local frame.
        forward = -rotOnly.getColumnAsVec3D(2);
        up = rotOnly.getColumnAsVec3D(1);
    }
    if (forward.norm2() < 1e-24 || up.norm2() < 1e-24) {
        return cam2world;
    }
    forward.normalize();
    up.normalize();

    ccGLMatrixd viewMat = ccGLMatrixd::FromViewDirAndUpDir(forward, up);
    ccGLMatrixd viewportCamera = viewMat;
    viewportCamera.invert();
    viewportCamera.setTranslation(cam2world.getTranslationAsVec3D());
    return viewportCamera;
}

ccGLMatrix ColmapInverseProjectionToGLMatrix(const float* invProj3x4RowMajor) {
    float cam2world[16] = {};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            cam2world[r * 4 + c] = invProj3x4RowMajor[r * 4 + c];
        }
    }
    cam2world[12] = 0.0f;
    cam2world[13] = 0.0f;
    cam2world[14] = 0.0f;
    cam2world[15] = 1.0f;
    return RowMajorCam2worldToGLMatrix(cam2world);
}

void ColmapWorldToDisplay(double x,
                          double y,
                          double z,
                          double& outX,
                          double& outY,
                          double& outZ) {
    outX = x;
    outY = -y;
    outZ = -z;
}

ccGLMatrix ColmapPoseGLMatrixToDisplayMatrix(const ccGLMatrix& colmapPose) {
    float rowMajor[16];
    const float* data = colmapPose.data();
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            rowMajor[r * 4 + c] = data[c * 4 + r];
        }
    }
    return Cam2WorldRowMajorToDisplayMatrix(rowMajor);
}

ccGLMatrix ColmapInverseProjectionToDisplayMatrix(
        const float* invProj3x4RowMajor) {
    float cam2world[16] = {};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            cam2world[r * 4 + c] = invProj3x4RowMajor[r * 4 + c];
        }
    }
    cam2world[12] = 0.0f;
    cam2world[13] = 0.0f;
    cam2world[14] = 0.0f;
    cam2world[15] = 1.0f;
    return Cam2WorldRowMajorToDisplayMatrix(cam2world);
}

float ComputeVerticalFovRad(float focalPx, int arrayHeight) {
    if (arrayHeight <= 0) {
        return ccCameraSensor::ComputeFovRadFromFocalPix(500.0f, 1024);
    }
    if (focalPx <= 0.0f) {
        return ccCameraSensor::ComputeFovRadFromFocalPix(500.0f, arrayHeight);
    }
    return ccCameraSensor::ComputeFovRadFromFocalPix(focalPx, arrayHeight);
}

float ComputeFrustumDisplayFocalMmFromExtent(float imageDisplaySize,
                                             int imageWidth,
                                             int imageHeight,
                                             float cameraExtentWorld) {
    if (imageWidth < 1 || imageHeight < 1) {
        return imageDisplaySize;
    }
    const float image_width = imageDisplaySize *
                              static_cast<float>(imageWidth) / kBaseCameraWidth;
    const float image_height = image_width * static_cast<float>(imageHeight) /
                               static_cast<float>(imageWidth);
    const float image_extent = std::max(image_width, image_height);
    if (cameraExtentWorld <= 1e-6f) {
        return image_extent;
    }
    return 2.0f * image_extent / cameraExtentWorld;
}

float ComputeFrustumDisplayFocalMm(float imageDisplaySize,
                                   int imageWidth,
                                   int imageHeight,
                                   float focalPx) {
    const float camera_extent =
            static_cast<float>(std::max(imageWidth, imageHeight));
    const float focal = focalPx > 0.0f ? focalPx : 500.0f;
    const float camera_extent_world = camera_extent / focal;
    return ComputeFrustumDisplayFocalMmFromExtent(
            imageDisplaySize, imageWidth, imageHeight, camera_extent_world);
}

}  // namespace ecvCameraSensorDisplay

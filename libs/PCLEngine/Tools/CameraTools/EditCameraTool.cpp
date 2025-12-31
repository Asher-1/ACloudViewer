// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "EditCameraTool.h"

// Local
#include "PclUtils/PCLVis.h"

// CV_CORE_LIB
#include <CVLog.h>

// ECV_DB_LIB
#include <ecvDisplayTools.h>

// VTK / ParaView Server Manager includes.
#include <vtkCamera.h>
#include <vtkCollection.h>
#include <vtkMath.h>
#include <vtkSmartPointer.h>
#include <vtkTransform.h>

// Qt includes.
#include <QDebug>
#include <QPointer>
#include <QSettings>
#include <QString>
#include <QToolButton>

// STL
#include <sstream>
#include <string>

namespace {
void RotateElevation(vtkCamera* camera, double angle) {
    vtkNew<vtkTransform> transform;

    double scale = vtkMath::Norm(camera->GetPosition());
    if (scale <= 0.0) {
        scale = vtkMath::Norm(camera->GetFocalPoint());
        if (scale <= 0.0) {
            scale = 1.0;
        }
    }
    double* temp = camera->GetFocalPoint();
    camera->SetFocalPoint(temp[0] / scale, temp[1] / scale, temp[2] / scale);
    temp = camera->GetPosition();
    camera->SetPosition(temp[0] / scale, temp[1] / scale, temp[2] / scale);

    double v2[3];
    // translate to center
    // we rotate around 0,0,0 rather than the center of rotation
    transform->Identity();

    // elevation
    camera->OrthogonalizeViewUp();
    double* viewUp = camera->GetViewUp();
    vtkMath::Cross(camera->GetDirectionOfProjection(), viewUp, v2);
    transform->RotateWXYZ(-angle, v2[0], v2[1], v2[2]);

    // translate back
    // we are already at 0,0,0

    camera->ApplyTransform(transform.GetPointer());
    camera->OrthogonalizeViewUp();

    // For rescale back.
    temp = camera->GetFocalPoint();
    camera->SetFocalPoint(temp[0] * scale, temp[1] * scale, temp[2] * scale);
    temp = camera->GetPosition();
    camera->SetPosition(temp[0] * scale, temp[1] * scale, temp[2] * scale);
}

};  // namespace

static vtkSmartPointer<vtkCamera> s_camera = nullptr;
static PclUtils::PCLVis* s_viewer = nullptr;

//-----------------------------------------------------------------------------
EditCameraTool::EditCameraTool(ecvGenericVisualizer3D* viewer)
    : ecvGenericCameraTool() {
    SetVisualizer(viewer);
    updateCameraParameters();
}

//-----------------------------------------------------------------------------
EditCameraTool::~EditCameraTool() {}

void EditCameraTool::SetVisualizer(ecvGenericVisualizer3D* viewer) {
    if (viewer) {
        s_viewer = reinterpret_cast<PclUtils::PCLVis*>(viewer);
        if (!s_viewer) {
            CVLog::Warning("[EditCameraTool::setVisualizer] viewer is Null!");
        }
    } else {
        CVLog::Warning("[EditCameraTool::setVisualizer] viewer is Null!");
    }
}

void EditCameraTool::UpdateCameraInfo() {
    if (!s_viewer) {
        SetVisualizer(ecvDisplayTools::GetVisualizer3D());
    }

    s_camera = s_viewer->getVtkCamera();
    OldCameraParam = CurrentCameraParam;

    s_camera->GetViewUp(CurrentCameraParam.viewUp.u);
    s_camera->GetFocalPoint(CurrentCameraParam.focal.u);
    s_camera->GetPosition(CurrentCameraParam.position.u);
    s_camera->GetClippingRange(CurrentCameraParam.clippRange.u);
    CurrentCameraParam.viewAngle = s_camera->GetViewAngle();
    CurrentCameraParam.eyeAngle = s_camera->GetEyeAngle();
    s_viewer->getCenterOfRotation(CurrentCameraParam.pivot.u);
    CurrentCameraParam.rotationFactor = s_viewer->getRotationFactor();
}

void EditCameraTool::UpdateCamera() {
    if (!s_viewer) {
        SetVisualizer(ecvDisplayTools::GetVisualizer3D());
    }

    s_camera = s_viewer->getVtkCamera();

    s_camera->SetViewUp(CurrentCameraParam.viewUp.u);
    s_camera->SetFocalPoint(CurrentCameraParam.focal.u);
    s_camera->SetPosition(CurrentCameraParam.position.u);
    s_camera->SetClippingRange(CurrentCameraParam.clippRange.u);

    s_camera->SetViewAngle(CurrentCameraParam.viewAngle);
    s_camera->SetEyeAngle(CurrentCameraParam.eyeAngle);
    s_viewer->setCenterOfRotation(CurrentCameraParam.pivot.u);
    s_viewer->setRotationFactor(CurrentCameraParam.rotationFactor);

    s_viewer->getCurrentRenderer()->SetActiveCamera(s_camera);
    s_viewer->UpdateScreen();
}

//-----------------------------------------------------------------------------
void EditCameraTool::resetViewDirection(double look_x,
                                        double look_y,
                                        double look_z,
                                        double up_x,
                                        double up_y,
                                        double up_z) {
    if (s_viewer) {
        s_viewer->setCameraPosition(0.0, 0.0, 0.0, look_x, look_y, look_z, up_x,
                                    up_y, up_z);
        s_viewer->synchronizeGeometryBounds();
        double bounds[6];
        s_viewer->getVisibleGeometryBounds().GetBounds(bounds);
        s_viewer->resetCamera(bounds);
        ecvDisplayTools::UpdateScreen();
    }
}

void EditCameraTool::updateCamera() { UpdateCamera(); }

void EditCameraTool::updateCameraParameters() { UpdateCameraInfo(); }

//-----------------------------------------------------------------------------
void EditCameraTool::adjustCamera(CameraAdjustmentType enType, double value) {
    if (s_viewer && s_camera) {
        switch (enType) {
            case ecvGenericCameraTool::Roll:
                s_camera->Roll(value);
                break;
            case ecvGenericCameraTool::Elevation:
                RotateElevation(s_camera, value);
                break;
            case ecvGenericCameraTool::Azimuth:
                s_camera->Azimuth(value);
                break;
            case ecvGenericCameraTool::Zoom: {
                if (s_camera->GetParallelProjection()) {
                    s_camera->SetParallelScale(s_camera->GetParallelScale() /
                                               value);
                } else {
                    s_camera->Dolly(value);
                }
            }  // if (EditCameraTool::Zoom)
            break;
            default:
                break;
        }

        s_viewer->UpdateScreen();
    }
}

//-----------------------------------------------------------------------------
void EditCameraTool::saveCameraConfiguration(const std::string& file) {
    if (s_viewer) {
        s_viewer->saveCameraParameters(file);
    }
}

//-----------------------------------------------------------------------------
void EditCameraTool::loadCameraConfiguration(const std::string& file) {
    if (s_viewer) {
        s_viewer->loadCameraParameters(file);
        updateCameraParameters();
    }
}
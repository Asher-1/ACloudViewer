// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "EditCameraTool.h"

// Local
#include "Visualization/VtkVis.h"

// CV_CORE_LIB
#include <CVLog.h>

// CV_DB_LIB
#include <ecvDisplayTools.h>

// VTK / ParaView Server Manager includes.
#include <vtkCamera.h>
#include <vtkCollection.h>
#include <vtkMath.h>
#include <vtkRenderer.h>
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

static EditCameraTool* s_activeTool = nullptr;

//-----------------------------------------------------------------------------
EditCameraTool::EditCameraTool(ecvGenericVisualizer3D* viewer)
    : ecvGenericCameraTool() {
    SetVisualizer(viewer);
    s_activeTool = this;
    updateCameraParameters();
}

//-----------------------------------------------------------------------------
EditCameraTool::~EditCameraTool() {
    if (s_activeTool == this) s_activeTool = nullptr;
}

EditCameraTool* EditCameraTool::ActiveTool() { return s_activeTool; }

void EditCameraTool::SetVisualizer(ecvGenericVisualizer3D* viewer) {
    auto* tool = s_activeTool;
    if (!tool) return;
    if (viewer) {
        tool->m_viewer = reinterpret_cast<Visualization::VtkVis*>(viewer);
        if (!tool->m_viewer) {
            CVLog::Warning("[EditCameraTool::setVisualizer] viewer is Null!");
        }
    } else {
        CVLog::Warning("[EditCameraTool::setVisualizer] viewer is Null!");
    }
}

void EditCameraTool::UpdateCameraInfo() {
    auto* tool = s_activeTool;
    if (!tool) return;
    if (!tool->m_viewer) {
        SetVisualizer(ecvDisplayTools::GetVisualizer3D());
    }
    if (!tool->m_viewer) return;

    tool->m_camera = tool->m_viewer->getVtkCamera();
    tool->OldCameraParam = tool->CurrentCameraParam;

    tool->m_camera->GetViewUp(tool->CurrentCameraParam.viewUp.u);
    tool->m_camera->GetFocalPoint(tool->CurrentCameraParam.focal.u);
    tool->m_camera->GetPosition(tool->CurrentCameraParam.position.u);
    tool->m_camera->GetClippingRange(tool->CurrentCameraParam.clippRange.u);
    tool->CurrentCameraParam.viewAngle = tool->m_camera->GetViewAngle();
    tool->CurrentCameraParam.eyeAngle = tool->m_camera->GetEyeAngle();
    tool->m_viewer->getCenterOfRotation(tool->CurrentCameraParam.pivot.u);
    tool->CurrentCameraParam.rotationFactor = tool->m_viewer->getRotationFactor();
}

void EditCameraTool::UpdateCamera() {
    auto* tool = s_activeTool;
    if (!tool) return;
    if (!tool->m_viewer) {
        SetVisualizer(ecvDisplayTools::GetVisualizer3D());
    }
    if (!tool->m_viewer) return;

    tool->m_camera = tool->m_viewer->getVtkCamera();

    tool->m_camera->SetViewUp(tool->CurrentCameraParam.viewUp.u);
    tool->m_camera->SetFocalPoint(tool->CurrentCameraParam.focal.u);
    tool->m_camera->SetPosition(tool->CurrentCameraParam.position.u);
    tool->m_camera->SetClippingRange(tool->CurrentCameraParam.clippRange.u);

    tool->m_camera->SetViewAngle(tool->CurrentCameraParam.viewAngle);
    tool->m_camera->SetEyeAngle(tool->CurrentCameraParam.eyeAngle);
    tool->m_viewer->setCenterOfRotation(tool->CurrentCameraParam.pivot.u);
    tool->m_viewer->setRotationFactor(tool->CurrentCameraParam.rotationFactor);

    tool->m_viewer->getCurrentRenderer()->SetActiveCamera(tool->m_camera);
    tool->m_viewer->UpdateScreen();
}

//-----------------------------------------------------------------------------
void EditCameraTool::resetViewDirection(double look_x,
                                        double look_y,
                                        double look_z,
                                        double up_x,
                                        double up_y,
                                        double up_z) {
    if (m_viewer) {
        m_viewer->setCameraPosition(0.0, 0.0, 0.0, look_x, look_y, look_z, up_x,
                                    up_y, up_z);
        m_viewer->synchronizeGeometryBounds();
        double bounds[6];
        m_viewer->getVisibleGeometryBounds().GetBounds(bounds);
        m_viewer->resetCamera(bounds);
        ecvDisplayTools::UpdateScreen();
    }
}

void EditCameraTool::updateCamera() { UpdateCamera(); }

void EditCameraTool::updateCameraParameters() { UpdateCameraInfo(); }

//-----------------------------------------------------------------------------
void EditCameraTool::adjustCamera(CameraAdjustmentType enType, double value) {
    if (m_viewer && m_camera) {
        switch (enType) {
            case ecvGenericCameraTool::Roll:
                m_camera->Roll(value);
                break;
            case ecvGenericCameraTool::Elevation:
                RotateElevation(m_camera, value);
                break;
            case ecvGenericCameraTool::Azimuth:
                m_camera->Azimuth(value);
                break;
            case ecvGenericCameraTool::Zoom: {
                if (m_camera->GetParallelProjection()) {
                    m_camera->SetParallelScale(m_camera->GetParallelScale() /
                                               value);
                } else {
                    m_camera->Dolly(value);
                }
            }
            break;
            default:
                break;
        }

        m_viewer->UpdateScreen();
    }
}

//-----------------------------------------------------------------------------
void EditCameraTool::saveCameraConfiguration(const std::string& file) {
    if (m_viewer) {
        m_viewer->saveCameraParameters(file);
    }
}

//-----------------------------------------------------------------------------
void EditCameraTool::loadCameraConfiguration(const std::string& file) {
    if (m_viewer) {
        m_viewer->loadCameraParameters(file);
        updateCameraParameters();
    }
}
// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VtkCameraLink.h"

#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>

#include "VtkVis.h"

namespace Visualization {

VtkCameraLink& VtkCameraLink::instance() {
    static VtkCameraLink s_instance;
    return s_instance;
}

VtkCameraLink::~VtkCameraLink() { clear(); }

void VtkCameraLink::setEnabled(bool enabled) {
    if (m_enabled == enabled) return;
    m_enabled = enabled;

    if (m_enabled) {
        installObservers();
    } else {
        removeObservers();
    }
}

void VtkCameraLink::addView(VtkVis* vis) {
    if (!vis) return;

    for (const auto& lv : m_views) {
        if (lv.vis == vis) return;
    }

    LinkedView lv;
    lv.vis = vis;
    m_views.push_back(lv);

    if (m_enabled) {
        auto& back = m_views.back();
        auto cb = vtkSmartPointer<vtkCallbackCommand>::New();
        cb->SetCallback(OnRenderEnd);
        cb->SetClientData(this);
        back.observer = cb;

        auto rw = vis->getRenderWindow();
        if (rw) {
            back.observerTag = rw->AddObserver(vtkCommand::EndEvent, cb);
        }
    }
}

void VtkCameraLink::removeView(VtkVis* vis) {
    for (auto it = m_views.begin(); it != m_views.end(); ++it) {
        if (it->vis == vis) {
            if (it->observerTag != 0 && vis) {
                auto rw = vis->getRenderWindow();
                if (rw) {
                    rw->RemoveObserver(it->observerTag);
                }
            }
            m_views.erase(it);
            return;
        }
    }
}

void VtkCameraLink::clear() {
    removeObservers();
    m_views.clear();
}

void VtkCameraLink::installObservers() {
    for (auto& lv : m_views) {
        if (lv.observerTag != 0 || !lv.vis) continue;

        auto cb = vtkSmartPointer<vtkCallbackCommand>::New();
        cb->SetCallback(OnRenderEnd);
        cb->SetClientData(this);
        lv.observer = cb;

        auto rw = lv.vis->getRenderWindow();
        if (rw) {
            lv.observerTag = rw->AddObserver(vtkCommand::EndEvent, cb);
        }
    }
}

void VtkCameraLink::removeObservers() {
    for (auto& lv : m_views) {
        if (lv.observerTag != 0 && lv.vis) {
            auto rw = lv.vis->getRenderWindow();
            if (rw) {
                rw->RemoveObserver(lv.observerTag);
            }
        }
        lv.observerTag = 0;
        lv.observer = nullptr;
    }
}

// static
void VtkCameraLink::OnRenderEnd(vtkObject* caller,
                                unsigned long /*eid*/,
                                void* clientData,
                                void* /*callData*/) {
    auto* self = static_cast<VtkCameraLink*>(clientData);
    if (!self || !self->m_enabled || self->m_updating) return;

    auto* rw = vtkRenderWindow::SafeDownCast(caller);
    if (!rw) return;

    VtkVis* source = nullptr;
    for (const auto& lv : self->m_views) {
        if (lv.vis && lv.vis->getRenderWindow().GetPointer() == rw) {
            source = lv.vis;
            break;
        }
    }

    if (source) {
        self->syncCamerasFrom(source);
    }
}

void VtkCameraLink::syncCamerasFrom(VtkVis* source) {
    if (!source || m_views.size() < 2) return;

    m_updating = true;

    auto srcCam = source->getVtkCamera();
    if (!srcCam) {
        m_updating = false;
        return;
    }

    double pos[3], foc[3], up[3], clip[2];
    srcCam->GetPosition(pos);
    srcCam->GetFocalPoint(foc);
    srcCam->GetViewUp(up);
    srcCam->GetClippingRange(clip);
    double viewAngle = srcCam->GetViewAngle();
    double parallelScale = srcCam->GetParallelScale();
    int parallelProj = srcCam->GetParallelProjection();

    double cor[3];
    source->getCenterOfRotation(cor);

    for (auto& lv : m_views) {
        if (lv.vis == source || !lv.vis) continue;

        auto dstCam = lv.vis->getVtkCamera();
        if (!dstCam) continue;

        dstCam->SetPosition(pos);
        dstCam->SetFocalPoint(foc);
        dstCam->SetViewUp(up);
        dstCam->SetClippingRange(clip);
        dstCam->SetViewAngle(viewAngle);
        dstCam->SetParallelScale(parallelScale);
        dstCam->SetParallelProjection(parallelProj);

        lv.vis->setCenterOfRotation(cor);

        auto targetRW = lv.vis->getRenderWindow();
        if (targetRW) {
            targetRW->Render();
        }
    }

    m_updating = false;
}

}  // namespace Visualization

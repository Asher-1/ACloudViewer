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
#include <vtkRendererCollection.h>

#include <algorithm>
#include <sstream>

#include "VtkVis.h"

namespace Visualization {

VtkCameraLink& VtkCameraLink::instance() {
    static VtkCameraLink s_instance;
    return s_instance;
}

VtkCameraLink::~VtkCameraLink() { clear(); }

void VtkCameraLink::registerView(VtkVis* vis) {
    if (!vis) return;
    for (auto* v : m_registeredViews) {
        if (v == vis) return;
    }
    m_registeredViews.push_back(vis);
}

void VtkCameraLink::unregisterView(VtkVis* vis) {
    if (!vis) return;
    removeLinksForView(vis);
    m_registeredViews.erase(
            std::remove(m_registeredViews.begin(), m_registeredViews.end(), vis),
            m_registeredViews.end());
}

std::string VtkCameraLink::addLink(const std::string& name,
                                   VtkVis* viewA, VtkVis* viewB) {
    if (!viewA || !viewB || viewA == viewB) return {};

    for (const auto& lp : m_links) {
        if (lp.name == name) return name;
        if ((lp.viewA == viewA && lp.viewB == viewB) ||
            (lp.viewA == viewB && lp.viewB == viewA)) {
            return lp.name;
        }
    }

    CameraLinkPair lp;
    lp.name = name;
    lp.viewA = viewA;
    lp.viewB = viewB;
    m_links.push_back(std::move(lp));
    installLinkObservers(static_cast<int>(m_links.size()) - 1);
    return name;
}

std::string VtkCameraLink::addLink(VtkVis* viewA, VtkVis* viewB) {
    std::ostringstream oss;
    oss << "CameraLink" << (++m_linkCounter);
    return addLink(oss.str(), viewA, viewB);
}

void VtkCameraLink::removeLink(const std::string& name) {
    for (int i = 0; i < static_cast<int>(m_links.size()); ++i) {
        if (m_links[i].name == name) {
            removeLinkObservers(i);
            m_links.erase(m_links.begin() + i);
            return;
        }
    }
}

void VtkCameraLink::removeLinksForView(VtkVis* vis) {
    if (!vis) return;
    for (int i = static_cast<int>(m_links.size()) - 1; i >= 0; --i) {
        if (m_links[i].viewA == vis || m_links[i].viewB == vis) {
            removeLinkObservers(i);
            m_links.erase(m_links.begin() + i);
        }
    }
}

std::vector<std::string> VtkCameraLink::linkNames() const {
    std::vector<std::string> names;
    names.reserve(m_links.size());
    for (const auto& lp : m_links) {
        names.push_back(lp.name);
    }
    return names;
}

bool VtkCameraLink::isLinked(VtkVis* vis) const {
    for (const auto& lp : m_links) {
        if (lp.viewA == vis || lp.viewB == vis) return true;
    }
    return false;
}

void VtkCameraLink::linkAll() {
    for (size_t i = 0; i < m_registeredViews.size(); ++i) {
        for (size_t j = i + 1; j < m_registeredViews.size(); ++j) {
            addLink(m_registeredViews[i], m_registeredViews[j]);
        }
    }
}

void VtkCameraLink::unlinkAll() {
    for (int i = static_cast<int>(m_links.size()) - 1; i >= 0; --i) {
        removeLinkObservers(i);
    }
    m_links.clear();
}

void VtkCameraLink::setEnabled(bool enabled) {
    if (enabled) {
        linkAll();
    } else {
        unlinkAll();
    }
}

void VtkCameraLink::clear() {
    unlinkAll();
    m_registeredViews.clear();
}

// --- Observer installation / removal ---

static vtkRenderer* firstRenderer(VtkVis* vis) {
    auto rc = vis ? vis->getRendererCollection() : nullptr;
    return rc ? rc->GetFirstRenderer() : nullptr;
}

void VtkCameraLink::installLinkObservers(int idx) {
    auto& lp = m_links[idx];

    auto makeCb = [](void (*fn)(vtkObject*, unsigned long, void*, void*),
                     void* data) {
        auto cb = vtkSmartPointer<vtkCallbackCommand>::New();
        cb->SetCallback(fn);
        cb->SetClientData(data);
        return cb;
    };

    auto installForView =
            [&](VtkVis* vis, ViewObservers& obs) {
                obs.renderCb = makeCb(OnRenderEnd, this);
                obs.resetCb = makeCb(OnResetCamera, this);

                auto rw = vis->getRenderWindow();
                if (rw) {
                    obs.renderTag =
                            rw->AddObserver(vtkCommand::EndEvent,
                                            obs.renderCb);
                }
                auto* ren = firstRenderer(vis);
                if (ren) {
                    obs.resetTag =
                            ren->AddObserver(vtkCommand::ResetCameraEvent,
                                             obs.resetCb);
                }
            };

    if (lp.viewA) installForView(lp.viewA, lp.obsA);
    if (lp.viewB) installForView(lp.viewB, lp.obsB);
}

void VtkCameraLink::removeLinkObservers(int idx) {
    auto& lp = m_links[idx];

    auto removeForView =
            [](VtkVis* vis, ViewObservers& obs) {
                if (!vis) return;
                if (obs.renderTag != 0) {
                    auto rw = vis->getRenderWindow();
                    if (rw) rw->RemoveObserver(obs.renderTag);
                    obs.renderTag = 0;
                }
                if (obs.resetTag != 0) {
                    auto* ren = firstRenderer(vis);
                    if (ren) ren->RemoveObserver(obs.resetTag);
                    obs.resetTag = 0;
                }
                obs.renderCb = nullptr;
                obs.resetCb = nullptr;
            };

    removeForView(lp.viewA, lp.obsA);
    removeForView(lp.viewB, lp.obsB);
}

// --- Callbacks ---

void VtkCameraLink::OnRenderEnd(vtkObject* caller, unsigned long /*eid*/,
                                void* clientData, void* /*callData*/) {
    auto* self = static_cast<VtkCameraLink*>(clientData);
    if (!self || self->m_updating) return;

    auto* rw = vtkRenderWindow::SafeDownCast(caller);
    if (!rw) return;

    for (const auto& lp : self->m_links) {
        if (lp.viewA && lp.viewA->getRenderWindow().GetPointer() == rw) {
            self->syncCamerasFrom(lp.viewA, lp.viewB);
        } else if (lp.viewB &&
                   lp.viewB->getRenderWindow().GetPointer() == rw) {
            self->syncCamerasFrom(lp.viewB, lp.viewA);
        }
    }
}

void VtkCameraLink::OnResetCamera(vtkObject* caller, unsigned long /*eid*/,
                                  void* clientData, void* /*callData*/) {
    auto* self = static_cast<VtkCameraLink*>(clientData);
    if (!self || self->m_updating) return;

    auto* ren = vtkRenderer::SafeDownCast(caller);
    if (!ren) return;

    for (const auto& lp : self->m_links) {
        auto* renA = firstRenderer(lp.viewA);
        auto* renB = firstRenderer(lp.viewB);
        if (renA == ren && lp.viewB) {
            self->syncCamerasFrom(lp.viewA, lp.viewB);
        } else if (renB == ren && lp.viewA) {
            self->syncCamerasFrom(lp.viewB, lp.viewA);
        }
    }
}

void VtkCameraLink::syncCamerasFrom(VtkVis* source, VtkVis* target) {
    if (!source || !target) return;

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
    double focalDisk = srcCam->GetFocalDisk();
    double focalDistance = srcCam->GetFocalDistance();

    double cor[3];
    source->getCenterOfRotation(cor);

    auto dstCam = target->getVtkCamera();
    if (dstCam) {
        dstCam->SetPosition(pos);
        dstCam->SetFocalPoint(foc);
        dstCam->SetViewUp(up);
        dstCam->SetClippingRange(clip);
        dstCam->SetViewAngle(viewAngle);
        dstCam->SetParallelScale(parallelScale);
        dstCam->SetParallelProjection(parallelProj);
        dstCam->SetFocalDisk(focalDisk);
        dstCam->SetFocalDistance(focalDistance);
    }

    target->setCenterOfRotation(cor);

    auto targetRW = target->getRenderWindow();
    if (targetRW) {
        targetRW->Render();
    }

    m_updating = false;
}

}  // namespace Visualization

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vtkSmartPointer.h>

#include <vector>

#include "qVTK.h"

class vtkCallbackCommand;
class vtkObject;

namespace Visualization {
class VtkVis;

/// Synchronizes cameras across multiple VtkVis views.
///
/// Inspired by ParaView's vtkSMCameraLink: when enabled, any camera change
/// in one view is automatically propagated to all other linked views.
///
/// The synchronization works by observing each view's vtkRenderWindow EndEvent.
/// An internal re-entry guard (m_updating) prevents infinite render loops.
class QVTK_ENGINE_LIB_API VtkCameraLink {
public:
    static VtkCameraLink& instance();

    void setEnabled(bool enabled);
    bool isEnabled() const { return m_enabled; }

    void addView(VtkVis* vis);
    void removeView(VtkVis* vis);
    void clear();

    void setSyncInteractiveRenders(bool sync) { m_syncInteractive = sync; }
    bool syncInteractiveRenders() const { return m_syncInteractive; }

private:
    VtkCameraLink() = default;
    ~VtkCameraLink();

    VtkCameraLink(const VtkCameraLink&) = delete;
    VtkCameraLink& operator=(const VtkCameraLink&) = delete;

    static void OnRenderEnd(vtkObject* caller,
                            unsigned long eid,
                            void* clientData,
                            void* callData);

    void syncCamerasFrom(VtkVis* source);
    void installObservers();
    void removeObservers();

    bool m_enabled = false;
    bool m_updating = false;
    bool m_syncInteractive = true;

    struct LinkedView {
        VtkVis* vis = nullptr;
        vtkSmartPointer<vtkCallbackCommand> observer;
        unsigned long observerTag = 0;
    };
    std::vector<LinkedView> m_views;
};

}  // namespace Visualization

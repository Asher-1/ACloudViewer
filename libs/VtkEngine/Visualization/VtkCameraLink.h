// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vtkSmartPointer.h>

#include <string>
#include <vector>

#include "qVTK.h"

class vtkCallbackCommand;
class vtkGLView;
class vtkObject;

namespace Visualization {
class VtkVis;

/// Named pairwise camera link registry.
///
/// Aligned with ParaView's vtkSMCameraLink + vtkSMSessionProxyManager model:
///   - Each link has a unique name and connects exactly two render views.
///   - Camera properties are synced bidirectionally on EndEvent.
///   - ResetCameraEvent is propagated across linked views.
///   - Additional properties (FocalDistance, FocalDisk) are synced.
///
/// Views must be registered first (registerView). Links are created between
/// registered views via addLink(). The global linkAll()/unlinkAll() convenience
/// methods match the toolbar toggle behavior.
class QVTK_ENGINE_LIB_API VtkCameraLink {
public:
    static VtkCameraLink& instance();

    /// Register a view as available for linking. Does NOT create any links.
    void registerView(VtkVis* vis);
    /// Unregister a view and remove all links that reference it.
    void unregisterView(VtkVis* vis);

    /// Legacy compatibility wrappers.
    void addView(VtkVis* vis) { registerView(vis); }
    void removeView(VtkVis* vis) { unregisterView(vis); }

    /// Create a named pairwise camera link (ParaView style).
    /// @return The name assigned to the link.
    std::string addLink(const std::string& name, VtkVis* viewA, VtkVis* viewB);
    /// Create a link with an auto-generated name "CameraLink<N>".
    std::string addLink(VtkVis* viewA, VtkVis* viewB);
    /// Remove a specific named link.
    void removeLink(const std::string& name);
    /// Remove all links that reference the given view.
    void removeLinksForView(VtkVis* vis);
    /// Query link names.
    std::vector<std::string> linkNames() const;
    /// Check if a view has any active links.
    bool isLinked(VtkVis* vis) const;
    /// Total number of active links.
    int linkCount() const { return static_cast<int>(m_links.size()); }

    /// Convenience: link ALL registered views pairwise (toolbar toggle ON).
    void linkAll();
    /// Convenience: remove all links (toolbar toggle OFF).
    void unlinkAll();
    /// Legacy toggle wrapping linkAll()/unlinkAll().
    void setEnabled(bool enabled);
    bool isEnabled() const { return !m_links.empty(); }

    /// Remove all links AND unregister all views.
    void clear();

    /// Get the list of registered views (for UI: pick link target).
    const std::vector<VtkVis*>& registeredViews() const {
        return m_registeredViews;
    }

private:
    VtkCameraLink() = default;
    ~VtkCameraLink();

    VtkCameraLink(const VtkCameraLink&) = delete;
    VtkCameraLink& operator=(const VtkCameraLink&) = delete;

    static void OnRenderEnd(vtkObject* caller,
                            unsigned long eid,
                            void* clientData,
                            void* callData);
    static void OnResetCamera(vtkObject* caller,
                              unsigned long eid,
                              void* clientData,
                              void* callData);

    void syncCamerasFrom(VtkVis* source, VtkVis* target);
    void initialSyncFromActive();
    ::vtkGLView* findGLViewForVis(VtkVis* vis) const;
    void installLinkObservers(int linkIndex);
    void removeLinkObservers(int linkIndex);

    struct ViewObservers {
        vtkSmartPointer<vtkCallbackCommand> renderCb;
        vtkSmartPointer<vtkCallbackCommand> resetCb;
        unsigned long renderTag = 0;
        unsigned long resetTag = 0;
    };

    struct CameraLinkPair {
        std::string name;
        VtkVis* viewA = nullptr;
        VtkVis* viewB = nullptr;
        ViewObservers obsA;
        ViewObservers obsB;
    };

    bool m_updating = false;
    int m_linkCounter = 0;

    std::vector<VtkVis*> m_registeredViews;
    std::vector<CameraLinkPair> m_links;
};

}  // namespace Visualization

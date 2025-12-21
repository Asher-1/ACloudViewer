// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvZoomToBoxTool.h"

// LOCAL
#include "PclUtils/PCLVis.h"

// VTK
#include <vtkCamera.h>
#include <vtkCommand.h>
#include <vtkInteractorObserver.h>
#include <vtkInteractorStyle.h>
#include <vtkInteractorStyleRubberBandZoom.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

// QT
#include <QPixmap>

#include "Tools/CameraTools/zoom.xpm"

//-----------------------------------------------------------------------------
// Callback class for zoom completion
// Follows ParaView's pqRenderViewSelectionReaction pattern
class vtkZoomToBoxCallback : public vtkCommand {
public:
    static vtkZoomToBoxCallback* New() { return new vtkZoomToBoxCallback; }

    void SetTool(cvZoomToBoxTool* tool) { m_tool = tool; }

    void Execute(vtkObject* caller,
                 unsigned long eventId,
                 void* callData) override {
        // ParaView approach: simply emit completion signal on LeftButtonRelease
        // VTK's vtkInteractorStyleRubberBandZoom handles the drag detection
        // internally (it only zooms if StartPosition != EndPosition) When this
        // callback is called, Zoom() and Render() have already been executed
        if (eventId == vtkCommand::LeftButtonReleaseEvent && m_tool) {
            emit m_tool->zoomCompleted();
        }
    }

private:
    cvZoomToBoxTool* m_tool = nullptr;
};

//-----------------------------------------------------------------------------
cvZoomToBoxTool::cvZoomToBoxTool(QObject* parent)
    : QObject(parent),
      m_viewer(nullptr),
      m_interactor(nullptr),
      m_renderer(nullptr),
      m_enabled(false),
      m_zoomCursor(QCursor(QPixmap((const char**)zoom_xpm))),
      m_observerId(0) {
    // m_zoomStyle will be created in enable() to ensure clean state
    // This follows ParaView's pattern to avoid state retention between uses
}

//-----------------------------------------------------------------------------
cvZoomToBoxTool::~cvZoomToBoxTool() { disable(); }

//-----------------------------------------------------------------------------
void cvZoomToBoxTool::setVisualizer(ecvGenericVisualizer3D* viewer) {
    if (m_viewer == viewer) {
        return;
    }

    // Disable if currently enabled
    if (m_enabled) {
        disable();
    }

    m_viewer = viewer;

    // Get PCLVis instance for VTK-specific operations
    PclUtils::PCLVis* pclVis = reinterpret_cast<PclUtils::PCLVis*>(m_viewer);
    if (pclVis) {
        m_interactor = pclVis->getRenderWindowInteractor();
        m_renderer = pclVis->getCurrentRenderer();
    } else {
        m_interactor = nullptr;
        m_renderer = nullptr;
    }
}

//-----------------------------------------------------------------------------
void cvZoomToBoxTool::storeCurrentStyle() {
    if (m_interactor) {
        vtkInteractorObserver* style = m_interactor->GetInteractorStyle();
        m_previousStyle = vtkInteractorStyle::SafeDownCast(style);
    }
}

//-----------------------------------------------------------------------------
void cvZoomToBoxTool::restoreStyle() {
    if (m_interactor && m_previousStyle) {
        m_interactor->SetInteractorStyle(m_previousStyle);
        m_previousStyle = nullptr;
    }
}

//-----------------------------------------------------------------------------
void cvZoomToBoxTool::enable() {
    if (m_enabled || !m_viewer || !m_interactor) {
        return;
    }

    // Store current style
    storeCurrentStyle();

    // Create a NEW zoom style instance each time to avoid state retention
    m_zoomStyle = vtkSmartPointer<vtkInteractorStyleRubberBandZoom>::New();

    // Set up zoom style
    if (m_renderer) {
        m_zoomStyle->SetDefaultRenderer(m_renderer);
    }

    // Switch to zoom style FIRST
    m_interactor->SetInteractorStyle(m_zoomStyle);

    // Add callback for zoom completion
    // Following ParaView pattern: observe Interactor's LeftButtonReleaseEvent
    // NOT the InteractorStyle's event!
    // VTK's vtkInteractorStyleRubberBandZoom internally checks if drag occurred
    vtkSmartPointer<vtkZoomToBoxCallback> callback =
            vtkSmartPointer<vtkZoomToBoxCallback>::New();
    callback->SetTool(this);

    // CRITICAL: Observe Interactor, not Style (matching ParaView
    // implementation)
    m_observerId = m_interactor->AddObserver(vtkCommand::LeftButtonReleaseEvent,
                                             callback);

    // Set the cursor
    PclUtils::PCLVis* pclVis = reinterpret_cast<PclUtils::PCLVis*>(m_viewer);
    if (pclVis && pclVis->getRenderWindow()) {
        // Note: VTK doesn't directly support custom cursors via API
        // The cursor will be managed at the Qt level in MainWindow
    }

    m_enabled = true;
    emit enabledChanged(true);
}

//-----------------------------------------------------------------------------
void cvZoomToBoxTool::disable() {
    if (!m_enabled) {
        return;
    }

    // Remove observer from Interactor (not Style!)
    // This matches ParaView's pattern
    if (m_observerId > 0 && m_interactor) {
        m_interactor->RemoveObserver(m_observerId);
        m_observerId = 0;
    }

    // Restore previous style
    restoreStyle();

    m_enabled = false;
    emit enabledChanged(false);
}

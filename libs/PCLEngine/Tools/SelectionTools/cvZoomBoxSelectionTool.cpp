// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvZoomBoxSelectionTool.h"

#include "cvSelectionToolHelper.h"
#include "cvSelectionTypes.h"  // For SelectionMode enum

// LOCAL
#include "PclUtils/PCLVis.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <ecvDisplayTools.h>

// VTK
#include <vtkCamera.h>
#include <vtkCommand.h>
#include <vtkHardwareSelector.h>  // Full definition needed for copy assignment operator
#include <vtkInteractorStyleRubberBandZoom.h>
#include <vtkMath.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

// QT
#include <QPixmap>

// Use zoom cursor from CameraTools
#include "zoom.xpm"

//-----------------------------------------------------------------------------
// VTK callback class for zoom events
//-----------------------------------------------------------------------------
class cvZoomBoxCallback : public vtkCommand {
public:
    static cvZoomBoxCallback* New() { return new cvZoomBoxCallback; }

    void SetTool(cvZoomBoxSelectionTool* tool) { m_tool = tool; }

    void Execute(vtkObject* caller,
                 unsigned long eventId,
                 void* callData) override {
        if (m_tool) {
            m_tool->onSelectionChanged(caller, eventId, callData);
        }
    }

protected:
    cvZoomBoxCallback() : m_tool(nullptr) {}
    ~cvZoomBoxCallback() override = default;

private:
    cvZoomBoxSelectionTool* m_tool;
};

//-----------------------------------------------------------------------------
cvZoomBoxSelectionTool::cvZoomBoxSelectionTool(QObject* parent)
    : cvRenderViewSelectionTool(SelectionMode::ZOOM_TO_BOX, parent),
      m_useDollyForPerspective(true) {
    m_startPosition[0] = m_startPosition[1] = 0;
    m_endPosition[0] = m_endPosition[1] = 0;

    // Set zoom cursor
    m_cursor = QCursor(QPixmap(zoom_xpm));

    CVLog::Print("[cvZoomBoxSelectionTool] Created");
}

//-----------------------------------------------------------------------------
cvZoomBoxSelectionTool::~cvZoomBoxSelectionTool() {
    CVLog::Print("[cvZoomBoxSelectionTool] Destroyed");
}

//-----------------------------------------------------------------------------
QCursor cvZoomBoxSelectionTool::getCursor() const { return m_cursor; }

//-----------------------------------------------------------------------------
void cvZoomBoxSelectionTool::setupInteractorStyle() {
    if (!m_interactor) {
        return;
    }

    CVLog::PrintDebug(
            "[cvZoomBoxSelectionTool] Setting up RubberBandZoom interactor "
            "style");

    // Use vtkInteractorStyleRubberBandZoom for zoom box interaction
    // Reference: pqRenderViewSelectionReaction.cxx, line 403-406
    vtkSmartPointer<vtkInteractorStyleRubberBandZoom> zoomStyle =
            vtkSmartPointer<vtkInteractorStyleRubberBandZoom>::New();

    if (m_renderer) {
        zoomStyle->SetDefaultRenderer(m_renderer);
    }

    // Configure zoom style (ParaView defaults)
    zoomStyle->SetUseDollyForPerspectiveProjection(m_useDollyForPerspective);

    m_selectionStyle = zoomStyle;
    m_interactor->SetInteractorStyle(zoomStyle);
    m_cursor = QCursor(QPixmap(zoom_xpm));
}

//-----------------------------------------------------------------------------
void cvZoomBoxSelectionTool::setupObservers() {
    if (!m_interactor) {
        return;
    }

    CVLog::PrintDebug("[cvZoomBoxSelectionTool] Setting up zoom observers");

    // Create callback
    cvZoomBoxCallback* callback = cvZoomBoxCallback::New();
    callback->SetTool(this);

    // For ZOOM_TO_BOX, observe LeftButtonReleaseEvent from the interactor
    // Reference: pqRenderViewSelectionReaction.cxx, line 443-447
    m_observedObject = m_interactor;
    m_observerIds[0] = m_interactor->AddObserver(
            vtkCommand::LeftButtonReleaseEvent, callback);

    // Note: callback ownership is transferred to VTK observer mechanism
}

//-----------------------------------------------------------------------------
void cvZoomBoxSelectionTool::showInstructionAndSetCursor() {
    // Set zoom cursor
    // Reference: pqRenderViewSelectionReaction.cxx, line 404
    // this->View->setCursor(this->ZoomCursor);
    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::GetCurrentScreen()->setCursor(m_cursor);
    }

    // No instruction dialog for zoom mode (ParaView doesn't show one)
}

//-----------------------------------------------------------------------------
void cvZoomBoxSelectionTool::onSelectionChanged(vtkObject* caller,
                                                unsigned long eventId,
                                                void* callData) {
    if (eventId != vtkCommand::LeftButtonReleaseEvent) {
        return;
    }

    CVLog::Print(
            "[cvZoomBoxSelectionTool] Left button released - zoom completed");

    // Get the rubber band zoom style to access the selection region
    vtkInteractorStyleRubberBandZoom* zoomStyle =
            vtkInteractorStyleRubberBandZoom::SafeDownCast(m_selectionStyle);

    if (!zoomStyle || !m_interactor) {
        CVLog::Warning(
                "[cvZoomBoxSelectionTool] Invalid zoom style or interactor");
        return;
    }

    // The zoom is already performed by vtkInteractorStyleRubberBandZoom
    // We just need to emit the signal with the region

    // Get the current event position as the end position
    int* eventPos = m_interactor->GetEventPosition();
    if (eventPos) {
        m_endPosition[0] = eventPos[0];
        m_endPosition[1] = eventPos[1];
    }

    // Emit zoom completed signal
    // Note: The actual zoom is performed by vtkInteractorStyleRubberBandZoom
    // We emit this signal for any additional processing (e.g., updating UI)
    emit zoomToBoxCompleted(std::min(m_startPosition[0], m_endPosition[0]),
                            std::min(m_startPosition[1], m_endPosition[1]),
                            std::max(m_startPosition[0], m_endPosition[0]),
                            std::max(m_startPosition[1], m_endPosition[1]));

    // NOTE: Do NOT emit selectionCompleted() for zoom mode!
    // Zoom doesn't produce selection data, and emitting selectionCompleted
    // would trigger the selection data flow causing infinite recursion:
    // selectionFinished -> setCurrentSelection -> selectionChanged ->
    // selectionFinished...
}

//-----------------------------------------------------------------------------
bool cvZoomBoxSelectionTool::performSelection(int region[4]) {
    if (!m_viewer || !m_renderer || !region) {
        CVLog::Warning(
                "[cvZoomBoxSelectionTool] Invalid viewer, renderer or region");
        return false;
    }

    CVLog::Print(
            QString("[cvZoomBoxSelectionTool] Perform zoom: [%1, %2, %3, %4]")
                    .arg(region[0])
                    .arg(region[1])
                    .arg(region[2])
                    .arg(region[3]));

    // Check if region is valid (not just a click)
    if (region[0] == region[2] && region[1] == region[3]) {
        CVLog::Print("[cvZoomBoxSelectionTool] Single point click - no zoom");
        return false;
    }

    // Get camera
    vtkCamera* cam = m_renderer->GetActiveCamera();
    if (!cam) {
        CVLog::Warning("[cvZoomBoxSelectionTool] No active camera");
        return false;
    }

    // Perform zoom based on projection type
    if (cam->GetParallelProjection() || m_useDollyForPerspective) {
        zoomTraditional(region);
    } else {
        zoomPerspective(region);
    }

    // Force render update
    if (m_interactor) {
        m_interactor->Render();
    }

    // Emit zoom completed
    emit zoomToBoxCompleted(region[0], region[1], region[2], region[3]);

    return true;
}

//-----------------------------------------------------------------------------
void cvZoomBoxSelectionTool::zoomTraditional(int region[4]) {
    // Reference: vtkInteractorStyleRubberBandZoom::ZoomTraditional()

    if (!m_renderer) return;

    const int* size = m_renderer->GetSize();
    const int* origin = m_renderer->GetOrigin();
    vtkCamera* cam = m_renderer->GetActiveCamera();

    // Calculate box center in display coordinates
    double rbCenterX = (region[0] + region[2]) / 2.0;
    double rbCenterY = (region[1] + region[3]) / 2.0;

    // Convert to world coordinates
    m_renderer->SetDisplayPoint(rbCenterX, rbCenterY, 0.0);
    m_renderer->DisplayToWorld();
    double worldRBCenter[4];
    m_renderer->GetWorldPoint(worldRBCenter);

    // Calculate window center
    double winCenterX = origin[0] + size[0] / 2.0;
    double winCenterY = origin[1] + size[1] / 2.0;

    m_renderer->SetDisplayPoint(winCenterX, winCenterY, 0.0);
    m_renderer->DisplayToWorld();
    double worldWinCenter[4];
    m_renderer->GetWorldPoint(worldWinCenter);

    // Calculate translation
    double translation[3];
    translation[0] = worldRBCenter[0] - worldWinCenter[0];
    translation[1] = worldRBCenter[1] - worldWinCenter[1];
    translation[2] = worldRBCenter[2] - worldWinCenter[2];

    // Get camera position and focal point
    double pos[3], fp[3];
    cam->GetPosition(pos);
    cam->GetFocalPoint(fp);

    // Apply translation
    pos[0] += translation[0];
    pos[1] += translation[1];
    pos[2] += translation[2];
    fp[0] += translation[0];
    fp[1] += translation[1];
    fp[2] += translation[2];

    cam->SetPosition(pos);
    cam->SetFocalPoint(fp);

    // Calculate zoom factor
    int boxWidth = std::abs(region[2] - region[0]);
    int boxHeight = std::abs(region[3] - region[1]);

    double zoomFactor;
    if (boxWidth > boxHeight) {
        zoomFactor = size[0] / static_cast<double>(boxWidth);
    } else {
        zoomFactor = size[1] / static_cast<double>(boxHeight);
    }

    if (cam->GetParallelProjection()) {
        cam->Zoom(zoomFactor);
    } else {
        // Perspective mode: dolly
        double initialDistance = cam->GetDistance();
        cam->Dolly(zoomFactor);

        double finalDistance = cam->GetDistance();
        double deltaDistance = initialDistance - finalDistance;
        double clippingRange[2];
        cam->GetClippingRange(clippingRange);
        clippingRange[0] -= deltaDistance;
        clippingRange[1] -= deltaDistance;

        // Correct clipping planes
        if (clippingRange[1] <= 0.0) {
            clippingRange[1] = 0.001;
        }
        clippingRange[0] = std::max(clippingRange[0], 0.001 * clippingRange[1]);
        cam->SetClippingRange(clippingRange);
    }

    CVLog::Print(QString("[cvZoomBoxSelectionTool] Traditional zoom: factor=%1")
                         .arg(zoomFactor));
}

//-----------------------------------------------------------------------------
void cvZoomBoxSelectionTool::zoomPerspective(int region[4]) {
    // Reference: vtkInteractorStyleRubberBandZoom::Zoom() perspective case

    if (!m_renderer) return;

    vtkCamera* cam = m_renderer->GetActiveCamera();

    // Calculate box center
    double rbCenterX = (region[0] + region[2]) / 2.0;
    double rbCenterY = (region[1] + region[3]) / 2.0;

    // Convert to world coordinates for new focal point
    m_renderer->SetDisplayPoint(rbCenterX, rbCenterY, 0.0);
    m_renderer->DisplayToWorld();
    double worldRBCenter[4];
    m_renderer->GetWorldPoint(worldRBCenter);

    // Set new focal point
    cam->SetFocalPoint(worldRBCenter[0], worldRBCenter[1], worldRBCenter[2]);

    // Calculate box dimensions
    int boxWidth = std::abs(region[2] - region[0]);
    int boxHeight = std::abs(region[3] - region[1]);

    // Use renderer's ZoomToBoxUsingViewAngle-like behavior
    // Adjust view angle based on box size
    const int* size = m_renderer->GetSize();
    double aspect = static_cast<double>(size[0]) / size[1];
    double boxAspect = static_cast<double>(boxWidth) / boxHeight;

    double viewAngle = cam->GetViewAngle();
    double zoomFactor;
    if (boxAspect > aspect) {
        zoomFactor = size[0] / static_cast<double>(boxWidth);
    } else {
        zoomFactor = size[1] / static_cast<double>(boxHeight);
    }

    // Reduce view angle (zoom in)
    double newViewAngle = viewAngle / zoomFactor;
    cam->SetViewAngle(newViewAngle);

    CVLog::Print(QString("[cvZoomBoxSelectionTool] Perspective zoom: "
                         "viewAngle=%1 -> %2")
                         .arg(viewAngle)
                         .arg(newViewAngle));
}

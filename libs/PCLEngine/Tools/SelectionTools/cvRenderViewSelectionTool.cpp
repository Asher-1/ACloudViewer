// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvRenderViewSelectionTool.h"

#include "cvSelectionToolHelper.h"

// LOCAL
#include "PclUtils/PCLVis.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <ecvDisplayTools.h>

// VTK
#include <vtkActor.h>
#include <vtkCommand.h>
#include <vtkDataSetMapper.h>
#include <vtkIntArray.h>
#include <vtkInteractorObserver.h>
#include <vtkInteractorStyle.h>
#include <vtkInteractorStyleDrawPolygon.h>
#include <vtkInteractorStyleRubberBandPick.h>
#include <vtkInteractorStyleRubberBandZoom.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPropCollection.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

// QT
#include <QPixmap>

//-----------------------------------------------------------------------------
// cvSelectionCallback implementation
//-----------------------------------------------------------------------------

/**
 * @brief VTK callback class for handling selection events
 *
 * This is a helper class that bridges VTK's callback system
 * with Qt's signal/slot mechanism.
 *
 * Reference: pqRenderViewSelectionReaction.cxx uses vtkCommand callbacks
 */
class cvSelectionCallback : public vtkCommand {
public:
    static cvSelectionCallback* New() { return new cvSelectionCallback; }

    void SetTool(cvRenderViewSelectionTool* tool) { m_tool = tool; }

    void Execute(vtkObject* caller,
                 unsigned long eventId,
                 void* callData) override {
        if (m_tool) {
            m_tool->onSelectionChanged(caller, eventId, callData);
        }
    }

protected:
    cvSelectionCallback() : m_tool(nullptr) {}
    ~cvSelectionCallback() override = default;

private:
    cvRenderViewSelectionTool* m_tool;
};

//-----------------------------------------------------------------------------
// cvRenderViewSelectionTool implementation
//-----------------------------------------------------------------------------

cvRenderViewSelectionTool::cvRenderViewSelectionTool(SelectionMode mode,
                                                     QObject* parent)
    : QObject(parent),
      cvGenericSelectionTool(),
      m_mode(mode),
      m_modifier(cvViewSelectionManager::SELECTION_DEFAULT),
      m_enabled(false),
      m_previousRenderViewMode(-1),
      m_cursor(Qt::CrossCursor) {
    // Note: m_interactor and m_renderer are now inherited from base class
    // and initialized there

    // Initialize observer IDs
    for (size_t i = 0; i < sizeof(m_observerIds) / sizeof(m_observerIds[0]);
         ++i) {
        m_observerIds[i] = 0;
    }
}

//-----------------------------------------------------------------------------
cvRenderViewSelectionTool::~cvRenderViewSelectionTool() {
    if (m_enabled) {
        disable();
    }
    cleanupObservers();
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionTool::setVisualizer(ecvGenericVisualizer3D* viewer) {
    if (getVisualizer() == viewer) {
        return;
    }

    // Disable if currently enabled
    if (m_enabled) {
        disable();
    }

    cvGenericSelectionTool::setVisualizer(viewer);

    PclUtils::PCLVis* pclVis = getPCLVis();
    if (pclVis) {
        m_interactor = pclVis->getRenderWindowInteractor();
        m_renderer = pclVis->getCurrentRenderer();
    } else {
        m_interactor = nullptr;
        m_renderer = nullptr;
        CVLog::Warning(
                "[cvRenderViewSelectionTool::setVisualizer] viewer is "
                "nullptr!");
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionTool::setSelectionModifier(
        SelectionModifier modifier) {
    m_modifier = modifier;
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionTool::enable() {
    if (m_enabled || !getVisualizer() || !m_interactor) {
        if (m_enabled) {
            CVLog::Warning(
                    "[cvRenderViewSelectionTool::enable] Already enabled, "
                    "returning");
        }
        if (!getVisualizer()) {
            CVLog::Warning(
                    "[cvRenderViewSelectionTool::enable] visualizer is "
                    "nullptr, "
                    "returning");
        }
        if (!m_interactor) {
            CVLog::Warning(
                    "[cvRenderViewSelectionTool::enable] m_interactor is "
                    "nullptr, returning");
        }
        return;
    }

    CVLog::PrintDebug(QString("[cvRenderViewSelectionTool] Enabling mode %1")
                              .arg(static_cast<int>(m_mode)));

    // CRITICAL: Disable PCLVis's PointPickingCallback to prevent
    // multi-threading crash PCLVis::pointPickingProcess uses OpenMP parallel
    // trianglePicking which conflicts with our selection tools' picking
    // operations (Ctrl+Shift+Click crash fix)
    PclUtils::PCLVis* pclVis = getPCLVis();
    if (pclVis && pclVis->isPointPickingEnabled()) {
        pclVis->setPointPickingEnabled(false);
        CVLog::PrintDebug(
                "[cvRenderViewSelectionTool] Disabled PCLVis PointPicking to "
                "prevent multi-threading conflicts");
    }

    // Show instruction dialog and set cursor (ParaView-style)
    showInstructionAndSetCursor();

    // Store current interactor style
    storeCurrentStyle();

    // Set up the interactor style for this mode
    setupInteractorStyle();

    // Set up event observers
    setupObservers();

    m_enabled = true;
    emit enabledChanged(true);
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionTool::disable() {
    if (!m_enabled) {
        return;
    }

    // Clean up observers
    cleanupObservers();

    // Restore previous style
    restoreStyle();

    // Re-enable PCLVis's PointPickingCallback (restore normal behavior)
    PclUtils::PCLVis* pclVis = getPCLVis();
    if (pclVis && !pclVis->isPointPickingEnabled()) {
        pclVis->setPointPickingEnabled(true);
    }

    // Restore cursor (ParaView-style)
    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::GetCurrentScreen()->setCursor(Qt::ArrowCursor);
    }

    m_enabled = false;
    emit enabledChanged(false);
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionTool::showInstructionAndSetCursor() {
    // Show instruction dialog (ParaView-style: pqCoreUtilities::promptUser)
    QString settingsKey, title, message;
    QCursor cursor;

    switch (m_mode) {
        case cvViewSelectionManager::HOVER_CELLS_TOOLTIP:
            settingsKey = "pqTooltipSelection";
            title = tr("Tooltip Selection Information");
            message = tr(
                    "You are entering tooltip selection mode to display cell "
                    "information. "
                    "Simply move the mouse point over the dataset to "
                    "interactively highlight "
                    "cells and display a tooltip with cell information.<br><br>"
                    "Use the <b>Esc</b> key or the same toolbar button to exit "
                    "this mode.");
            cursor = Qt::CrossCursor;
            break;

        case cvViewSelectionManager::HOVER_POINTS_TOOLTIP:
            settingsKey = "pqTooltipSelection";
            title = tr("Tooltip Selection Information");
            message =
                    tr("You are entering tooltip selection mode to display "
                       "points information. "
                       "Simply move the mouse point over the dataset to "
                       "interactively highlight "
                       "points and display a tooltip with points "
                       "information.<br><br>"
                       "Use the <b>Esc</b> key or the same toolbar button to "
                       "exit this mode.");
            cursor = Qt::CrossCursor;
            break;

        case cvViewSelectionManager::SELECT_SURFACE_CELLS_INTERACTIVELY:
        case cvViewSelectionManager::SELECT_SURFACE_POINTS_INTERACTIVELY:
            settingsKey = "pqInteractiveSelection";
            title = tr("Interactive Selection Information");
            message =
                    tr("You are entering interactive selection mode to "
                       "highlight cells (or points). "
                       "Simply move the mouse point over the dataset to "
                       "interactively highlight elements.<br><br>"
                       "<b>Click to add</b> the currently highlighted element "
                       "to the active "
                       "selection (default behavior is <b>accumulation</b>, no "
                       "Ctrl needed).<br><br>"
                       "You can use modifier keys:<br>"
                       "• <b>Shift + Click</b>: Remove from selection<br>"
                       "• <b>Ctrl+Shift + Click</b>: Toggle selection<br><br>"
                       "Click outside of mesh to clear selection.<br><br>"
                       "Use the <b>Esc</b> key or the same toolbar button to "
                       "exit this mode.");
            cursor = Qt::CrossCursor;
            break;

        case cvViewSelectionManager::SELECT_SURFACE_CELLS:
        case cvViewSelectionManager::SELECT_SURFACE_POINTS:
        case cvViewSelectionManager::SELECT_FRUSTUM_CELLS:
        case cvViewSelectionManager::SELECT_FRUSTUM_POINTS:
            cursor = Qt::CrossCursor;
            // No dialog for these standard selection modes (ParaView doesn't
            // show one)
            break;

        case cvViewSelectionManager::SELECT_SURFACE_CELLS_POLYGON:
        case cvViewSelectionManager::SELECT_SURFACE_POINTS_POLYGON:
            cursor = Qt::PointingHandCursor;
            break;
        default:
            cursor = Qt::ArrowCursor;
            break;
    }

    // Show instruction dialog if settings key is provided
    if (!settingsKey.isEmpty()) {
        cvSelectionToolHelper::promptUser(settingsKey, title, message, nullptr);
    }

    // Set cursor (ParaView-style: this->View->setCursor(...))
    if (ecvDisplayTools::GetCurrentScreen()) {
        ecvDisplayTools::GetCurrentScreen()->setCursor(cursor);
    }
}

//-----------------------------------------------------------------------------
QCursor cvRenderViewSelectionTool::getCursor() const {
    // Return mode-specific cursor
    // This can be customized by subclasses
    return m_cursor;
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionTool::storeCurrentStyle() {
    if (m_interactor) {
        vtkInteractorObserver* style = m_interactor->GetInteractorStyle();
        m_previousStyle = vtkInteractorStyle::SafeDownCast(style);
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionTool::restoreStyle() {
    if (m_interactor && m_previousStyle) {
        m_interactor->SetInteractorStyle(m_previousStyle);
        m_previousStyle = nullptr;
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionTool::setupInteractorStyle() {
    if (!m_interactor) {
        return;
    }

    // Set up interactor style based on mode
    // Reference: pqRenderViewSelectionReaction.cxx, line 338-437

    switch (m_mode) {
        case SelectionMode::SELECT_SURFACE_CELLS:
        case SelectionMode::SELECT_SURFACE_POINTS:
        case SelectionMode::SELECT_FRUSTUM_CELLS:
        case SelectionMode::SELECT_FRUSTUM_POINTS:
        case SelectionMode::SELECT_BLOCKS:
        case SelectionMode::SELECT_FRUSTUM_BLOCKS: {
            // Rectangle selection mode
            vtkSmartPointer<vtkInteractorStyleRubberBandPick> rubberBandStyle =
                    vtkSmartPointer<vtkInteractorStyleRubberBandPick>::New();

            if (m_renderer) {
                rubberBandStyle->SetDefaultRenderer(m_renderer);
            }

            m_selectionStyle = rubberBandStyle;
            m_interactor->SetInteractorStyle(rubberBandStyle);
            m_cursor = Qt::CrossCursor;
            break;
        }

        case SelectionMode::SELECT_SURFACE_CELLS_POLYGON:
        case SelectionMode::SELECT_SURFACE_POINTS_POLYGON:
        case SelectionMode::SELECT_CUSTOM_POLYGON: {
            // Polygon selection mode
            vtkSmartPointer<vtkInteractorStyleDrawPolygon> polygonStyle =
                    vtkSmartPointer<vtkInteractorStyleDrawPolygon>::New();

            if (m_renderer) {
                polygonStyle->SetDefaultRenderer(m_renderer);
            }

            m_selectionStyle = polygonStyle;
            m_interactor->SetInteractorStyle(polygonStyle);
            m_cursor = Qt::PointingHandCursor;
            break;
        }

        case SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_CELLDATA_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTDATA_INTERACTIVELY:
        case SelectionMode::HOVER_POINTS_TOOLTIP:
        case SelectionMode::HOVER_CELLS_TOOLTIP: {
            // Interactive selection mode
            // Use rubber band pick style but handle mouse move events
            vtkSmartPointer<vtkInteractorStyleRubberBandPick> interactiveStyle =
                    vtkSmartPointer<vtkInteractorStyleRubberBandPick>::New();

            if (m_renderer) {
                interactiveStyle->SetDefaultRenderer(m_renderer);
            }

            m_selectionStyle = interactiveStyle;
            m_interactor->SetInteractorStyle(interactiveStyle);
            m_cursor = Qt::CrossCursor;
            break;
        }

        default:
            CVLog::Warning(
                    QString("[cvRenderViewSelectionTool] Unknown mode: %1")
                            .arg(static_cast<int>(m_mode)));
            m_cursor = Qt::ArrowCursor;
            break;
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionTool::setupObservers() {
    if (!m_selectionStyle) {
        return;
    }

    // Create callback
    cvSelectionCallback* callback = cvSelectionCallback::New();
    callback->SetTool(this);

    // Set up observers based on mode
    // Reference: pqRenderViewSelectionReaction.cxx, line 439-491

    switch (m_mode) {
        case SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_CELLDATA_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTDATA_INTERACTIVELY:
            // Observe mouse move and button events
            m_observedObject = m_interactor;
            m_observerIds[0] = m_interactor->AddObserver(
                    vtkCommand::MouseMoveEvent, callback);
            m_observerIds[1] = m_interactor->AddObserver(
                    vtkCommand::LeftButtonReleaseEvent, callback);
            m_observerIds[2] = m_interactor->AddObserver(
                    vtkCommand::MouseWheelForwardEvent, callback);
            m_observerIds[3] = m_interactor->AddObserver(
                    vtkCommand::MouseWheelBackwardEvent, callback);
            m_observerIds[4] = m_interactor->AddObserver(
                    vtkCommand::RightButtonPressEvent, callback);
            m_observerIds[5] = m_interactor->AddObserver(
                    vtkCommand::RightButtonReleaseEvent, callback);
            break;

        case SelectionMode::HOVER_POINTS_TOOLTIP:
        case SelectionMode::HOVER_CELLS_TOOLTIP:
            // Observe mouse move for tooltip
            m_observedObject = m_interactor;
            m_observerIds[0] = m_interactor->AddObserver(
                    vtkCommand::MouseMoveEvent, callback);
            m_observerIds[1] = m_interactor->AddObserver(
                    vtkCommand::RightButtonPressEvent, callback);
            m_observerIds[2] = m_interactor->AddObserver(
                    vtkCommand::RightButtonReleaseEvent, callback);
            break;

        default:
            // Default: observe selection changed event from the style
            m_observedObject = m_selectionStyle;
            m_observerIds[0] = m_selectionStyle->AddObserver(
                    vtkCommand::SelectionChangedEvent, callback);
            break;
    }

    // Note: callback is a raw pointer returned by New(), but VTK's observer
    // mechanism takes ownership. We should NOT delete it here as VTK manages
    // it. The callback will be automatically deleted when the observer is
    // removed.
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionTool::cleanupObservers() {
    CVLog::PrintDebug(QString("[cvRenderViewSelectionTool::cleanupObservers] "
                              "m_observedObject=%1")
                              .arg((quintptr)m_observedObject.Get(), 0, 16));

    int removedCount = 0;
    for (size_t i = 0; i < sizeof(m_observerIds) / sizeof(m_observerIds[0]);
         ++i) {
        if (m_observedObject && m_observerIds[i] > 0) {
            m_observedObject->RemoveObserver(m_observerIds[i]);
            CVLog::PrintDebug(
                    QString("[cleanupObservers] Removed observer ID: %1")
                            .arg(m_observerIds[i]));
            removedCount++;
        }
        m_observerIds[i] = 0;
    }

    CVLog::PrintDebug(QString("[cleanupObservers] Removed %1 observers")
                              .arg(removedCount));
    m_observedObject = nullptr;
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionTool::onSelectionChanged(vtkObject* caller,
                                                   unsigned long eventId,
                                                   void* callData) {
    CVLog::Print(
            QString("[cvRenderViewSelectionTool] Selection changed, event: %1")
                    .arg(eventId));

    // Get the selection modifier (may be modified by keyboard)
    SelectionModifier modifier = getSelectionModifierFromKeyboard();
    if (modifier == cvViewSelectionManager::SELECTION_DEFAULT &&
        m_modifier != cvViewSelectionManager::SELECTION_DEFAULT) {
        modifier = m_modifier;
    }

    // Handle based on event type
    if (eventId == vtkCommand::SelectionChangedEvent) {
        // Rectangle or polygon selection completed
        int* region = reinterpret_cast<int*>(callData);
        vtkObject* obj = reinterpret_cast<vtkObject*>(callData);

        // Try rectangle selection first
        if (region) {
            performSelection(region);
        }
        // Try polygon selection
        else if (vtkIntArray* polygon = vtkIntArray::SafeDownCast(obj)) {
            performPolygonSelection(polygon);
        }

        emit selectionCompleted();
    } else if (eventId == vtkCommand::LeftButtonReleaseEvent) {
        // Interactive selection click
        // Handled by specific tool subclasses (cvSurfaceSelectionTool, etc.)
        emit selectionChanged();
    } else if (eventId == vtkCommand::MouseMoveEvent) {
        // Interactive selection or tooltip
        // Handled by cvTooltipSelectionTool for hover highlighting
    }
}

//-----------------------------------------------------------------------------
bool cvRenderViewSelectionTool::performSelection(int region[4]) {
    if (!getVisualizer() || !region) {
        return false;
    }

    CVLog::Print(QString("[cvRenderViewSelectionTool] Perform selection: [%1, "
                         "%2, %3, %4]")
                         .arg(region[0])
                         .arg(region[1])
                         .arg(region[2])
                         .arg(region[3]));

    // Base implementation - subclasses override for specific selection logic
    // - cvSurfaceSelectionTool: Uses vtkHardwareSelector for surface picking
    // - cvFrustumSelectionTool: Uses frustum planes for volume selection
    // - cvPolygonSelectionTool: Uses polygon-based hardware selection
    // - cvBlockSelectionTool: Uses block/composite selection

    emit selectionChanged();
    return true;
}

//-----------------------------------------------------------------------------
bool cvRenderViewSelectionTool::performPolygonSelection(vtkIntArray* polygon) {
    if (!getVisualizer() || !polygon) {
        return false;
    }

    CVLog::Print(QString("[cvRenderViewSelectionTool] Perform polygon "
                         "selection: %1 points")
                         .arg(polygon->GetNumberOfTuples()));

    // Base implementation - subclasses override for specific polygon selection
    // cvPolygonSelectionTool implements pixel-precise polygon selection using
    // vtkHardwareSelector::GeneratePolygonSelection() (ParaView-aligned)

    emit selectionChanged();
    return true;
}

//-----------------------------------------------------------------------------
cvRenderViewSelectionTool::SelectionModifier
cvRenderViewSelectionTool::getSelectionModifierFromKeyboard() const {
    if (!m_interactor) {
        return cvViewSelectionManager::SELECTION_DEFAULT;
    }

    // Check keyboard modifiers
    // Reference: pqRenderViewSelectionReaction.cxx, line 1013-1035
    bool ctrl = m_interactor->GetControlKey() == 1;
    bool shift = m_interactor->GetShiftKey() == 1;

    if (ctrl && shift) {
        return cvViewSelectionManager::SELECTION_TOGGLE;
    } else if (ctrl) {
        return cvViewSelectionManager::SELECTION_ADDITION;
    } else if (shift) {
        return cvViewSelectionManager::SELECTION_SUBTRACTION;
    }

    return cvViewSelectionManager::SELECTION_DEFAULT;
}

// Note: initializePickers() and pickAtCursor() methods have been moved to
// base class cvGenericSelectionTool to unify picking across all selection
// tools. Now using inherited methods.

//-----------------------------------------------------------------------------
bool cvRenderViewSelectionTool::isSelectingCells() const {
    // Check if the mode is for cell selection
    return (m_mode == cvViewSelectionManager::SELECT_SURFACE_CELLS ||
            m_mode == cvViewSelectionManager::SELECT_SURFACE_CELLS_POLYGON ||
            m_mode == cvViewSelectionManager::
                              SELECT_SURFACE_CELLS_INTERACTIVELY ||
            m_mode == cvViewSelectionManager::SELECT_FRUSTUM_CELLS ||
            m_mode == cvViewSelectionManager::HOVER_CELLS_TOOLTIP ||
            m_mode == cvViewSelectionManager::SELECT_BLOCKS ||
            m_mode == cvViewSelectionManager::SELECT_FRUSTUM_BLOCKS);
}

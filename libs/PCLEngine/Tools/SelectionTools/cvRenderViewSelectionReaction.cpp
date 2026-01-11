// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file cvRenderViewSelectionReaction.cpp
 * @brief Implementation of the simplified selection reaction class
 *
 * This implementation directly mirrors ParaView's pqRenderViewSelectionReaction
 * for clean, maintainable selection handling.
 *
 * Reference:
 * ParaView/Qt/ApplicationComponents/pqRenderViewSelectionReaction.cxx
 */

#include "cvRenderViewSelectionReaction.h"

#include "cvSelectionHighlighter.h"
#include "cvSelectionPipeline.h"
// cvSelectionToolHelper merged into cvSelectionPipeline
#include "cvViewSelectionManager.h"

// LOCAL
#include "PclUtils/PCLVis.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <ecvDisplayTools.h>

// VTK
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkCellType.h>
#include <vtkCommand.h>
#include <vtkDataArray.h>
#include <vtkFieldData.h>
#include <vtkHardwareSelector.h>
#include <vtkIdTypeArray.h>
#include <vtkIntArray.h>
#include <vtkInteractorStyle.h>
#include <vtkInteractorStyleDrawPolygon.h>
#include <vtkInteractorStyleRubberBand3D.h>
#include <vtkInteractorStyleRubberBandPick.h>
#include <vtkInteractorStyleRubberBandZoom.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkStringArray.h>
#include <vtkVector.h>

// QT
#include <QActionGroup>
#include <QApplication>
#include <QMessageBox>
#include <QPixmap>
#include <QTimer>
#include <QToolTip>
#include <QWidget>

// Include zoom cursor XPM
#include "zoom.xpm"

//-----------------------------------------------------------------------------
// Static member initialization
//-----------------------------------------------------------------------------
QPointer<cvRenderViewSelectionReaction>
        cvRenderViewSelectionReaction::ActiveReaction;

//-----------------------------------------------------------------------------
// Constructor/Destructor
//-----------------------------------------------------------------------------

cvRenderViewSelectionReaction::cvRenderViewSelectionReaction(
        QAction* parentAction, SelectionMode mode, QActionGroup* modifierGroup)
    : QObject(parentAction),
      m_parentAction(parentAction),
      m_modifierGroup(modifierGroup),
      m_mode(mode),
      m_zoomCursor(QCursor(QPixmap((const char**)zoom_xpm))),
      m_mouseMovingTimer(this) {
    // Initialize observer IDs
    for (size_t i = 0; i < sizeof(m_observerIds) / sizeof(m_observerIds[0]);
         ++i) {
        m_observerIds[i] = 0;
    }

    // Connect action triggered signal
    // Reference: pqRenderViewSelectionReaction.cxx line 68
    connect(parentAction, &QAction::triggered, this,
            &cvRenderViewSelectionReaction::actionTriggered);

    // For non-interactive modes (Clear, Grow, Shrink), connect to selection
    // manager Reference: pqRenderViewSelectionReaction.cxx lines 80-88
    if (m_mode == SelectionMode::CLEAR_SELECTION ||
        m_mode == SelectionMode::GROW_SELECTION ||
        m_mode == SelectionMode::SHRINK_SELECTION) {
        cvViewSelectionManager* manager = cvViewSelectionManager::instance();
        connect(manager,
                QOverload<>::of(&cvViewSelectionManager::selectionChanged),
                this, &cvRenderViewSelectionReaction::updateEnableState);
    }

    // For interactive/tooltip modes, connect to data update signal
    // Reference: pqRenderViewSelectionReaction.cxx lines 99-112
    switch (m_mode) {
        case SelectionMode::SELECT_SURFACE_CELLDATA_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTDATA_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY:
        case SelectionMode::HOVER_POINTS_TOOLTIP:
        case SelectionMode::HOVER_CELLS_TOOLTIP:
            // Connect to data update for cache invalidation
            // This would be connected to pqActiveObjects::dataUpdated in
            // ParaView
            break;
        default:
            break;
    }

    // Setup mouse moving timer for tooltip
    // Reference: pqRenderViewSelectionReaction.cxx line 115-116
    m_mouseMovingTimer.setSingleShot(true);
    connect(&m_mouseMovingTimer, &QTimer::timeout, this,
            &cvRenderViewSelectionReaction::onMouseStop);

    // Initial enable state update
    updateEnableState();
}

//-----------------------------------------------------------------------------
cvRenderViewSelectionReaction::~cvRenderViewSelectionReaction() {
    // Clean up observers before destruction
    // Reference: pqRenderViewSelectionReaction.cxx line 124-127
    cleanupObservers();
}

//-----------------------------------------------------------------------------
// Public methods
//-----------------------------------------------------------------------------

void cvRenderViewSelectionReaction::setVisualizer(
        ecvGenericVisualizer3D* viewer) {
    if (m_viewer == viewer) {
        return;
    }

    // End any active selection before changing visualizer
    if (isActive()) {
        endSelection();
    }

    m_viewer = viewer;
    m_interactor = nullptr;
    m_renderer = nullptr;

    // Get VTK objects from visualizer
    PclUtils::PCLVis* pclVis = getPCLVis();
    if (pclVis) {
        m_interactor = pclVis->getRenderWindowInteractor();
        m_renderer = pclVis->getCurrentRenderer();
    }

    // Update enable state with new visualizer
    updateEnableState();
}

//-----------------------------------------------------------------------------
bool cvRenderViewSelectionReaction::isActive() const {
    return ActiveReaction == this;
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::endActiveSelection() {
    if (ActiveReaction) {
        ActiveReaction->endSelection();
    }
}

//-----------------------------------------------------------------------------
// Public event handlers (forwarded to protected methods)
//-----------------------------------------------------------------------------

void cvRenderViewSelectionReaction::handleSelectionChanged(
        vtkObject* caller, unsigned long eventId, void* callData) {
    selectionChanged(caller, eventId, callData);
}

void cvRenderViewSelectionReaction::handleMouseMove() { onMouseMove(); }

void cvRenderViewSelectionReaction::handleLeftButtonPress() {
    onLeftButtonPress();
}

void cvRenderViewSelectionReaction::handleLeftButtonRelease() {
    onLeftButtonRelease();
}

void cvRenderViewSelectionReaction::handleWheelRotate() { onWheelRotate(); }

void cvRenderViewSelectionReaction::handleRightButtonPressed() {
    onRightButtonPressed();
}

void cvRenderViewSelectionReaction::handleRightButtonRelease() {
    onRightButtonRelease();
}

//-----------------------------------------------------------------------------
// VTK callback functions (ParaView-style direct member function callbacks)
//-----------------------------------------------------------------------------

// VTK observer callback adapters - required signature for AddObserver
void cvRenderViewSelectionReaction::vtkOnSelectionChanged(vtkObject* caller,
                                                          unsigned long eventId,
                                                          void* callData) {
    selectionChanged(caller, eventId, callData);
}

void cvRenderViewSelectionReaction::vtkOnMouseMove(vtkObject*,
                                                   unsigned long,
                                                   void*) {
    onMouseMove();
}

void cvRenderViewSelectionReaction::vtkOnLeftButtonPress(vtkObject*,
                                                         unsigned long,
                                                         void*) {
    onLeftButtonPress();
}

void cvRenderViewSelectionReaction::vtkOnLeftButtonRelease(vtkObject*,
                                                           unsigned long,
                                                           void*) {
    onLeftButtonRelease();
}

void cvRenderViewSelectionReaction::vtkOnWheelRotate(vtkObject*,
                                                     unsigned long,
                                                     void*) {
    onWheelRotate();
}

void cvRenderViewSelectionReaction::vtkOnRightButtonPressed(vtkObject*,
                                                            unsigned long,
                                                            void*) {
    onRightButtonPressed();
}

void cvRenderViewSelectionReaction::vtkOnRightButtonRelease(vtkObject*,
                                                            unsigned long,
                                                            void*) {
    onRightButtonRelease();
}

void cvRenderViewSelectionReaction::vtkOnMiddleButtonPressed(vtkObject*,
                                                             unsigned long,
                                                             void*) {
    onMiddleButtonPressed();
}

void cvRenderViewSelectionReaction::vtkOnMiddleButtonRelease(vtkObject*,
                                                             unsigned long,
                                                             void*) {
    onMiddleButtonRelease();
}

//-----------------------------------------------------------------------------
// Public slots
//-----------------------------------------------------------------------------

void cvRenderViewSelectionReaction::actionTriggered(bool val) {
    // Reference: pqRenderViewSelectionReaction::actionTriggered()
    // lines 144-163

    QAction* actn = m_parentAction;
    if (!actn) {
        return;
    }

    if (actn->isCheckable()) {
        if (val) {
            beginSelection();
        } else {
            endSelection();
        }
    } else {
        // Non-checkable actions (like Clear, Grow, Shrink)
        // Reference: ParaView pqRenderViewSelectionReaction.cxx lines 158-162
        // Call both beginSelection() and endSelection() in sequence
        beginSelection();
        endSelection();
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::updateEnableState() {
    // Reference: pqRenderViewSelectionReaction::updateEnableState()
    // lines 166-259

    if (!m_parentAction) {
        return;
    }

    // End selection if currently active and updating state
    // Reference: line 168
    endSelection();

    cvViewSelectionManager* manager = cvViewSelectionManager::instance();

    switch (m_mode) {
        case SelectionMode::CLEAR_SELECTION:
        case SelectionMode::GROW_SELECTION:
            // Enable if there's an active selection
            m_parentAction->setEnabled(manager->hasSelection());
            break;

        case SelectionMode::SHRINK_SELECTION:
            // Enable if selection can be shrunk (has layers >= 1)
            // Reference: pqRenderViewSelectionReaction::updateEnableState()
            // lines 184-212 - checks NumberOfLayers property
            m_parentAction->setEnabled(manager->canShrinkSelection());
            break;

        case SelectionMode::SELECT_SURFACE_CELLDATA_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTDATA_INTERACTIVELY:
            // These require a colored representation
            // Simplified: enable if visualizer is available
            m_parentAction->setEnabled(m_viewer != nullptr);
            break;

        default:
            // Other modes: enable if visualizer is available
            m_parentAction->setEnabled(m_viewer != nullptr);
            break;
    }
}

//-----------------------------------------------------------------------------
// Protected slots
//-----------------------------------------------------------------------------

void cvRenderViewSelectionReaction::beginSelection() {
    // Reference: pqRenderViewSelectionReaction::beginSelection()
    // lines 307-494

    if (!m_viewer || !m_interactor) {
        CVLog::Warning(
                "[cvRenderViewSelectionReaction::beginSelection] No viewer or "
                "interactor");
        return;
    }

    // Already active - nothing to do
    // Reference: lines 314-318
    if (ActiveReaction == this) {
        return;
    }

    // Check compatibility with current selection and handle modifier group
    // Reference: lines 320-331
    if (ActiveReaction) {
        if (!ActiveReaction->isCompatible(m_mode)) {
            uncheckSelectionModifiers();
        }
        // End other active reaction
        ActiveReaction->endSelection();
    }

    // Set this as the active reaction
    // Reference: line 333
    ActiveReaction = this;

    // Store previous interaction mode for restoration
    // Reference: lines 335-336
    // In ParaView this uses vtkSMPropertyHelper to get InteractionMode
    // We store the interactor style instead
    m_previousRenderViewMode = 0;  // Placeholder (valid value, not -1)

    // Handle each mode type
    // Reference: lines 340-437 (first switch statement)
    cvViewSelectionManager* manager = cvViewSelectionManager::instance();

    switch (m_mode) {
        // CLEAR/GROW/SHRINK are one-shot commands
        // Reference: lines 415-431
        case SelectionMode::CLEAR_SELECTION:
            if (manager) {
                manager->clearSelection();
            }
            break;

        case SelectionMode::GROW_SELECTION:
            if (manager) {
                manager->growSelection();
            }
            break;

        case SelectionMode::SHRINK_SELECTION:
            if (manager) {
                manager->shrinkSelection();
            }
            break;

        // Interactive and tooltip modes
        case SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_CELLDATA_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTDATA_INTERACTIVELY:
        case SelectionMode::HOVER_POINTS_TOOLTIP:
        case SelectionMode::HOVER_CELLS_TOOLTIP:
            // Show instruction dialog for interactive/tooltip modes
            // Reference: lines 352-401
            showInstructionDialog();
            // Fall through to setup cursor and style
            [[fallthrough]];

        default:
            // CRITICAL: Disable PCLVis's PointPickingCallback to prevent
            // conflicts
            if (PclUtils::PCLVis* pclVis = getPCLVis()) {
                if (pclVis->isPointPickingEnabled()) {
                    pclVis->setPointPickingEnabled(false);
                }
            }

            // Enter selection mode for cache optimization
            if (cvSelectionPipeline* pipeline = getSelectionPipeline()) {
                pipeline->enterSelectionMode();
            }

            // Set cursor and interactor style based on mode
            // Reference: lines 340-437
            // NOTE: setupInteractorStyle() calls storeCurrentStyle() internally
            setupInteractorStyle();
            break;
    }

    // Set up event observers based on mode
    // Reference: lines 439-491 (second switch statement)
    setupObservers();

    // Update action state
    // Reference: line 493
    if (m_parentAction) {
        m_parentAction->setChecked(true);
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::endSelection() {
    // Reference: pqRenderViewSelectionReaction::endSelection()
    // lines 497-533

    if (!m_viewer) {
        return;
    }

    // Only end if this is the active reaction
    // Reference: lines 504-507
    if (ActiveReaction != this || m_previousRenderViewMode == -1) {
        return;
    }

    // Clear active reaction FIRST to prevent re-entry
    // Reference: line 509
    ActiveReaction = nullptr;
    m_previousRenderViewMode = -1;

    // Stop mouse moving timer
    // Reference: lines 517-518
    m_mouseMovingTimer.stop();
    m_mouseMoving = false;

    // Clear hover state
    m_hoveredId = -1;
    m_currentPolyData = nullptr;

    // CRITICAL FIX: Defer cleanup operations to avoid crashing when called
    // from within VTK event handlers (e.g.,
    // vtkInteractorStyleRubberBand3D::OnLeftButtonUp) The VTK event loop may
    // still invoke additional events after SelectionChangedEvent, so we must
    // not cleanup observers until the event loop completes.
    QTimer::singleShot(0, this, [this]() {
        // If a new mode has started, only skip the mode-exit operations
        // but still clean up observers from this mode
        bool newModeActive = (ActiveReaction != nullptr);

        if (newModeActive) {
            // Only clean up observers - the new mode will handle style/cursor
            cleanupObservers();
            return;
        }

        // Full cleanup when no new mode is active:

        // Restore previous interactor style
        // Reference: lines 510-513
        restoreStyle();

        // Restore cursor to default arrow
        // Reference: line 514
        unsetCursor();

        // Clean up observers - SAFE now since we're outside VTK event loop
        // Reference: line 515
        cleanupObservers();

        // Update tooltip
        // Reference: line 519
        updateTooltip();

        // Exit selection mode to release cached buffers
        cvSelectionPipeline* pipeline = getSelectionPipeline();
        if (pipeline) {
            pipeline->exitSelectionMode();
        }

        // Re-enable PCLVis's PointPickingCallback
        PclUtils::PCLVis* pclVis = getPCLVis();
        if (pclVis && !pclVis->isPointPickingEnabled()) {
            pclVis->setPointPickingEnabled(true);
        }
    });

    // Uncheck the action immediately (safe to do)
    // Reference: line 516
    if (m_parentAction && m_parentAction->isCheckable() &&
        m_parentAction->isChecked()) {
        m_parentAction->blockSignals(true);
        m_parentAction->setChecked(false);
        m_parentAction->blockSignals(false);
    }

    CVLog::PrintDebug(
            QString("[cvRenderViewSelectionReaction] Selection mode %1 ended")
                    .arg(static_cast<int>(m_mode)));
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::onMouseStop() {
    // Reference: pqRenderViewSelectionReaction::onMouseStop()
    // lines 875-879
    m_mouseMoving = false;
    updateTooltip();
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::clearSelectionCache() {
    // Reference: pqRenderViewSelectionReaction::clearSelectionCache()
    // lines 882-888
    if (ActiveReaction == this) {
        cvSelectionPipeline* pipeline = getSelectionPipeline();
        if (pipeline) {
            pipeline->invalidateCachedSelection();
        }
    }
}

//-----------------------------------------------------------------------------
// Protected methods
//-----------------------------------------------------------------------------

void cvRenderViewSelectionReaction::selectionChanged(vtkObject* caller,
                                                     unsigned long eventId,
                                                     void* callData) {
    // Reference: pqRenderViewSelectionReaction::selectionChanged()
    // lines 536-603

    if (ActiveReaction != this) {
        // This can happen due to delayed cleanup of VTK observers
        // It's not an error, just ignore it
        return;
    }

    int selectionModifier = getSelectionModifier();

    // For polygon selection modes, we need to get polygon points from the
    // interactor style, not from callData. vtkInteractorStyleDrawPolygon fires
    // SelectionChangedEvent without passing polygon data - we need to call
    // GetPolygonPoints() on the style.
    // Reference: ParaView vtkPVRenderView.cxx lines 919-935
    bool isPolygonMode =
            (m_mode == SelectionMode::SELECT_SURFACE_CELLS_POLYGON ||
             m_mode == SelectionMode::SELECT_SURFACE_POINTS_POLYGON ||
             m_mode == SelectionMode::SELECT_CUSTOM_POLYGON);

    if (isPolygonMode) {
        vtkInteractorStyleDrawPolygon* polygonStyle =
                vtkInteractorStyleDrawPolygon::SafeDownCast(m_selectionStyle);
        if (!polygonStyle) {
            CVLog::Warning(
                    "[cvRenderViewSelectionReaction] Polygon mode but style is "
                    "not vtkInteractorStyleDrawPolygon");

            // Try to get style from interactor directly
            if (m_interactor) {
                vtkInteractorObserver* currentStyle =
                        m_interactor->GetInteractorStyle();
                polygonStyle = vtkInteractorStyleDrawPolygon::SafeDownCast(
                        currentStyle);
            }

            if (!polygonStyle) {
                return;
            }
        }

        // Get polygon points from the style (ParaView approach)
        std::vector<vtkVector2i> points = polygonStyle->GetPolygonPoints();

        if (points.size() < 3) {
            CVLog::Warning(QString("[cvRenderViewSelectionReaction] Polygon "
                                   "has fewer than 3 "
                                   "vertices (%1), ignoring")
                                   .arg(points.size()));
            return;
        }

        // Construct vtkIntArray with polygon points (like ParaView)
        vtkSmartPointer<vtkIntArray> polygonPointsArray =
                vtkSmartPointer<vtkIntArray>::New();
        polygonPointsArray->SetNumberOfComponents(2);
        polygonPointsArray->SetNumberOfTuples(
                static_cast<vtkIdType>(points.size()));
        for (unsigned int j = 0; j < points.size(); ++j) {
            const vtkVector2i& v = points[j];
            int pos[2] = {v[0], v[1]};
            polygonPointsArray->SetTypedTuple(j, pos);
        }

        // Handle polygon selection
        switch (m_mode) {
            case SelectionMode::SELECT_SURFACE_CELLS_POLYGON:
                selectPolygonCells(polygonPointsArray, selectionModifier);
                break;
            case SelectionMode::SELECT_SURFACE_POINTS_POLYGON:
                selectPolygonPoints(polygonPointsArray, selectionModifier);
                break;
            case SelectionMode::SELECT_CUSTOM_POLYGON:
                emit selectedCustomPolygon(polygonPointsArray);
                break;
            default:
                break;
        }

        // Polygon selection is complete - end the selection mode
        // Reference: ParaView behavior - polygon selection auto-exits after
        // completion
        endSelection();
        return;
    }

    // For non-polygon modes, we need valid callData
    if (!callData) {
        CVLog::Warning(
                "[cvRenderViewSelectionReaction] selectionChanged: callData is "
                "null");
        return;
    }

    int* region = reinterpret_cast<int*>(callData);

    // Handle based on mode
    // Reference: lines 550-598
    switch (m_mode) {
        case SelectionMode::SELECT_SURFACE_CELLS:
            selectCellsOnSurface(region, selectionModifier);
            break;

        case SelectionMode::SELECT_SURFACE_POINTS:
            selectPointsOnSurface(region, selectionModifier);
            break;

            // NOTE: Interactive modes are NOT handled here!
            // They use LeftButtonReleaseEvent and are handled in
            // onLeftButtonRelease() Reference: ParaView
            // pqRenderViewSelectionReaction.cxx lines 938-1000

        case SelectionMode::SELECT_FRUSTUM_CELLS:
            selectFrustumCells(region, selectionModifier);
            break;

        case SelectionMode::SELECT_FRUSTUM_POINTS:
            selectFrustumPoints(region, selectionModifier);
            break;

        case SelectionMode::SELECT_BLOCKS:
            selectBlock(region, selectionModifier);
            break;

        case SelectionMode::ZOOM_TO_BOX:
            // Zoom handled separately
            break;

        default:
            break;
    }

    // Only end selection for non-interactive modes
    // Interactive modes should stay active to allow cumulative selection
    // Reference: ParaView pqRenderViewSelectionReaction.cxx lines 602-612
    bool isInteractiveMode =
            (m_mode == SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY ||
             m_mode == SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY ||
             m_mode == SelectionMode::SELECT_SURFACE_CELLDATA_INTERACTIVELY ||
             m_mode == SelectionMode::SELECT_SURFACE_POINTDATA_INTERACTIVELY);

    if (!isInteractiveMode) {
        endSelection();
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::onMouseMove() {
    // Reference: pqRenderViewSelectionReaction::onMouseMove()
    // lines 606-642

    if (m_disablePreSelection) {
        return;
    }

    switch (m_mode) {
        case SelectionMode::HOVER_POINTS_TOOLTIP:
        case SelectionMode::HOVER_CELLS_TOOLTIP:
            m_mouseMovingTimer.start(TOOLTIP_WAITING_TIME);
            m_mouseMoving = true;
            preSelection();
            break;

        case SelectionMode::SELECT_SURFACE_CELLDATA_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTDATA_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY:
            // Use preSelection for highlighting
            preSelection();
            break;

        case SelectionMode::SELECT_SURFACE_CELLS_POLYGON:
        case SelectionMode::SELECT_SURFACE_POINTS_POLYGON:
        case SelectionMode::SELECT_CUSTOM_POLYGON:
            // Polygon modes: Mouse events handled by
            // vtkInteractorStyleDrawPolygon No action needed here
            break;

        default:
            CVLog::Warning(
                    "[cvRenderViewSelectionReaction] Invalid call to "
                    "onMouseMove");
            break;
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::onLeftButtonPress() {
    // Record starting position for ZoomToBox
    if (ActiveReaction != this) {
        return;
    }

    if (!m_interactor) {
        return;
    }

    // For ZoomToBox mode, track the start position
    if (m_mode == SelectionMode::ZOOM_TO_BOX) {
        int* pos = m_interactor->GetEventPosition();
        if (pos) {
            m_zoomStartPosition[0] = pos[0];
            m_zoomStartPosition[1] = pos[1];
            m_zoomTracking = true;
        }
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::onLeftButtonRelease() {
    // Reference: pqRenderViewSelectionReaction::onLeftButtonRelease()
    // lines 938-1010

    if (ActiveReaction != this) {
        return;
    }

    if (!m_interactor) {
        return;
    }

    int x = m_interactor->GetEventPosition()[0];
    int y = m_interactor->GetEventPosition()[1];

    if (x < 0 || y < 0) {
        return;
    }

    // Handle ZoomToBox mode
    if (m_mode == SelectionMode::ZOOM_TO_BOX) {
        if (m_zoomTracking) {
            // Emit zoom completed signal
            // Note: The actual zoom is performed by
            // vtkInteractorStyleRubberBandZoom
            int xmin = std::min(m_zoomStartPosition[0], x);
            int ymin = std::min(m_zoomStartPosition[1], y);
            int xmax = std::max(m_zoomStartPosition[0], x);
            int ymax = std::max(m_zoomStartPosition[1], y);

            emit zoomToBoxCompleted(xmin, ymin, xmax, ymax);
            m_zoomTracking = false;
        }
        // End selection after zoom is complete
        endSelection();
        return;
    }

    int selectionModifier = getSelectionModifier();

    // In interactive mode, default is ADDITION (not replace)
    // Reference: lines 958-962
    if (selectionModifier ==
        static_cast<int>(SelectionModifier::SELECTION_DEFAULT)) {
        selectionModifier =
                static_cast<int>(SelectionModifier::SELECTION_ADDITION);
    }

    int region[4] = {x, y, x, y};

    switch (m_mode) {
        case SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_CELLDATA_INTERACTIVELY:
            selectCellsOnSurface(region, selectionModifier);
            break;

        case SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTDATA_INTERACTIVELY:
            selectPointsOnSurface(region, selectionModifier);
            break;

        default:
            CVLog::Warning(QString("[cvRenderViewSelectionReaction::"
                                   "onLeftButtonRelease] "
                                   "Mode %1 not handled in switch")
                                   .arg(static_cast<int>(m_mode)));
            break;
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::onWheelRotate() {
    // Reference: pqRenderViewSelectionReaction::onWheelRotate()
    // lines 1063-1069
    if (ActiveReaction == this) {
        // CRITICAL: Invalidate selection cache after zoom
        // Camera changed, so cached selection buffer is no longer valid
        cvSelectionPipeline* pipeline = getSelectionPipeline();
        if (pipeline) {
            pipeline->invalidateCachedSelection();
        }
        onMouseMove();
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::onRightButtonPressed() {
    // Reference: pqRenderViewSelectionReaction::onRightButtonPressed()
    // lines 1072-1075
    m_disablePreSelection = true;
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::onRightButtonRelease() {
    // Reference: pqRenderViewSelectionReaction::onRightButtonRelease()
    // lines 1077-1080
    m_disablePreSelection = false;

    // CRITICAL: Invalidate selection cache after rotate/zoom
    // Camera changed, so cached selection buffer is no longer valid
    cvSelectionPipeline* pipeline = getSelectionPipeline();
    if (pipeline) {
        pipeline->invalidateCachedSelection();
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::onMiddleButtonPressed() {
    // Middle button is used for panning in vtkInteractorStyleRubberBand3D
    m_disablePreSelection = true;
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::onMiddleButtonRelease() {
    // Middle button released after panning
    m_disablePreSelection = false;

    // CRITICAL: Invalidate selection cache after pan
    // Camera changed, so cached selection buffer is no longer valid
    cvSelectionPipeline* pipeline = getSelectionPipeline();
    if (pipeline) {
        pipeline->invalidateCachedSelection();
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::preSelection() {
    // Reference: pqRenderViewSelectionReaction::preSelection()
    // lines 645-765

    if (ActiveReaction != this || !m_interactor || !m_renderer) {
        return;
    }

    int x = m_interactor->GetEventPosition()[0];
    int y = m_interactor->GetEventPosition()[1];
    int* size = m_interactor->GetSize();

    m_mousePosition[0] = x;
    m_mousePosition[1] = y;

    // Check bounds
    if (x < 0 || y < 0 || x >= size[0] || y >= size[1]) {
        // Hide highlight when cursor is out of view
        cvSelectionHighlighter* highlighter = getSelectionHighlighter();
        if (highlighter) {
            highlighter->clearHoverHighlight();
        }
        updateTooltip();
        return;
    }

    // Perform single-pixel selection for highlighting
    cvSelectionPipeline* pipeline = getSelectionPipeline();
    if (!pipeline) {
        return;
    }

    bool selectCells = isSelectingCells();
    cvSelectionPipeline::PixelSelectionInfo info =
            pipeline->getPixelSelectionInfo(x, y, selectCells);

    if (info.valid && info.attributeID >= 0) {
        m_hoveredId = info.attributeID;
        m_currentPolyData = info.polyData;

        // Show highlight using highlightElement
        cvSelectionHighlighter* highlighter = getSelectionHighlighter();
        if (highlighter && info.polyData) {
            // fieldAssociation: 0 for cells, 1 for points
            int fieldAssociation = selectCells ? 0 : 1;
            highlighter->highlightElement(info.polyData, info.attributeID,
                                          fieldAssociation);
        }
    } else {
        m_hoveredId = -1;
        m_currentPolyData = nullptr;

        cvSelectionHighlighter* highlighter = getSelectionHighlighter();
        if (highlighter) {
            highlighter->clearHoverHighlight();
        }
    }

    // ParaView-style: Trigger render after selection update
    // Reference: pqRenderViewSelectionReaction::fastPreSelection() line 871
    if (m_interactor) {
        vtkRenderWindow* renWin = m_interactor->GetRenderWindow();
        if (renWin) {
            renWin->Render();
        }
    }

    updateTooltip();
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::fastPreSelection() {
    // Reference: pqRenderViewSelectionReaction::fastPreSelection()
    // lines 768-872

    // For now, use regular preSelection
    // Fast pre-selection optimization can be added later
    preSelection();
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::updateTooltip() {
    // Reference: pqRenderViewSelectionReaction::UpdateTooltip()
    // lines 891-935

    if (!isTooltipMode()) {
        return;
    }

    // Don't show tooltip while mouse is moving
    if (m_mouseMoving) {
        QToolTip::hideText();
        return;
    }

    // Check if we have valid hover data
    if (m_hoveredId < 0) {
        QToolTip::hideText();
        return;
    }

    if (!m_currentPolyData) {
        QToolTip::hideText();
        return;
    }

    // Build tooltip text
    QString tooltipText;
    bool selectCells = isSelectingCells();

    // Helper lambda to format array values for tooltip
    // ParaView reference: vtkSMTooltipSelectionPipeline.cxx lines 246-264
    // ParaView shows ALL components for multi-component arrays in format:
    // "Name: (v1, v2, ...)"
    auto formatArrayValue = [](vtkDataArray* arr, vtkIdType id) -> QString {
        if (!arr || !arr->GetName()) return QString();

        QString name = QString::fromUtf8(arr->GetName());
        int numComp = arr->GetNumberOfComponents();

        // Skip internal VTK arrays (ParaView skips vtkOriginalPointIds/CellIds)
        if (name.startsWith("vtk", Qt::CaseInsensitive) ||
            name == "vtkOriginalPointIds" || name == "vtkOriginalCellIds") {
            return QString();
        }

        // ParaView format: always show all components
        // Single component: "Name: value"
        // Multi-component: "Name: (v1, v2, ...)"
        if (numComp == 1) {
            double value = arr->GetTuple1(id);
            return QString("\n  %1: %2").arg(name).arg(value, 0, 'g', 6);
        }

        // Multi-component arrays - check for special handling
        double* tuple = arr->GetTuple(id);

        // RGB/RGBA colors: show as integer values
        if ((numComp == 3 || numComp == 4) &&
            (name.compare("RGB", Qt::CaseInsensitive) == 0 ||
             name.compare("RGBA", Qt::CaseInsensitive) == 0 ||
             name.compare("Colors", Qt::CaseInsensitive) == 0)) {
            QString result = QString("\n  %1: (").arg(name);
            for (int i = 0; i < numComp; ++i) {
                if (i > 0) result += ", ";
                result += QString::number(static_cast<int>(tuple[i]));
            }
            result += ")";
            return result;
        }

        // All other multi-component arrays (Normals, TCoords, etc.)
        // ParaView shows all components with full precision
        QString result = QString("\n  %1: (").arg(name);
        for (int i = 0; i < numComp; ++i) {
            if (i > 0) result += ", ";
            result += QString::number(tuple[i], 'g', 6);
        }
        result += ")";
        return result;
    };

    // Get dataset name from FieldData (ParaView-style)
    QString datasetName;
    vtkFieldData* fieldData = m_currentPolyData->GetFieldData();
    if (fieldData) {
        vtkStringArray* nameArray = vtkStringArray::SafeDownCast(
                fieldData->GetAbstractArray("DatasetName"));
        if (nameArray && nameArray->GetNumberOfTuples() > 0) {
            datasetName = QString::fromStdString(nameArray->GetValue(0));
        }
    }

    if (selectCells) {
        vtkIdType numCells = m_currentPolyData->GetNumberOfCells();
        if (m_hoveredId >= 0 && m_hoveredId < numCells) {
            // ParaView format: Dataset name first (if available)
            if (!datasetName.isEmpty()) {
                tooltipText = datasetName;
                tooltipText += QString("\n  Id: %1").arg(m_hoveredId);
            } else {
                tooltipText = QString("Cell ID: %1").arg(m_hoveredId);
            }

            // Add cell type (ParaView-style)
            vtkCell* cell = m_currentPolyData->GetCell(m_hoveredId);
            if (cell) {
                QString cellType;
                switch (cell->GetCellType()) {
                    case VTK_VERTEX:
                        cellType = "Vertex";
                        break;
                    case VTK_POLY_VERTEX:
                        cellType = "Poly Vertex";
                        break;
                    case VTK_LINE:
                        cellType = "Line";
                        break;
                    case VTK_POLY_LINE:
                        cellType = "Poly Line";
                        break;
                    case VTK_TRIANGLE:
                        cellType = "Triangle";
                        break;
                    case VTK_TRIANGLE_STRIP:
                        cellType = "Triangle Strip";
                        break;
                    case VTK_POLYGON:
                        cellType = "Polygon";
                        break;
                    case VTK_QUAD:
                        cellType = "Quad";
                        break;
                    case VTK_TETRA:
                        cellType = "Tetra";
                        break;
                    case VTK_HEXAHEDRON:
                        cellType = "Hexahedron";
                        break;
                    default:
                        cellType = QString("Type %1").arg(cell->GetCellType());
                        break;
                }
                tooltipText += QString("\n  Type: %1").arg(cellType);

                // Add number of points
                vtkIdType npts = cell->GetNumberOfPoints();
                tooltipText += QString("\n  Number of Points: %1").arg(npts);

                // Add cell center
                if (npts > 0) {
                    double center[3] = {0, 0, 0};
                    for (vtkIdType i = 0; i < npts; ++i) {
                        double pt[3];
                        m_currentPolyData->GetPoint(cell->GetPointId(i), pt);
                        center[0] += pt[0];
                        center[1] += pt[1];
                        center[2] += pt[2];
                    }
                    center[0] /= npts;
                    center[1] /= npts;
                    center[2] /= npts;
                    tooltipText += QString("\n  Center: (%1, %2, %3)")
                                           .arg(center[0], 0, 'g', 6)
                                           .arg(center[1], 0, 'g', 6)
                                           .arg(center[2], 0, 'g', 6);
                }
            }

            // Add cell data arrays to tooltip
            // Track shown color arrays to avoid duplicates (RGB and Colors are
            // the same)
            vtkCellData* cellData = m_currentPolyData->GetCellData();
            if (cellData) {
                bool hasShownColorArray = false;
                for (int i = 0; i < cellData->GetNumberOfArrays(); ++i) {
                    vtkDataArray* arr = cellData->GetArray(i);
                    if (!arr || !arr->GetName()) continue;

                    QString arrName = QString::fromUtf8(arr->GetName());
                    int numComp = arr->GetNumberOfComponents();

                    // Skip duplicate color arrays (RGB and Colors are the same
                    // thing)
                    if ((numComp == 3 || numComp == 4) &&
                        (arrName.compare("RGB", Qt::CaseInsensitive) == 0 ||
                         arrName.compare("RGBA", Qt::CaseInsensitive) == 0 ||
                         arrName.compare("Colors", Qt::CaseInsensitive) == 0)) {
                        if (hasShownColorArray) {
                            continue;  // Skip duplicate color array
                        }
                        hasShownColorArray = true;
                    }

                    QString arrayText = formatArrayValue(arr, m_hoveredId);
                    if (!arrayText.isEmpty()) {
                        tooltipText += arrayText;
                    }
                }
            }
        }
    } else {
        vtkIdType numPoints = m_currentPolyData->GetNumberOfPoints();
        if (m_hoveredId >= 0 && m_hoveredId < numPoints) {
            double pt[3];
            m_currentPolyData->GetPoint(m_hoveredId, pt);

            // ParaView format: Dataset name first (if available)
            if (!datasetName.isEmpty()) {
                tooltipText = datasetName;
                tooltipText += QString("\n  Id: %1").arg(m_hoveredId);
                tooltipText += QString("\n  Coords: (%1, %2, %3)")
                                       .arg(pt[0], 0, 'g', 6)
                                       .arg(pt[1], 0, 'g', 6)
                                       .arg(pt[2], 0, 'g', 6);
            } else {
                tooltipText = QString("Point ID: %1\nPosition: (%2, %3, %4)")
                                      .arg(m_hoveredId)
                                      .arg(pt[0], 0, 'f', 6)
                                      .arg(pt[1], 0, 'f', 6)
                                      .arg(pt[2], 0, 'f', 6);
            }

            // Add point data arrays to tooltip
            // Track shown color arrays to avoid duplicates (RGB and Colors are
            // the same)
            vtkPointData* pointData = m_currentPolyData->GetPointData();
            if (pointData) {
                bool hasShownColorArray = false;
                for (int i = 0; i < pointData->GetNumberOfArrays(); ++i) {
                    vtkDataArray* arr = pointData->GetArray(i);
                    if (!arr || !arr->GetName()) continue;

                    QString arrName = QString::fromUtf8(arr->GetName());
                    int numComp = arr->GetNumberOfComponents();

                    // Skip duplicate color arrays (RGB and Colors are the same
                    // thing)
                    if ((numComp == 3 || numComp == 4) &&
                        (arrName.compare("RGB", Qt::CaseInsensitive) == 0 ||
                         arrName.compare("RGBA", Qt::CaseInsensitive) == 0 ||
                         arrName.compare("Colors", Qt::CaseInsensitive) == 0)) {
                        if (hasShownColorArray) {
                            continue;  // Skip duplicate color array
                        }
                        hasShownColorArray = true;
                    }

                    QString arrayText = formatArrayValue(arr, m_hoveredId);
                    if (!arrayText.isEmpty()) {
                        tooltipText += arrayText;
                    }
                }
            }
        }
    }

    if (!tooltipText.isEmpty()) {
        // Get widget position for tooltip
        // Reference: ParaView uses DPI scaling - we should too
        QWidget* widget = ecvDisplayTools::GetCurrentScreen();
        if (widget) {
            // Take DPI scaling into account (ParaView-style)
            qreal dpr = widget->devicePixelRatioF();

            // Convert VTK coordinates (origin bottom-left) to Qt coordinates
            // (origin top-left) VTK Y increases upward, Qt Y increases downward
            QPoint localPos(static_cast<int>(m_mousePosition[0] / dpr),
                            widget->height() -
                                    static_cast<int>(m_mousePosition[1] / dpr));
            QPoint globalPos = widget->mapToGlobal(localPos);

            QToolTip::showText(globalPos, tooltipText);
            m_plainTooltipText = tooltipText;
        }
    } else {
        QToolTip::hideText();
    }
}

//-----------------------------------------------------------------------------
int cvRenderViewSelectionReaction::getSelectionModifier() {
    // Reference: pqSelectionReaction::getSelectionModifier() and
    // pqRenderViewSelectionReaction::getSelectionModifier()
    // lines 1013-1035

    // First check modifier group
    // IMPORTANT: We cannot use QActionGroup::checkedAction() since the
    // ModifierGroup may not be exclusive (exclusive=false). We need to iterate
    // through all actions to find which one is checked, following ParaView's
    // pqSelectionReaction pattern.
    int selectionModifier =
            static_cast<int>(SelectionModifier::SELECTION_DEFAULT);

    if (m_modifierGroup) {
        // ParaView pattern: iterate through all actions in the group
        for (QAction* maction : m_modifierGroup->actions()) {
            if (maction && maction->isChecked() && maction->data().isValid()) {
                selectionModifier = maction->data().toInt();
                CVLog::PrintDebug(
                        QString("[getSelectionModifier] From modifierGroup: "
                                "action='%1', modifier=%2")
                                .arg(maction->text())
                                .arg(selectionModifier));
                break;  // Only one should be checked at a time
            }
        }
    }

    // Check keyboard modifiers (override button selection)
    if (m_interactor) {
        bool ctrl = m_interactor->GetControlKey() == 1;
        bool shift = m_interactor->GetShiftKey() == 1;

        if (ctrl && shift) {
            selectionModifier =
                    static_cast<int>(SelectionModifier::SELECTION_TOGGLE);
        } else if (ctrl) {
            selectionModifier =
                    static_cast<int>(SelectionModifier::SELECTION_ADDITION);
        } else if (shift) {
            selectionModifier =
                    static_cast<int>(SelectionModifier::SELECTION_SUBTRACTION);
        }
    }

    return selectionModifier;
}

//-----------------------------------------------------------------------------
bool cvRenderViewSelectionReaction::isCompatible(SelectionMode otherMode) {
    // Reference: pqRenderViewSelectionReaction::isCompatible()
    // lines 1037-1060

    if (m_mode == otherMode) {
        return true;
    }

    // Cell selection modes are compatible with each other
    if ((m_mode == SelectionMode::SELECT_SURFACE_CELLS ||
         m_mode == SelectionMode::SELECT_SURFACE_CELLS_POLYGON ||
         m_mode == SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY ||
         m_mode == SelectionMode::SELECT_FRUSTUM_CELLS) &&
        (otherMode == SelectionMode::SELECT_SURFACE_CELLS ||
         otherMode == SelectionMode::SELECT_SURFACE_CELLS_POLYGON ||
         otherMode == SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY ||
         otherMode == SelectionMode::SELECT_FRUSTUM_CELLS)) {
        return true;
    }

    // Point selection modes are compatible with each other
    if ((m_mode == SelectionMode::SELECT_SURFACE_POINTS ||
         m_mode == SelectionMode::SELECT_SURFACE_POINTS_POLYGON ||
         m_mode == SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY ||
         m_mode == SelectionMode::SELECT_FRUSTUM_POINTS) &&
        (otherMode == SelectionMode::SELECT_SURFACE_POINTS ||
         otherMode == SelectionMode::SELECT_SURFACE_POINTS_POLYGON ||
         otherMode == SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY ||
         otherMode == SelectionMode::SELECT_FRUSTUM_POINTS)) {
        return true;
    }

    return false;
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::cleanupObservers() {
    // Reference: pqRenderViewSelectionReaction::cleanupObservers()
    // lines 130-141
    // CRITICAL: Remove observers from their respective objects
    // m_observedObject is typically m_interactor for most events
    // m_styleObserverId is used for SelectionChangedEvent on m_selectionStyle

    // Remove observers from the main observed object (interactor)
    for (size_t i = 0; i < sizeof(m_observerIds) / sizeof(m_observerIds[0]);
         ++i) {
        if (m_observerIds[i] > 0 && m_observedObject) {
            m_observedObject->RemoveObserver(m_observerIds[i]);
        }
        m_observerIds[i] = 0;
    }

    // Remove observer from selection style separately (if any)
    if (m_styleObserverId > 0 && m_selectionStyle) {
        m_selectionStyle->RemoveObserver(m_styleObserverId);
        m_styleObserverId = 0;
    }

    m_observedObject = nullptr;
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::uncheckSelectionModifiers() {
    if (m_modifierGroup) {
        QAction* checkedAction = m_modifierGroup->checkedAction();
        if (checkedAction) {
            checkedAction->setChecked(false);
        }
    }
}

//-----------------------------------------------------------------------------
// Selection execution methods
//-----------------------------------------------------------------------------

void cvRenderViewSelectionReaction::selectCellsOnSurface(
        int region[4], int selectionModifier) {
    cvSelectionPipeline* pipeline = getSelectionPipeline();
    if (!pipeline) {
        CVLog::Warning(
                "[cvRenderViewSelectionReaction::selectCellsOnSurface] "
                "No pipeline available");
        return;
    }

    // CRITICAL: Temporarily hide highlight actors before hardware selection
    // This prevents the highlight actor from occluding the original data in
    // the depth buffer, which would cause subtract selection to select wrong
    // points.
    cvSelectionHighlighter* highlighter = getSelectionHighlighter();
    if (highlighter) {
        highlighter->setHighlightsVisible(false);
    }

    cvSelectionData selection = pipeline->selectCellsOnSurface(region);

    // Restore highlight visibility
    if (highlighter) {
        highlighter->setHighlightsVisible(true);
    }

    finalizeSelection(selection, selectionModifier, "SurfaceCells");
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::selectPointsOnSurface(
        int region[4], int selectionModifier) {
    cvSelectionPipeline* pipeline = getSelectionPipeline();
    if (!pipeline) {
        CVLog::Warning(
                "[cvRenderViewSelectionReaction::selectPointsOnSurface] "
                "No pipeline available");
        return;
    }

    // CRITICAL: Temporarily hide highlight actors before hardware selection
    // This prevents the highlight actor from occluding the original data in
    // the depth buffer, which would cause subtract selection to select wrong
    // points.
    cvSelectionHighlighter* highlighter = getSelectionHighlighter();
    if (highlighter) {
        highlighter->setHighlightsVisible(false);
    }

    cvSelectionData selection = pipeline->selectPointsOnSurface(region);

    // Restore highlight visibility
    if (highlighter) {
        highlighter->setHighlightsVisible(true);
    }

    finalizeSelection(selection, selectionModifier, "SurfacePoints");
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::selectFrustumCells(int region[4],
                                                       int selectionModifier) {
    // Frustum selection implementation
    // For now, use surface selection as fallback
    selectCellsOnSurface(region, selectionModifier);
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::selectFrustumPoints(int region[4],
                                                        int selectionModifier) {
    // Frustum selection implementation
    // For now, use surface selection as fallback
    selectPointsOnSurface(region, selectionModifier);
}

//-----------------------------------------------------------------------------
// Unified selection finalization (ParaView-style)
// This method handles combining with existing selection and updating highlights
//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::finalizeSelection(
        const cvSelectionData& newSelection,
        int selectionModifier,
        const QString& description) {
    Q_UNUSED(description);

    if (newSelection.isEmpty()) {
        return;
    }

    cvViewSelectionManager* manager = cvViewSelectionManager::instance();
    if (!manager) {
        return;
    }
    cvSelectionData currentSel = manager->currentSelection();

    // Combine with existing selection based on modifier
    cvSelectionData combined = cvSelectionPipeline::combineSelections(
            currentSel, newSelection,
            static_cast<cvSelectionPipeline::CombineOperation>(
                    selectionModifier));

    // Update source object from PCLVis for direct extraction
    // This allows extraction operations to use the original ccPointCloud/ccMesh
    // instead of converting from VTK polydata
    PclUtils::PCLVis* pclVis = getPCLVis();
    if (pclVis && combined.hasActorInfo()) {
        // Get viewID from the primary (front-most) actor
        vtkActor* primaryActor = combined.primaryActor();
        if (primaryActor) {
            std::string viewID = pclVis->getIdByActor(primaryActor);
            if (!viewID.empty()) {
                ccHObject* sourceObj = pclVis->getSourceObject(viewID);
                if (sourceObj) {
                    manager->setSourceObject(sourceObj);
                } else {
                    CVLog::Warning(
                            QString("[finalizeSelection] No source object "
                                    "found "
                                    "for viewID='%1'")
                                    .arg(QString::fromStdString(viewID)));
                }
            } else {
                CVLog::Warning(
                        "[finalizeSelection] Could not find viewID for "
                        "primary actor");
            }
        }
    }

    // Update manager's current selection
    manager->setCurrentSelection(combined);

    // Update persistent SELECTED highlight (ParaView-style)
    // This shows accumulated selection as a persistent visual indicator
    cvSelectionHighlighter* highlighter = getSelectionHighlighter();
    if (highlighter && !combined.isEmpty()) {
        highlighter->highlightSelection(combined,
                                        cvSelectionHighlighter::SELECTED);
    } else if (highlighter && combined.isEmpty()) {
        // Clear highlight if selection is empty
        highlighter->clearHighlights();
    }

    emit selectionFinished(combined);
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::selectPolygonCells(vtkIntArray* polygon,
                                                       int selectionModifier) {
    cvSelectionPipeline* pipeline = getSelectionPipeline();
    if (!pipeline || !polygon) return;

    // CRITICAL: Temporarily hide highlight actors before hardware selection
    cvSelectionHighlighter* highlighter = getSelectionHighlighter();
    if (highlighter) {
        highlighter->setHighlightsVisible(false);
    }

    cvSelectionData selection = pipeline->selectCellsInPolygon(polygon);

    if (highlighter) {
        highlighter->setHighlightsVisible(true);
    }

    finalizeSelection(selection, selectionModifier, "PolygonCells");
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::selectPolygonPoints(vtkIntArray* polygon,
                                                        int selectionModifier) {
    cvSelectionPipeline* pipeline = getSelectionPipeline();
    if (!pipeline || !polygon) return;

    // CRITICAL: Temporarily hide highlight actors before hardware selection
    cvSelectionHighlighter* highlighter = getSelectionHighlighter();
    if (highlighter) {
        highlighter->setHighlightsVisible(false);
    }

    cvSelectionData selection = pipeline->selectPointsInPolygon(polygon);

    if (highlighter) {
        highlighter->setHighlightsVisible(true);
    }

    finalizeSelection(selection, selectionModifier, "PolygonPoints");
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::selectBlock(int region[4],
                                                int selectionModifier) {
    // Block selection - for now delegate to cell selection
    selectCellsOnSurface(region, selectionModifier);
}

//-----------------------------------------------------------------------------
// Internal helpers
//-----------------------------------------------------------------------------

PclUtils::PCLVis* cvRenderViewSelectionReaction::getPCLVis() const {
    if (!m_viewer) return nullptr;

    // Dynamic cast to PCLVis
    return dynamic_cast<PclUtils::PCLVis*>(m_viewer);
}

//-----------------------------------------------------------------------------
cvSelectionPipeline* cvRenderViewSelectionReaction::getSelectionPipeline()
        const {
    cvViewSelectionManager* manager = cvViewSelectionManager::instance();
    return manager ? manager->getPipeline() : nullptr;
}

//-----------------------------------------------------------------------------
cvSelectionHighlighter* cvRenderViewSelectionReaction::getSelectionHighlighter()
        const {
    cvViewSelectionManager* manager = cvViewSelectionManager::instance();
    return manager ? manager->getHighlighter() : nullptr;
}

//-----------------------------------------------------------------------------
bool cvRenderViewSelectionReaction::isSelectingCells() const {
    switch (m_mode) {
        case SelectionMode::SELECT_SURFACE_CELLS:
        case SelectionMode::SELECT_SURFACE_CELLS_POLYGON:
        case SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_CELLDATA_INTERACTIVELY:
        case SelectionMode::SELECT_FRUSTUM_CELLS:
        case SelectionMode::SELECT_BLOCKS:
        case SelectionMode::SELECT_FRUSTUM_BLOCKS:
        case SelectionMode::HOVER_CELLS_TOOLTIP:
            return true;
        default:
            return false;
    }
}

//-----------------------------------------------------------------------------
bool cvRenderViewSelectionReaction::isInteractiveMode() const {
    switch (m_mode) {
        case SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_CELLDATA_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTDATA_INTERACTIVELY:
            return true;
        default:
            return false;
    }
}

//-----------------------------------------------------------------------------
bool cvRenderViewSelectionReaction::isTooltipMode() const {
    return m_mode == SelectionMode::HOVER_POINTS_TOOLTIP ||
           m_mode == SelectionMode::HOVER_CELLS_TOOLTIP;
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::setCursor(const QCursor& cursor) {
    QWidget* widget = ecvDisplayTools::GetCurrentScreen();
    if (widget) {
        widget->setCursor(cursor);
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::unsetCursor() {
    QWidget* widget = ecvDisplayTools::GetCurrentScreen();
    if (widget) {
        widget->setCursor(Qt::ArrowCursor);  // Explicitly set arrow cursor
        widget->unsetCursor();  // Also unset to remove any override
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::storeCurrentStyle() {
    if (m_interactor) {
        vtkInteractorObserver* currentStyle =
                m_interactor->GetInteractorStyle();
        vtkInteractorStyle* style =
                vtkInteractorStyle::SafeDownCast(currentStyle);

        // Don't store selection-specific styles as the "previous" style
        // This prevents issues when switching between selection modes
        if (style) {
            // Check if current style is a selection style (RubberBand3D,
            // RubberBandZoom, DrawPolygon) - these should not be stored as
            // "previous"
            bool isSelectionStyle =
                    (vtkInteractorStyleRubberBand3D::SafeDownCast(style) !=
                     nullptr) ||
                    (vtkInteractorStyleRubberBandZoom::SafeDownCast(style) !=
                     nullptr) ||
                    (vtkInteractorStyleDrawPolygon::SafeDownCast(style) !=
                     nullptr);

            if (!isSelectionStyle) {
                m_previousStyle = style;
                CVLog::PrintDebug(QString("[storeCurrentStyle] Stored "
                                          "non-selection style: %1")
                                          .arg(style->GetClassName()));
            }
        }
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::restoreStyle() {
    if (m_interactor && m_previousStyle) {
        m_interactor->SetInteractorStyle(m_previousStyle);
        m_previousStyle = nullptr;
    } else if (m_interactor) {
        // Fallback: If no previous style saved, try to get PCLVis's default
        // style
        PclUtils::PCLVis* pclVis = getPCLVis();
        if (pclVis) {
            // PCLVis typically uses vtkInteractorStyleTrackballCamera
            auto defaultStyle =
                    vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
            if (m_renderer) {
                defaultStyle->SetDefaultRenderer(m_renderer);
            }
            m_interactor->SetInteractorStyle(defaultStyle);
        }
    }
    m_selectionStyle = nullptr;
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::setRubberBand3DStyle(
        bool renderOnMouseMove) {
    // Helper: Create and set vtkInteractorStyleRubberBand3D
    // This style: Left=RubberBand, Middle=Pan, Right=Zoom (NO rotation!)
    auto style = vtkSmartPointer<vtkInteractorStyleRubberBand3D>::New();
    if (!renderOnMouseMove) {
        style->RenderOnMouseMoveOff();
    }
    if (m_renderer) {
        style->SetDefaultRenderer(m_renderer);
    }
    m_selectionStyle = style;
    m_interactor->SetInteractorStyle(style);
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::setupInteractorStyle() {
    // Reference: pqRenderViewSelectionReaction::beginSelection() lines 338-437
    // ParaView uses INTERACTION_MODE_SELECTION which maps to
    // vtkInteractorStyleRubberBand3D This style: Left=RubberBand, Middle=Pan,
    // Right=Zoom (NO rotation!)

    if (!m_interactor) return;
    storeCurrentStyle();

    switch (m_mode) {
        case SelectionMode::SELECT_SURFACE_CELLS:
        case SelectionMode::SELECT_SURFACE_POINTS:
        case SelectionMode::SELECT_FRUSTUM_CELLS:
        case SelectionMode::SELECT_FRUSTUM_POINTS:
        case SelectionMode::SELECT_BLOCKS:
        case SelectionMode::SELECT_FRUSTUM_BLOCKS:
            // Rectangle selection - cross cursor
            setCursor(Qt::CrossCursor);
            setRubberBand3DStyle(false);  // No RenderOnMouseMove
            break;

        case SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_CELLDATA_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTDATA_INTERACTIVELY:
            // Interactive selection - cross cursor, no render on mouse move
            setCursor(Qt::CrossCursor);
            setRubberBand3DStyle(false);
            break;

        case SelectionMode::HOVER_POINTS_TOOLTIP:
        case SelectionMode::HOVER_CELLS_TOOLTIP:
            // Tooltip mode - default arrow cursor, no render on mouse move
            unsetCursor();
            setRubberBand3DStyle(false);
            break;

        case SelectionMode::ZOOM_TO_BOX: {
            setCursor(m_zoomCursor);
            auto zoomStyle =
                    vtkSmartPointer<vtkInteractorStyleRubberBandZoom>::New();
            m_selectionStyle = zoomStyle;
            m_interactor->SetInteractorStyle(zoomStyle);
            break;
        }

        case SelectionMode::SELECT_SURFACE_CELLS_POLYGON:
        case SelectionMode::SELECT_SURFACE_POINTS_POLYGON:
        case SelectionMode::SELECT_CUSTOM_POLYGON: {
            // Polygon selection
            setCursor(Qt::PointingHandCursor);
            vtkSmartPointer<vtkInteractorStyleDrawPolygon> polygonStyle =
                    vtkSmartPointer<vtkInteractorStyleDrawPolygon>::New();
            if (m_renderer) {
                polygonStyle->SetDefaultRenderer(m_renderer);
            }
            m_selectionStyle = polygonStyle;
            m_interactor->SetInteractorStyle(polygonStyle);
            break;
        }

        default:
            break;
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::setupObservers() {
    // Reference: pqRenderViewSelectionReaction::beginSelection() lines 439-491
    if (!m_interactor) return;

    m_observedObject = m_interactor;

    switch (m_mode) {
        // One-shot commands - no observers needed
        // Reference: lines 449-452
        case SelectionMode::CLEAR_SELECTION:
        case SelectionMode::GROW_SELECTION:
        case SelectionMode::SHRINK_SELECTION:
            break;

        case SelectionMode::ZOOM_TO_BOX:
            m_observerIds[0] = m_interactor->AddObserver(
                    vtkCommand::LeftButtonPressEvent, this,
                    &cvRenderViewSelectionReaction::vtkOnLeftButtonPress);
            m_observerIds[1] = m_interactor->AddObserver(
                    vtkCommand::LeftButtonReleaseEvent, this,
                    &cvRenderViewSelectionReaction::vtkOnLeftButtonRelease);
            break;

        case SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_CELLDATA_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTDATA_INTERACTIVELY:
            // Interactive selection: MouseMove + LeftButtonRelease for click
            // selection Plus camera movement observers for cache invalidation
            m_observerIds[0] = m_interactor->AddObserver(
                    vtkCommand::MouseMoveEvent, this,
                    &cvRenderViewSelectionReaction::vtkOnMouseMove);
            m_observerIds[1] = m_interactor->AddObserver(
                    vtkCommand::LeftButtonReleaseEvent, this,
                    &cvRenderViewSelectionReaction::vtkOnLeftButtonRelease);
            addCameraMovementObservers(2);  // Start from index 2
            break;

        case SelectionMode::HOVER_POINTS_TOOLTIP:
        case SelectionMode::HOVER_CELLS_TOOLTIP:
            // Tooltip mode: MouseMove only, plus camera movement observers
            // No LeftButtonRelease - tooltip doesn't do selection
            m_observerIds[0] = m_interactor->AddObserver(
                    vtkCommand::MouseMoveEvent, this,
                    &cvRenderViewSelectionReaction::vtkOnMouseMove);
            addCameraMovementObservers(1);  // Start from index 1
            break;

        default:
            // Rectangle selection: SelectionChangedEvent on style for
            // rubber-band completion Plus camera movement observers for cache
            // invalidation
            if (m_selectionStyle) {
                m_styleObserverId = m_selectionStyle->AddObserver(
                        vtkCommand::SelectionChangedEvent, this,
                        &cvRenderViewSelectionReaction::vtkOnSelectionChanged);
                addCameraMovementObservers(0);  // Start from index 0
            }
            break;
    }
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::addCameraMovementObservers(int startIndex) {
    // Add observers for camera movements that require selection cache
    // invalidation vtkInteractorStyleRubberBand3D: Middle=Pan, Right=Zoom,
    // Wheel=Zoom
    if (!m_interactor) return;

    m_observerIds[startIndex++] = m_interactor->AddObserver(
            vtkCommand::MouseWheelForwardEvent, this,
            &cvRenderViewSelectionReaction::vtkOnWheelRotate);
    m_observerIds[startIndex++] = m_interactor->AddObserver(
            vtkCommand::MouseWheelBackwardEvent, this,
            &cvRenderViewSelectionReaction::vtkOnWheelRotate);
    m_observerIds[startIndex++] = m_interactor->AddObserver(
            vtkCommand::RightButtonPressEvent, this,
            &cvRenderViewSelectionReaction::vtkOnRightButtonPressed);
    m_observerIds[startIndex++] = m_interactor->AddObserver(
            vtkCommand::RightButtonReleaseEvent, this,
            &cvRenderViewSelectionReaction::vtkOnRightButtonRelease);
    m_observerIds[startIndex++] = m_interactor->AddObserver(
            vtkCommand::MiddleButtonPressEvent, this,
            &cvRenderViewSelectionReaction::vtkOnMiddleButtonPressed);
    m_observerIds[startIndex] = m_interactor->AddObserver(
            vtkCommand::MiddleButtonReleaseEvent, this,
            &cvRenderViewSelectionReaction::vtkOnMiddleButtonRelease);
}

//-----------------------------------------------------------------------------
void cvRenderViewSelectionReaction::showInstructionDialog() {
    // Reference: pqRenderViewSelectionReaction::beginSelection()
    // Shows instruction dialogs for interactive/tooltip modes

    QString settingsKey, title, message;

    switch (m_mode) {
        case SelectionMode::HOVER_CELLS_TOOLTIP:
            settingsKey = "pqTooltipSelection";
            title = tr("Tooltip Selection Information");
            message = tr(
                    "You are entering tooltip selection mode to display cell "
                    "information. "
                    "Simply move the mouse point over the dataset to "
                    "interactively highlight "
                    "cells and display a tooltip with cell "
                    "information.\n\n"
                    "Use the 'Esc' key or the same toolbar button to exit this "
                    "mode.");
            break;

        case SelectionMode::HOVER_POINTS_TOOLTIP:
            settingsKey = "pqTooltipSelection";
            title = tr("Tooltip Selection Information");
            message =
                    tr("You are entering tooltip selection mode to display "
                       "points information. "
                       "Simply move the mouse point over the dataset to "
                       "interactively highlight "
                       "points and display a tooltip with points "
                       "information.\n\n"
                       "Use the 'Esc' key or the same toolbar button to exit "
                       "this mode.");
            break;

        case SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY:
        case SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY:
            settingsKey = "pqInteractiveSelection";
            title = tr("Interactive Selection Information");
            message =
                    tr("You are entering interactive selection mode to "
                       "highlight cells (or points). "
                       "Simply move the mouse point over the dataset to "
                       "interactively highlight elements.\n\n"
                       "To add the currently highlighted element to the active "
                       "selection, simply click on that element.\n\n"
                       "You can click on selection modifier button or use "
                       "modifier keys to subtract or "
                       "even toggle the selection. Click outside of mesh to "
                       "clear selection.\n\n"
                       "Use the 'Esc' key or the same toolbar button to exit "
                       "this mode.");
            break;

        default:
            // No dialog for other modes
            return;
    }

    // Use helper to show dialog (respects "don't show again" setting)
    if (!settingsKey.isEmpty()) {
        cvSelectionPipeline::promptUser(settingsKey, title, message, nullptr);
    }
}

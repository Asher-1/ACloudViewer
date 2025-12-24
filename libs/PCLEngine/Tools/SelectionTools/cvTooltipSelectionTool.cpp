// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvTooltipSelectionTool.h"

#include "cvSelectionData.h"
#include "cvSelectionPipeline.h"

// LOCAL
#include "PclUtils/PCLVis.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <ecvDisplayTools.h>

// VTK
#include <vtkActor.h>
#include <vtkCommand.h>
#include <vtkFieldData.h>
#include <vtkIdTypeArray.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPropCollection.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkStringArray.h>

// QT
#include <QApplication>
#include <QClipboard>
#include <QCursor>
#include <QToolTip>

//-----------------------------------------------------------------------------
cvTooltipSelectionTool::cvTooltipSelectionTool(SelectionMode mode,
                                               QObject* parent)
    : cvRenderViewSelectionTool(mode, parent),
      m_hoveredId(-1),
      m_rightButtonPressed(false),
      m_mouseMoving(false),
      m_leftButtonPressed(false),
      m_currentPolyData(nullptr),
      m_tooltipHelper(new cvSelectionTooltipHelper()),
      m_copyShortcut(nullptr),
      m_enableSelection(false),
      m_tooltipEnabled(true) {  // Tooltips enabled by default
    m_fieldAssociation = isSelectingCells() ? 0 : 1;
    m_leftButtonPressPos[0] = 0;
    m_leftButtonPressPos[1] = 0;

    // Determine mode: interactive (click-to-select) or tooltip-only
    m_enableSelection = isInteractiveMode();

    // Initialize pickers (now in base class)
    initializePickers();

    // Setup tooltip display timer (400ms delay)
    // Note: Highlight updates are immediate (ParaView-style), no throttling
    m_mouseStopTimer.setSingleShot(true);
    m_mouseStopTimer.setInterval(TOOLTIP_WAITING_TIME);
    connect(&m_mouseStopTimer, &QTimer::timeout, this,
            &cvTooltipSelectionTool::onMouseStop);

    CVLog::PrintDebug(
            QString("[cvTooltipSelectionTool] Created, mode: %1, field: %2")
                    .arg(m_enableSelection ? "INTERACTIVE" : "TOOLTIP")
                    .arg(m_fieldAssociation == 0 ? "CELLS" : "POINTS"));

    // Note: Highlighter is obtained from cvViewSelectionManager via
    // getHighlighter() This ensures all tools share the same highlighter
    // instance, so color settings from cvSelectionPropertiesWidget are
    // automatically synchronized.
}

//-----------------------------------------------------------------------------
cvTooltipSelectionTool::~cvTooltipSelectionTool() {
    // Just hide the Qt tooltip - don't try to access highlighter during
    // destruction as the cvViewSelectionManager singleton might already be
    // destroyed
    QToolTip::hideText();

    // Clean up shortcuts
    if (m_copyShortcut) {
        delete m_copyShortcut;
        m_copyShortcut = nullptr;
    }

    CVLog::Print("[cvTooltipSelectionTool] Destroyed");
}

//-----------------------------------------------------------------------------
cvSelectionHighlighter* cvTooltipSelectionTool::getHighlighter() const {
    // Get shared highlighter from the selection manager singleton
    cvViewSelectionManager* manager = cvViewSelectionManager::instance();
    if (manager) {
        return manager->getHighlighter();
    }
    return nullptr;
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::disable() {
    // Stop any running timers
    m_mouseStopTimer.stop();

    // Hide tooltip and clear all highlights
    hideTooltip();

    // Call base class disable
    cvRenderViewSelectionTool::disable();
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::setTooltipEnabled(bool enabled) {
    m_tooltipEnabled = enabled;

    if (!enabled) {
        // Hide any currently displayed tooltip
        hideTooltip();
    }

    CVLog::PrintDebug(QString("[cvTooltipSelectionTool] Tooltips %1")
                              .arg(enabled ? "enabled" : "disabled"));
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::setMaxTooltipAttributes(int maxAttributes) {
    if (m_tooltipHelper) {
        m_tooltipHelper->setMaxAttributes(maxAttributes);
        CVLog::PrintDebug(QString("[cvTooltipSelectionTool] Max tooltip "
                                  "attributes set to %1")
                                  .arg(maxAttributes));
    }
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::setupInteractorStyle() {
    // Tooltip mode doesn't change interactor style
    // It works with the default camera manipulation style
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::setupObservers() {
    if (!m_interactor) {
        CVLog::Error("[cvTooltipSelectionTool] No interactor available");
        return;
    }

    // CRITICAL: Set m_observedObject so cleanupObservers() can remove observers
    m_observedObject = m_interactor;

    // Note: Hardware selection buffers are now captured on-demand in
    // updateHighlight() to ensure they always reflect the current scene state
    // (see updateHighlight fix) This prevents stale buffer issues that caused
    // multiple ghost candidate points

    // ParaView-style: Observe MouseMoveEvent for tooltip updates
    m_observerIds[0] = m_interactor->AddObserver(
            vtkCommand::MouseMoveEvent, this,
            &cvTooltipSelectionTool::onSelectionChanged);

    // Additional observers for right button (to pause tooltip)
    m_observerIds[1] = m_interactor->AddObserver(
            vtkCommand::RightButtonPressEvent, this,
            &cvTooltipSelectionTool::onSelectionChanged);

    m_observerIds[2] = m_interactor->AddObserver(
            vtkCommand::RightButtonReleaseEvent, this,
            &cvTooltipSelectionTool::onSelectionChanged);

    // Interactive mode: observe left button press/release for click-to-select
    // ParaView-style: distinguish click from drag by checking distance
    if (m_enableSelection) {
        m_observerIds[3] = m_interactor->AddObserver(
                vtkCommand::LeftButtonPressEvent, this,
                &cvTooltipSelectionTool::onSelectionChanged);
        m_observerIds[4] = m_interactor->AddObserver(
                vtkCommand::LeftButtonReleaseEvent, this,
                &cvTooltipSelectionTool::onSelectionChanged);
    }
    // ParaView-style: Setup Ctrl-C/Cmd-C shortcut for copying tooltip
    // Note: Use ecvDisplayTools::GetCurrentScreen() as parent for proper Qt
    // lifecycle
    QWidget* parentWidget = ecvDisplayTools::GetCurrentScreen();
    if (!m_copyShortcut && parentWidget) {
        m_copyShortcut = new QShortcut(QKeySequence::Copy, parentWidget);
        m_copyShortcut->setContext(Qt::ApplicationShortcut);
        connect(m_copyShortcut, &QShortcut::activated, [this]() {
            if (!m_currentPlainText.isEmpty()) {
                QApplication::clipboard()->setText(m_currentPlainText);
                CVLog::Print(
                        "[cvTooltipSelectionTool] Tooltip text copied to "
                        "clipboard");
            }
        });
    }

    // NOTE: ESC key handling is centralized in MainWindow::handleEscapeKey()
    // which calls disableAllSelectionTools(). No need for per-tool ESC
    // shortcuts.
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::onSelectionChanged(vtkObject* caller,
                                                unsigned long eventId,
                                                void* callData) {
    Q_UNUSED(caller);
    Q_UNUSED(callData);

    // Removed verbose logging for MouseMoveEvent to reduce lag
    if (eventId == vtkCommand::MouseMoveEvent) {
        onMouseMove();
    } else if (eventId == vtkCommand::RightButtonPressEvent) {
        onRightButtonPress();
    } else if (eventId == vtkCommand::RightButtonReleaseEvent) {
        onRightButtonRelease();
    } else if (eventId == vtkCommand::LeftButtonPressEvent) {
        // Interactive mode only: record press position (ParaView-style)
        if (m_enableSelection) {
            onLeftButtonPress();
        }
    } else if (eventId == vtkCommand::LeftButtonReleaseEvent) {
        // Interactive mode only: check if it's a click (not drag) and select
        if (m_enableSelection) {
            onLeftButtonRelease();
        }
    }
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::onMouseMove() {
    if (m_rightButtonPressed) {
        // Don't update tooltip while right button is pressed (camera
        // manipulation)
        return;
    }

    // Start tooltip timer (400ms delay for text display)
    m_mouseMoving = true;
    m_mouseStopTimer.start();

    // ParaView-style: Execute highlight immediately (no throttling)
    // ParaView uses hardware selection with cached buffers for fast performance
    // Reference: pqRenderViewSelectionReaction::onMouseMove() calls
    // preSelection()
    updateHighlight();
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::updateHighlight() {
    // Get shared highlighter from manager
    cvSelectionHighlighter* highlighter = getHighlighter();

    // Performance optimization: Skip if highlighter is not available
    if (!highlighter) {
        return;
    }

    // CRITICAL FIX: Refresh hardware selection buffers before each pick
    // This ensures we pick from the current scene state, not stale cached
    // buffers Without this, we get multiple ghost candidates from old buffer
    // data
    cvSelectionPipeline* pipeline = getSelectionPipeline();
    if (pipeline) {
        // Recapture buffers to match current scene
        // Note: This adds minimal overhead (~1-2ms) and is essential for
        // correct picking
        pipeline->captureBuffersForFastPreSelection();
    }

    // ParaView-style: Use fast pre-selection with cached hardware buffers
    // Reference: pqRenderViewSelectionReaction::fastPreSelection()
    bool selectCells = isSelectingCells();
    vtkIdType id = -1;
    vtkPolyData* polyData = nullptr;

    // Try fast hardware selection first (uses cached buffers)
    // Use new getPixelSelectionInfo to get correct actor and polyData
    if (pipeline && m_interactor) {
        int* pos = m_interactor->GetEventPosition();
        cvSelectionPipeline::PixelSelectionInfo selInfo =
                pipeline->getPixelSelectionInfo(pos[0], pos[1], selectCells);

        if (selInfo.valid) {
            id = selInfo.attributeID;
            polyData = selInfo.polyData;

            // Cache the polyData for tooltip display
            m_currentPolyData = polyData;
        }
    }

    // Fallback to software picking if hardware selection fails
    if (id < 0) {
        id = pickAtCursor(selectCells);
        // For software picking, use the old method to get polyData
        if (id >= 0) {
            polyData = getPolyDataForSelection();
            m_currentPolyData = polyData;
        }
    }

    // Only update if the hovered element changed
    if (id != m_hoveredId) {
        m_hoveredId = id;

        // Only update highlight, don't show tooltip yet
        if (id >= 0 && polyData) {
            highlighter->highlightElement(polyData, id, m_fieldAssociation);
        } else {
            // Clear ONLY hover highlight, keep selected highlights
            highlighter->clearHoverHighlight();
        }

        emit hoverChanged(static_cast<qint64>(id), m_fieldAssociation);
    }
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::onMouseStop() {
    m_mouseMoving = false;

    // Show tooltip for the currently hovered element
    if (m_hoveredId >= 0) {
        updateTooltip(m_hoveredId);
    } else {
        hideTooltip();
    }
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::onRightButtonPress() {
    m_rightButtonPressed = true;
    hideTooltip();
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::onRightButtonRelease() {
    m_rightButtonPressed = false;
}

// Note: pickAtCursor() method moved to base class cvRenderViewSelectionTool
// to eliminate code duplication. Now using inherited method.

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::updateTooltip(vtkIdType id) {
    // Check if tooltips are enabled
    if (!m_tooltipEnabled) {
        hideTooltip();
        return;
    }

    if (id < 0) {
        CVLog::Warning(
                "[cvTooltipSelectionTool::updateTooltip] id < 0, hiding "
                "tooltip");
        hideTooltip();
        return;
    }

    // Use cached polyData from updateHighlight() (fixes "Invalid cell ID"
    // issue) This ensures we use the same polyData that was selected by
    // hardware selector
    vtkPolyData* polyData = m_currentPolyData;

    // Fallback to getPolyDataForSelection if cache is empty (should not happen)
    if (!polyData) {
        polyData = getPolyDataForSelection();
        CVLog::PrintDebug(
                "[cvTooltipSelectionTool::updateTooltip] Using fallback "
                "polyData");
    }

    if (!polyData) {
        CVLog::Warning(
                "[cvTooltipSelectionTool::updateTooltip] polyData is nullptr, "
                "hiding tooltip");
        hideTooltip();
        return;
    }

    // Validate ID is within range for this polyData
    vtkIdType maxId = (m_fieldAssociation == 0) ? polyData->GetNumberOfCells()
                                                : polyData->GetNumberOfPoints();
    if (id >= maxId) {
        CVLog::Error(
                QString("[cvTooltipSelectionTool::updateTooltip] ID %1 out of "
                        "range "
                        "(max %2) for %3")
                        .arg(id)
                        .arg(maxId - 1)
                        .arg(m_fieldAssociation == 0 ? "cells" : "points"));
        hideTooltip();
        return;
    }

    if (!m_tooltipHelper) {
        CVLog::Warning(
                "[cvTooltipSelectionTool::updateTooltip] m_tooltipHelper is "
                "nullptr, hiding tooltip");
        hideTooltip();
        return;
    }

    // Extract dataset name from field data (optional, for display purposes)
    QString datasetName;
    vtkFieldData* fieldData = polyData->GetFieldData();
    if (fieldData) {
        vtkStringArray* nameArray = vtkStringArray::SafeDownCast(
                fieldData->GetAbstractArray("DatasetName"));
        if (nameArray && nameArray->GetNumberOfTuples() > 0) {
            datasetName = QString::fromStdString(nameArray->GetValue(0));
        } else {
            // DatasetName is optional - not a problem if missing
            CVLog::PrintDebug(
                    "[cvTooltipSelectionTool::updateTooltip] DatasetName array "
                    "not found (using default)");
        }
    } else {
        CVLog::PrintDebug(
                "[cvTooltipSelectionTool::updateTooltip] FieldData is nullptr");
    }

    // Generate tooltip text (ParaView format)
    QString htmlTooltip = m_tooltipHelper->getTooltipInfo(
            polyData, id,
            m_fieldAssociation == 0 ? cvSelectionTooltipHelper::CELLS
                                    : cvSelectionTooltipHelper::POINTS,
            datasetName);

    QString plainTooltip = m_tooltipHelper->getPlainTooltipInfo(
            polyData, id,
            m_fieldAssociation == 0 ? cvSelectionTooltipHelper::CELLS
                                    : cvSelectionTooltipHelper::POINTS,
            datasetName);

    if (!htmlTooltip.isEmpty()) {
        // Update cached tooltip text
        m_currentTooltipText = htmlTooltip;
        m_currentPlainText = plainTooltip;

        // Show tooltip at cursor position
        QPoint globalPos = QCursor::pos();
        QToolTip::showText(globalPos, htmlTooltip);

        // Note: Highlight is already shown by onHighlightUpdate(), no need to
        // highlight again

        emit tooltipChanged(htmlTooltip, plainTooltip);
    } else {
        CVLog::Print(
                "[cvTooltipSelectionTool::updateTooltip] htmlTooltip is empty, "
                "hiding tooltip");
        hideTooltip();
    }
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::hideTooltip() {
    QToolTip::hideText();

    cvSelectionHighlighter* hl = getHighlighter();
    if (hl) {
        // Only clear hover highlight, keep selected highlights
        hl->clearHoverHighlight();
    }

    m_currentTooltipText.clear();
    m_currentPlainText.clear();
    m_hoveredId = -1;
    m_currentPolyData = nullptr;  // Clear cached polyData

    emit tooltipChanged(QString(), QString());
}

// Note: isSelectingCells() method moved to base class cvRenderViewSelectionTool
// to eliminate code duplication. Now using inherited method.

//-----------------------------------------------------------------------------
bool cvTooltipSelectionTool::isInteractiveMode() const {
    // Interactive modes allow click-to-select behavior
    // Reference: pqRenderViewSelectionReaction lines 101-104
    return (m_mode == cvViewSelectionManager::
                              SELECT_SURFACE_CELLS_INTERACTIVELY ||
            m_mode == cvViewSelectionManager::
                              SELECT_SURFACE_POINTS_INTERACTIVELY ||
            m_mode == cvViewSelectionManager::
                              SELECT_SURFACE_CELLDATA_INTERACTIVELY ||
            m_mode == cvViewSelectionManager::
                              SELECT_SURFACE_POINTDATA_INTERACTIVELY);
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::onLeftButtonPress() {
    // ParaView-style: Record press position, don't select yet
    // This allows us to distinguish between click (select) and drag (rotate
    // camera)

    if (!m_interactor) {
        return;
    }

    int* pos = m_interactor->GetEventPosition();
    m_leftButtonPressed = true;
    m_leftButtonPressPos[0] = pos[0];
    m_leftButtonPressPos[1] = pos[1];

    CVLog::PrintDebug(QString("[cvTooltipSelectionTool::onLeftButtonPress] "
                              "Recorded press at (%1, %2)")
                              .arg(pos[0])
                              .arg(pos[1]));
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::onLeftButtonRelease() {
    // ParaView-style: Check if this was a click (not drag) before selecting
    // Reference: pqRenderViewSelectionReaction checks mouse movement distance

    if (!m_enableSelection || !m_interactor) {
        m_leftButtonPressed = false;
        return;
    }

    if (!m_leftButtonPressed) {
        // No matching press event, ignore
        return;
    }

    // Get release position
    int* releasePos = m_interactor->GetEventPosition();

    // Calculate distance between press and release
    int dx = releasePos[0] - m_leftButtonPressPos[0];
    int dy = releasePos[1] - m_leftButtonPressPos[1];
    int distanceSquared = dx * dx + dy * dy;

    CVLog::PrintDebug(
            QString("[cvTooltipSelectionTool::onLeftButtonRelease] "
                    "Press: (%1, %2), Release: (%3, %4), Distance²: %5")
                    .arg(m_leftButtonPressPos[0])
                    .arg(m_leftButtonPressPos[1])
                    .arg(releasePos[0])
                    .arg(releasePos[1])
                    .arg(distanceSquared));

    // Reset state
    m_leftButtonPressed = false;

    // CRITICAL FIX: Only perform selection if distance < threshold
    // This prevents selection during camera rotation (ParaView-style)
    int thresholdSquared = CLICK_THRESHOLD * CLICK_THRESHOLD;
    if (distanceSquared > thresholdSquared) {
        CVLog::Print(QString("[cvTooltipSelectionTool] Mouse moved too much "
                             "(distance²: %1 > %2) - "
                             "this is a DRAG (camera rotation), not a click. "
                             "Skipping selection.")
                             .arg(distanceSquared)
                             .arg(thresholdSquared));
        return;
    }

    CVLog::Print(
            "[cvTooltipSelectionTool] Mouse movement < threshold - this is a "
            "CLICK, performing selection");

    // Now perform the actual selection (ParaView-style precise picking)
    bool selectCells = isSelectingCells();
    vtkIdType id = -1;
    vtkPolyData* polyData = nullptr;

    // PRIMARY: Use precise picker for accurate click selection (ParaView
    // method)
    id = pickAtCursor(selectCells);
    if (id >= 0) {
        polyData = getPolyDataForSelection();
        m_currentPolyData = polyData;
        CVLog::Print(QString("[cvTooltipSelectionTool] Precise picker selected "
                             "ID: %1")
                             .arg(id));
    } else {
        CVLog::PrintDebug(
                "[cvTooltipSelectionTool] Precise picker failed, trying "
                "hardware selector");

        // FALLBACK: Try hardware selector if precise picker fails
        cvSelectionPipeline* pipeline = getSelectionPipeline();
        if (pipeline) {
            cvSelectionPipeline::PixelSelectionInfo selInfo =
                    pipeline->getPixelSelectionInfo(releasePos[0],
                                                    releasePos[1], selectCells);

            if (selInfo.valid) {
                id = selInfo.attributeID;
                polyData = selInfo.polyData;
                m_currentPolyData = polyData;
                CVLog::PrintDebug(QString("[cvTooltipSelectionTool] Hardware "
                                          "selector fallback selected ID: %1")
                                          .arg(id));
            }
        }
    }

    if (id >= 0) {
        toggleSelection(id);
        CVLog::Print(
                QString("[cvTooltipSelectionTool] Clicked and selected ID: %1")
                        .arg(id));
    } else {
        CVLog::Warning(
                "[cvTooltipSelectionTool] Click did not pick any element");
    }
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::toggleSelection(vtkIdType id) {
    if (!m_currentSelection) {
        m_currentSelection = vtkSmartPointer<vtkIdTypeArray>::New();
    }

    // Check if ID is already selected
    SelectionModifier modifier = getSelectionModifierFromKeyboard();

    // ParaView behavior: In interactive selection mode, DEFAULT becomes
    // ADDITION Reference: pqRenderViewSelectionReaction.cxx, line 958-962
    if (modifier == cvViewSelectionManager::SELECTION_DEFAULT) {
        modifier = cvViewSelectionManager::SELECTION_ADDITION;
    }

    bool isSelected = false;
    bool needUpdate = false;  // Track if selection actually changed

    for (vtkIdType i = 0; i < m_currentSelection->GetNumberOfTuples(); ++i) {
        if (m_currentSelection->GetValue(i) == id) {
            isSelected = true;

            // Handle modifier keys for already-selected items
            if (modifier == cvViewSelectionManager::SELECTION_SUBTRACTION ||
                modifier == cvViewSelectionManager::SELECTION_TOGGLE) {
                // Remove from selection
                m_currentSelection->RemoveTuple(i);
                needUpdate = true;
                CVLog::Print(QString("[cvTooltipSelectionTool] Removed ID: %1 "
                                     "from selection")
                                     .arg(id));
            } else {
                // ADDITION on already-selected item: do nothing (ParaView
                // style)
                CVLog::Print(QString("[cvTooltipSelectionTool] ID: %1 already "
                                     "selected, skipping")
                                     .arg(id));
                // Don't update - selection unchanged
                return;  // Early return to avoid emitting signals
            }
            break;
        }
    }

    if (!isSelected) {
        // Add to selection (ParaView style: default is ADDITION in interactive
        // mode)
        SelectionModifier modifier = getSelectionModifierFromKeyboard();

        // ParaView behavior: In interactive selection mode, DEFAULT becomes
        // ADDITION Reference: pqRenderViewSelectionReaction.cxx, line 958-962
        if (modifier == cvViewSelectionManager::SELECTION_DEFAULT) {
            modifier = cvViewSelectionManager::SELECTION_ADDITION;
        }

        if (modifier == cvViewSelectionManager::SELECTION_ADDITION) {
            // Add to existing selection (ParaView default for interactive mode)
            m_currentSelection->InsertNextValue(id);
            needUpdate = true;
            CVLog::Print(QString("[cvTooltipSelectionTool] Added ID: %1 to "
                                 "selection (total: %2)")
                                 .arg(id)
                                 .arg(m_currentSelection->GetNumberOfTuples()));
        } else if (modifier == cvViewSelectionManager::SELECTION_SUBTRACTION) {
            // This case won't happen since we already checked !isSelected
            // But keep it for completeness
            CVLog::Warning(QString("[cvTooltipSelectionTool] ID: %1 not in "
                                   "selection, cannot subtract")
                                   .arg(id));
            return;  // No update needed
        } else if (modifier == cvViewSelectionManager::SELECTION_TOGGLE) {
            // Add since it's not selected
            m_currentSelection->InsertNextValue(id);
            needUpdate = true;
            CVLog::Print(QString("[cvTooltipSelectionTool] Toggled ID: %1 "
                                 "(added, total: %2)")
                                 .arg(id)
                                 .arg(m_currentSelection->GetNumberOfTuples()));
        }
    }

    // Only update if selection actually changed
    if (!needUpdate) {
        CVLog::PrintDebug(
                "[cvTooltipSelectionTool] Selection unchanged, skipping "
                "update");
        return;
    }

    // CRITICAL FIX: Validate m_currentSelection before creating cvSelectionData
    if (!m_currentSelection || !m_currentSelection.GetPointer()) {
        CVLog::Error("[cvTooltipSelectionTool] m_currentSelection is invalid!");
        return;
    }

    // Verify the array is valid
    vtkIdType numTuples = m_currentSelection->GetNumberOfTuples();
    if (numTuples < 0) {
        CVLog::Error(
                "[cvTooltipSelectionTool] m_currentSelection has invalid tuple "
                "count!");
        return;
    }

    CVLog::PrintDebug(QString("[cvTooltipSelectionTool] Creating selection "
                              "data with %1 elements")
                              .arg(numTuples));

    // Create selection data - this will do a DeepCopy
    cvSelectionData selectionData(m_currentSelection.Get(), m_fieldAssociation);

    // Set actor info for the selection (required for getPolyDataForSelection)
    vtkPolyData* polyData =
            m_currentPolyData;  // Use cached polyData from picking
    if (!polyData) {
        polyData = getPolyDataForSelection();
    }
    if (polyData) {
        selectionData.setActorInfo(nullptr, polyData, 0.0);
    }

    // Store selection in manager for retrieval
    if (m_manager) {
        CVLog::PrintDebug(
                "[cvTooltipSelectionTool] Storing selection in manager");
        m_manager->setCurrentSelection(selectionData);
    }

    // CRITICAL FIX: Emit base class signal that manager listens to
    // The manager connects to selectionCompleted, not selectionFinished
    emit selectionCompleted();

    // Also emit custom signal for backward compatibility
    emit selectionFinished(selectionData);

    CVLog::Print(
            QString("[cvTooltipSelectionTool] Selection updated: %1 elements")
                    .arg(numTuples));
}
// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvTooltipSelectionTool.h"

#include "cvSelectionData.h"

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
      m_tooltipHelper(new cvSelectionTooltipHelper()),
      m_hoverHighlighter(new cvSelectionHighlighter()),
      m_copyShortcut(nullptr),
      m_escapeShortcut(nullptr),
      m_enableSelection(false),
      m_tooltipEnabled(true) {  // Tooltips enabled by default
    m_fieldAssociation = isSelectingCells() ? 0 : 1;

    // Determine mode: interactive (click-to-select) or tooltip-only
    m_enableSelection = isInteractiveMode();

    // Initialize pickers (now in base class)
    initializePickers();

    // Setup tooltip display timer (400ms delay)
    m_mouseStopTimer.setSingleShot(true);
    m_mouseStopTimer.setInterval(TOOLTIP_WAITING_TIME);
    connect(&m_mouseStopTimer, &QTimer::timeout, this,
            &cvTooltipSelectionTool::onMouseStop);

    CVLog::PrintDebug(
            QString("[cvTooltipSelectionTool] Created, mode: %1, field: %2")
                    .arg(m_enableSelection ? "INTERACTIVE" : "TOOLTIP")
                    .arg(m_fieldAssociation == 0 ? "CELLS" : "POINTS"));

    // Set visualizer for highlighter
    PclUtils::PCLVis* viewer = reinterpret_cast<PclUtils::PCLVis*>(
            ecvDisplayTools::GetVisualizer3D());
    if (viewer) {
        m_hoverHighlighter->setVisualizer(viewer);
    }
}

//-----------------------------------------------------------------------------
cvTooltipSelectionTool::~cvTooltipSelectionTool() {
    hideTooltip();

    // Clear hover highlight
    if (m_hoverHighlighter) {
        m_hoverHighlighter->clearHighlights();
    }

    // Clean up shortcuts
    if (m_copyShortcut) {
        delete m_copyShortcut;
        m_copyShortcut = nullptr;
    }

    if (m_escapeShortcut) {
        delete m_escapeShortcut;
        m_escapeShortcut = nullptr;
    }

    CVLog::Print("[cvTooltipSelectionTool] Destroyed");
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

    // Interactive mode: observe left button press for selection
    if (m_enableSelection) {
        m_observerIds[3] = m_interactor->AddObserver(
                vtkCommand::LeftButtonPressEvent, this,
                &cvTooltipSelectionTool::onSelectionChanged);
    }
    // ParaView-style: Setup Ctrl-C/Cmd-C shortcut for copying tooltip
    if (!m_copyShortcut) {
        m_copyShortcut = new QShortcut(QKeySequence::Copy, nullptr);
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

    // Setup ESC key to exit/disable the tool
    if (!m_escapeShortcut) {
        m_escapeShortcut = new QShortcut(QKeySequence(Qt::Key_Escape), nullptr);
        m_escapeShortcut->setContext(Qt::ApplicationShortcut);
        connect(m_escapeShortcut, &QShortcut::activated, [this]() {
            CVLog::Print(
                    "[cvTooltipSelectionTool] ESC pressed, disabling tool");
            // Request tool to be disabled - this will emit signal to MainWindow
            emit requestDisable();
        });
    }
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
        // Interactive mode only
        if (m_enableSelection) {
            onLeftButtonPress();
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

    // ParaView-style: Execute highlight immediately, no timer
    // This is exactly how ParaView implements it in
    // pqRenderViewSelectionReaction::onMouseMove()
    updateHighlight();
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::updateHighlight() {
    // Pick at current cursor position for immediate highlight (ParaView-style)
    bool selectCells =
            isSelectingCells();  // Determine if selecting cells or points
    vtkIdType id = pickAtCursor(selectCells);

    if (id != m_hoveredId) {
        m_hoveredId = id;

        // Only update highlight, don't show tooltip yet
        if (id >= 0 && m_hoverHighlighter) {
            vtkPolyData* polyData = getPolyDataForSelection();
            if (polyData) {
                m_hoverHighlighter->highlightElement(polyData, id,
                                                     m_fieldAssociation);
            }
        } else {
            // Clear highlight if nothing is picked
            if (m_hoverHighlighter) {
                m_hoverHighlighter->clearHighlights();
            }
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

    // Get polyData using centralized ParaView-style method
    vtkPolyData* polyData = getPolyDataForSelection();

    if (!polyData) {
        CVLog::Warning(
                "[cvTooltipSelectionTool::updateTooltip] polyData is nullptr, "
                "hiding tooltip");
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

    // Extract dataset name from field data
    QString datasetName;
    vtkFieldData* fieldData = polyData->GetFieldData();
    if (fieldData) {
        vtkStringArray* nameArray = vtkStringArray::SafeDownCast(
                fieldData->GetAbstractArray("DatasetName"));
        if (nameArray && nameArray->GetNumberOfTuples() > 0) {
            datasetName = QString::fromStdString(nameArray->GetValue(0));
        } else {
            CVLog::Warning(
                    "[cvTooltipSelectionTool::updateTooltip] DatasetName array "
                    "not found or empty");
        }
    } else {
        CVLog::Warning(
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

    if (m_hoverHighlighter) {
        m_hoverHighlighter->clearHighlights();
    }

    m_currentTooltipText.clear();
    m_currentPlainText.clear();
    m_hoveredId = -1;

    emit tooltipChanged(QString(), QString());
}

// Note: isSelectingCells() method moved to base class cvRenderViewSelectionTool
// to eliminate code duplication. Now using inherited method.

//-----------------------------------------------------------------------------
bool cvTooltipSelectionTool::isInteractiveMode() const {
    return (m_mode == cvViewSelectionManager::
                              SELECT_SURFACE_CELLS_INTERACTIVELY ||
            m_mode == cvViewSelectionManager::
                              SELECT_SURFACE_POINTS_INTERACTIVELY);
}

//-----------------------------------------------------------------------------
void cvTooltipSelectionTool::onLeftButtonPress() {
    // Interactive mode only: perform selection on click
    if (!m_enableSelection) {
        return;
    }

    bool selectCells =
            isSelectingCells();  // Determine if selecting cells or points
    vtkIdType id = pickAtCursor(selectCells);

    if (id >= 0) {
        toggleSelection(id);
        CVLog::PrintDebug(
                QString("[cvTooltipSelectionTool] Selected ID: %1").arg(id));
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
    for (vtkIdType i = 0; i < m_currentSelection->GetNumberOfTuples(); ++i) {
        if (m_currentSelection->GetValue(i) == id) {
            isSelected = true;

            // Handle modifier keys for already-selected items
            if (modifier == cvViewSelectionManager::SELECTION_SUBTRACTION ||
                modifier == cvViewSelectionManager::SELECTION_TOGGLE) {
                // Remove from selection
                m_currentSelection->RemoveTuple(i);
                CVLog::Print(QString("[cvTooltipSelectionTool] Removed ID: %1 "
                                     "from selection")
                                     .arg(id));
            } else {
                // ADDITION on already-selected item: do nothing (ParaView
                // style)
                CVLog::Print(QString("[cvTooltipSelectionTool] ID: %1 already "
                                     "selected")
                                     .arg(id));
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
            CVLog::PrintDebug(
                    QString("[cvTooltipSelectionTool] Added ID: %1 to "
                            "selection")
                            .arg(id));
        } else if (modifier == cvViewSelectionManager::SELECTION_SUBTRACTION) {
            // This case won't happen since we already checked !isSelected
            // But keep it for completeness
            CVLog::Warning(QString("[cvTooltipSelectionTool] ID: %1 not in "
                                   "selection, cannot subtract")
                                   .arg(id));
        } else if (modifier == cvViewSelectionManager::SELECTION_TOGGLE) {
            // Add since it's not selected
            m_currentSelection->InsertNextValue(id);
            CVLog::Print(
                    QString("[cvTooltipSelectionTool] Toggled ID: %1 (added)")
                            .arg(id));
        }
    }

    // Emit selection finished signal
    cvSelectionData selectionData(m_currentSelection, m_fieldAssociation);
    emit selectionFinished(selectionData);
}
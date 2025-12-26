// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionReaction.h"

#include "cvRenderViewSelectionTool.h"
#include "cvSelectionData.h"
#include "cvSelectionTypes.h"  // For SelectionMode and SelectionModifier enums
#include "cvViewSelectionManager.h"
#include "cvZoomBoxSelectionTool.h"

// CV_CORE_LIB
#include <CVLog.h>

// QT
#include <QActionGroup>
#include <QApplication>
#include <QWidget>

QPointer<cvSelectionReaction> cvSelectionReaction::ActiveReaction;

//-----------------------------------------------------------------------------
cvSelectionReaction::cvSelectionReaction(QAction* parentAction,
                                         SelectionMode mode,
                                         QActionGroup* modifierGroup)
    : QObject(parentAction),
      m_parentAction(parentAction),
      m_modifierGroup(modifierGroup),
      m_mode(mode),
      m_tool(nullptr) {
    // Connect action triggered signal
    connect(parentAction, &QAction::triggered, this,
            &cvSelectionReaction::actionTriggered);

    // Connect to selection manager for state updates
    if (m_mode == SelectionMode::CLEAR_SELECTION ||
        m_mode == SelectionMode::GROW_SELECTION ||
        m_mode == SelectionMode::SHRINK_SELECTION) {
        cvViewSelectionManager* manager = cvViewSelectionManager::instance();
        // Use the overload that takes no arguments for simple state updates
        connect(manager,
                QOverload<>::of(&cvViewSelectionManager::selectionChanged),
                this, &cvSelectionReaction::updateEnableState);
    }

    updateEnableState();
}

//-----------------------------------------------------------------------------
cvSelectionReaction::~cvSelectionReaction() { endSelection(); }

//-----------------------------------------------------------------------------
void cvSelectionReaction::setVisualizer(ecvGenericVisualizer3D* viewer) {
    cvViewSelectionManager::instance()->setVisualizer(viewer);
}

//-----------------------------------------------------------------------------
bool cvSelectionReaction::isActive() const { return ActiveReaction == this; }

//-----------------------------------------------------------------------------
void cvSelectionReaction::actionTriggered(bool val) {
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
        beginSelection();
        endSelection();
    }
}

//-----------------------------------------------------------------------------
void cvSelectionReaction::updateEnableState() {
    if (!m_parentAction) {
        return;
    }

    cvViewSelectionManager* manager = cvViewSelectionManager::instance();

    switch (m_mode) {
        case SelectionMode::CLEAR_SELECTION:
        case SelectionMode::GROW_SELECTION:
        case SelectionMode::SHRINK_SELECTION:
            m_parentAction->setEnabled(manager->hasSelection());
            break;
        default:
            // Other modes are always enabled when visualizer is available
            m_parentAction->setEnabled(manager->getVisualizer() != nullptr);
            break;
    }
}

//-----------------------------------------------------------------------------
void cvSelectionReaction::beginSelection() {
    cvViewSelectionManager* manager = cvViewSelectionManager::instance();

    // Special handling for non-interactive modes
    if (m_mode == SelectionMode::CLEAR_SELECTION) {
        manager->clearSelection();
        return;
    }
    if (m_mode == SelectionMode::GROW_SELECTION) {
        manager->growSelection();
        return;
    }
    if (m_mode == SelectionMode::SHRINK_SELECTION) {
        manager->shrinkSelection();
        return;
    }

    // End any other active reaction first
    if (ActiveReaction && ActiveReaction != this) {
        ActiveReaction->endSelection();
    }

    // Get or create the tool
    cvRenderViewSelectionTool* tool = getOrCreateTool();
    if (!tool) {
        CVLog::Warning(
                QString("[cvSelectionReaction] Failed to create tool for "
                        "mode %1")
                        .arg(static_cast<int>(m_mode)));
        return;
    }

    // Store previous cursor
    if (QWidget* widget = QApplication::activeWindow()) {
        m_previousCursor = widget->cursor();
    }

    // Set selection modifier
    int modifierInt = getSelectionModifier();
    tool->setSelectionModifier(
            static_cast<SelectionModifier>(modifierInt));

    // Enable the tool
    tool->enable();

    // Set appropriate cursor
    // For ZOOM_TO_BOX, use the tool's custom cursor (zoom cursor from XPM)
    // For other modes, use standard Qt cursors
    if (m_mode == SelectionMode::ZOOM_TO_BOX) {
        // Zoom cursor is set by the tool itself (cvZoomBoxSelectionTool)
        if (QWidget* widget = QApplication::activeWindow()) {
            widget->setCursor(tool->getCursor());
        }
    } else {
        Qt::CursorShape cursor = Qt::CrossCursor;
        switch (m_mode) {
            case SelectionMode::SELECT_SURFACE_CELLS_POLYGON:
            case SelectionMode::SELECT_SURFACE_POINTS_POLYGON:
            case SelectionMode::SELECT_CUSTOM_POLYGON:
                cursor = Qt::PointingHandCursor;
                break;
            case SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY:
            case SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY:
            case SelectionMode::HOVER_CELLS_TOOLTIP:
            case SelectionMode::HOVER_POINTS_TOOLTIP:
                cursor = Qt::PointingHandCursor;
                break;
            default:
                cursor = Qt::CrossCursor;
                break;
        }

        if (QWidget* widget = QApplication::activeWindow()) {
            widget->setCursor(cursor);
        }
    }

    // Mark this as the active reaction
    ActiveReaction = this;

    CVLog::PrintDebug(
            QString("[cvSelectionReaction] Selection mode %1 activated")
                    .arg(static_cast<int>(m_mode)));
}

//-----------------------------------------------------------------------------
void cvSelectionReaction::endSelection() {
    if (ActiveReaction != this) {
        return;
    }

    // Disable the tool
    if (m_tool) {
        m_tool->disable();
    }

    // Restore cursor
    if (QWidget* widget = QApplication::activeWindow()) {
        widget->setCursor(m_previousCursor);
    }

    // Uncheck the action if checkable
    if (m_parentAction && m_parentAction->isCheckable() &&
        m_parentAction->isChecked()) {
        m_parentAction->blockSignals(true);
        m_parentAction->setChecked(false);
        m_parentAction->blockSignals(false);
    }

    ActiveReaction = nullptr;

    CVLog::PrintDebug(
            QString("[cvSelectionReaction] Selection mode %1 deactivated")
                    .arg(static_cast<int>(m_mode)));
}

//-----------------------------------------------------------------------------
void cvSelectionReaction::onToolSelectionFinished(
        const cvSelectionData& selectionData) {
    emit selectionFinished(selectionData);
}

//-----------------------------------------------------------------------------
int cvSelectionReaction::getSelectionModifier() {
    if (!m_modifierGroup) {
        return static_cast<int>(SelectionModifier::SELECTION_DEFAULT);
    }

    QAction* checkedAction = m_modifierGroup->checkedAction();
    if (!checkedAction) {
        return static_cast<int>(SelectionModifier::SELECTION_DEFAULT);
    }

    // The action's data should contain the modifier value
    QVariant data = checkedAction->data();
    if (data.isValid()) {
        return data.toInt();
    }

    return static_cast<int>(SelectionModifier::SELECTION_DEFAULT);
}

//-----------------------------------------------------------------------------
bool cvSelectionReaction::isCompatible(SelectionMode mode) {
    return cvViewSelectionManager::instance()->isCompatible(m_mode, mode);
}

//-----------------------------------------------------------------------------
cvRenderViewSelectionTool* cvSelectionReaction::getOrCreateTool() {
    if (m_tool) {
        return m_tool;
    }

    // Use the manager to get or create the tool
    cvViewSelectionManager* manager = cvViewSelectionManager::instance();
    m_tool = manager->getOrCreateTool(m_mode);

    if (m_tool) {
        // Connect selection completed signal
        // Note: cvRenderViewSelectionTool emits selectionCompleted, not
        // selectionFinished We need to get the selection data from the manager
        // when this happens
        connect(m_tool, &cvRenderViewSelectionTool::selectionCompleted, this,
                [this]() {
                    cvViewSelectionManager* manager =
                            cvViewSelectionManager::instance();
                    if (manager) {
                        const cvSelectionData& data =
                                manager->currentSelection();
                        emit selectionFinished(data);
                    }
                });

        // For zoom to box mode, connect the zoomToBoxCompleted signal
        if (m_mode == SelectionMode::ZOOM_TO_BOX) {
            cvZoomBoxSelectionTool* zoomTool =
                    qobject_cast<cvZoomBoxSelectionTool*>(m_tool);
            if (zoomTool) {
                connect(zoomTool, &cvZoomBoxSelectionTool::zoomToBoxCompleted,
                        this, &cvSelectionReaction::zoomToBoxRequested);
                // Auto-exit zoom mode after zoom is completed (ParaView
                // behavior) Reference:
                // pqRenderViewSelectionReaction::selectionChanged() calls
                // endSelection()
                connect(zoomTool, &cvZoomBoxSelectionTool::zoomToBoxCompleted,
                        this, &cvSelectionReaction::endSelection);
            }
        }
    }

    return m_tool;
}

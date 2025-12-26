// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionToolController.h"

#include "cvSelectionBookmarks.h"
#include "cvSelectionData.h"
#include "cvSelectionHighlighter.h"
#include "cvSelectionHistory.h"
#include "cvSelectionReaction.h"
#include "cvSelectionTypes.h"  // For SelectionMode and SelectionModifier enums
#include "cvViewSelectionManager.h"

// CV_CORE_LIB
#include <CVLog.h>

// QT
#include <QAction>
#include <QActionGroup>
#include <QWidget>

//-----------------------------------------------------------------------------
cvSelectionToolController* cvSelectionToolController::instance() {
    static cvSelectionToolController _instance;
    return &_instance;
}

//-----------------------------------------------------------------------------
cvSelectionToolController::cvSelectionToolController(QObject* parent)
    : QObject(parent),
      m_parentWidget(nullptr),
      m_manager(cvViewSelectionManager::instance()),
      m_selectionToolsActive(false),
      m_modifierGroup(nullptr) {
    // Connect to manager signals
    // Use QOverload to select the correct signal overload
    connect(m_manager,
            QOverload<const cvSelectionData&>::of(
                    &cvViewSelectionManager::selectionChanged),
            this, [this](const cvSelectionData& data) {
                emit selectionFinished(data);
            });

    connect(m_manager,
            QOverload<>::of(&cvViewSelectionManager::selectionChanged), this,
            &cvSelectionToolController::selectionHistoryChanged);

    CVLog::PrintDebug("[cvSelectionToolController] Initialized");
}

//-----------------------------------------------------------------------------
cvSelectionToolController::~cvSelectionToolController() {
    // Clean up reactions
    for (auto reaction : m_reactions) {
        if (reaction) {
            delete reaction;
        }
    }
    m_reactions.clear();

    CVLog::PrintDebug("[cvSelectionToolController] Destroyed");
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::initialize(QWidget* parent) {
    m_parentWidget = parent;

    CVLog::PrintDebug("[cvSelectionToolController] Initialized with parent widget");
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::setVisualizer(ecvGenericVisualizer3D* viewer) {
    if (m_manager) {
        m_manager->setVisualizer(viewer);
    }
}

//-----------------------------------------------------------------------------
cvSelectionReaction* cvSelectionToolController::registerAction(
        QAction* action, SelectionMode mode) {
    if (!action) {
        CVLog::Warning(
                "[cvSelectionToolController] Cannot register null action");
        return nullptr;
    }

    // Create the reaction
    cvSelectionReaction* reaction =
            new cvSelectionReaction(action, mode, m_modifierGroup);

    // Connect selection finished signal
    connect(reaction, &cvSelectionReaction::selectionFinished, this,
            &cvSelectionToolController::onSelectionFinished);

    // Connect zoom to box signal if applicable
    if (mode == SelectionMode::ZOOM_TO_BOX) {
        connect(reaction, &cvSelectionReaction::zoomToBoxRequested, this,
                &cvSelectionToolController::zoomToBoxRequested);
    }

    // Monitor action state changes to track when selection tools are active
    // This enables selection properties panel display
    // NOTE: ZOOM_TO_BOX is excluded from selection tools - it doesn't produce
    // selection data and shouldn't show selection properties panel
    if (action->isCheckable()) {
        connect(action, &QAction::toggled, this, [this, mode](bool checked) {
            CVLog::PrintDebug(
                    QString("[cvSelectionToolController] Action for mode %1 %2")
                            .arg(static_cast<int>(mode))
                            .arg(checked ? "checked" : "unchecked"));

            // For ZOOM_TO_BOX, don't update selection properties state
            // It's not a selection tool and shouldn't affect the properties panel
            if (mode == SelectionMode::ZOOM_TO_BOX) {
                return;
            }

            // Update selection tools active state
            // A tool is active if any reaction's action is checked
            // EXCLUDE ZOOM_TO_BOX from this check - it's not a selection tool
            bool anyActive = false;
            for (auto it = m_reactions.constBegin();
                 it != m_reactions.constEnd(); ++it) {
                SelectionMode reactionMode = it.key();
                // Skip ZOOM_TO_BOX - it doesn't produce selection data
                if (reactionMode == SelectionMode::ZOOM_TO_BOX) {
                    continue;
                }
                
                QPointer<cvSelectionReaction> reaction = it.value();
                if (reaction && reaction->parentAction() &&
                    reaction->parentAction()->isCheckable() &&
                    reaction->parentAction()->isChecked()) {
                    anyActive = true;
                    break;
                }
            }

            setSelectionPropertiesActive(anyActive);
        });
    }

    // Store the reaction
    m_reactions[mode] = reaction;

    CVLog::PrintDebug(
            QString("[cvSelectionToolController] Registered action for "
                    "mode %1")
                    .arg(static_cast<int>(mode)));

    return reaction;
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::registerModifierActions(QAction* addAction,
                                                        QAction* subtractAction,
                                                        QAction* toggleAction) {
    m_addAction = addAction;
    m_subtractAction = subtractAction;
    m_toggleAction = toggleAction;

    // Create action group for mutual exclusivity
    m_modifierGroup = new QActionGroup(this);
    m_modifierGroup->setExclusive(false);  // Allow none to be checked

    if (addAction) {
        addAction->setData(
                static_cast<int>(SelectionModifier::SELECTION_ADDITION));
        m_modifierGroup->addAction(addAction);
    }
    if (subtractAction) {
        subtractAction->setData(
                static_cast<int>(SelectionModifier::SELECTION_SUBTRACTION));
        m_modifierGroup->addAction(subtractAction);
    }
    if (toggleAction) {
        toggleAction->setData(
                static_cast<int>(SelectionModifier::SELECTION_TOGGLE));
        m_modifierGroup->addAction(toggleAction);
    }

    // Connect modifier changes
    connect(m_modifierGroup, &QActionGroup::triggered, this,
            &cvSelectionToolController::onModifierChanged);

    CVLog::PrintDebug(
            "[cvSelectionToolController] Registered modifier actions");
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::registerManipulationActions(
        QAction* growAction, QAction* shrinkAction, QAction* clearAction) {
    m_growAction = growAction;
    m_shrinkAction = shrinkAction;
    m_clearAction = clearAction;

    // Register these with reactions
    if (growAction) {
        registerAction(growAction, SelectionMode::GROW_SELECTION);
    }
    if (shrinkAction) {
        registerAction(shrinkAction, SelectionMode::SHRINK_SELECTION);
    }
    if (clearAction) {
        registerAction(clearAction, SelectionMode::CLEAR_SELECTION);
    }

    CVLog::PrintDebug(
            "[cvSelectionToolController] Registered manipulation actions");
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::disableAllTools(cvSelectionReaction* except) {
    for (auto reaction : m_reactions) {
        if (reaction && reaction != except && reaction->isActive()) {
            // Trigger the action to toggle it off
            if (reaction->parentAction() &&
                reaction->parentAction()->isCheckable()) {
                reaction->parentAction()->blockSignals(true);
                reaction->parentAction()->setChecked(false);
                reaction->parentAction()->blockSignals(false);
            }
        }
    }

    // End the active reaction if it's not the exception
    if (cvSelectionReaction::activeReaction() &&
        cvSelectionReaction::activeReaction() != except) {
        // The endSelection is called automatically when action is unchecked
    }

    // Update selection properties panel
    if (except == nullptr) {
        setSelectionPropertiesActive(false);
    }

    emit selectionToolStateChanged(except != nullptr);
}

//-----------------------------------------------------------------------------
bool cvSelectionToolController::isAnyToolActive() const {
    return cvSelectionReaction::activeReaction() != nullptr;
}

//-----------------------------------------------------------------------------
cvSelectionToolController::SelectionMode
cvSelectionToolController::currentMode() const {
    if (m_manager) {
        return m_manager->getCurrentMode();
    }
    return static_cast<SelectionMode>(-1);
}

//-----------------------------------------------------------------------------
bool cvSelectionToolController::handleEscapeKey() {
    if (!isAnyToolActive()) {
        return false;
    }

    // Disable all selection tools via the reaction system
    disableAllTools(nullptr);

    // Also explicitly uncheck all actions to keep UI synchronized
    auto uncheckAction = [](QAction* action) {
        if (action && action->isCheckable() && action->isChecked()) {
            action->blockSignals(true);
            action->setChecked(false);
            action->blockSignals(false);
        }
    };

    uncheckAction(m_actions.selectSurfaceCells);
    uncheckAction(m_actions.selectSurfacePoints);
    uncheckAction(m_actions.selectFrustumCells);
    uncheckAction(m_actions.selectFrustumPoints);
    uncheckAction(m_actions.selectPolygonCells);
    uncheckAction(m_actions.selectPolygonPoints);
    uncheckAction(m_actions.selectBlocks);
    uncheckAction(m_actions.selectFrustumBlocks);
    uncheckAction(m_actions.interactiveSelectCells);
    uncheckAction(m_actions.interactiveSelectPoints);
    uncheckAction(m_actions.hoverCells);
    uncheckAction(m_actions.hoverPoints);
    uncheckAction(m_actions.zoomToBox);

    CVLog::PrintDebug(
            "[cvSelectionToolController] ESC key handled - disabled all "
            "selection tools");
    return true;
}

//-----------------------------------------------------------------------------
cvSelectionHighlighter* cvSelectionToolController::highlighter() const {
    if (m_manager) {
        return m_manager->getHighlighter();
    }
    return nullptr;
}

//-----------------------------------------------------------------------------
cvSelectionHistory* cvSelectionToolController::history() const {
    if (m_manager) {
        return m_manager->getHistory();
    }
    return nullptr;
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::setSelectionPropertiesActive(bool active) {
    if (m_selectionToolsActive != active) {
        m_selectionToolsActive = active;
        emit selectionToolStateChanged(active);
    }
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::onSelectionFinished(
        const cvSelectionData& selectionData) {
    // Update selection in manager
    if (m_manager) {
        m_manager->setCurrentSelection(selectionData);
    }

    // Update manipulation action states
    bool hasSelection = !selectionData.isEmpty();
    if (m_growAction) {
        m_growAction->setEnabled(hasSelection);
    }
    if (m_shrinkAction) {
        m_shrinkAction->setEnabled(hasSelection);
    }
    if (m_clearAction) {
        m_clearAction->setEnabled(hasSelection);
    }

    // Update selection properties widget via signal
    if (m_selectionToolsActive) {
        emit selectionPropertiesUpdateRequested(selectionData);
    }

    // Emit signal
    emit selectionFinished(selectionData);

    CVLog::PrintDebug(
            QString("[cvSelectionToolController] Selection finished: %1 %2")
                    .arg(selectionData.count())
                    .arg(selectionData.fieldTypeString()));
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::undoSelection() {
    if (m_manager && m_manager->canUndo()) {
        m_manager->undo();
    }
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::redoSelection() {
    if (m_manager && m_manager->canRedo()) {
        m_manager->redo();
    }
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::onModifierChanged(QAction* action) {
    if (!action || !m_manager) {
        return;
    }

    QVariant data = action->data();
    if (!data.isValid()) {
        return;
    }

    if (action->isChecked()) {
        m_manager->setSelectionModifier(
                static_cast<SelectionModifier>(data.toInt()));

        QString modeName;
        switch (data.toInt()) {
            case static_cast<int>(SelectionModifier::SELECTION_ADDITION):
                modeName = "ADD (Ctrl)";
                break;
            case static_cast<int>(SelectionModifier::SELECTION_SUBTRACTION):
                modeName = "SUBTRACT (Shift)";
                break;
            case static_cast<int>(SelectionModifier::SELECTION_TOGGLE):
                modeName = "TOGGLE (Ctrl+Shift)";
                break;
            default:
                modeName = "DEFAULT";
                break;
        }
        CVLog::PrintDebug(
                QString("[cvSelectionToolController] Selection modifier: %1")
                        .arg(modeName));
    } else {
        m_manager->setSelectionModifier(SelectionModifier::SELECTION_DEFAULT);
    }
}

//-----------------------------------------------------------------------------
// Note: onTooltipSettingsChanged has been removed as tooltip settings
// are now managed through cvSelectionLabelPropertiesDialog

//-----------------------------------------------------------------------------
bool cvSelectionToolController::isPCLBackendAvailable() {
#ifdef USE_PCL_BACKEND
    return true;
#else
    return false;
#endif
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::setupActions(const SelectionActions& actions) {
    m_actions = actions;

    // Register selection modifier actions first (used by other reactions)
    if (actions.addSelection || actions.subtractSelection ||
        actions.toggleSelection) {
        registerModifierActions(actions.addSelection, actions.subtractSelection,
                                actions.toggleSelection);
    }

    // Register surface selection actions
    if (actions.selectSurfaceCells) {
        registerAction(actions.selectSurfaceCells,
                       SelectionMode::SELECT_SURFACE_CELLS);
    }
    if (actions.selectSurfacePoints) {
        registerAction(actions.selectSurfacePoints,
                       SelectionMode::SELECT_SURFACE_POINTS);
    }

    // Register frustum selection actions
    if (actions.selectFrustumCells) {
        registerAction(actions.selectFrustumCells,
                       SelectionMode::SELECT_FRUSTUM_CELLS);
    }
    if (actions.selectFrustumPoints) {
        registerAction(actions.selectFrustumPoints,
                       SelectionMode::SELECT_FRUSTUM_POINTS);
    }

    // Register polygon selection actions
    if (actions.selectPolygonCells) {
        registerAction(actions.selectPolygonCells,
                       SelectionMode::SELECT_SURFACE_CELLS_POLYGON);
    }
    if (actions.selectPolygonPoints) {
        registerAction(actions.selectPolygonPoints,
                       SelectionMode::SELECT_SURFACE_POINTS_POLYGON);
    }

    // Register block selection actions
    if (actions.selectBlocks) {
        registerAction(actions.selectBlocks, SelectionMode::SELECT_BLOCKS);
    }
    if (actions.selectFrustumBlocks) {
        registerAction(actions.selectFrustumBlocks,
                       SelectionMode::SELECT_FRUSTUM_BLOCKS);
    }

    // Register interactive selection actions
    if (actions.interactiveSelectCells) {
        registerAction(actions.interactiveSelectCells,
                       SelectionMode::SELECT_SURFACE_CELLS_INTERACTIVELY);
    }
    if (actions.interactiveSelectPoints) {
        registerAction(actions.interactiveSelectPoints,
                       SelectionMode::SELECT_SURFACE_POINTS_INTERACTIVELY);
    }

    // Register hover/tooltip actions
    if (actions.hoverCells) {
        registerAction(actions.hoverCells, SelectionMode::HOVER_CELLS_TOOLTIP);
    }
    if (actions.hoverPoints) {
        registerAction(actions.hoverPoints,
                       SelectionMode::HOVER_POINTS_TOOLTIP);
    }

    // Register zoom to box action
    if (actions.zoomToBox) {
        registerAction(actions.zoomToBox, SelectionMode::ZOOM_TO_BOX);
    }

    // Register manipulation actions
    registerManipulationActions(actions.growSelection, actions.shrinkSelection,
                                actions.clearSelection);

    CVLog::PrintDebug(
            "[cvSelectionToolController] All selection actions registered");
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::connectHighlighter() {
    // Highlighter is now managed by cvViewSelectionManager
    // This method is kept for compatibility but does nothing
    CVLog::PrintDebug(
            "[cvSelectionToolController] Highlighter connection established "
            "via manager");
}

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionToolController.h"

#include "cvRenderViewSelectionReaction.h"
#include "cvSelectionData.h"
#include "cvSelectionHighlighter.h"
#include "cvSelectionPipeline.h"  // For invalidateCachedSelection
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

    CVLog::PrintVerbose("[cvSelectionToolController] Initialized");
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

    CVLog::PrintVerbose("[cvSelectionToolController] Destroyed");
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::initialize(QWidget* parent) {
    m_parentWidget = parent;

    CVLog::PrintVerbose(
            "[cvSelectionToolController] Initialized with parent widget");
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::setVisualizer(ecvGenericVisualizer3D* viewer) {
    if (m_manager) {
        m_manager->setVisualizer(viewer);
    }

    // Update all registered reactions with the visualizer
    // This is necessary because reactions may be registered before
    // the visualizer is set
    for (auto reaction : m_reactions) {
        if (reaction) {
            reaction->setVisualizer(viewer);
        }
    }

    CVLog::PrintVerbose(
            QString("[cvSelectionToolController] Visualizer set, updated %1 "
                    "reactions")
                    .arg(m_reactions.size()));
}

//-----------------------------------------------------------------------------
cvRenderViewSelectionReaction* cvSelectionToolController::registerAction(
        QAction* action, SelectionMode mode) {
    if (!action) {
        CVLog::Warning(
                "[cvSelectionToolController] Cannot register null action "
                "(new)");
        return nullptr;
    }

    // Create the new simplified reaction (ParaView-style)
    cvRenderViewSelectionReaction* reaction =
            new cvRenderViewSelectionReaction(action, mode, m_modifierGroup);

    // Set visualizer if already available
    if (m_manager && m_manager->getVisualizer()) {
        reaction->setVisualizer(m_manager->getVisualizer());
    }

    // Connect selection finished signal
    connect(reaction, &cvRenderViewSelectionReaction::selectionFinished, this,
            &cvSelectionToolController::onSelectionFinished);

    // Connect zoom to box signal if applicable
    if (mode == SelectionMode::ZOOM_TO_BOX) {
        connect(reaction, &cvRenderViewSelectionReaction::zoomToBoxCompleted,
                this, &cvSelectionToolController::zoomToBoxRequested);
    }

    // Monitor action state changes to track when selection tools are active
    if (action->isCheckable()) {
        connect(action, &QAction::toggled, this, [this, mode](bool checked) {
            CVLog::PrintVerbose(
                    QString("[cvSelectionToolController] Action (new) for mode "
                            "%1 %2")
                            .arg(static_cast<int>(mode))
                            .arg(checked ? "checked" : "unchecked"));

            // For ZOOM_TO_BOX, don't update selection properties state
            if (mode == SelectionMode::ZOOM_TO_BOX) {
                return;
            }

            // Update selection tools active state
            bool anyActive = false;
            for (auto it = m_reactions.constBegin();
                 it != m_reactions.constEnd(); ++it) {
                SelectionMode reactionMode = it.key();
                if (reactionMode == SelectionMode::ZOOM_TO_BOX) {
                    continue;
                }

                QPointer<cvRenderViewSelectionReaction> r = it.value();
                if (r && r->parentAction() &&
                    r->parentAction()->isCheckable() &&
                    r->parentAction()->isChecked()) {
                    anyActive = true;
                    break;
                }
            }

            setSelectionPropertiesActive(anyActive);
        });
    }

    // Store the reaction
    m_reactions[mode] = reaction;

    CVLog::PrintVerbose(
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
    // Reference: pqStandardViewFrameActionsImplementation.cxx lines 266-285
    m_modifierGroup = new QActionGroup(this);
    // Set non-exclusive to allow unchecking by clicking on the same button
    // We manually manage exclusivity in onModifierChanged
    m_modifierGroup->setExclusive(false);

    if (addAction) {
        addAction->setCheckable(true);
        addAction->setData(
                static_cast<int>(SelectionModifier::SELECTION_ADDITION));
        m_modifierGroup->addAction(addAction);
    }
    if (subtractAction) {
        subtractAction->setCheckable(true);
        subtractAction->setData(
                static_cast<int>(SelectionModifier::SELECTION_SUBTRACTION));
        m_modifierGroup->addAction(subtractAction);
    }
    if (toggleAction) {
        toggleAction->setCheckable(true);
        toggleAction->setData(
                static_cast<int>(SelectionModifier::SELECTION_TOGGLE));
        m_modifierGroup->addAction(toggleAction);
    }

    // Connect modifier changes using triggered signal
    // Reference: pqStandardViewFrameActionsImplementation.cxx lines 284-285
    connect(m_modifierGroup, &QActionGroup::triggered, this,
            &cvSelectionToolController::onModifierChanged);

    CVLog::PrintVerbose(
            "[cvSelectionToolController] Registered modifier actions");
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::registerManipulationActions(
        QAction* growAction, QAction* shrinkAction, QAction* clearAction) {
    m_growAction = growAction;
    m_shrinkAction = shrinkAction;
    m_clearAction = clearAction;

    // Register these with new architecture reactions
    // These are non-checkable instant actions, handled properly by
    // cvRenderViewSelectionReaction
    if (growAction) {
        registerAction(growAction, SelectionMode::GROW_SELECTION);
    }
    if (shrinkAction) {
        registerAction(shrinkAction, SelectionMode::SHRINK_SELECTION);
    }
    if (clearAction) {
        registerAction(clearAction, SelectionMode::CLEAR_SELECTION);
    }

    CVLog::PrintVerbose(
            "[cvSelectionToolController] Registered manipulation actions "
            "(using new architecture)");
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::disableAllTools(
        cvRenderViewSelectionReaction* except) {
    // Disable all reactions
    for (auto reaction : m_reactions) {
        if (reaction && reaction != except && reaction->isActive()) {
            if (reaction->parentAction() &&
                reaction->parentAction()->isCheckable()) {
                reaction->parentAction()->blockSignals(true);
                reaction->parentAction()->setChecked(false);
                reaction->parentAction()->blockSignals(false);
            }
        }
    }

    // End any active reaction
    cvRenderViewSelectionReaction::endActiveSelection();

    // Update selection properties panel
    if (except == nullptr) {
        setSelectionPropertiesActive(false);
    }

    emit selectionToolStateChanged(except != nullptr);
}

//-----------------------------------------------------------------------------
bool cvSelectionToolController::isAnyToolActive() const {
    return cvRenderViewSelectionReaction::activeReaction() != nullptr;
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

    CVLog::PrintVerbose(
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
// history() removed - UI not implemented

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
    // CRITICAL: Block manager's signal to prevent double emission of
    // selectionFinished The manager's selectionChanged is connected to emit
    // selectionFinished, and we're already inside onSelectionFinished called
    // from reaction's emit. This prevents:
    // reaction->selectionFinished->here->manager->selectionChanged->selectionFinished(again!)
    if (m_manager) {
        m_manager->blockSignals(true);
        m_manager->setCurrentSelection(selectionData);
        m_manager->blockSignals(false);
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

    CVLog::PrintVerbose(
            QString("[cvSelectionToolController] Selection finished: %1 %2")
                    .arg(selectionData.count())
                    .arg(selectionData.fieldTypeString()));
}

// undoSelection/redoSelection removed - UI not implemented

//-----------------------------------------------------------------------------
void cvSelectionToolController::onModifierChanged(QAction* action) {
    // Reference:
    // pqStandardViewFrameActionsImplementation::manageGroupExclusivity() lines
    // 851-866
    if (!action) {
        return;
    }

    // Handle non-checkable actions (shouldn't happen but be safe)
    if (!action->isCheckable()) {
        return;
    }

    // Implement ParaView-style group exclusivity management
    // When an action is checked, uncheck all others in the group
    // When an action is unchecked (by clicking it again), revert to default
    if (action->isChecked()) {
        // Manually uncheck other actions in the group
        // This is the key to ParaView's "manageGroupExclusivity" behavior
        if (m_modifierGroup) {
            for (QAction* groupAction : m_modifierGroup->actions()) {
                if (groupAction != action && groupAction->isChecked()) {
                    groupAction->blockSignals(true);
                    groupAction->setChecked(false);
                    groupAction->blockSignals(false);
                }
            }
        }

        // Set the modifier
        if (m_manager) {
            QVariant data = action->data();
            if (data.isValid()) {
                m_manager->setSelectionModifier(
                        static_cast<SelectionModifier>(data.toInt()));

                QString modeName;
                switch (data.toInt()) {
                    case static_cast<int>(
                            SelectionModifier::SELECTION_ADDITION):
                        modeName = "ADD (Ctrl)";
                        break;
                    case static_cast<int>(
                            SelectionModifier::SELECTION_SUBTRACTION):
                        modeName = "SUBTRACT (Shift)";
                        break;
                    case static_cast<int>(SelectionModifier::SELECTION_TOGGLE):
                        modeName = "TOGGLE (Ctrl+Shift)";
                        break;
                    default:
                        modeName = "DEFAULT";
                        break;
                }
                CVLog::PrintVerbose(
                        QString("[cvSelectionToolController] Selection "
                                "modifier: %1")
                                .arg(modeName));
            }
        }
    } else {
        // Action was unchecked - revert to default
        if (m_manager) {
            m_manager->setSelectionModifier(
                    SelectionModifier::SELECTION_DEFAULT);
            CVLog::PrintVerbose(
                    "[cvSelectionToolController] Selection modifier: DEFAULT");
        }
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

    // NOTE: Using registerAction() for simplified ParaView-aligned architecture
    // cvRenderViewSelectionReaction handles all selection logic directly
    // without the intermediate cvRenderViewSelectionTool layer

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

    // Register manipulation actions (still using old API as they are simple)
    registerManipulationActions(actions.growSelection, actions.shrinkSelection,
                                actions.clearSelection);

    CVLog::PrintVerbose(
            "[cvSelectionToolController] All selection actions registered "
            "(using new ParaView-aligned architecture)");
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::connectHighlighter() {
    // Highlighter is now managed by cvViewSelectionManager
    // This method is kept for compatibility but does nothing
    CVLog::PrintVerbose(
            "[cvSelectionToolController] Highlighter connection established "
            "via manager");
}

//-----------------------------------------------------------------------------
void cvSelectionToolController::invalidateCache() {
    if (m_manager) {
        // Invalidate cached selection in the pipeline
        cvSelectionPipeline* pipeline = m_manager->getPipeline();
        if (pipeline) {
            pipeline->invalidateCachedSelection();
            CVLog::PrintVerbose(
                    "[cvSelectionToolController] Selection cache invalidated "
                    "(scene content changed)");
        }

        // Clear source object since scene changed
        m_manager->setSourceObject(nullptr);
    }
}

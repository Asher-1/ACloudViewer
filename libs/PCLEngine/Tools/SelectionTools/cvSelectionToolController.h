// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Qt - must be included before other headers for MOC to work correctly
#include <QtGlobal>

// clang-format off
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
#include <QActionGroup>
#include <QMap>
#include <QObject>
#include <QPointer>
#else
#include <QtWidgets/QActionGroup>
#include <QtCore/QMap>
#include <QtCore/QObject>
#include <QtCore/QPointer>
#endif
// clang-format on

#include "cvSelectionData.h"  // Contains SelectionMode, SelectionModifier enums  // For SelectionMode and SelectionModifier enums
#include "cvViewSelectionManager.h"
#include "qPCL.h"

class QAction;
class QMenu;
class QToolBar;
class cvRenderViewSelectionReaction;
class cvSelectionData;
class cvSelectionHighlighter;
class cvSelectionHistory;
class cvSelectionBookmarks;
class ecvGenericVisualizer3D;

/**
 * @brief Controller class that manages all selection tools and their UI
 *
 * This class follows ParaView's pattern of separating selection tool
 * management from the main window. It encapsulates all the selection
 * tool creation, activation, and UI synchronization logic.
 *
 * Reference: ParaView's pqStandardViewFrameActionsImplementation
 */
class QPCL_ENGINE_LIB_API cvSelectionToolController : public QObject {
    Q_OBJECT

public:
    using SelectionMode = ::SelectionMode;
    using SelectionModifier = ::SelectionModifier;

    /**
     * @brief Get the singleton instance
     */
    static cvSelectionToolController* instance();

    /**
     * @brief Initialize the controller with UI elements
     * @param parent Parent widget (usually MainWindow)
     */
    void initialize(QWidget* parent);

    /**
     * @brief Set the visualizer for all selection operations
     */
    void setVisualizer(ecvGenericVisualizer3D* viewer);

    /**
     * @brief Setup all selection actions from a UI struct
     *
     * This method registers all selection-related QActions and creates
     * the corresponding reactions. Call this once during MainWindow
     * initialization.
     *
     * @param ui Pointer to the UI object containing action pointers
     *
     * The UI object should have these actions:
     * - actionSelectSurfaceCells, actionSelectSurfacePoints
     * - actionSelectFrustumCells, actionSelectFrustumPoints
     * - actionSelectPolygonCells, actionSelectPolygonPoints
     * - actionSelectBlocks, actionSelectFrustumBlocks
     * - actionInteractiveSelectCells, actionInteractiveSelectPoints
     * - actionHoverCells, actionHoverPoints
     * - actionAddSelection, actionSubtractSelection, actionToggleSelection
     * - actionGrowSelection, actionShrinkSelection, actionClearSelection
     * - actionZoomToBox
     */
    template <typename UiType>
    void setupActions(UiType* ui);

    /**
     * @brief Check if PCL backend is available
     */
    static bool isPCLBackendAvailable();

    /**
     * @brief Structure to hold all selection action pointers
     *
     * This provides a clean interface for registering actions
     * without depending on specific UI class types.
     */
    struct SelectionActions {
        // Surface selection
        QAction* selectSurfaceCells = nullptr;
        QAction* selectSurfacePoints = nullptr;

        // Frustum selection
        QAction* selectFrustumCells = nullptr;
        QAction* selectFrustumPoints = nullptr;

        // Polygon selection
        QAction* selectPolygonCells = nullptr;
        QAction* selectPolygonPoints = nullptr;

        // Block selection
        QAction* selectBlocks = nullptr;
        QAction* selectFrustumBlocks = nullptr;

        // Interactive selection
        QAction* interactiveSelectCells = nullptr;
        QAction* interactiveSelectPoints = nullptr;

        // Hover/tooltip selection
        QAction* hoverCells = nullptr;
        QAction* hoverPoints = nullptr;

        // Selection modifiers
        QAction* addSelection = nullptr;
        QAction* subtractSelection = nullptr;
        QAction* toggleSelection = nullptr;

        // Selection manipulation
        QAction* growSelection = nullptr;
        QAction* shrinkSelection = nullptr;
        QAction* clearSelection = nullptr;

        // Zoom to box
        QAction* zoomToBox = nullptr;
    };

    /**
     * @brief Setup all selection actions
     * @param actions Structure containing all action pointers
     */
    void setupActions(const SelectionActions& actions);

    /**
     * @brief Connect highlighter for visual feedback
     */
    void connectHighlighter();

    /**
     * @brief Get db root delegate (if connected)
     */
    QObject* getPropertiesDelegate() const { return m_propertiesDelegate; }

    /**
     * @brief Set the properties delegate for selection properties updates
     */
    void setPropertiesDelegate(QObject* delegate) {
        m_propertiesDelegate = delegate;
    }

    /**
     * @brief Register an action for a selection mode
     * @param action The QAction to register
     * @param mode The selection mode
     * @return The created cvRenderViewSelectionReaction
     *
     * This method uses the simplified architecture that aligns with ParaView's
     * pqRenderViewSelectionReaction pattern. All selection logic is contained
     * in a single class without the need for intermediate manager/tool layers.
     */
    cvRenderViewSelectionReaction* registerAction(QAction* action,
                                                  SelectionMode mode);

    /**
     * @brief Register the selection modifier action group
     * @param addAction Add selection action (Ctrl)
     * @param subtractAction Subtract selection action (Shift)
     * @param toggleAction Toggle selection action (Ctrl+Shift)
     */
    void registerModifierActions(QAction* addAction,
                                 QAction* subtractAction,
                                 QAction* toggleAction);

    /**
     * @brief Register grow/shrink/clear actions
     */
    void registerManipulationActions(QAction* growAction,
                                     QAction* shrinkAction,
                                     QAction* clearAction);

    /**
     * @brief Disable all selection tools
     * @param except Tool to keep active (nullptr to disable all)
     */
    void disableAllTools(cvRenderViewSelectionReaction* except = nullptr);

    /**
     * @brief Check if any selection tool is active
     */
    bool isAnyToolActive() const;

    /**
     * @brief Get the current active selection mode
     */
    SelectionMode currentMode() const;

    /**
     * @brief Handle ESC key to exit selection tools
     * @return true if handled, false otherwise
     */
    bool handleEscapeKey();

    /**
     * @brief Get the selection manager
     */
    cvViewSelectionManager* manager() const { return m_manager; }

    /**
     * @brief Get the selection highlighter
     */
    cvSelectionHighlighter* highlighter() const;

    // history() removed - UI not implemented

    /**
     * @brief Update selection properties widget state
     * @param active Whether selection tools are active
     */
    void setSelectionPropertiesActive(bool active);

    /**
     * @brief Invalidate cached selection data
     *
     * Call this when scene content changes (e.g., new entity added/removed)
     * to ensure stale selection data is not used.
     */
    void invalidateCache();

signals:
    /**
     * @brief Emitted when a selection operation is completed
     */
    void selectionFinished(const cvSelectionData& selectionData);

    /**
     * @brief Emitted when selection tool state changes
     * Connect this to ecvPropertiesTreeDelegate::setSelectionToolsActive()
     */
    void selectionToolStateChanged(bool anyToolActive);

    /**
     * @brief Emitted when selection properties should be updated
     * Connect this to ecvPropertiesTreeDelegate::updateSelectionProperties()
     */
    void selectionPropertiesUpdateRequested(
            const cvSelectionData& selectionData);

    /**
     * @brief Emitted when selection history changes
     */
    void selectionHistoryChanged();

    /**
     * @brief Emitted when bookmarks change
     */
    void bookmarksChanged();

    /**
     * @brief Emitted for zoom to box requests
     */
    void zoomToBoxRequested(int xmin, int ymin, int xmax, int ymax);

public slots:
    /**
     * @brief Handle selection finished from any tool
     */
    void onSelectionFinished(const cvSelectionData& selectionData);

    // undoSelection/redoSelection removed - UI not implemented

    /**
     * @brief Handle modifier action changes
     */
    void onModifierChanged(QAction* action);

    // Note: onTooltipSettingsChanged has been removed as tooltip settings
    // are now managed through cvSelectionLabelPropertiesDialog

private:
    explicit cvSelectionToolController(QObject* parent = nullptr);
    ~cvSelectionToolController() override;

    Q_DISABLE_COPY(cvSelectionToolController)

    // Core components
    QPointer<QWidget> m_parentWidget;
    cvViewSelectionManager* m_manager;
    bool m_selectionToolsActive;
    QPointer<QObject> m_propertiesDelegate;

    // Reactions for each selection mode
    QMap<SelectionMode, QPointer<cvRenderViewSelectionReaction>> m_reactions;

    // Store actions for ESC handling
    SelectionActions m_actions;

    // Modifier action group
    QPointer<QActionGroup> m_modifierGroup;
    QPointer<QAction> m_addAction;
    QPointer<QAction> m_subtractAction;
    QPointer<QAction> m_toggleAction;

    // Manipulation actions
    QPointer<QAction> m_growAction;
    QPointer<QAction> m_shrinkAction;
    QPointer<QAction> m_clearAction;
};

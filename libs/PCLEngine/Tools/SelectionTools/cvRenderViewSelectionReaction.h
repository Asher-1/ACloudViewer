// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file cvRenderViewSelectionReaction.h
 * @brief Simplified selection reaction class aligned with ParaView architecture
 *
 * This class combines the functionality of the original cvSelectionReaction and
 * cvRenderViewSelectionTool into a single, streamlined class that directly
 * mirrors ParaView's pqRenderViewSelectionReaction pattern.
 *
 * Architecture Goals:
 * - Single class manages all selection logic (like ParaView)
 * - Static ActiveReaction ensures only one selection mode is active
 * - Direct cursor/style management without external dependencies
 * - Clean state transitions via beginSelection()/endSelection()
 *
 * Reference: ParaView/Qt/ApplicationComponents/pqRenderViewSelectionReaction.h
 */

// clang-format off
// Qt - must be included before other headers for MOC to work correctly
#include <QtGlobal>
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
#include <QAction>
#include <QCursor>
#include <QObject>
#include <QPointer>
#include <QShortcut>
#include <QTimer>
#else
#include <QtWidgets/QAction>
#include <QtGui/QCursor>
#include <QtCore/QObject>
#include <QtCore/QPointer>
#include <QtWidgets/QShortcut>
#include <QtCore/QTimer>
#endif
// clang-format on

#include "cvSelectionData.h"
#include "cvSelectionData.h"  // Contains SelectionMode, SelectionModifier enums
#include "qPCL.h"

// VTK
#include <vtkSmartPointer.h>
#include <vtkWeakPointer.h>

// Forward declarations
class QActionGroup;
class ecvGenericVisualizer3D;
class cvSelectionPipeline;
class cvSelectionHighlighter;
class vtkObject;
class vtkRenderWindowInteractor;
class vtkRenderer;
class vtkInteractorStyle;
class vtkIntArray;
class vtkPolyData;

namespace PclUtils {
class PCLVis;
}

/**
 * @brief cvRenderViewSelectionReaction handles all selection modes in a single
 * class
 *
 * This class follows ParaView's pqRenderViewSelectionReaction pattern:
 * - One instance per selection action (button/menu item)
 * - Static ActiveReaction tracks the currently active selection
 * - beginSelection()/endSelection() handle all state transitions
 * - All cursor, interactor style, and observer management is internal
 *
 * Key improvements over the previous multi-layer architecture:
 * - No separate cvViewSelectionManager layer
 * - No separate cvRenderViewSelectionTool layer
 * - All selection logic in one place for easier maintenance
 * - Direct alignment with ParaView's proven architecture
 *
 * Reference: pqRenderViewSelectionReaction.{h,cxx}
 */
class QPCL_ENGINE_LIB_API cvRenderViewSelectionReaction : public QObject {
    Q_OBJECT

public:
    using SelectionMode = ::SelectionMode;
    using SelectionModifier = ::SelectionModifier;

    /**
     * @brief Constructor
     * @param parentAction The QAction that triggers this reaction
     * @param mode The selection mode for this reaction
     * @param modifierGroup Optional action group for selection modifiers
     */
    cvRenderViewSelectionReaction(QAction* parentAction,
                                  SelectionMode mode,
                                  QActionGroup* modifierGroup = nullptr);

    ~cvRenderViewSelectionReaction() override;

    /**
     * @brief Get the parent action
     */
    QAction* parentAction() const { return m_parentAction; }

    /**
     * @brief Get the selection mode
     */
    SelectionMode mode() const { return m_mode; }

    /**
     * @brief Set the visualizer for selection operations
     * @param viewer The generic 3D visualizer
     *
     * This should be called once before any selection operations.
     */
    void setVisualizer(ecvGenericVisualizer3D* viewer);

    /**
     * @brief Get the current visualizer
     */
    ecvGenericVisualizer3D* getVisualizer() const { return m_viewer; }

    /**
     * @brief Check if this reaction's selection is currently active
     */
    bool isActive() const;

    /**
     * @brief Get the currently active reaction (static)
     * @return The currently active reaction, or nullptr if none
     *
     * Reference: pqRenderViewSelectionReaction::ActiveReaction
     */
    static cvRenderViewSelectionReaction* activeReaction() {
        return ActiveReaction;
    }

    /**
     * @brief Force end any active selection
     *
     * Utility method to ensure clean state when switching views or
     * performing operations that require no active selection.
     */
    static void endActiveSelection();

    //-------------------------------------------------------------------------
    // Public event handlers (called by VTK callbacks)
    //-------------------------------------------------------------------------

    /**
     * @brief Handle selection changed event from VTK (public for callback
     * access)
     */
    void handleSelectionChanged(vtkObject* caller,
                                unsigned long eventId,
                                void* callData);

    /**
     * @brief Handle mouse move event (public for callback access)
     */
    Q_INVOKABLE void handleMouseMove();

    /**
     * @brief Handle left button press (public for callback access)
     */
    Q_INVOKABLE void handleLeftButtonPress();

    /**
     * @brief Handle left button release (public for callback access)
     */
    Q_INVOKABLE void handleLeftButtonRelease();

    /**
     * @brief Handle wheel rotation (public for callback access)
     */
    Q_INVOKABLE void handleWheelRotate();

    /**
     * @brief Handle right button press (public for callback access)
     */
    Q_INVOKABLE void handleRightButtonPressed();

    /**
     * @brief Handle right button release (public for callback access)
     */
    Q_INVOKABLE void handleRightButtonRelease();

protected:
    ///@{
    /**
     * @brief VTK callback functions for observer pattern
     *
     * These methods follow VTK's callback signature and are used with
     * vtkObject::AddObserver() to directly bind member functions.
     * This matches ParaView's implementation pattern.
     *
     * Reference: pqRenderViewSelectionReaction uses the same pattern
     */
    void vtkOnSelectionChanged(vtkObject* caller,
                               unsigned long eventId,
                               void* callData);
    void vtkOnMouseMove(vtkObject* caller,
                        unsigned long eventId,
                        void* callData);
    void vtkOnLeftButtonPress(vtkObject* caller,
                              unsigned long eventId,
                              void* callData);
    void vtkOnLeftButtonRelease(vtkObject* caller,
                                unsigned long eventId,
                                void* callData);
    void vtkOnWheelRotate(vtkObject* caller,
                          unsigned long eventId,
                          void* callData);
    void vtkOnRightButtonPressed(vtkObject* caller,
                                 unsigned long eventId,
                                 void* callData);
    void vtkOnRightButtonRelease(vtkObject* caller,
                                 unsigned long eventId,
                                 void* callData);
    void vtkOnMiddleButtonPressed(vtkObject* caller,
                                  unsigned long eventId,
                                  void* callData);
    void vtkOnMiddleButtonRelease(vtkObject* caller,
                                  unsigned long eventId,
                                  void* callData);
    ///@}

signals:
    /**
     * @brief Emitted when selection is finished
     * @param selectionData The resulting selection data
     */
    void selectionFinished(const cvSelectionData& selectionData);

    /**
     * @brief Emitted for custom box selection
     * @param region The selected region [x1, y1, x2, y2]
     */
    void selectedCustomBox(int xmin, int ymin, int xmax, int ymax);
    void selectedCustomBox(const int region[4]);

    /**
     * @brief Emitted for custom polygon selection
     * @param polygon Array of polygon vertices
     */
    void selectedCustomPolygon(vtkIntArray* polygon);

    /**
     * @brief Emitted when zoom to box is completed
     */
    void zoomToBoxCompleted(int xmin, int ymin, int xmax, int ymax);

public slots:
    /**
     * @brief Called when the action is triggered
     *
     * For checkable actions, calls beginSelection() or endSelection().
     * For non-checkable actions, calls both in sequence.
     *
     * Reference: pqRenderViewSelectionReaction::actionTriggered()
     */
    virtual void actionTriggered(bool val);

    /**
     * @brief Updates the enabled state of the action
     *
     * Handles enable state for CLEAR_SELECTION, GROW_SELECTION,
     * and SHRINK_SELECTION modes.
     *
     * Reference: pqRenderViewSelectionReaction::updateEnableState()
     */
    virtual void updateEnableState();

protected slots:
    /**
     * @brief Starts the selection mode
     *
     * This will:
     * 1. End any other active selection
     * 2. Store the current interactor style
     * 3. Set up the selection-specific interactor style
     * 4. Change the cursor
     * 5. Register event observers
     * 6. Set parentAction as checked
     *
     * Reference: pqRenderViewSelectionReaction::beginSelection()
     */
    virtual void beginSelection();

    /**
     * @brief Ends the selection mode
     *
     * This will:
     * 1. Restore the previous interactor style
     * 2. Restore the cursor
     * 3. Remove event observers
     * 4. Set parentAction as unchecked
     * 5. Clean up resources
     *
     * Reference: pqRenderViewSelectionReaction::endSelection()
     */
    virtual void endSelection();

    /**
     * @brief Handle mouse stop event (for tooltip display)
     *
     * Reference: pqRenderViewSelectionReaction::onMouseStop()
     */
    virtual void onMouseStop();

    /**
     * @brief Clear selection cache when data changes
     *
     * Reference: pqRenderViewSelectionReaction::clearSelectionCache()
     */
    virtual void clearSelectionCache();

protected:
    /**
     * @brief Callback for selection changed event from VTK
     *
     * This is the main callback that executes the selection logic.
     *
     * @param caller The VTK object that triggered the event
     * @param eventId The event ID
     * @param callData Event-specific data (region or polygon)
     *
     * Reference: pqRenderViewSelectionReaction::selectionChanged()
     */
    virtual void selectionChanged(vtkObject* caller,
                                  unsigned long eventId,
                                  void* callData);

    /**
     * @brief Handle mouse move events (for interactive selection)
     *
     * Reference: pqRenderViewSelectionReaction::onMouseMove()
     */
    virtual void onMouseMove();

    /**
     * @brief Handle left button press (for zoom tracking)
     */
    virtual void onLeftButtonPress();

    /**
     * @brief Handle left button release (for interactive selection)
     *
     * Reference: pqRenderViewSelectionReaction::onLeftButtonRelease()
     */
    virtual void onLeftButtonRelease();

    /**
     * @brief Handle wheel rotation events
     *
     * Reference: pqRenderViewSelectionReaction::onWheelRotate()
     */
    virtual void onWheelRotate();

    /**
     * @brief Handle right button press (disable pre-selection)
     */
    virtual void onRightButtonPressed();

    /**
     * @brief Handle right button release (re-enable pre-selection)
     */
    virtual void onRightButtonRelease();

    /**
     * @brief Handle middle button press (disable pre-selection during pan)
     */
    virtual void onMiddleButtonPressed();

    /**
     * @brief Handle middle button release (re-enable pre-selection, invalidate
     * cache)
     */
    virtual void onMiddleButtonRelease();

    /**
     * @brief Perform pre-selection highlighting
     *
     * Reference: pqRenderViewSelectionReaction::preSelection()
     */
    virtual void preSelection();

    /**
     * @brief Perform fast pre-selection using cached buffers
     *
     * Reference: pqRenderViewSelectionReaction::fastPreSelection()
     */
    virtual void fastPreSelection();

    /**
     * @brief Update tooltip display
     *
     * Reference: pqRenderViewSelectionReaction::UpdateTooltip()
     */
    virtual void updateTooltip();

    /**
     * @brief Get the selection modifier from keyboard state
     * @return The current selection modifier
     *
     * Checks Ctrl, Shift keys and modifier group to determine modifier.
     *
     * Reference: pqRenderViewSelectionReaction::getSelectionModifier()
     */
    virtual int getSelectionModifier();

    /**
     * @brief Check if this selection is compatible with another mode
     *
     * Used to determine if selection modifier buttons should remain active.
     *
     * Reference: pqRenderViewSelectionReaction::isCompatible()
     */
    virtual bool isCompatible(SelectionMode mode);

    /**
     * @brief Clean up event observers
     *
     * Reference: pqRenderViewSelectionReaction::cleanupObservers()
     */
    virtual void cleanupObservers();

    /**
     * @brief Uncheck selection modifier buttons
     */
    void uncheckSelectionModifiers();

    //-------------------------------------------------------------------------
    // Selection execution methods
    //-------------------------------------------------------------------------

    /**
     * @brief Select cells on surface
     * @param region Screen-space rectangle [x1, y1, x2, y2]
     * @param selectionModifier Selection modifier
     */
    void selectCellsOnSurface(int region[4], int selectionModifier);

    /**
     * @brief Select points on surface
     * @param region Screen-space rectangle [x1, y1, x2, y2]
     * @param selectionModifier Selection modifier
     */
    void selectPointsOnSurface(int region[4], int selectionModifier);

    /**
     * @brief Select cells in frustum
     * @param region Screen-space rectangle [x1, y1, x2, y2]
     * @param selectionModifier Selection modifier
     */
    void selectFrustumCells(int region[4], int selectionModifier);

    /**
     * @brief Select points in frustum
     * @param region Screen-space rectangle [x1, y1, x2, y2]
     * @param selectionModifier Selection modifier
     */
    void selectFrustumPoints(int region[4], int selectionModifier);

    /**
     * @brief Unified selection finalization (ParaView-style)
     * Combines new selection with existing, updates manager, and shows
     * highlights
     * @param newSelection The newly selected elements
     * @param selectionModifier How to combine with existing selection
     * @param description Debug description for logging
     */
    void finalizeSelection(const cvSelectionData& newSelection,
                           int selectionModifier,
                           const QString& description);

    /**
     * @brief Select cells in polygon
     * @param polygon Polygon vertices
     * @param selectionModifier Selection modifier
     */
    void selectPolygonCells(vtkIntArray* polygon, int selectionModifier);

    /**
     * @brief Select points in polygon
     * @param polygon Polygon vertices
     * @param selectionModifier Selection modifier
     */
    void selectPolygonPoints(vtkIntArray* polygon, int selectionModifier);

    /**
     * @brief Select blocks
     * @param region Screen-space rectangle [x1, y1, x2, y2]
     * @param selectionModifier Selection modifier
     */
    void selectBlock(int region[4], int selectionModifier);

    //-------------------------------------------------------------------------
    // Internal helpers
    //-------------------------------------------------------------------------

    /**
     * @brief Get the PCLVis instance
     */
    PclUtils::PCLVis* getPCLVis() const;

    /**
     * @brief Get the selection pipeline
     */
    cvSelectionPipeline* getSelectionPipeline() const;

    /**
     * @brief Get the selection highlighter
     */
    cvSelectionHighlighter* getSelectionHighlighter() const;

    /**
     * @brief Check if currently selecting cells (vs points)
     */
    bool isSelectingCells() const;

    /**
     * @brief Check if this is an interactive selection mode
     */
    bool isInteractiveMode() const;

    /**
     * @brief Check if this is a tooltip mode
     */
    bool isTooltipMode() const;

    /**
     * @brief Set cursor on the view
     */
    void setCursor(const QCursor& cursor);

    /**
     * @brief Restore default cursor
     */
    void unsetCursor();

    /**
     * @brief Store current interactor style
     */
    void storeCurrentStyle();

    /**
     * @brief Restore previous interactor style
     */
    void restoreStyle();

    /**
     * @brief Set up interactor style for this selection mode
     */
    void setupInteractorStyle();

    /**
     * @brief Set up event observers for this selection mode
     */
    void setupObservers();

    /**
     * @brief Add camera movement observers for cache invalidation
     * @param startIndex Starting index in m_observerIds array
     *
     * Adds observers for: WheelForward, WheelBackward, RightButtonPress,
     * RightButtonRelease, MiddleButtonPress, MiddleButtonRelease
     */
    void addCameraMovementObservers(int startIndex);

    /**
     * @brief Set vtkInteractorStyleRubberBand3D as the interactor style
     * @param renderOnMouseMove Whether to render on mouse move events
     *
     * This style disables rotation: Left=RubberBand, Middle=Pan, Right=Zoom
     */
    void setRubberBand3DStyle(bool renderOnMouseMove);

    /**
     * @brief Show instruction dialog for interactive modes
     */
    void showInstructionDialog();

private:
    // Qt objects
    QPointer<QAction> m_parentAction;
    QPointer<QActionGroup> m_modifierGroup;

    // Selection mode
    SelectionMode m_mode;

    // Visualizer reference (not owned)
    ecvGenericVisualizer3D* m_viewer = nullptr;

    // VTK objects
    vtkRenderWindowInteractor* m_interactor = nullptr;
    vtkRenderer* m_renderer = nullptr;
    vtkSmartPointer<vtkInteractorStyle> m_previousStyle;
    vtkSmartPointer<vtkInteractorStyle> m_selectionStyle;

    // Previous render view mode (for restoration)
    int m_previousRenderViewMode = -1;

    // Event observer IDs for interactor events (increased to 8 for pan/zoom)
    unsigned long m_observerIds[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    vtkWeakPointer<vtkObject> m_observedObject;

    // Separate observer ID for selection style (SelectionChangedEvent)
    unsigned long m_styleObserverId = 0;

    // Cursor
    QCursor m_zoomCursor;

    // Mouse move state (for tooltip timing)
    QTimer m_mouseMovingTimer;
    bool m_mouseMoving = false;
    int m_mousePosition[2] = {0, 0};

    // Zoom box tracking (for zoomToBoxCompleted signal)
    int m_zoomStartPosition[2] = {0, 0};
    bool m_zoomTracking = false;

    // Pre-selection state
    bool m_disablePreSelection = false;

    // Tooltip state
    QString m_plainTooltipText;
    QShortcut* m_copyTooltipShortcut = nullptr;

    // Current highlight state (for interactive modes)
    vtkIdType m_hoveredId = -1;
    vtkPolyData* m_currentPolyData = nullptr;

    // Current selection data
    cvSelectionData m_currentSelection;

    // Static: only one reaction can be active at a time
    // Reference: pqRenderViewSelectionReaction::ActiveReaction
    static QPointer<cvRenderViewSelectionReaction> ActiveReaction;

    // Tooltip display wait time (ms)
    static const int TOOLTIP_WAITING_TIME = 400;
};

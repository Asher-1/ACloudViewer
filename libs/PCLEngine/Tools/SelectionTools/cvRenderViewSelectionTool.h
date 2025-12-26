// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// clang-format off
// QT - must be included before other headers for MOC to work correctly
#include <QtGui/QCursor>
#include <QtCore/QObject>
// clang-format on

// LOCAL
#include "cvGenericSelectionTool.h"
#include "qPCL.h"

// Forward declaration - full definition included in cvRenderViewSelectionTool.cpp
// to avoid circular dependency with cvViewSelectionManager.h
class cvViewSelectionManager;

// VTK
#include <vtkSmartPointer.h>

// Forward declarations
class vtkRenderWindowInteractor;
class vtkRenderer;
class vtkInteractorStyle;
class vtkCommand;
class vtkObject;
class vtkIntArray;
class vtkPolyData;

/**
 * @brief Base class for all selection tools
 *
 * This class provides the common infrastructure for implementing
 * different selection modes. Each specific selection mode should
 * inherit from this class and implement the selection logic.
 *
 * Based on ParaView's pqRenderViewSelectionReaction.
 *
 * Reference: pqRenderViewSelectionReaction.{h,cxx}
 */
class QPCL_ENGINE_LIB_API cvRenderViewSelectionTool
    : public QObject,
      public cvGenericSelectionTool {
    Q_OBJECT

    friend class cvSelectionCallback;  // Allow callback to access protected
                                       // methods

public:
    // SelectionMode and SelectionModifier are now defined in cvSelectionTypes.h
    // to avoid circular dependency with cvViewSelectionManager

    explicit cvRenderViewSelectionTool(SelectionMode mode,
                                       QObject* parent = nullptr);
    ~cvRenderViewSelectionTool() override;

    /**
     * @brief Set the visualizer for this tool
     * @param viewer The generic 3D visualizer
     *
     * Overrides cvGenericSelectionTool::setVisualizer to also update
     * VTK-specific renderer and interactor references.
     */
    void setVisualizer(ecvGenericVisualizer3D* viewer) override;

    /**
     * @brief Set the selection modifier
     * @param modifier The modifier to use
     */
    void setSelectionModifier(SelectionModifier modifier);

    /**
     * @brief Get the current selection modifier
     */
    SelectionModifier getSelectionModifier() const { return m_modifier; }

    /**
     * @brief Enable this selection tool
     *
     * This will:
     * 1. Store the current interactor style
     * 2. Set up the selection-specific interactor style
     * 3. Change the cursor
     * 4. Register event observers
     *
     * Reference: pqRenderViewSelectionReaction.cxx, beginSelection()
     */
    virtual void enable();

    /**
     * @brief Disable this selection tool
     *
     * This will:
     * 1. Restore the previous interactor style
     * 2. Restore the cursor
     * 3. Remove event observers
     * 4. Clean up resources
     *
     * Reference: pqRenderViewSelectionReaction.cxx, endSelection()
     */
    virtual void disable();

    /**
     * @brief Check if this tool is currently enabled
     */
    bool isEnabled() const { return m_enabled; }

    /**
     * @brief Get the selection mode
     */
    SelectionMode getMode() const { return m_mode; }

    /**
     * @brief Get the cursor for this selection mode
     */
    virtual QCursor getCursor() const;

signals:
    /**
     * @brief Emitted when a selection operation is completed
     */
    void selectionCompleted();

    /**
     * @brief Emitted when the selection has changed
     */
    void selectionChanged();

    /**
     * @brief Emitted when the tool is enabled/disabled
     */
    void enabledChanged(bool enabled);

protected:
    /**
     * @brief Store the current interactor style before switching
     *
     * Reference: pqRenderViewSelectionReaction.cxx, line 335-336
     */
    virtual void storeCurrentStyle();

    /**
     * @brief Restore the previous interactor style
     *
     * Reference: pqRenderViewSelectionReaction.cxx, line 510-513
     */
    virtual void restoreStyle();

    /**
     * @brief Set up the interactor style for this selection mode
     *
     * Subclasses should override this to set up mode-specific styles.
     * For example:
     * - Rectangle selection: vtkInteractorStyleRubberBandPick
     * - Polygon selection: vtkInteractorStyleDrawPolygon
     * - Zoom: vtkInteractorStyleRubberBandZoom
     *
     * Reference: pqRenderViewSelectionReaction.cxx, line 338-437
     */
    virtual void setupInteractorStyle();

    /**
     * @brief Set up event observers
     *
     * Subclasses should override this to observe mode-specific events.
     *
     * Reference: pqRenderViewSelectionReaction.cxx, line 439-491
     */
    virtual void setupObservers();

    /**
     * @brief Clean up event observers
     *
     * Reference: pqRenderViewSelectionReaction.cxx, line 130-141
     */
    virtual void cleanupObservers();

    /**
     * @brief Show instruction dialog and set cursor (ParaView-style)
     *
     * Shows mode-specific instruction dialog (using
     * cvSelectionToolHelper::promptUser) and sets appropriate cursor for the
     * selection mode.
     *
     * Reference: pqRenderViewSelectionReaction.cxx, line 311-437
     */
    virtual void showInstructionAndSetCursor();

    /**
     * @brief Handle selection changed event from VTK
     *
     * This is the main callback that executes the selection logic.
     * Subclasses should override this to implement specific selection
     * algorithms.
     *
     * @param caller The VTK object that triggered the event
     * @param eventId The event ID
     * @param callData Event-specific data (e.g., region array, polygon array)
     *
     * Reference: pqRenderViewSelectionReaction.cxx, line 536-603
     */
    virtual void onSelectionChanged(vtkObject* caller,
                                    unsigned long eventId,
                                    void* callData);

    /**
     * @brief Perform the actual selection operation
     *
     * This method should be implemented by subclasses to execute
     * the specific selection algorithm (e.g., surface cells, frustum points).
     *
     * @param region Screen-space region [x1, y1, x2, y2]
     * @return true if selection was successful
     */
    virtual bool performSelection(int region[4]);

    /**
     * @brief Perform polygon selection
     *
     * For polygon-based selection modes.
     *
     * @param polygon Array of polygon vertices
     * @return true if selection was successful
     */
    virtual bool performPolygonSelection(vtkIntArray* polygon);

    /**
     * @brief Get the selection modifier from keyboard state
     *
     * Checks Ctrl, Shift keys to determine the modifier.
     *
     * Reference: pqRenderViewSelectionReaction.cxx, line 1013-1035
     */
    virtual SelectionModifier getSelectionModifierFromKeyboard() const;

protected:
    // Note: initializePickers(), pickAtCursor(), pickAtPosition() have been
    // moved to base class cvGenericSelectionTool to unify picking across all
    // selection tools. Now using inherited methods.

    // Note: m_cellPicker, m_pointPicker, m_interactor, m_renderer have been
    // moved to base class cvGenericSelectionTool. Use inherited members.

    /**
     * @brief Check if currently selecting cells (vs points)
     *
     * Determines the field association based on the selection mode.
     * Extracted to base class to eliminate code duplication.
     *
     * @return true if selecting cells, false if selecting points
     */
    bool isSelectingCells() const;

    SelectionMode m_mode;
    SelectionModifier m_modifier;
    // m_viewer, m_cellPicker, m_pointPicker, m_interactor, m_renderer
    // are inherited from cvGenericSelectionTool

    vtkSmartPointer<vtkInteractorStyle> m_previousStyle;
    vtkSmartPointer<vtkInteractorStyle> m_selectionStyle;

    bool m_enabled;
    int m_previousRenderViewMode;

    // Event observer IDs
    unsigned long m_observerIds[6];
    vtkSmartPointer<vtkObject> m_observedObject;

    // Cursor
    QCursor m_cursor;
};

// Forward declare callback class
class cvSelectionCallback;

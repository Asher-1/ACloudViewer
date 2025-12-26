// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

#include "cvSelectionBase.h"
#include "cvSelectionData.h"
#include "cvSelectionTypes.h"  // For SelectionModifier enum

// VTK includes (need full definitions for template instantiation)
#include <vtkCellPicker.h>        // Full definition needed for vtkSmartPointer<vtkCellPicker>
#include <vtkHardwareSelector.h>  // Full definition needed for vtkSmartPointer<vtkHardwareSelector>
#include <vtkPointPicker.h>       // Full definition needed for vtkSmartPointer<vtkPointPicker>
#include <vtkSmartPointer.h>      // For vtkSmartPointer template
#include <vtkType.h>              // For vtkIdType

// Forward declarations
class vtkRenderer;
class vtkSelection;
class vtkRenderWindowInteractor;
class cvViewSelectionManager;
class cvSelectionPipeline;

/**
 * @brief Base class for selection tools with picking capabilities
 *
 * Inherits from cvSelectionBase and adds:
 * - Software picking (cell/point pickers)
 * - Hardware selection (vtkHardwareSelector)
 * - Selection modifiers
 *
 * Use this class when you need picking/selection functionality.
 * For components that only need visualizer access, use cvSelectionBase instead.
 *
 * ParaView-Aligned Features:
 * - Hardware-accelerated selection using vtkHardwareSelector
 * - Multi-actor support with Z-value ordering
 * - Selection modifiers (Add/Subtract/Toggle)
 * - Unified API for both software and hardware selection
 */
class QPCL_ENGINE_LIB_API cvGenericSelectionTool : public cvSelectionBase {
public:
    /**
     * @brief Selection mode
     * Now using global SelectionMode enum from cvSelectionTypes.h
     */
    using SelectionMode = ::SelectionMode;

    /**
     * @brief Selection modifier (for multi-selection)
     * Now using global SelectionModifier enum from cvSelectionTypes.h
     */
    using SelectionModifier = ::SelectionModifier;

    cvGenericSelectionTool()
        : cvSelectionBase(),
          m_manager(nullptr),
          m_useHardwareSelection(false),
          m_captureZValues(true),
          m_multipleSelectionMode(false),
          m_interactor(nullptr),
          m_renderer(nullptr) {}

    ~cvGenericSelectionTool() override = default;

    /**
     * @brief Set the selection manager (to access pipeline)
     * @param manager The view selection manager
     */
    void setSelectionManager(cvViewSelectionManager* manager) {
        m_manager = manager;
    }

    /**
     * @brief Get the selection manager
     * @return Pointer to selection manager, or nullptr
     */
    cvViewSelectionManager* getSelectionManager() const { return m_manager; }

    /**
     * @brief Get the selection pipeline from manager
     * @return Pointer to pipeline, or nullptr
     */
    cvSelectionPipeline* getSelectionPipeline() const;

    /**
     * @brief Get polyData for a selection using ParaView-style priority
     * (override)
     * @param selectionData Optional selection data to extract from (highest
     * priority)
     * @return PolyData pointer, or nullptr if not available
     *
     * This method extends cvSelectionBase::getPolyDataForSelection() by adding:
     * Priority 2: From current selection's actor info (m_currentSelection)
     *
     * Full priority order:
     * Priority 1: From selectionData's actor info (if provided and has actor
     * info) Priority 2: From current selection's actor info
     * (m_currentSelection) [NEW] Priority 3: From selection manager's
     * getPolyData() (uses last selection) Priority 4: From first data actor
     * (fallback for non-selection operations)
     *
     * This override provides enhanced functionality for selection tools.
     */
    vtkPolyData* getPolyDataForSelection(
            const cvSelectionData* selectionData = nullptr) override;

protected:
    // Note: Visualizer access methods (setVisualizer, getVisualizer, getPCLVis,
    // hasValidPCLVis, getAllPolyDataFromVisualizer)
    // are inherited from cvSelectionBase

    ///@{
    /**
     * @brief Hardware-accelerated selection (ParaView-aligned)
     *
     * These methods provide GPU-accelerated selection using
     * vtkHardwareSelector. They return cvSelectionData with full actor
     * information and Z-value ordering.
     */

    /**
     * @brief Perform hardware selection at a point
     * @param x X coordinate in window space
     * @param y Y coordinate in window space
     * @param mode Selection mode (cells/points)
     * @param modifier Selection modifier (replace/add/subtract/toggle)
     * @return Selection data with all actors at that location
     *
     * Uses vtkHardwareSelector for precise, GPU-accelerated selection.
     * Returns selection with actor information sorted by Z-value.
     */
    cvSelectionData hardwareSelectAtPoint(
            int x,
            int y,
            SelectionMode mode = SelectionMode::SELECT_SURFACE_CELLS,
            SelectionModifier modifier = SelectionModifier::SELECTION_DEFAULT);

    /**
     * @brief Perform hardware selection in a region
     * @param region Selection region [x1, y1, x2, y2]
     * @param mode Selection mode
     * @param modifier Selection modifier
     * @return Selection data with all selected items
     */
    cvSelectionData hardwareSelectInRegion(
            const int region[4],
            SelectionMode mode = SelectionMode::SELECT_SURFACE_CELLS,
            SelectionModifier modifier = SelectionModifier::SELECTION_DEFAULT);

    /**
     * @brief Get all actors at a screen location (without full selection)
     * @param x X coordinate
     * @param y Y coordinate
     * @return Vector of actor info, sorted by depth (front to back)
     *
     * Fast query to find what actors are at a given screen position.
     * Useful for tooltips and hover highlighting.
     */
    QVector<cvActorSelectionInfo> getActorsAtPoint(int x, int y);

    ///@}

    ///@{
    /**
     * @brief Hardware selection configuration
     */

    /**
     * @brief Enable/disable hardware selection
     * @param enable If true, use vtkHardwareSelector; if false, use software
     * methods
     *
     * When enabled, selection tools will use GPU-accelerated hardware selection
     * for better accuracy and multi-actor support.
     */
    void setUseHardwareSelection(bool enable) {
        m_useHardwareSelection = enable;
    }

    /**
     * @brief Check if hardware selection is enabled
     */
    bool useHardwareSelection() const { return m_useHardwareSelection; }

    /**
     * @brief Set whether to capture Z-buffer values
     * @param capture If true, capture Z values for depth sorting
     */
    void setCaptureZValues(bool capture) { m_captureZValues = capture; }

    /**
     * @brief Enable/disable multi-selection mode
     * @param enable If true, allow multiple selections
     */
    void setMultipleSelectionMode(bool enable) {
        m_multipleSelectionMode = enable;
    }

    /**
     * @brief Check if multi-selection is enabled
     */
    bool multipleSelectionMode() const { return m_multipleSelectionMode; }

    ///@}

    ///@{
    /**
     * @brief Software picking methods (unified from subclasses)
     *
     * These methods provide software-based picking using VTK pickers.
     * All selection tools can use these methods without duplicating code.
     */

    /**
     * @brief Initialize cell and point pickers
     *
     * Creates and configures vtkCellPicker and vtkPointPicker with default
     * tolerances. Call this before using pickAtPosition.
     *
     * Tolerance defaults:
     * - Cell picker: 0.005
     * - Point picker: 0.01
     */
    void initializePickers();

    /**
     * @brief Pick element at screen position
     * @param x X coordinate in screen space
     * @param y Y coordinate in screen space
     * @param selectCells If true, pick cells; if false, pick points
     * @return Picked element ID, or -1 if nothing was picked
     *
     * This method uses vtkCellPicker or vtkPointPicker based on selectCells.
     * The interactor and renderer must be set before calling this method.
     */
    vtkIdType pickAtPosition(int x, int y, bool selectCells);

    /**
     * @brief Pick element at current cursor position
     * @param selectCells If true, pick cells; if false, pick points
     * @return Picked element ID, or -1 if nothing was picked
     *
     * Convenience method that gets the current cursor position from
     * the interactor and calls pickAtPosition.
     */
    vtkIdType pickAtCursor(bool selectCells);

    /**
     * @brief Set the interactor for picking
     * @param interactor The render window interactor
     */
    void setInteractor(vtkRenderWindowInteractor* interactor) {
        m_interactor = interactor;
    }

    /**
     * @brief Get the interactor
     */
    vtkRenderWindowInteractor* getInteractor() const { return m_interactor; }

    /**
     * @brief Set the renderer for picking
     * @param renderer The renderer
     */
    void setRenderer(vtkRenderer* renderer) { m_renderer = renderer; }

    /**
     * @brief Get the renderer (for picking)
     */
    vtkRenderer* getPickingRenderer() const { return m_renderer; }

    /**
     * @brief Set picker tolerance
     * @param cellTolerance Tolerance for cell picker
     * @param pointTolerance Tolerance for point picker
     */
    void setPickerTolerance(double cellTolerance, double pointTolerance);

    /**
     * @brief Get the last picked actor
     * @param selectCells If true, get from cell picker; if false, from point
     * picker
     * @return Picked actor, or nullptr if no pick was made
     *
     * Call this after pickAtPosition/pickAtCursor to get the actor that was
     * picked. Useful for multi-actor scenarios to identify which actor was
     * clicked.
     */
    vtkActor* getPickedActor(bool selectCells);

    /**
     * @brief Get the last picked polyData
     * @param selectCells If true, get from cell picker; if false, from point
     * picker
     * @return Picked polyData, or nullptr if no pick was made
     *
     * Convenience method to get the polyData from the picked actor's mapper.
     */
    vtkPolyData* getPickedPolyData(bool selectCells);

    /**
     * @brief Get pick position in world coordinates
     * @param selectCells If true, get from cell picker; if false, from point
     * picker
     * @param position Output: world position [x, y, z]
     * @return true if position is valid
     */
    bool getPickedPosition(bool selectCells, double position[3]);

    /**
     * @brief Create cvSelectionData from software picking result
     * @param pickedId The picked element ID
     * @param selectCells If true, selecting cells; if false, selecting points
     * @return Selection data with actor information
     *
     * Convenience method to convert software picking result to cvSelectionData
     * with full actor information. This bridges software and hardware
     * selection.
     */
    cvSelectionData createSelectionFromPick(vtkIdType pickedId,
                                            bool selectCells);

    /**
     * @brief Apply selection modifier to combine selections
     * @param newSelection New selection to combine
     * @param currentSelection Current selection (or empty for replace)
     * @param modifier Modifier defining combination operation
     * @param fieldAssociation Field association (0=cells, 1=points)
     * @return Combined selection result
     *
     * ParaView-aligned: Uses cvSelectionPipeline::combineSelections()
     * Reference: pqRenderViewSelectionReaction selection modifier handling
     */
    cvSelectionData applySelectionModifierUnified(
            const cvSelectionData& newSelection,
            const cvSelectionData& currentSelection,
            int modifier,
            int fieldAssociation);

    ///@}

private:
    /**
     * @brief Get or create vtkHardwareSelector (reused for performance)
     * @return Hardware selector instance (managed by smart pointer)
     */
    vtkHardwareSelector* getHardwareSelector();

    /**
     * @brief Configure vtkHardwareSelector for current selection
     * @param region Selection region [x1, y1, x2, y2]
     * @param fieldAssociation Field association (cells/points)
     * @return Configured hardware selector, or nullptr if failed
     */
    vtkHardwareSelector* configureHardwareSelector(const int region[4],
                                                   int fieldAssociation);

    /**
     * @brief Extract actor information from hardware selection
     */
    QVector<cvActorSelectionInfo> extractActorInfo(
            vtkHardwareSelector* selector, vtkSelection* vtkSel);

    /**
     * @brief Convert vtkSelection to cvSelectionData
     */
    cvSelectionData convertSelection(
            vtkSelection* vtkSel,
            const QVector<cvActorSelectionInfo>& actorInfos);

    /**
     * @brief Apply selection modifier
     */
    cvSelectionData applyModifier(const cvSelectionData& newSelection,
                                  const cvSelectionData& currentSelection,
                                  SelectionModifier modifier);

    /**
     * @brief Get vtkRenderer from visualizer
     */
    vtkRenderer* getRenderer();

protected:
    // Note: m_viewer is inherited from cvSelectionBase

    //! Manager reference (for pipeline access)
    cvViewSelectionManager* m_manager;  ///< Selection manager (weak pointer)

    //! Software picking components (unified from subclasses)
    vtkSmartPointer<vtkCellPicker> m_cellPicker;    ///< Cell picker
    vtkSmartPointer<vtkPointPicker> m_pointPicker;  ///< Point picker
    vtkRenderWindowInteractor* m_interactor;  ///< Interactor (weak pointer)
    vtkRenderer* m_renderer;                  ///< Renderer (weak pointer)

    //! Hardware selection components (reused for performance)
    bool m_useHardwareSelection;         ///< Use hardware selection
    bool m_captureZValues;               ///< Capture Z-buffer values
    bool m_multipleSelectionMode;        ///< Allow multiple selections
    cvSelectionData m_currentSelection;  ///< Current selection (for modifiers)
    vtkSmartPointer<vtkHardwareSelector>
            m_hardwareSelector;  ///< Hardware selector (reused)
};

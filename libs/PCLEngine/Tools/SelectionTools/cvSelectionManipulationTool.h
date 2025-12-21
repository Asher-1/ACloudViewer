// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cvRenderViewSelectionTool.h"

// VTK
#include <vtkSmartPointer.h>

// Forward declarations
class vtkIdTypeArray;
class vtkDataSet;
class vtkPolyData;

/**
 * @brief Tool for selection manipulation (clear, grow, shrink)
 *
 * This tool provides utilities to manipulate existing selections:
 * - CLEAR_SELECTION: Clear current selection
 * - GROW_SELECTION: Expand selection by one layer of neighbors
 * - SHRINK_SELECTION: Contract selection by removing boundary items
 *
 * Based on ParaView's selection manipulation.
 *
 * Reference: pqRenderViewSelectionReaction.cxx, grow/shrink selection
 */
class QPCL_ENGINE_LIB_API cvSelectionManipulationTool
    : public cvRenderViewSelectionTool {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param mode Selection mode (CLEAR_SELECTION, GROW_SELECTION, or
     * SHRINK_SELECTION)
     * @param parent Parent QObject
     */
    explicit cvSelectionManipulationTool(SelectionMode mode,
                                         QObject* parent = nullptr);
    ~cvSelectionManipulationTool() override;

    /**
     * @brief Execute the manipulation immediately
     * @param currentSelection Current selection IDs (smart pointer)
     * @param fieldAssociation 0 for cells, 1 for points
     * @return Manipulated selection IDs (managed by caller)
     *
     * @note Returns raw pointer for backward compatibility, but internally
     *       uses smart pointers. Caller is responsible for managing the
     * returned pointer.
     */
    vtkIdTypeArray* execute(
            const vtkSmartPointer<vtkIdTypeArray>& currentSelection,
            int fieldAssociation);

    /**
     * @brief Execute manipulation and return VTK-independent result
     *
     * This is the preferred method for UI layer to call.
     * It returns cvSelectionData instead of raw vtkIdTypeArray*,
     * keeping VTK types isolated.
     *
     * @param currentData Current selection data
     * @return Manipulated selection data (VTK-independent)
     */
    cvSelectionData executeData(const cvSelectionData& currentData);

    // ===== Public Helper Methods (ParaView-style) ===== //

    /**
     * @brief Find boundary items of selection (public access for visualization)
     *
     * Items on the boundary of the selection region.
     * Useful for displaying boundary highlights in UI.
     *
     * @param selection Current selection (smart pointer)
     * @param fieldAssociation 0 for cells, 1 for points
     * @return Array of boundary item IDs (caller must delete)
     */
    vtkIdTypeArray* computeBoundary(
            const vtkSmartPointer<vtkIdTypeArray>& selection,
            int fieldAssociation);

    /**
     * @brief Find neighbors of selected items (public access for visualization)
     *
     * For cells: cells sharing edges/vertices
     * For points: points connected by edges
     *
     * @param selection Current selection (smart pointer)
     * @param fieldAssociation 0 for cells, 1 for points
     * @return Array of neighbor item IDs (caller must delete)
     */
    vtkIdTypeArray* computeNeighbors(
            const vtkSmartPointer<vtkIdTypeArray>& selection,
            int fieldAssociation);

    // ===== ParaView Advanced Parameters ===== //

    /**
     * @brief Set whether to remove the seed (original selection)
     *
     * When enabled in GROW mode:
     * - Only the new layer of neighbors is returned
     * - Original selection is removed
     *
     * Example: If you select cells [1,2,3] and grow with removeSeed=true,
     * you get only the new neighbors, not [1,2,3].
     *
     * @param remove True to remove seed, false to keep it
     */
    void setRemoveSeed(bool remove) { m_removeSeed = remove; }

    /**
     * @brief Get whether seed removal is enabled
     */
    bool getRemoveSeed() const { return m_removeSeed; }

    /**
     * @brief Set whether to remove intermediate layers
     *
     * When enabled in multi-layer GROW/SHRINK:
     * - Only the outermost layer is returned
     * - Intermediate layers are removed
     *
     * Example: With numberOfLayers=3 and removeIntermediateLayers=true,
     * only the 3rd layer is returned, not layers 1 and 2.
     *
     * @param remove True to keep only outermost layer
     */
    void setRemoveIntermediateLayers(bool remove) {
        m_removeIntermediateLayers = remove;
    }

    /**
     * @brief Get whether intermediate layer removal is enabled
     */
    bool getRemoveIntermediateLayers() const {
        return m_removeIntermediateLayers;
    }

    /**
     * @brief Set number of layers for multi-layer grow/shrink
     *
     * - 0 (default): Single layer operation (ParaView default behavior)
     * - N > 0: Grow/shrink by N layers
     *
     * Example: setNumberOfLayers(3) will grow 3 layers of neighbors.
     *
     * @param layers Number of layers (0 = single layer)
     */
    void setNumberOfLayers(int layers) { m_numberOfLayers = qMax(0, layers); }

    /**
     * @brief Get number of layers
     */
    int getNumberOfLayers() const { return m_numberOfLayers; }

signals:
    /**
     * @brief Emitted when manipulation is completed
     * @param selectionData Selection data after manipulation (VTK-independent)
     */
    void selectionFinished(const cvSelectionData& selectionData);

protected:
    void setupInteractorStyle() override;
    void setupObservers() override;

private:
    /**
     * @brief Clear all selection
     */
    vtkIdTypeArray* clearSelection();

    /**
     * @brief Grow selection by one layer
     */
    vtkIdTypeArray* growSelection(
            const vtkSmartPointer<vtkIdTypeArray>& currentSelection,
            int fieldAssociation);

    /**
     * @brief Shrink selection by one layer
     */
    vtkIdTypeArray* shrinkSelection(
            const vtkSmartPointer<vtkIdTypeArray>& currentSelection,
            int fieldAssociation);

    /**
     * @brief Find neighbors of selected items
     *
     * For cells: cells sharing edges/vertices
     * For points: points connected by edges
     */
    vtkIdTypeArray* findNeighbors(
            const vtkSmartPointer<vtkIdTypeArray>& selection,
            int fieldAssociation);

    /**
     * @brief Find boundary items of selection
     *
     * Items on the boundary of the selection region
     */
    vtkIdTypeArray* findBoundary(
            const vtkSmartPointer<vtkIdTypeArray>& selection,
            int fieldAssociation);

    /**
     * @brief Find topological neighbors for cells
     * @param cellId The cell ID
     * @param polyData The mesh data
     * @param neighbors Output set of neighbor cell IDs
     */
    void findCellNeighbors(vtkIdType cellId,
                           vtkPolyData* polyData,
                           QSet<vtkIdType>& neighbors);

    /**
     * @brief Find topological neighbors for points
     * @param pointId The point ID
     * @param polyData The mesh data
     * @param neighbors Output set of neighbor point IDs
     */
    void findPointNeighbors(vtkIdType pointId,
                            vtkPolyData* polyData,
                            QSet<vtkIdType>& neighbors);

private:
    // ParaView-style parameters (for future implementation)
    bool m_removeSeed = false;
    bool m_removeIntermediateLayers = false;
    int m_numberOfLayers = 0;
};

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
class vtkIntArray;
class vtkIdTypeArray;
class vtkHardwareSelector;
class vtkSelection;

/**
 * @brief Tool for polygon selection (cells and points)
 *
 * This tool implements polygon-based selection where users
 * draw a polygon by clicking multiple points, then selecting
 * items within that polygon.
 *
 * Interaction:
 * - Left click to add polygon vertex
 * - Double click or Enter to complete
 * - ESC to cancel
 *
 * Based on ParaView's polygon selection implementation.
 *
 * Reference: pqRenderViewSelectionReaction.cxx, polygon selection
 */
class QPCL_ENGINE_LIB_API cvPolygonSelectionTool
    : public cvRenderViewSelectionTool {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param mode Selection mode (SELECT_SURFACE_CELLS_POLYGON or
     * SELECT_SURFACE_POINTS_POLYGON)
     * @param parent Parent QObject
     */
    explicit cvPolygonSelectionTool(SelectionMode mode,
                                    QObject* parent = nullptr);
    ~cvPolygonSelectionTool() override;

signals:
    /**
     * @brief Emitted when selection is completed with selection results
     * @param selectionData Selection data (VTK-independent)
     */
    void selectionFinished(const cvSelectionData& selectionData);

    /**
     * @brief Emitted when polygon drawing is completed (for custom polygon
     * mode)
     * @param polygon Array of polygon vertices [x1, y1, x2, y2, ...]
     *
     * Reference: pqRenderViewSelectionReaction::selectedCustomPolygon()
     */
    void polygonCompleted(vtkIntArray* polygon);

protected:
    // Note: setupInteractorStyle() and setupObservers() are handled by base
    // class

    /**
     * @brief Perform polygon selection
     *
     * @param polygon Array of polygon vertices [x1, y1, x2, y2, ...]
     * @return true if selection was successful
     */
    bool performPolygonSelection(vtkIntArray* polygon) override;

private:
    /**
     * @brief Apply selection modifier (add, subtract, toggle)
     * @param newSelection New selection data
     * @param modifier Selection modifier
     * @return Final selection data after applying modifier
     */
    cvSelectionData applySelectionModifier(const cvSelectionData& newSelection,
                                           SelectionModifier modifier);

    /**
     * @brief Check if we are selecting cells or points
     */
    bool isSelectingCells() const;

private:
    // Note: m_hardwareSelector is now in base class (cvGenericSelectionTool)
    // Use base class methods: hardwareSelectInRegion(), hardwareSelectAtPoint()

    vtkSmartPointer<vtkIdTypeArray> m_currentSelection;

    // Field association: 0 for cells, 1 for points
    int m_fieldAssociation;
};

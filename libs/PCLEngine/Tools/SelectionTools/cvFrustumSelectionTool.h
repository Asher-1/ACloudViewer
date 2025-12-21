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
class vtkExtractSelectedFrustum;
class vtkPlanes;
class vtkIdTypeArray;
class vtkPolyData;
class vtkUnstructuredGrid;
class vtkDataSet;

/**
 * @brief Tool for frustum selection (cells and points)
 *
 * This tool implements frustum-based selection where all items
 * within the viewing frustum defined by the screen selection
 * are selected (not just visible surface items).
 *
 * Based on ParaView's frustum selection implementation.
 *
 * Reference: pqRenderViewSelectionReaction.cxx, selectFrustum()
 */
class QPCL_ENGINE_LIB_API cvFrustumSelectionTool
    : public cvRenderViewSelectionTool {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param mode Selection mode (SELECT_FRUSTUM_CELLS or
     * SELECT_FRUSTUM_POINTS)
     * @param parent Parent QObject
     */
    explicit cvFrustumSelectionTool(SelectionMode mode,
                                    QObject* parent = nullptr);
    ~cvFrustumSelectionTool() override;

signals:
    /**
     * @brief Emitted when selection is completed with selection results
     * @param selectionData Selection data (VTK-independent)
     */
    void selectionFinished(const cvSelectionData& selectionData);

protected:
    // Note: setupInteractorStyle() and setupObservers() are handled by base
    // class

    /**
     * @brief Perform the actual frustum selection
     *
     * @param region Screen space region [x1, y1, x2, y2]
     * @return true if selection was successful
     *
     * Reference: pqRenderView.cxx, selectFrustum()
     */
    bool performSelection(int region[4]) override;

private:
    /**
     * @brief Calculate frustum planes from screen region
     *
     * @param region Screen space region
     * @return Frustum planes
     *
     * Reference: pqRenderView.cxx, line 1251-1304
     */
    vtkPlanes* calculateFrustumPlanes(int region[4]);

    /**
     * @brief Perform frustum extraction on dataset
     *
     * @param dataset Input dataset
     * @param planes Frustum planes
     * @return Array of selected IDs
     */
    vtkIdTypeArray* extractFrustumSelection(vtkDataSet* dataset,
                                            vtkPlanes* planes);

    /**
     * @brief Apply selection modifier (add, subtract, toggle)
     * @return Smart pointer to result array (automatic memory management)
     */
    vtkSmartPointer<vtkIdTypeArray> applySelectionModifier(
            vtkIdTypeArray* newIds, SelectionModifier modifier);

    /**
     * @brief Check if we are selecting cells or points
     */
    bool isSelectingCells() const;

private:
    vtkSmartPointer<vtkExtractSelectedFrustum> m_frustumExtractor;
    vtkSmartPointer<vtkIdTypeArray> m_currentSelection;

    // Field association: 0 for cells, 1 for points
    int m_fieldAssociation;
};

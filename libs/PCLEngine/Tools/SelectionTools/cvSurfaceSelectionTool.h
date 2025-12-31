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
class vtkAreaPicker;
class vtkHardwareSelector;
class vtkSelection;
class vtkIdTypeArray;
class vtkProp;

/**
 * @brief Tool for surface selection (cells and points)
 *
 * This tool implements surface-based selection using VTK's picking
 * capabilities. It supports both cell and point selection modes
 * with selection modifiers (add, subtract, toggle).
 *
 * Based on ParaView's surface selection implementation.
 *
 * Reference: pqRenderViewSelectionReaction.cxx, selectOnSurface()
 */
class QPCL_ENGINE_LIB_API cvSurfaceSelectionTool
    : public cvRenderViewSelectionTool {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param mode Selection mode (SELECT_SURFACE_CELLS or
     * SELECT_SURFACE_POINTS)
     * @param parent Parent QObject
     */
    explicit cvSurfaceSelectionTool(SelectionMode mode,
                                    QObject* parent = nullptr);
    ~cvSurfaceSelectionTool() override;

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
     * @brief Perform the actual selection based on screen region
     *
     * @param region Screen space region [x1, y1, x2, y2]
     * @return true if selection was successful
     *
     * Reference: pqRenderView.cxx, selectOnSurface()
     */
    bool performSelection(int region[4]) override;

private:
    /**
     * @brief Apply selection modifier (add, subtract, toggle)
     *
     * @param newSelection New selection data
     * @param modifier Selection modifier
     * @return Final selection data after applying modifier
     *
     * Reference: pqRenderView.cxx, line 1197-1250
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

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"

// LOCAL
#include "cvSelectionData.h"

// Qt
#include <QObject>
#include <QSet>
#include <QString>

// VTK
#include <vtkType.h>

// Forward declarations
class vtkPolyData;

/**
 * @brief Selection algebra operations
 *
 * Provides set-theoretic operations on selections:
 * - Union (A ∪ B): Combine two selections
 * - Intersection (A ∩ B): Common elements
 * - Difference (A - B): Elements in A but not in B
 * - Symmetric Difference (A △ B): Elements in A or B but not both
 * - Complement (~A): All elements not in A
 *
 * Based on ParaView's selection algebra functionality.
 */
class QPCL_ENGINE_LIB_API cvSelectionAlgebra : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Algebra operations
     */
    enum Operation {
        UNION,           ///< A ∪ B
        INTERSECTION,    ///< A ∩ B
        DIFFERENCE,      ///< A - B
        SYMMETRIC_DIFF,  ///< A △ B (elements in either but not both)
        COMPLEMENT       ///< ~A (all elements not in A)
    };

    explicit cvSelectionAlgebra(QObject* parent = nullptr);
    ~cvSelectionAlgebra() override;

    /**
     * @brief Compute union of two selections
     * @param a First selection
     * @param b Second selection
     * @return Union result (A ∪ B)
     */
    static cvSelectionData unionOf(const cvSelectionData& a,
                                   const cvSelectionData& b);

    /**
     * @brief Compute intersection of two selections
     * @param a First selection
     * @param b Second selection
     * @return Intersection result (A ∩ B)
     */
    static cvSelectionData intersectionOf(const cvSelectionData& a,
                                          const cvSelectionData& b);

    /**
     * @brief Compute difference of two selections
     * @param a First selection
     * @param b Second selection
     * @return Difference result (A - B)
     */
    static cvSelectionData differenceOf(const cvSelectionData& a,
                                        const cvSelectionData& b);

    /**
     * @brief Compute symmetric difference of two selections
     * @param a First selection
     * @param b Second selection
     * @return Symmetric difference result (A △ B)
     */
    static cvSelectionData symmetricDifferenceOf(const cvSelectionData& a,
                                                 const cvSelectionData& b);

    /**
     * @brief Compute complement of a selection
     * @param polyData The mesh data (to determine total element count)
     * @param input Input selection
     * @return Complement result (~A)
     */
    static cvSelectionData complementOf(vtkPolyData* polyData,
                                        const cvSelectionData& input);

    /**
     * @brief Perform algebra operation on two selections
     * @param op Operation to perform
     * @param a First selection
     * @param b Second selection (not used for COMPLEMENT)
     * @param polyData Mesh data (required for COMPLEMENT)
     * @return Result selection
     */
    static cvSelectionData performOperation(Operation op,
                                            const cvSelectionData& a,
                                            const cvSelectionData& b,
                                            vtkPolyData* polyData = nullptr);

    /**
     * @brief Grow selection by adding neighbors
     * @param polyData The mesh data
     * @param input Input selection (must be CELLS)
     * @param layers Number of growth layers (negative for shrink)
     * @param removeSeed If true, removes the original seed cells (ParaView-aligned)
     * @param removeIntermediateLayers If true, keeps only outermost layer (ParaView-aligned)
     * @return Grown selection
     * 
     * Reference: vtkPVRenderViewSettings::GrowSelectionRemoveSeed
     *            vtkPVRenderViewSettings::GrowSelectionRemoveIntermediateLayers
     */
    static cvSelectionData growSelection(vtkPolyData* polyData,
                                         const cvSelectionData& input,
                                         int layers = 1,
                                         bool removeSeed = false,
                                         bool removeIntermediateLayers = false);

    /**
     * @brief Shrink selection by removing boundary elements
     * @param polyData The mesh data
     * @param input Input selection (must be CELLS)
     * @param iterations Number of shrink iterations
     * @return Shrunk selection
     */
    static cvSelectionData shrinkSelection(vtkPolyData* polyData,
                                           const cvSelectionData& input,
                                           int iterations = 1);
    
    /**
     * @brief Expand selection (ParaView-compatible)
     * @param polyData The mesh data
     * @param input Input selection
     * @param layers Number of layers to expand (positive = grow, negative = shrink)
     * @param removeSeed If true, removes the original seed elements
     * @param removeIntermediateLayers If true, keeps only the outermost layer
     * @return Expanded selection
     * 
     * This is the ParaView-compatible API matching vtkSMSelectionHelper::ExpandSelection
     */
    static cvSelectionData expandSelection(vtkPolyData* polyData,
                                           const cvSelectionData& input,
                                           int layers,
                                           bool removeSeed = false,
                                           bool removeIntermediateLayers = false);

    /**
     * @brief Extract boundary elements of selection
     * @param polyData The mesh data
     * @param input Input selection (must be CELLS)
     * @return Boundary selection (cells at the edge of the selection)
     */
    static cvSelectionData extractBoundary(vtkPolyData* polyData,
                                           const cvSelectionData& input);

    /**
     * @brief Validate that two selections are compatible for operations
     * @return True if compatible
     */
    static bool areCompatible(const cvSelectionData& a,
                              const cvSelectionData& b);

    /**
     * @brief Grow point selection by adding neighbor points
     * @param polyData The mesh data
     * @param input Input selection (must be POINTS)
     * @param layers Number of growth layers
     * @param removeSeed If true, removes the original seed points
     * @param removeIntermediateLayers If true, keeps only outermost layer
     * @return Grown selection
     * 
     * Point neighbors are determined by shared cells.
     */
    static cvSelectionData growPointSelection(vtkPolyData* polyData,
                                               const cvSelectionData& input,
                                               int layers = 1,
                                               bool removeSeed = false,
                                               bool removeIntermediateLayers = false);

    /**
     * @brief Shrink point selection by removing boundary points
     * @param polyData The mesh data
     * @param input Input selection (must be POINTS)
     * @param iterations Number of shrink iterations
     * @return Shrunk selection
     */
    static cvSelectionData shrinkPointSelection(vtkPolyData* polyData,
                                                 const cvSelectionData& input,
                                                 int iterations = 1);

signals:
    /**
     * @brief Emitted when operation is complete
     */
    void operationComplete(const cvSelectionData& result);

private:
    static QSet<vtkIdType> getCellNeighbors(vtkPolyData* polyData,
                                            vtkIdType cellId);
    static QSet<vtkIdType> getPointNeighbors(vtkPolyData* polyData,
                                             vtkIdType pointId);
    static bool isBoundaryCell(vtkPolyData* polyData,
                               vtkIdType cellId,
                               const QSet<vtkIdType>& selectedSet);
    static bool isBoundaryPoint(vtkPolyData* polyData,
                                vtkIdType pointId,
                                const QSet<vtkIdType>& selectedSet);
};

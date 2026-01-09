// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Protect against Windows macro conflicts - must be before any includes
// Note: Windows headers may define DIFFERENCE as a macro
#ifdef DIFFERENCE
#undef DIFFERENCE
#endif

// clang-format off
// Qt - must be included before qPCL.h and cvSelectionData.h for MOC to work correctly
// QObject must be fully defined before Q_ENUM macro
#include <QtCore/QObject>
#include <QtCore/QSet>
#include <QtCore/QString>
// clang-format on

// LOCAL
// Note: cvSelectionData.h includes qPCL.h, so we include it after Qt headers
#include "cvSelectionData.h"

// VTK
#include <vtkType.h>

// Forward declarations
class vtkPolyData;

/**
 * @brief Selection algebra operations
 *
 * Provides set-theoretic operations on selections:
 * - Union (A U B): Combine two selections
 * - Intersection (A & B): Common elements
 * - Difference (A - B): Elements in A but not in B
 * - Symmetric Difference (A ^ B): Elements in A or B but not both
 * - Complement (~A): All elements not in A
 *
 * Based on ParaView's selection algebra functionality.
 */
class QPCL_ENGINE_LIB_API cvSelectionAlgebra : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Algebra operations
     * Using enum class to avoid macro conflicts (e.g., DIFFERENCE may be
     * defined as a macro on Windows)
     * Note: Q_ENUM requires QObject to be fully defined before the enum
     * Note: DIFFERENCE macro protection is handled at file scope (top of file)
     */
    enum class Operation {
        UNION,
        INTERSECTION,
        DIFFERENCE,
        SYMMETRIC_DIFF,
        COMPLEMENT
    };
    Q_ENUM(Operation)

    explicit cvSelectionAlgebra(QObject* parent = nullptr);

    /**
     * @brief Compute union of two selections
     * @param a First selection
     * @param b Second selection
     * @return Union result (A U B)
     */
    static cvSelectionData unionOf(const cvSelectionData& a,
                                   const cvSelectionData& b);

    /**
     * @brief Compute intersection of two selections
     * @param a First selection
     * @param b Second selection
     * @return Intersection result (A & B)
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
     * @return Symmetric difference result (A ^ B)
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
     * @param removeSeed If true, removes the original seed cells
     * (ParaView-aligned)
     * @param removeIntermediateLayers If true, keeps only outermost layer
     * (ParaView-aligned)
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
     * @param layers Number of layers to expand (positive = grow, negative =
     * shrink)
     * @param removeSeed If true, removes the original seed elements
     * @param removeIntermediateLayers If true, keeps only the outermost layer
     * @return Expanded selection
     *
     * This is the ParaView-compatible API matching
     * vtkSMSelectionHelper::ExpandSelection
     */
    static cvSelectionData expandSelection(
            vtkPolyData* polyData,
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
    static cvSelectionData growPointSelection(
            vtkPolyData* polyData,
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

    /**
     * @brief Emitted when filtering progress changes
     */
    void progressChanged(int percentage);

    /**
     * @brief Emitted when filtering is complete
     */
    void filteringComplete(const cvSelectionData& result);

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
    double computeCellArea(vtkPolyData* polyData, vtkIdType cellId);
    double computeAngleBetweenNormals(const double n1[3], const double n2[3]);
    bool isPointInBounds(const double point[3], const double bounds[6]);
    double computeDistance(const double p1[3], const double p2[3]);
    int countCellNeighbors(vtkPolyData* polyData, vtkIdType cellId);
};

//=============================================================================
// Selection Filter (merged from cvSelectionFilter.h)
//=============================================================================

/**
 * @brief Advanced selection filtering system
 *
 * Provides various filters to refine selections:
 * - Attribute-based filtering (scalar values, colors, etc.)
 * - Geometric filtering (area, angle, distance)
 * - Spatial filtering (bounding box, distance from point)
 * - Combinatorial filtering (AND, OR, NOT operations)
 *
 * Based on ParaView's selection filter functionality.
 */
class QPCL_ENGINE_LIB_API cvSelectionFilter : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Filter types
     */
    enum FilterType {
        ATTRIBUTE_RANGE,    ///< Filter by attribute value range
        GEOMETRIC_AREA,     ///< Filter by cell area
        GEOMETRIC_ANGLE,    ///< Filter by normal angle
        SPATIAL_BBOX,       ///< Filter by bounding box
        SPATIAL_DISTANCE,   ///< Filter by distance from point
        TOPOLOGY_NEIGHBORS  ///< Filter by neighbor count
    };

    /**
     * @brief Comparison operators
     */
    enum ComparisonOp {
        EQUAL,
        NOT_EQUAL,
        LESS_THAN,
        LESS_EQUAL,
        GREATER_THAN,
        GREATER_EQUAL,
        BETWEEN,
        OUTSIDE
    };

    explicit cvSelectionFilter(QObject* parent = nullptr);
    ~cvSelectionFilter() override;

    /**
     * @brief Filter by attribute value range
     */
    cvSelectionData filterByAttributeRange(vtkPolyData* polyData,
                                           const cvSelectionData& input,
                                           const QString& attributeName,
                                           double minValue,
                                           double maxValue);

    /**
     * @brief Filter by attribute comparison
     */
    cvSelectionData filterByAttributeComparison(vtkPolyData* polyData,
                                                const cvSelectionData& input,
                                                const QString& attributeName,
                                                ComparisonOp op,
                                                double value);

    /**
     * @brief Filter cells by area
     */
    cvSelectionData filterByArea(vtkPolyData* polyData,
                                 const cvSelectionData& input,
                                 double minArea,
                                 double maxArea);

    /**
     * @brief Filter by normal angle relative to reference direction
     */
    cvSelectionData filterByNormalAngle(vtkPolyData* polyData,
                                        const cvSelectionData& input,
                                        double refX,
                                        double refY,
                                        double refZ,
                                        double minAngleDeg,
                                        double maxAngleDeg);

    /**
     * @brief Filter by bounding box
     */
    cvSelectionData filterByBoundingBox(vtkPolyData* polyData,
                                        const cvSelectionData& input,
                                        const double bounds[6]);

    /**
     * @brief Filter by distance from point
     */
    cvSelectionData filterByDistanceFromPoint(vtkPolyData* polyData,
                                              const cvSelectionData& input,
                                              double x,
                                              double y,
                                              double z,
                                              double minDistance,
                                              double maxDistance);

    /**
     * @brief Filter by neighbor count (topology)
     */
    cvSelectionData filterByNeighborCount(vtkPolyData* polyData,
                                          const cvSelectionData& input,
                                          int minNeighbors,
                                          int maxNeighbors);

    /**
     * @brief Combine two selections with AND operation
     */
    static cvSelectionData combineAND(const cvSelectionData& a,
                                      const cvSelectionData& b);

    /**
     * @brief Combine two selections with OR operation
     */
    static cvSelectionData combineOR(const cvSelectionData& a,
                                     const cvSelectionData& b);

    /**
     * @brief Invert selection (NOT operation)
     */
    static cvSelectionData invert(vtkPolyData* polyData,
                                  const cvSelectionData& input);

    /**
     * @brief Get attribute names available in polyData
     */
    static QStringList getAttributeNames(vtkPolyData* polyData,
                                         bool pointData = true);

signals:
    /**
     * @brief Emitted when filtering progress changes
     */
    void progressChanged(int percentage);

    /**
     * @brief Emitted when filtering is complete
     */
    void filteringComplete(const cvSelectionData& result);

private:
    double computeCellArea(vtkPolyData* polyData, vtkIdType cellId);
    double computeAngleBetweenNormals(const double n1[3], const double n2[3]);
    bool isPointInBounds(const double point[3], const double bounds[6]);
    double computeDistance(const double p1[3], const double p2[3]);
    int countCellNeighbors(vtkPolyData* polyData, vtkIdType cellId);
};

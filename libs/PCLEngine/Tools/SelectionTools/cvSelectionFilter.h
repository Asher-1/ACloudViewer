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
#include <QString>
#include <QVariant>

// VTK
#include <vtkType.h>

// Forward declarations
class vtkPolyData;

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
     * @param attributeName Name of the attribute (e.g., "Intensity", "RGB")
     * @param minValue Minimum value (inclusive)
     * @param maxValue Maximum value (inclusive)
     * @return Filtered selection data
     */
    cvSelectionData filterByAttributeRange(vtkPolyData* polyData,
                                           const cvSelectionData& input,
                                           const QString& attributeName,
                                           double minValue,
                                           double maxValue);

    /**
     * @brief Filter by attribute comparison
     * @param attributeName Name of the attribute
     * @param op Comparison operator
     * @param value Comparison value
     * @return Filtered selection data
     */
    cvSelectionData filterByAttributeComparison(vtkPolyData* polyData,
                                                const cvSelectionData& input,
                                                const QString& attributeName,
                                                ComparisonOp op,
                                                double value);

    /**
     * @brief Filter cells by area
     * @param minArea Minimum area (inclusive)
     * @param maxArea Maximum area (inclusive)
     * @return Filtered selection data
     */
    cvSelectionData filterByArea(vtkPolyData* polyData,
                                 const cvSelectionData& input,
                                 double minArea,
                                 double maxArea);

    /**
     * @brief Filter by normal angle relative to reference direction
     * @param refX Reference direction X
     * @param refY Reference direction Y
     * @param refZ Reference direction Z
     * @param minAngleDeg Minimum angle in degrees
     * @param maxAngleDeg Maximum angle in degrees
     * @return Filtered selection data
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
     * @param bounds Bounding box [xmin, xmax, ymin, ymax, zmin, zmax]
     * @return Filtered selection data
     */
    cvSelectionData filterByBoundingBox(vtkPolyData* polyData,
                                        const cvSelectionData& input,
                                        const double bounds[6]);

    /**
     * @brief Filter by distance from point
     * @param x Point X coordinate
     * @param y Point Y coordinate
     * @param z Point Z coordinate
     * @param minDistance Minimum distance
     * @param maxDistance Maximum distance
     * @return Filtered selection data
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
     * @param minNeighbors Minimum number of neighbors
     * @param maxNeighbors Maximum number of neighbors
     * @return Filtered selection data
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

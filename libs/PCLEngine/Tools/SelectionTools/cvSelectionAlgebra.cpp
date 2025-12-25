// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionAlgebra.h"

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkCell.h>
#include <vtkIdList.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

// Qt
#include <QSet>

//-----------------------------------------------------------------------------
cvSelectionAlgebra::cvSelectionAlgebra(QObject* parent) : QObject(parent) {
    CVLog::PrintDebug("[cvSelectionAlgebra] Initialized");
}

//-----------------------------------------------------------------------------
cvSelectionAlgebra::~cvSelectionAlgebra() {}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionAlgebra::unionOf(const cvSelectionData& a,
                                            const cvSelectionData& b) {
    if (a.isEmpty()) return b;
    if (b.isEmpty()) return a;

    if (!areCompatible(a, b)) {
        CVLog::Error("[cvSelectionAlgebra] Incompatible selections for union");
        return cvSelectionData();
    }

    // Union of IDs (remove duplicates)
    QSet<qint64> setA = QSet<qint64>(a.ids().begin(), a.ids().end());
    QSet<qint64> setB = QSet<qint64>(b.ids().begin(), b.ids().end());
    QSet<qint64> result = setA.unite(setB);

    QVector<qint64> resultIds = QVector<qint64>(result.begin(), result.end());

    CVLog::Print(QString("[cvSelectionAlgebra] Union: %1 ∪ %2 = %3")
                         .arg(a.count())
                         .arg(b.count())
                         .arg(resultIds.size()));

    return cvSelectionData(resultIds, a.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionAlgebra::intersectionOf(const cvSelectionData& a,
                                                   const cvSelectionData& b) {
    if (a.isEmpty() || b.isEmpty()) {
        return cvSelectionData();
    }

    if (!areCompatible(a, b)) {
        CVLog::Error(
                "[cvSelectionAlgebra] Incompatible selections for "
                "intersection");
        return cvSelectionData();
    }

    // Intersection of IDs
    QSet<qint64> setA = QSet<qint64>(a.ids().begin(), a.ids().end());
    QSet<qint64> setB = QSet<qint64>(b.ids().begin(), b.ids().end());
    QSet<qint64> result = setA.intersect(setB);

    QVector<qint64> resultIds = QVector<qint64>(result.begin(), result.end());

    CVLog::Print(QString("[cvSelectionAlgebra] Intersection: %1 ∩ %2 = %3")
                         .arg(a.count())
                         .arg(b.count())
                         .arg(resultIds.size()));

    return cvSelectionData(resultIds, a.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionAlgebra::differenceOf(const cvSelectionData& a,
                                                 const cvSelectionData& b) {
    if (a.isEmpty()) {
        return cvSelectionData();
    }

    if (b.isEmpty()) {
        return a;
    }

    if (!areCompatible(a, b)) {
        CVLog::Error(
                "[cvSelectionAlgebra] Incompatible selections for difference");
        return cvSelectionData();
    }

    // Difference: elements in A but not in B
    QSet<qint64> setA = QSet<qint64>(a.ids().begin(), a.ids().end());
    QSet<qint64> setB = QSet<qint64>(b.ids().begin(), b.ids().end());
    QSet<qint64> result = setA.subtract(setB);

    QVector<qint64> resultIds = QVector<qint64>(result.begin(), result.end());

    CVLog::Print(QString("[cvSelectionAlgebra] Difference: %1 - %2 = %3")
                         .arg(a.count())
                         .arg(b.count())
                         .arg(resultIds.size()));

    return cvSelectionData(resultIds, a.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionAlgebra::symmetricDifferenceOf(
        const cvSelectionData& a, const cvSelectionData& b) {
    if (a.isEmpty() && b.isEmpty()) {
        return cvSelectionData();
    }

    if (!areCompatible(a, b)) {
        CVLog::Error(
                "[cvSelectionAlgebra] Incompatible selections for symmetric "
                "difference");
        return cvSelectionData();
    }

    // Symmetric difference: (A - B) ∪ (B - A)
    // Or equivalently: (A ∪ B) - (A ∩ B)
    QSet<qint64> setA = QSet<qint64>(a.ids().begin(), a.ids().end());
    QSet<qint64> setB = QSet<qint64>(b.ids().begin(), b.ids().end());

    // Elements in A but not B
    QSet<qint64> aMinusB = setA;
    aMinusB.subtract(setB);

    // Elements in B but not A
    QSet<qint64> bMinusA = setB;
    bMinusA.subtract(setA);

    // Union of both
    QSet<qint64> result = aMinusB.unite(bMinusA);

    QVector<qint64> resultIds = QVector<qint64>(result.begin(), result.end());

    CVLog::Print(
            QString("[cvSelectionAlgebra] Symmetric Difference: %1 △ %2 = %3")
                    .arg(a.count())
                    .arg(b.count())
                    .arg(resultIds.size()));

    return cvSelectionData(resultIds, a.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionAlgebra::complementOf(vtkPolyData* polyData,
                                                 const cvSelectionData& input) {
    if (!polyData) {
        CVLog::Error("[cvSelectionAlgebra] polyData is nullptr for complement");
        return cvSelectionData();
    }

    if (input.isEmpty()) {
        // Return all elements
        vtkIdType totalCount =
                (input.fieldAssociation() == cvSelectionData::POINTS)
                        ? polyData->GetNumberOfPoints()
                        : polyData->GetNumberOfCells();

        QVector<qint64> allIds;
        for (vtkIdType i = 0; i < totalCount; ++i) {
            allIds.append(i);
        }

        CVLog::Print(QString("[cvSelectionAlgebra] Complement: 0 -> %1 (all)")
                             .arg(totalCount));

        return cvSelectionData(allIds, input.fieldAssociation());
    }

    QSet<qint64> selectedSet =
            QSet<qint64>(input.ids().begin(), input.ids().end());
    QVector<qint64> complementIds;

    vtkIdType totalCount = (input.fieldAssociation() == cvSelectionData::POINTS)
                                   ? polyData->GetNumberOfPoints()
                                   : polyData->GetNumberOfCells();

    for (vtkIdType i = 0; i < totalCount; ++i) {
        if (!selectedSet.contains(i)) {
            complementIds.append(i);
        }
    }

    CVLog::Print(QString("[cvSelectionAlgebra] Complement: ~%1 = %2")
                         .arg(input.count())
                         .arg(complementIds.size()));

    return cvSelectionData(complementIds, input.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionAlgebra::performOperation(Operation op,
                                                     const cvSelectionData& a,
                                                     const cvSelectionData& b,
                                                     vtkPolyData* polyData) {
    switch (op) {
        case Operation::UNION:
            return unionOf(a, b);
        case Operation::INTERSECTION:
            return intersectionOf(a, b);
        case Operation::DIFFERENCE:
            return differenceOf(a, b);
        case Operation::SYMMETRIC_DIFF:
            return symmetricDifferenceOf(a, b);
        case Operation::COMPLEMENT:
            if (!polyData) {
                CVLog::Error(
                        "[cvSelectionAlgebra] polyData required for "
                        "complement");
                return cvSelectionData();
            }
            return complementOf(polyData, a);
        default:
            CVLog::Error(QString("[cvSelectionAlgebra] Unknown operation: %1")
                                 .arg(static_cast<int>(op)));
            return cvSelectionData();
    }
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionAlgebra::growSelection(
        vtkPolyData* polyData,
        const cvSelectionData& input,
        int layers,
        bool removeSeed,
        bool removeIntermediateLayers) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    if (input.fieldAssociation() != cvSelectionData::CELLS) {
        CVLog::Warning(
                "[cvSelectionAlgebra] Grow only works with cell selection");
        return input;
    }

    // Store the original seed for removeSeed option
    QSet<vtkIdType> seedSet =
            QSet<vtkIdType>(input.ids().begin(), input.ids().end());
    QSet<vtkIdType> currentSet = seedSet;
    QSet<vtkIdType> previousLayerSet = seedSet;

    // Grow layer by layer
    for (int iter = 0; iter < layers; ++iter) {
        QSet<vtkIdType> newSet = currentSet;

        // For each selected cell, add its neighbors
        for (vtkIdType cellId : currentSet) {
            QSet<vtkIdType> neighbors = getCellNeighbors(polyData, cellId);
            newSet.unite(neighbors);
        }

        // Track the previous layer for removeIntermediateLayers
        previousLayerSet = currentSet;
        currentSet = newSet;
    }

    // ParaView-style: Apply removal options
    QSet<vtkIdType> resultSet = currentSet;

    if (removeIntermediateLayers) {
        // Keep only the outermost layer (elements added in the last iteration)
        // Outermost = currentSet - previousLayerSet
        resultSet = currentSet;
        resultSet.subtract(previousLayerSet);
        CVLog::Print(
                QString("[cvSelectionAlgebra] Removed intermediate layers, "
                        "keeping outermost: %1 cells")
                        .arg(resultSet.size()));
    }

    if (removeSeed) {
        // Remove the original seed elements
        resultSet.subtract(seedSet);
        CVLog::Print(
                QString("[cvSelectionAlgebra] Removed seed, result: %1 cells")
                        .arg(resultSet.size()));
    }

    QVector<qint64> resultIds;
    for (vtkIdType id : resultSet) {
        resultIds.append(id);
    }

    CVLog::Print(QString("[cvSelectionAlgebra] Grow %1 layers (removeSeed=%2, "
                         "removeIntermediate=%3): %4 -> %5 cells")
                         .arg(layers)
                         .arg(removeSeed)
                         .arg(removeIntermediateLayers)
                         .arg(input.count())
                         .arg(resultIds.size()));

    return cvSelectionData(resultIds, cvSelectionData::CELLS);
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionAlgebra::shrinkSelection(
        vtkPolyData* polyData, const cvSelectionData& input, int iterations) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    if (input.fieldAssociation() != cvSelectionData::CELLS) {
        CVLog::Warning(
                "[cvSelectionAlgebra] Shrink only works with cell selection");
        return input;
    }

    QSet<vtkIdType> currentSet =
            QSet<vtkIdType>(input.ids().begin(), input.ids().end());

    for (int iter = 0; iter < iterations; ++iter) {
        QSet<vtkIdType> newSet;

        // Only keep cells that are not on the boundary
        for (vtkIdType cellId : currentSet) {
            if (!isBoundaryCell(polyData, cellId, currentSet)) {
                newSet.insert(cellId);
            }
        }

        currentSet = newSet;

        if (currentSet.isEmpty()) {
            break;
        }
    }

    QVector<qint64> resultIds;
    for (vtkIdType id : currentSet) {
        resultIds.append(id);
    }

    CVLog::Print(
            QString("[cvSelectionAlgebra] Shrink %1 iterations: %2 -> %3 cells")
                    .arg(iterations)
                    .arg(input.count())
                    .arg(resultIds.size()));

    return cvSelectionData(resultIds, cvSelectionData::CELLS);
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionAlgebra::extractBoundary(
        vtkPolyData* polyData, const cvSelectionData& input) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    if (input.fieldAssociation() != cvSelectionData::CELLS) {
        CVLog::Warning(
                "[cvSelectionAlgebra] Boundary extraction only works with cell "
                "selection");
        return input;
    }

    QSet<vtkIdType> selectedSet =
            QSet<vtkIdType>(input.ids().begin(), input.ids().end());
    QVector<qint64> boundaryIds;

    for (vtkIdType cellId : selectedSet) {
        if (isBoundaryCell(polyData, cellId, selectedSet)) {
            boundaryIds.append(cellId);
        }
    }

    CVLog::Print(
            QString("[cvSelectionAlgebra] Boundary extraction: %1 -> %2 cells")
                    .arg(input.count())
                    .arg(boundaryIds.size()));

    return cvSelectionData(boundaryIds, cvSelectionData::CELLS);
}

//-----------------------------------------------------------------------------
bool cvSelectionAlgebra::areCompatible(const cvSelectionData& a,
                                       const cvSelectionData& b) {
    if (a.isEmpty() || b.isEmpty()) {
        return true;  // Empty selections are compatible with anything
    }

    return a.fieldAssociation() == b.fieldAssociation();
}

//-----------------------------------------------------------------------------
// Private helper methods
//-----------------------------------------------------------------------------

QSet<vtkIdType> cvSelectionAlgebra::getCellNeighbors(vtkPolyData* polyData,
                                                     vtkIdType cellId) {
    QSet<vtkIdType> neighbors;

    vtkCell* cell = polyData->GetCell(cellId);
    if (!cell) return neighbors;

    vtkIdType npts = cell->GetNumberOfPoints();

    // For each edge, find neighbor cells
    for (vtkIdType i = 0; i < npts; ++i) {
        vtkIdType p1 = cell->GetPointId(i);
        vtkIdType p2 = cell->GetPointId((i + 1) % npts);

        vtkSmartPointer<vtkIdList> cellIds = vtkSmartPointer<vtkIdList>::New();
        polyData->GetCellEdgeNeighbors(cellId, p1, p2, cellIds);

        for (vtkIdType j = 0; j < cellIds->GetNumberOfIds(); ++j) {
            neighbors.insert(cellIds->GetId(j));
        }
    }

    return neighbors;
}

//-----------------------------------------------------------------------------
bool cvSelectionAlgebra::isBoundaryCell(vtkPolyData* polyData,
                                        vtkIdType cellId,
                                        const QSet<vtkIdType>& selectedSet) {
    QSet<vtkIdType> neighbors = getCellNeighbors(polyData, cellId);

    // A cell is on the boundary if at least one of its neighbors is not
    // selected
    for (vtkIdType neighborId : neighbors) {
        if (!selectedSet.contains(neighborId)) {
            return true;
        }
    }

    // Also boundary if has no neighbors (isolated cell)
    return neighbors.isEmpty();
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionAlgebra::expandSelection(
        vtkPolyData* polyData,
        const cvSelectionData& input,
        int layers,
        bool removeSeed,
        bool removeIntermediateLayers) {
    // ParaView-compatible expand selection API
    // Reference: vtkSMSelectionHelper::ExpandSelection

    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    // Handle both cell and point selections
    bool isPointSelection =
            (input.fieldAssociation() == cvSelectionData::POINTS);

    if (layers > 0) {
        // Grow selection
        if (isPointSelection) {
            return growPointSelection(polyData, input, layers, removeSeed,
                                      removeIntermediateLayers);
        } else {
            return growSelection(polyData, input, layers, removeSeed,
                                 removeIntermediateLayers);
        }
    } else if (layers < 0) {
        // Shrink selection
        if (isPointSelection) {
            return shrinkPointSelection(polyData, input, -layers);
        } else {
            return shrinkSelection(polyData, input, -layers);
        }
    } else {
        // layers == 0, return input unchanged
        return input;
    }
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionAlgebra::growPointSelection(
        vtkPolyData* polyData,
        const cvSelectionData& input,
        int layers,
        bool removeSeed,
        bool removeIntermediateLayers) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    if (input.fieldAssociation() != cvSelectionData::POINTS) {
        CVLog::Warning(
                "[cvSelectionAlgebra] growPointSelection only works with point "
                "selection");
        return input;
    }

    // Store the original seed for removeSeed option
    QSet<vtkIdType> seedSet =
            QSet<vtkIdType>(input.ids().begin(), input.ids().end());
    QSet<vtkIdType> currentSet = seedSet;
    QSet<vtkIdType> previousLayerSet = seedSet;

    // Grow layer by layer
    for (int iter = 0; iter < layers; ++iter) {
        QSet<vtkIdType> newSet = currentSet;

        // For each selected point, add its neighbors
        for (vtkIdType pointId : currentSet) {
            QSet<vtkIdType> neighbors = getPointNeighbors(polyData, pointId);
            newSet.unite(neighbors);
        }

        previousLayerSet = currentSet;
        currentSet = newSet;
    }

    // Apply removal options
    QSet<vtkIdType> resultSet = currentSet;

    if (removeIntermediateLayers) {
        resultSet = currentSet;
        resultSet.subtract(previousLayerSet);
    }

    if (removeSeed) {
        resultSet.subtract(seedSet);
    }

    QVector<qint64> resultIds;
    for (vtkIdType id : resultSet) {
        resultIds.append(id);
    }

    CVLog::Print(QString("[cvSelectionAlgebra] Grow points %1 layers: %2 -> %3 "
                         "points")
                         .arg(layers)
                         .arg(input.count())
                         .arg(resultIds.size()));

    return cvSelectionData(resultIds, cvSelectionData::POINTS);
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionAlgebra::shrinkPointSelection(
        vtkPolyData* polyData, const cvSelectionData& input, int iterations) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    if (input.fieldAssociation() != cvSelectionData::POINTS) {
        CVLog::Warning(
                "[cvSelectionAlgebra] shrinkPointSelection only works with "
                "point selection");
        return input;
    }

    QSet<vtkIdType> currentSet =
            QSet<vtkIdType>(input.ids().begin(), input.ids().end());

    for (int iter = 0; iter < iterations; ++iter) {
        QSet<vtkIdType> newSet;

        // Only keep points that are not on the boundary
        for (vtkIdType pointId : currentSet) {
            if (!isBoundaryPoint(polyData, pointId, currentSet)) {
                newSet.insert(pointId);
            }
        }

        currentSet = newSet;

        if (currentSet.isEmpty()) {
            break;
        }
    }

    QVector<qint64> resultIds;
    for (vtkIdType id : currentSet) {
        resultIds.append(id);
    }

    CVLog::Print(QString("[cvSelectionAlgebra] Shrink points %1 iterations: %2 "
                         "-> %3 points")
                         .arg(iterations)
                         .arg(input.count())
                         .arg(resultIds.size()));

    return cvSelectionData(resultIds, cvSelectionData::POINTS);
}

//-----------------------------------------------------------------------------
QSet<vtkIdType> cvSelectionAlgebra::getPointNeighbors(vtkPolyData* polyData,
                                                      vtkIdType pointId) {
    QSet<vtkIdType> neighbors;

    // Get all cells that use this point
    vtkSmartPointer<vtkIdList> cellIds = vtkSmartPointer<vtkIdList>::New();
    polyData->GetPointCells(pointId, cellIds);

    // For each cell, get all its points (they are neighbors)
    for (vtkIdType i = 0; i < cellIds->GetNumberOfIds(); ++i) {
        vtkIdType cellId = cellIds->GetId(i);
        vtkCell* cell = polyData->GetCell(cellId);
        if (cell) {
            for (vtkIdType j = 0; j < cell->GetNumberOfPoints(); ++j) {
                vtkIdType neighborPtId = cell->GetPointId(j);
                if (neighborPtId != pointId) {
                    neighbors.insert(neighborPtId);
                }
            }
        }
    }

    return neighbors;
}

//-----------------------------------------------------------------------------
bool cvSelectionAlgebra::isBoundaryPoint(vtkPolyData* polyData,
                                         vtkIdType pointId,
                                         const QSet<vtkIdType>& selectedSet) {
    QSet<vtkIdType> neighbors = getPointNeighbors(polyData, pointId);

    // A point is on the boundary if at least one of its neighbors is not
    // selected
    for (vtkIdType neighborId : neighbors) {
        if (!selectedSet.contains(neighborId)) {
            return true;
        }
    }

    // Also boundary if has no neighbors (isolated point)
    return neighbors.isEmpty();
}

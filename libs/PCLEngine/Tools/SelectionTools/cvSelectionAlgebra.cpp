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
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkFieldData.h>
#include <vtkIdList.h>
#include <vtkKdTreePointLocator.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkSignedCharArray.h>
#include <vtkSmartPointer.h>

// Qt
#include <QSet>

// Qt5/Qt6 Compatibility
#include <QtCompat.h>

//-----------------------------------------------------------------------------
cvSelectionAlgebra::cvSelectionAlgebra(QObject* parent) : QObject(parent) {
    CVLog::PrintDebug("[cvSelectionAlgebra] Initialized");
}

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
    QSet<qint64> setA = qSetFromVector(a.ids());
    QSet<qint64> setB = qSetFromVector(b.ids());
    QSet<qint64> result = setA.unite(setB);

    QVector<qint64> resultIds = qVectorFromSet(result);

    CVLog::Print(QString("[cvSelectionAlgebra] Union: %1 U %2 = %3")
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
    QSet<qint64> setA = qSetFromVector(a.ids());
    QSet<qint64> setB = qSetFromVector(b.ids());
    QSet<qint64> result = setA.intersect(setB);

    QVector<qint64> resultIds = qVectorFromSet(result);

    CVLog::Print(QString("[cvSelectionAlgebra] Intersection: %1 & %2 = %3")
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
    QSet<qint64> setA = qSetFromVector(a.ids());
    QSet<qint64> setB = qSetFromVector(b.ids());
    QSet<qint64> result = setA.subtract(setB);

    QVector<qint64> resultIds = qVectorFromSet(result);

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

    // Symmetric difference: (A - B) U (B - A)
    // Or equivalently: (A U B) - (A & B)
    QSet<qint64> setA = qSetFromVector(a.ids());
    QSet<qint64> setB = qSetFromVector(b.ids());

    // Elements in A but not B
    QSet<qint64> aMinusB = setA;
    aMinusB.subtract(setB);

    // Elements in B but not A
    QSet<qint64> bMinusA = setB;
    bMinusA.subtract(setA);

    // Union of both
    QSet<qint64> result = aMinusB.unite(bMinusA);

    QVector<qint64> resultIds = qVectorFromSet(result);

    CVLog::Print(
            QString("[cvSelectionAlgebra] Symmetric Difference: %1 ^ %2 = %3")
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

    QSet<qint64> selectedSet = qSetFromVector(input.ids());
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
    QSet<qint64> idsSet = qSetFromVector(input.ids());
    QSet<vtkIdType> seedSet;
    for (qint64 id : idsSet) {
        seedSet.insert(static_cast<vtkIdType>(id));
    }
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
        CVLog::PrintDebug(
                QString("[cvSelectionAlgebra] Removed intermediate layers, "
                        "keeping outermost: %1 cells")
                        .arg(resultSet.size()));
    }

    if (removeSeed) {
        // Remove the original seed elements
        resultSet.subtract(seedSet);
        CVLog::PrintDebug(
                QString("[cvSelectionAlgebra] Removed seed, result: %1 cells")
                        .arg(resultSet.size()));
    }

    QVector<qint64> resultIds;
    for (vtkIdType id : resultSet) {
        resultIds.append(id);
    }

    CVLog::PrintDebug(
            QString("[cvSelectionAlgebra] Grow %1 layers (removeSeed=%2, "
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

    QSet<qint64> idsSet = qSetFromVector(input.ids());
    QSet<vtkIdType> currentSet;
    for (qint64 id : idsSet) {
        currentSet.insert(static_cast<vtkIdType>(id));
    }

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

    CVLog::PrintDebug(
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

    QSet<qint64> idsSet = qSetFromVector(input.ids());
    QSet<vtkIdType> selectedSet;
    for (qint64 id : idsSet) {
        selectedSet.insert(static_cast<vtkIdType>(id));
    }
    QVector<qint64> boundaryIds;

    for (vtkIdType cellId : selectedSet) {
        if (isBoundaryCell(polyData, cellId, selectedSet)) {
            boundaryIds.append(cellId);
        }
    }

    CVLog::PrintDebug(
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
    // Reference: vtkSelector.cxx line 139-148

    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    if (layers == 0) {
        return input;  // No expansion needed
    }

    // Determine field association
    bool isPointSelection =
            (input.fieldAssociation() == cvSelectionData::POINTS);

    // Handle grow (layers > 0) and shrink (layers < 0) separately
    if (layers > 0) {
        QSet<qint64> currentSet;
        // Convert input ids to QSet
        for (qint64 id : input.ids()) {
            currentSet.insert(id);
        }

        QSet<qint64> seedSet =
                currentSet;  // Save original selection for removeSeed

        // Iteratively expand
        for (int iter = 0; iter < layers; ++iter) {
            QSet<qint64> newElements;

            // Find all neighbors of current selection
            for (qint64 id : currentSet) {
                QSet<vtkIdType> neighbors;
                if (isPointSelection) {
                    neighbors = getPointNeighbors(polyData,
                                                  static_cast<vtkIdType>(id));
                } else {
                    neighbors = getCellNeighbors(polyData,
                                                 static_cast<vtkIdType>(id));
                }

                // Add neighbors that are not already in the set
                for (vtkIdType neighborId : neighbors) {
                    if (!currentSet.contains(static_cast<qint64>(neighborId))) {
                        newElements.insert(static_cast<qint64>(neighborId));
                    }
                }
            }

            // Add new elements to current set
            if (newElements.isEmpty()) {
                CVLog::PrintDebug(QString("[cvSelectionAlgebra] Grow stopped "
                                          "at iteration %1: no new neighbors")
                                          .arg(iter + 1));
                break;
            }

            currentSet.unite(newElements);
        }

        // Apply removeSeed if requested
        if (removeSeed) {
            currentSet.subtract(seedSet);
        }

        // Apply removeIntermediateLayers if requested
        // This keeps only the outermost layer
        if (removeIntermediateLayers && layers > 1) {
            // We need to keep only elements that were added in the last
            // iteration This requires re-doing the expansion layer by layer
            QSet<qint64> layerSet = seedSet;

            for (int iter = 0; iter < layers; ++iter) {
                QSet<qint64> nextLayerSet;

                // Find neighbors of current layer
                for (qint64 id : layerSet) {
                    QSet<vtkIdType> neighbors;
                    if (isPointSelection) {
                        neighbors = getPointNeighbors(
                                polyData, static_cast<vtkIdType>(id));
                    } else {
                        neighbors = getCellNeighbors(
                                polyData, static_cast<vtkIdType>(id));
                    }

                    for (vtkIdType neighborId : neighbors) {
                        if (!currentSet.contains(
                                    static_cast<qint64>(neighborId)) ||
                            layerSet.contains(
                                    static_cast<qint64>(neighborId))) {
                            continue;
                        }
                        nextLayerSet.insert(static_cast<qint64>(neighborId));
                    }
                }

                if (iter == layers - 1) {
                    // This is the final layer
                    currentSet = nextLayerSet;
                    break;
                }

                layerSet = nextLayerSet;
                if (layerSet.isEmpty()) {
                    break;
                }
            }
        }

        // Convert back to QVector
        QVector<qint64> expandedIds;
        for (qint64 id : currentSet) {
            expandedIds.append(id);
        }

        CVLog::PrintDebug(
                QString("[cvSelectionAlgebra] Grow: %1 layers, %2 -> %3 "
                        "%4 (removeSeed=%5, removeIntermediate=%6)")
                        .arg(layers)
                        .arg(input.ids().size())
                        .arg(expandedIds.size())
                        .arg(isPointSelection ? "points" : "cells")
                        .arg(removeSeed)
                        .arg(removeIntermediateLayers));

        // Create result selection data
        cvSelectionData resultSelection(expandedIds, input.fieldAssociation());

        // Copy actor info if available
        if (input.hasActorInfo()) {
            resultSelection.setActorInfo(input.primaryActor(),
                                         input.primaryPolyData());
        }

        return resultSelection;
    } else {
        // SHRINK (layers < 0): Remove boundary layers iteratively
        // ParaView's shrink is conceptually "remove N boundary layers"
        // This is the standard morphological erosion operation
        int shrinkLayers = -layers;  // Convert negative to positive

        // Check if this is a mesh with topology (has cells)
        // Note: For point selections on meshes, we still need cells to
        // determine connectivity
        vtkIdType numCells = polyData->GetNumberOfCells();
        bool hasMeshTopology = (numCells > 0);

        if (!hasMeshTopology) {
            CVLog::Warning(
                    "[cvSelectionAlgebra] Shrink operation requires mesh "
                    "topology. "
                    "Pure point clouds without cells are not supported. "
                    "Returning input unchanged.");
            return input;
        }

        CVLog::PrintDebug(
                QString("[cvSelectionAlgebra] Starting shrink: %1 %2, %3 "
                        "layers")
                        .arg(input.ids().size())
                        .arg(isPointSelection ? "points" : "cells")
                        .arg(shrinkLayers));

        // Convert input ids to QSet with correct type
        QSet<qint64> currentSet;
        for (qint64 id : input.ids()) {
            currentSet.insert(id);
        }

        // Improved shrink logic: Find elements that have AT LEAST ONE
        // unselected neighbor This ensures we can properly erode the selection
        // boundary
        for (int iter = 0; iter < shrinkLayers; ++iter) {
            QSet<qint64>
                    boundaryElements;  // Elements to remove (on the boundary)
            QSet<qint64>
                    interiorElements;  // Elements to keep (in the interior)

            // Classify each element as boundary or interior
            for (qint64 id : currentSet) {
                QSet<vtkIdType> neighbors;

                if (isPointSelection) {
                    neighbors = getPointNeighbors(polyData,
                                                  static_cast<vtkIdType>(id));
                } else {
                    neighbors = getCellNeighbors(polyData,
                                                 static_cast<vtkIdType>(id));
                }

                // Count how many neighbors are outside the current selection
                int neighborsOutside = 0;
                for (vtkIdType neighborId : neighbors) {
                    if (!currentSet.contains(static_cast<qint64>(neighborId))) {
                        neighborsOutside++;
                    }
                }

                // An element is on the boundary if:
                // 1. It has no neighbors (isolated element), OR
                // 2. At least one of its neighbors is NOT in the selection
                bool isBoundary = (neighbors.isEmpty() || neighborsOutside > 0);

                if (isBoundary) {
                    boundaryElements.insert(id);
                } else {
                    interiorElements.insert(id);
                }
            }

            CVLog::PrintDebug(
                    QString("[cvSelectionAlgebra] Shrink iteration %1: "
                            "found %2 boundary, %3 interior elements")
                            .arg(iter + 1)
                            .arg(boundaryElements.size())
                            .arg(interiorElements.size()));

            // If no boundary elements found, we cannot shrink further
            if (boundaryElements.isEmpty()) {
                CVLog::Warning(
                        QString("[cvSelectionAlgebra] Shrink stopped at "
                                "iteration %1: "
                                "no boundary elements found (all %2 %3 have "
                                "all neighbors selected)")
                                .arg(iter + 1)
                                .arg(currentSet.size())
                                .arg(isPointSelection ? "points" : "cells"));
                break;
            }

            // Remove boundary elements (keep only interior)
            currentSet = interiorElements;

            // If selection becomes empty, stop
            if (currentSet.isEmpty()) {
                CVLog::PrintDebug(
                        QString("[cvSelectionAlgebra] Shrink iteration "
                                "%1: selection shrunk to empty")
                                .arg(iter + 1));
                break;
            }
        }

        QVector<qint64> shrunkIds;
        for (qint64 id : currentSet) {
            shrunkIds.append(id);
        }

        CVLog::PrintDebug(
                QString("[cvSelectionAlgebra] Shrink: %1 layers, %2 -> %3 %4")
                        .arg(shrinkLayers)
                        .arg(input.ids().size())
                        .arg(shrunkIds.size())
                        .arg(isPointSelection ? "points" : "cells"));

        // Create result selection data
        cvSelectionData resultSelection(shrunkIds, input.fieldAssociation());

        // Copy actor info if available
        if (input.hasActorInfo()) {
            resultSelection.setActorInfo(input.primaryActor(),
                                         input.primaryPolyData());
        }

        return resultSelection;
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
    QSet<qint64> idsSet = qSetFromVector(input.ids());
    QSet<vtkIdType> seedSet;
    for (qint64 id : idsSet) {
        seedSet.insert(static_cast<vtkIdType>(id));
    }
    QSet<vtkIdType> currentSet = seedSet;
    QSet<vtkIdType> previousLayerSet = seedSet;

    // Grow layer by layer
    for (int iter = 0; iter < layers; ++iter) {
        QSet<vtkIdType> newSet = currentSet;
        bool hasAnyNeighbors = false;

        // For each selected point, add its neighbors
        for (vtkIdType pointId : currentSet) {
            QSet<vtkIdType> neighbors = getPointNeighbors(polyData, pointId);
            if (!neighbors.isEmpty()) {
                hasAnyNeighbors = true;
            }
            newSet.unite(neighbors);
        }

        // Early exit if no neighbors found (pure point cloud case)
        // This prevents infinite loops or unnecessary iterations
        if (!hasAnyNeighbors && iter == 0) {
            CVLog::Warning(
                    "[cvSelectionAlgebra] No topological neighbors found for "
                    "grow operation. "
                    "This appears to be a pure point cloud. Returning input "
                    "unchanged.");
            return input;
        }

        // If no new points were added, we can't grow further
        if (newSet.size() == currentSet.size()) {
            CVLog::PrintDebug(
                    QString("[cvSelectionAlgebra] Grow stopped at layer %1: "
                            "no new neighbors found")
                            .arg(iter + 1));
            break;
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

    CVLog::PrintDebug(
            QString("[cvSelectionAlgebra] Grow points %1 layers: %2 -> %3 "
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

    QSet<qint64> idsSet = qSetFromVector(input.ids());
    QSet<vtkIdType> currentSet;
    for (qint64 id : idsSet) {
        currentSet.insert(static_cast<vtkIdType>(id));
    }

    // Quick check: if no points have topological neighbors, this is a pure
    // point cloud In a pure point cloud, all points are boundary points, so
    // shrink would remove everything
    bool hasTopology = false;
    for (vtkIdType pointId : currentSet) {
        vtkSmartPointer<vtkIdList> cellIds = vtkSmartPointer<vtkIdList>::New();
        polyData->GetPointCells(pointId, cellIds);
        for (vtkIdType i = 0; i < cellIds->GetNumberOfIds(); ++i) {
            vtkCell* cell = polyData->GetCell(cellIds->GetId(i));
            if (cell && cell->GetNumberOfPoints() > 1) {
                hasTopology = true;
                break;
            }
        }
        if (hasTopology) {
            break;
        }
    }

    if (!hasTopology) {
        CVLog::Warning(
                "[cvSelectionAlgebra] No topological neighbors found for "
                "shrink operation. "
                "This appears to be a pure point cloud. Returning input "
                "unchanged.");
        return input;
    }

    for (int iter = 0; iter < iterations; ++iter) {
        QSet<vtkIdType> newSet;

        // Only keep points that are not on the boundary
        for (vtkIdType pointId : currentSet) {
            if (!isBoundaryPoint(polyData, pointId, currentSet)) {
                newSet.insert(pointId);
            }
        }

        // If no points were removed, we can't shrink further
        if (newSet.size() == currentSet.size()) {
            CVLog::PrintDebug(QString("[cvSelectionAlgebra] Shrink stopped at "
                                      "iteration %1: "
                                      "no boundary points found")
                                      .arg(iter + 1));
            break;
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

    CVLog::PrintDebug(
            QString("[cvSelectionAlgebra] Shrink points %1 iterations: %2 "
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

    // Safety checks
    if (!polyData) {
        CVLog::Warning(
                "[cvSelectionAlgebra::getPointNeighbors] polyData is null");
        return neighbors;
    }

    vtkIdType numPoints = polyData->GetNumberOfPoints();
    if (numPoints == 0) {
        CVLog::Warning(
                "[cvSelectionAlgebra::getPointNeighbors] polyData has no "
                "points");
        return neighbors;
    }

    if (pointId < 0 || pointId >= numPoints) {
        CVLog::Warning(QString("[cvSelectionAlgebra::getPointNeighbors] "
                               "pointId %1 out of range [0, %2)")
                               .arg(pointId)
                               .arg(numPoints));
        return neighbors;
    }

    // First try topology-based neighbors (works for meshes)
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

    // If no topological neighbors found (pure point cloud case),
    // For point clouds, each point is typically its own vertex cell with no
    // shared edges, so we skip KD-tree based neighbor search as it's expensive
    // and may cause issues. Grow/Shrink for pure point clouds is not
    // well-defined without a mesh topology.
    if (neighbors.isEmpty()) {
        // Log that we can't find neighbors for this point cloud
        // Don't use KD-tree as it can be expensive and cause crashes
        CVLog::PrintDebug(
                QString("[cvSelectionAlgebra] Point %1: no topological "
                        "neighbors "
                        "(pure point cloud - grow/shrink not applicable)")
                        .arg(pointId));
    }

    return neighbors;
}

//-----------------------------------------------------------------------------
bool cvSelectionAlgebra::isBoundaryPoint(vtkPolyData* polyData,
                                         vtkIdType pointId,
                                         const QSet<vtkIdType>& selectedSet) {
    // Safety checks
    if (!polyData) {
        return true;  // Treat as boundary if invalid
    }

    vtkIdType numPoints = polyData->GetNumberOfPoints();
    if (pointId < 0 || pointId >= numPoints) {
        return true;  // Treat as boundary if out of range
    }

    // For point clouds, we use spatial neighbors to determine boundary
    // A point is on the boundary if at least one of its spatial neighbors
    // is not in the selection

    // First try topology-based neighbors
    vtkSmartPointer<vtkIdList> cellIds = vtkSmartPointer<vtkIdList>::New();
    polyData->GetPointCells(pointId, cellIds);

    QSet<vtkIdType> topoNeighbors;
    for (vtkIdType i = 0; i < cellIds->GetNumberOfIds(); ++i) {
        vtkIdType cellId = cellIds->GetId(i);
        vtkCell* cell = polyData->GetCell(cellId);
        if (cell) {
            for (vtkIdType j = 0; j < cell->GetNumberOfPoints(); ++j) {
                vtkIdType neighborPtId = cell->GetPointId(j);
                if (neighborPtId != pointId) {
                    topoNeighbors.insert(neighborPtId);
                }
            }
        }
    }

    if (!topoNeighbors.isEmpty()) {
        // Use topology-based boundary detection
        for (vtkIdType neighborId : topoNeighbors) {
            if (!selectedSet.contains(neighborId)) {
                return true;
            }
        }
        return false;
    }

    // For point clouds (no topological neighbors), all points are considered
    // boundary Shrink operation will remove all points, which is the expected
    // behavior for a point cloud without mesh topology
    return true;
}

//=============================================================================
// cvSelectionFilter Implementation (merged from cvSelectionFilter.cpp)
//=============================================================================

#include <vtkTriangle.h>

//-----------------------------------------------------------------------------
cvSelectionFilter::cvSelectionFilter(QObject* parent) : QObject(parent) {
    CVLog::PrintDebug("[cvSelectionFilter] Initialized");
}

//-----------------------------------------------------------------------------
cvSelectionFilter::~cvSelectionFilter() {}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::filterByAttributeRange(
        vtkPolyData* polyData,
        const cvSelectionData& input,
        const QString& attributeName,
        double minValue,
        double maxValue) {
    if (!polyData || input.isEmpty() || attributeName.isEmpty()) {
        return cvSelectionData();
    }

    vtkDataArray* array = nullptr;
    if (input.fieldAssociation() == cvSelectionData::POINTS) {
        array = polyData->GetPointData()->GetArray(
                attributeName.toUtf8().constData());
    } else {
        array = polyData->GetCellData()->GetArray(
                attributeName.toUtf8().constData());
    }

    if (!array) {
        CVLog::Warning(QString("[cvSelectionFilter] Attribute '%1' not found")
                               .arg(attributeName));
        return cvSelectionData();
    }

    QVector<qint64> filteredIds;
    QVector<qint64> inputIds = input.ids();

    for (qint64 id : inputIds) {
        if (id < 0 || id >= array->GetNumberOfTuples()) {
            continue;
        }

        double value = array->GetComponent(id, 0);

        if (value >= minValue && value <= maxValue) {
            filteredIds.append(id);
        }
    }

    CVLog::PrintDebug(
            QString("[cvSelectionFilter] Attribute range filter: %1 -> %2 "
                    "items")
                    .arg(inputIds.size())
                    .arg(filteredIds.size()));

    return cvSelectionData(filteredIds, input.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::filterByAttributeComparison(
        vtkPolyData* polyData,
        const cvSelectionData& input,
        const QString& attributeName,
        ComparisonOp op,
        double value) {
    if (!polyData || input.isEmpty() || attributeName.isEmpty()) {
        return cvSelectionData();
    }

    vtkDataArray* array = nullptr;
    if (input.fieldAssociation() == cvSelectionData::POINTS) {
        array = polyData->GetPointData()->GetArray(
                attributeName.toUtf8().constData());
    } else {
        array = polyData->GetCellData()->GetArray(
                attributeName.toUtf8().constData());
    }

    if (!array) {
        CVLog::Warning(QString("[cvSelectionFilter] Attribute '%1' not found")
                               .arg(attributeName));
        return cvSelectionData();
    }

    QVector<qint64> filteredIds;
    QVector<qint64> inputIds = input.ids();

    for (qint64 id : inputIds) {
        if (id < 0 || id >= array->GetNumberOfTuples()) {
            continue;
        }

        double attrValue = array->GetComponent(id, 0);
        bool pass = false;

        switch (op) {
            case EQUAL:
                pass = (qAbs(attrValue - value) < 1e-6);
                break;
            case NOT_EQUAL:
                pass = (qAbs(attrValue - value) >= 1e-6);
                break;
            case LESS_THAN:
                pass = (attrValue < value);
                break;
            case LESS_EQUAL:
                pass = (attrValue <= value);
                break;
            case GREATER_THAN:
                pass = (attrValue > value);
                break;
            case GREATER_EQUAL:
                pass = (attrValue >= value);
                break;
            default:
                break;
        }

        if (pass) {
            filteredIds.append(id);
        }
    }

    CVLog::PrintDebug(
            QString("[cvSelectionFilter] Attribute comparison filter: %1 "
                    "-> %2 items")
                    .arg(inputIds.size())
                    .arg(filteredIds.size()));

    return cvSelectionData(filteredIds, input.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::filterByArea(vtkPolyData* polyData,
                                                const cvSelectionData& input,
                                                double minArea,
                                                double maxArea) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    if (input.fieldAssociation() != cvSelectionData::CELLS) {
        CVLog::Warning(
                "[cvSelectionFilter] Area filter only works with cell "
                "selection");
        return cvSelectionData();
    }

    QVector<qint64> filteredIds;
    QVector<qint64> inputIds = input.ids();

    for (qint64 id : inputIds) {
        if (id < 0 || id >= polyData->GetNumberOfCells()) {
            continue;
        }

        double area = computeCellArea(polyData, id);
        if (area >= minArea && area <= maxArea) {
            filteredIds.append(id);
        }
    }

    CVLog::PrintDebug(QString("[cvSelectionFilter] Area filter: %1 -> %2 cells")
                              .arg(inputIds.size())
                              .arg(filteredIds.size()));

    return cvSelectionData(filteredIds, cvSelectionData::CELLS);
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::filterByNormalAngle(
        vtkPolyData* polyData,
        const cvSelectionData& input,
        double refX,
        double refY,
        double refZ,
        double minAngleDeg,
        double maxAngleDeg) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    if (input.fieldAssociation() != cvSelectionData::CELLS) {
        CVLog::Warning(
                "[cvSelectionFilter] Normal angle filter only works with cell "
                "selection");
        return cvSelectionData();
    }

    vtkDataArray* normals = polyData->GetCellData()->GetNormals();
    if (!normals) {
        CVLog::Warning("[cvSelectionFilter] No cell normals available");
        return cvSelectionData();
    }

    double refNormal[3] = {refX, refY, refZ};
    vtkMath::Normalize(refNormal);

    QVector<qint64> filteredIds;
    QVector<qint64> inputIds = input.ids();

    for (qint64 id : inputIds) {
        if (id < 0 || id >= polyData->GetNumberOfCells()) {
            continue;
        }

        double* normal = normals->GetTuple3(id);
        double angleDeg = computeAngleBetweenNormals(normal, refNormal);

        if (angleDeg >= minAngleDeg && angleDeg <= maxAngleDeg) {
            filteredIds.append(id);
        }
    }

    CVLog::PrintDebug(
            QString("[cvSelectionFilter] Normal angle filter: %1 -> %2 cells")
                    .arg(inputIds.size())
                    .arg(filteredIds.size()));

    return cvSelectionData(filteredIds, cvSelectionData::CELLS);
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::filterByBoundingBox(
        vtkPolyData* polyData,
        const cvSelectionData& input,
        const double bounds[6]) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    QVector<qint64> filteredIds;
    QVector<qint64> inputIds = input.ids();

    if (input.fieldAssociation() == cvSelectionData::POINTS) {
        for (qint64 id : inputIds) {
            if (id < 0 || id >= polyData->GetNumberOfPoints()) {
                continue;
            }

            double point[3];
            polyData->GetPoint(id, point);

            if (isPointInBounds(point, bounds)) {
                filteredIds.append(id);
            }
        }
    } else {
        for (qint64 id : inputIds) {
            if (id < 0 || id >= polyData->GetNumberOfCells()) {
                continue;
            }

            vtkCell* cell = polyData->GetCell(id);
            if (!cell) continue;

            double center[3] = {0, 0, 0};
            vtkIdType npts = cell->GetNumberOfPoints();
            for (vtkIdType i = 0; i < npts; ++i) {
                double pt[3];
                polyData->GetPoint(cell->GetPointId(i), pt);
                center[0] += pt[0];
                center[1] += pt[1];
                center[2] += pt[2];
            }
            if (npts > 0) {
                center[0] /= npts;
                center[1] /= npts;
                center[2] /= npts;
            }

            if (isPointInBounds(center, bounds)) {
                filteredIds.append(id);
            }
        }
    }

    CVLog::PrintDebug(
            QString("[cvSelectionFilter] Bounding box filter: %1 -> %2 items")
                    .arg(inputIds.size())
                    .arg(filteredIds.size()));

    return cvSelectionData(filteredIds, input.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::filterByDistanceFromPoint(
        vtkPolyData* polyData,
        const cvSelectionData& input,
        double x,
        double y,
        double z,
        double minDistance,
        double maxDistance) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    double refPoint[3] = {x, y, z};
    QVector<qint64> filteredIds;
    QVector<qint64> inputIds = input.ids();

    if (input.fieldAssociation() == cvSelectionData::POINTS) {
        for (qint64 id : inputIds) {
            if (id < 0 || id >= polyData->GetNumberOfPoints()) {
                continue;
            }

            double point[3];
            polyData->GetPoint(id, point);

            double dist = computeDistance(point, refPoint);
            if (dist >= minDistance && dist <= maxDistance) {
                filteredIds.append(id);
            }
        }
    } else {
        for (qint64 id : inputIds) {
            if (id < 0 || id >= polyData->GetNumberOfCells()) {
                continue;
            }

            vtkCell* cell = polyData->GetCell(id);
            if (!cell) continue;

            double center[3] = {0, 0, 0};
            vtkIdType npts = cell->GetNumberOfPoints();
            for (vtkIdType i = 0; i < npts; ++i) {
                double pt[3];
                polyData->GetPoint(cell->GetPointId(i), pt);
                center[0] += pt[0];
                center[1] += pt[1];
                center[2] += pt[2];
            }
            if (npts > 0) {
                center[0] /= npts;
                center[1] /= npts;
                center[2] /= npts;
            }

            double dist = computeDistance(center, refPoint);
            if (dist >= minDistance && dist <= maxDistance) {
                filteredIds.append(id);
            }
        }
    }

    CVLog::PrintDebug(
            QString("[cvSelectionFilter] Distance filter: %1 -> %2 items")
                    .arg(inputIds.size())
                    .arg(filteredIds.size()));

    return cvSelectionData(filteredIds, input.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::filterByNeighborCount(
        vtkPolyData* polyData,
        const cvSelectionData& input,
        int minNeighbors,
        int maxNeighbors) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    if (input.fieldAssociation() != cvSelectionData::CELLS) {
        CVLog::Warning(
                "[cvSelectionFilter] Neighbor count filter only works with "
                "cell selection");
        return cvSelectionData();
    }

    QVector<qint64> filteredIds;
    QVector<qint64> inputIds = input.ids();

    for (qint64 id : inputIds) {
        if (id < 0 || id >= polyData->GetNumberOfCells()) {
            continue;
        }

        int neighborCount = countCellNeighbors(polyData, id);
        if (neighborCount >= minNeighbors && neighborCount <= maxNeighbors) {
            filteredIds.append(id);
        }
    }

    CVLog::PrintDebug(
            QString("[cvSelectionFilter] Neighbor count filter: %1 -> %2 cells")
                    .arg(inputIds.size())
                    .arg(filteredIds.size()));

    return cvSelectionData(filteredIds, cvSelectionData::CELLS);
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::combineAND(const cvSelectionData& a,
                                              const cvSelectionData& b) {
    if (a.isEmpty() || b.isEmpty()) {
        return cvSelectionData();
    }

    if (a.fieldAssociation() != b.fieldAssociation()) {
        return cvSelectionData();
    }

    QSet<qint64> setA = qSetFromVector(a.ids());
    QSet<qint64> setB = qSetFromVector(b.ids());
    QSet<qint64> result = setA.intersect(setB);

    QVector<qint64> resultIds = qVectorFromSet(result);

    CVLog::PrintDebug(QString("[cvSelectionFilter] AND: %1 & %2 = %3")
                              .arg(a.count())
                              .arg(b.count())
                              .arg(resultIds.size()));

    return cvSelectionData(resultIds, a.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::combineOR(const cvSelectionData& a,
                                             const cvSelectionData& b) {
    if (a.isEmpty()) return b;
    if (b.isEmpty()) return a;

    if (a.fieldAssociation() != b.fieldAssociation()) {
        return cvSelectionData();
    }

    QSet<qint64> setA = qSetFromVector(a.ids());
    QSet<qint64> setB = qSetFromVector(b.ids());
    QSet<qint64> result = setA.unite(setB);

    QVector<qint64> resultIds = qVectorFromSet(result);

    CVLog::PrintDebug(QString("[cvSelectionFilter] OR: %1 U %2 = %3")
                              .arg(a.count())
                              .arg(b.count())
                              .arg(resultIds.size()));

    return cvSelectionData(resultIds, a.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::invert(vtkPolyData* polyData,
                                          const cvSelectionData& input) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    QSet<qint64> selectedSet = qSetFromVector(input.ids());
    QVector<qint64> invertedIds;

    vtkIdType totalCount = (input.fieldAssociation() == cvSelectionData::POINTS)
                                   ? polyData->GetNumberOfPoints()
                                   : polyData->GetNumberOfCells();

    for (vtkIdType i = 0; i < totalCount; ++i) {
        if (!selectedSet.contains(i)) {
            invertedIds.append(i);
        }
    }

    CVLog::PrintDebug(QString("[cvSelectionFilter] Invert: %1 -> %2")
                              .arg(input.count())
                              .arg(invertedIds.size()));

    return cvSelectionData(invertedIds, input.fieldAssociation());
}

//-----------------------------------------------------------------------------
QStringList cvSelectionFilter::getAttributeNames(vtkPolyData* polyData,
                                                 bool pointData) {
    QStringList names;

    if (!polyData) {
        return names;
    }

    vtkFieldData* data = pointData ? (vtkFieldData*)polyData->GetPointData()
                                   : (vtkFieldData*)polyData->GetCellData();

    if (!data) {
        return names;
    }

    int numArrays = data->GetNumberOfArrays();
    for (int i = 0; i < numArrays; ++i) {
        vtkDataArray* array = data->GetArray(i);
        if (array && array->GetName()) {
            names.append(QString::fromUtf8(array->GetName()));
        }
    }

    return names;
}

//-----------------------------------------------------------------------------
// cvSelectionFilter Private helper methods
//-----------------------------------------------------------------------------

double cvSelectionFilter::computeCellArea(vtkPolyData* polyData,
                                          vtkIdType cellId) {
    vtkCell* cell = polyData->GetCell(cellId);
    if (!cell) return 0.0;

    if (cell->GetCellType() == VTK_TRIANGLE) {
        double p0[3], p1[3], p2[3];
        polyData->GetPoint(cell->GetPointId(0), p0);
        polyData->GetPoint(cell->GetPointId(1), p1);
        polyData->GetPoint(cell->GetPointId(2), p2);

        return vtkTriangle::TriangleArea(p0, p1, p2);
    }

    vtkIdType npts = cell->GetNumberOfPoints();
    if (npts < 3) return 0.0;

    double area = 0.0;
    double p0[3], p1[3], p2[3];
    polyData->GetPoint(cell->GetPointId(0), p0);

    for (vtkIdType i = 1; i < npts - 1; ++i) {
        polyData->GetPoint(cell->GetPointId(i), p1);
        polyData->GetPoint(cell->GetPointId(i + 1), p2);
        area += vtkTriangle::TriangleArea(p0, p1, p2);
    }

    return area;
}

//-----------------------------------------------------------------------------
double cvSelectionFilter::computeAngleBetweenNormals(const double n1[3],
                                                     const double n2[3]) {
    double dot = vtkMath::Dot(n1, n2);
    dot = qBound(-1.0, dot, 1.0);
    double angleRad = std::acos(dot);
    return vtkMath::DegreesFromRadians(angleRad);
}

//-----------------------------------------------------------------------------
bool cvSelectionFilter::isPointInBounds(const double point[3],
                                        const double bounds[6]) {
    return (point[0] >= bounds[0] && point[0] <= bounds[1] &&
            point[1] >= bounds[2] && point[1] <= bounds[3] &&
            point[2] >= bounds[4] && point[2] <= bounds[5]);
}

//-----------------------------------------------------------------------------
double cvSelectionFilter::computeDistance(const double p1[3],
                                          const double p2[3]) {
    double dx = p1[0] - p2[0];
    double dy = p1[1] - p2[1];
    double dz = p1[2] - p2[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

//-----------------------------------------------------------------------------
int cvSelectionFilter::countCellNeighbors(vtkPolyData* polyData,
                                          vtkIdType cellId) {
    vtkCell* cell = polyData->GetCell(cellId);
    if (!cell) return 0;

    QSet<vtkIdType> neighbors;

    vtkIdType npts = cell->GetNumberOfPoints();
    for (vtkIdType i = 0; i < npts; ++i) {
        vtkIdType p1 = cell->GetPointId(i);
        vtkIdType p2 = cell->GetPointId((i + 1) % npts);

        vtkSmartPointer<vtkIdList> cellIds = vtkSmartPointer<vtkIdList>::New();
        polyData->GetCellEdgeNeighbors(cellId, p1, p2, cellIds);

        for (vtkIdType j = 0; j < cellIds->GetNumberOfIds(); ++j) {
            neighbors.insert(cellIds->GetId(j));
        }
    }

    return neighbors.size();
}

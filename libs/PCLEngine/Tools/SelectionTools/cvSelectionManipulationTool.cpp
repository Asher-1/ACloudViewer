// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionManipulationTool.h"

#include "cvSelectionData.h"

// LOCAL
#include "PclUtils/PCLVis.h"

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkActor.h>
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkDataSet.h>
#include <vtkIdList.h>
#include <vtkIdTypeArray.h>
#include <vtkMapper.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>

// QT
#include <QSet>

//-----------------------------------------------------------------------------
cvSelectionManipulationTool::cvSelectionManipulationTool(SelectionMode mode,
                                                         QObject* parent)
    : cvRenderViewSelectionTool(mode, parent) {
    CVLog::Print(QString("[cvSelectionManipulationTool] Created with mode: %1")
                         .arg(static_cast<int>(mode)));
}

//-----------------------------------------------------------------------------
cvSelectionManipulationTool::~cvSelectionManipulationTool() {
    CVLog::Print("[cvSelectionManipulationTool] Destroyed");
}

//-----------------------------------------------------------------------------
vtkIdTypeArray* cvSelectionManipulationTool::execute(
        const vtkSmartPointer<vtkIdTypeArray>& currentSelection,
        int fieldAssociation) {
    vtkIdTypeArray* result = nullptr;

    switch (m_mode) {
        case cvViewSelectionManager::CLEAR_SELECTION:
            result = clearSelection();
            CVLog::Print("[cvSelectionManipulationTool] Selection cleared");
            break;

        case cvViewSelectionManager::GROW_SELECTION:
            result = growSelection(currentSelection, fieldAssociation);
            if (result) {
                CVLog::Print(QString("[cvSelectionManipulationTool] Selection "
                                     "grown to %1 items")
                                     .arg(result->GetNumberOfTuples()));
            }
            break;

        case cvViewSelectionManager::SHRINK_SELECTION:
            result = shrinkSelection(currentSelection, fieldAssociation);
            if (result) {
                CVLog::Print(QString("[cvSelectionManipulationTool] Selection "
                                     "shrunk to %1 items")
                                     .arg(result->GetNumberOfTuples()));
            }
            break;

        default:
            CVLog::Warning(
                    QString("[cvSelectionManipulationTool] Unknown mode: %1")
                            .arg(static_cast<int>(m_mode)));
            break;
    }

    cvSelectionData selectionData(result, fieldAssociation);
    emit selectionFinished(selectionData);
    return result;
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionManipulationTool::executeData(
        const cvSelectionData& currentData) {
    // VTK-independent interface for UI layer
    // This method wraps execute() and returns cvSelectionData directly

    if (currentData.isEmpty() &&
        m_mode != cvViewSelectionManager::CLEAR_SELECTION) {
        CVLog::Warning(
                "[cvSelectionManipulationTool] Cannot execute on empty "
                "selection");
        return cvSelectionData();
    }

    vtkIdTypeArray* result =
            execute(currentData.vtkArray(), currentData.fieldAssociation());

    if (!result) {
        // For CLEAR_SELECTION, result is nullptr (empty selection)
        if (m_mode == cvViewSelectionManager::CLEAR_SELECTION) {
            return cvSelectionData();  // Empty selection
        }
        CVLog::Warning(
                "[cvSelectionManipulationTool] Execute returned null result");
        return cvSelectionData();
    }

    // Create cvSelectionData from result (encapsulates VTK)
    cvSelectionData resultData(result, currentData.fieldAssociation());
    result->Delete();  // Clean up VTK object

    return resultData;
}

//-----------------------------------------------------------------------------
void cvSelectionManipulationTool::setupInteractorStyle() {
    // Selection manipulation doesn't need special interactor style
    CVLog::Print("[cvSelectionManipulationTool] No interactor style needed");
}

//-----------------------------------------------------------------------------
void cvSelectionManipulationTool::setupObservers() {
    // No observers needed for immediate operations
    CVLog::Print("[cvSelectionManipulationTool] No observers needed");
}

//-----------------------------------------------------------------------------
vtkIdTypeArray* cvSelectionManipulationTool::computeBoundary(
        const vtkSmartPointer<vtkIdTypeArray>& selection,
        int fieldAssociation) {
    // Public wrapper for findBoundary (for UI visualization)
    return findBoundary(selection, fieldAssociation);
}

//-----------------------------------------------------------------------------
vtkIdTypeArray* cvSelectionManipulationTool::computeNeighbors(
        const vtkSmartPointer<vtkIdTypeArray>& selection,
        int fieldAssociation) {
    // Public wrapper for findNeighbors (for UI visualization)
    return findNeighbors(selection, fieldAssociation);
}

//-----------------------------------------------------------------------------
vtkIdTypeArray* cvSelectionManipulationTool::clearSelection() {
    // ParaView-consistent implementation
    // Based on pqSelectionManager::clearSelection()
    //
    // ParaView's approach:
    // 1. CleanSelectionInputs() - clear selection proxy inputs
    // 2. renderAllViews(false) - re-render all views to remove highlights
    // 3. SelectedPorts.clear() - clear internal selection state
    // 4. Q_EMIT selectionChanged(nullptr) - emit signal
    //
    // In our simplified architecture:
    // 1. Clear any internal state (if any)
    // 2. Return nullptr to indicate empty selection
    // 3. The caller (MainWindow) will handle:
    //    - Updating the selection manager
    //    - Re-rendering the view (via UpdateScreen())
    //    - Emitting signals
    //    - Disabling selection manipulation actions

    // Return nullptr to indicate empty selection
    // This matches ParaView's behavior where clearSelection() results in
    // an empty selection being set
    return nullptr;
}

//-----------------------------------------------------------------------------
vtkIdTypeArray* cvSelectionManipulationTool::growSelection(
        const vtkSmartPointer<vtkIdTypeArray>& currentSelection,
        int fieldAssociation) {
    if (!currentSelection || currentSelection->GetNumberOfTuples() == 0) {
        CVLog::Warning(
                "[cvSelectionManipulationTool] No current selection to grow");
        return nullptr;
    }

    // Get the visualizer and data
    if (!m_viewer) {
        CVLog::Error("[cvSelectionManipulationTool] No visualizer available");
        return nullptr;
    }

    // Get the actual mesh data (using centralized ParaView-style method)
    vtkPolyData* polyData = getPolyDataForSelection();
    if (!polyData) {
        CVLog::Warning(
                "[cvSelectionManipulationTool] No polyData available - "
                "falling back to simplified algorithm");

        // Fallback: simple ID-based heuristic
        vtkIdTypeArray* result = vtkIdTypeArray::New();
        QSet<vtkIdType> selectedSet;
        for (vtkIdType i = 0; i < currentSelection->GetNumberOfTuples(); ++i) {
            selectedSet.insert(currentSelection->GetValue(i));
        }
        for (vtkIdType i = 0; i < currentSelection->GetNumberOfTuples(); ++i) {
            vtkIdType id = currentSelection->GetValue(i);
            for (int j = -1; j <= 1; ++j) {
                if (j != 0 && id + j >= 0) {
                    selectedSet.insert(id + j);
                }
            }
        }
        for (vtkIdType id : selectedSet) {
            result->InsertNextValue(id);
        }
        return result;
    }

    // TOPOLOGY-BASED IMPLEMENTATION with ParaView Advanced Parameters
    // Based on ParaView's vtkSMSelectionHelper::ExpandSelection()

    CVLog::Print(
            QString("[cvSelectionManipulationTool] Growing selection using "
                    "topology - "
                    "fieldAssociation=%1 (0=cells, 1=points), removeSeed=%2, "
                    "removeIntermediateLayers=%3, numberOfLayers=%4")
                    .arg(fieldAssociation)
                    .arg(m_removeSeed)
                    .arg(m_removeIntermediateLayers)
                    .arg(m_numberOfLayers));

    // Determine actual number of layers (0 means single layer, i.e. 1)
    int effectiveLayers = (m_numberOfLayers > 0) ? m_numberOfLayers : 1;

    // Keep track of all layers for advanced options
    QVector<QSet<vtkIdType>>
            layers;  // layers[0] = seed, layers[1] = 1st ring, etc.

    // Layer 0: original selection (seed)
    QSet<vtkIdType> seedLayer;
    for (vtkIdType i = 0; i < currentSelection->GetNumberOfTuples(); ++i) {
        seedLayer.insert(currentSelection->GetValue(i));
    }
    layers.append(seedLayer);

    // All selected items so far (for neighbor finding)
    QSet<vtkIdType> allSelected = seedLayer;

    // Grow layer by layer
    for (int layer = 1; layer <= effectiveLayers; ++layer) {
        // Find neighbors of the previous layer
        vtkIdTypeArray* prevLayerArray = vtkIdTypeArray::New();
        for (vtkIdType id : layers[layer - 1]) {
            prevLayerArray->InsertNextValue(id);
        }

        vtkSmartPointer<vtkIdTypeArray> neighbors =
                findNeighbors(prevLayerArray, fieldAssociation);
        prevLayerArray->Delete();  // Still need to delete prevLayerArray
                                   // (created with New())

        if (!neighbors) {
            CVLog::Warning(QString("[cvSelectionManipulationTool] No neighbors "
                                   "found in layer %1")
                                   .arg(layer));
            break;
        }

        // New layer: neighbors that are not already selected
        QSet<vtkIdType> newLayer;
        for (vtkIdType i = 0; i < neighbors->GetNumberOfTuples(); ++i) {
            vtkIdType id = neighbors->GetValue(i);
            if (!allSelected.contains(id)) {
                newLayer.insert(id);
                allSelected.insert(id);
            }
        }
        // Smart pointer handles cleanup automatically

        layers.append(newLayer);

        if (newLayer.isEmpty()) {
            CVLog::Print(QString("[cvSelectionManipulationTool] No more "
                                 "neighbors at layer %1, stopping")
                                 .arg(layer));
            break;
        }
    }

    // Build result based on advanced parameters
    vtkIdTypeArray* result = vtkIdTypeArray::New();
    QSet<vtkIdType> resultSet;

    if (m_removeIntermediateLayers) {
        // Only keep the outermost layer
        int lastLayer = layers.size() - 1;
        resultSet = layers[lastLayer];
        CVLog::Print(QString("[cvSelectionManipulationTool] Keeping only "
                             "outermost layer %1 (%2 items)")
                             .arg(lastLayer)
                             .arg(resultSet.size()));
    } else {
        // Keep all layers
        int startLayer = m_removeSeed ? 1 : 0;  // Skip seed if removeSeed=true
        for (int i = startLayer; i < layers.size(); ++i) {
            resultSet.unite(layers[i]);
        }
        CVLog::Print(QString("[cvSelectionManipulationTool] Keeping layers "
                             "%1-%2 (%3 items total)")
                             .arg(startLayer)
                             .arg(layers.size() - 1)
                             .arg(resultSet.size()));
    }

    // Convert QSet to vtkIdTypeArray
    for (vtkIdType id : resultSet) {
        result->InsertNextValue(id);
    }

    CVLog::Print(QString("[cvSelectionManipulationTool] Selection grown from "
                         "%1 to %2 items")
                         .arg(currentSelection->GetNumberOfTuples())
                         .arg(result->GetNumberOfTuples()));

    return result;
}

//-----------------------------------------------------------------------------
vtkIdTypeArray* cvSelectionManipulationTool::shrinkSelection(
        const vtkSmartPointer<vtkIdTypeArray>& currentSelection,
        int fieldAssociation) {
    if (!currentSelection || currentSelection->GetNumberOfTuples() == 0) {
        CVLog::Warning(
                "[cvSelectionManipulationTool] No current selection to shrink");
        return nullptr;
    }

    // Get the visualizer and data
    if (!m_viewer) {
        CVLog::Error("[cvSelectionManipulationTool] No visualizer available");
        return nullptr;
    }

    // Get the actual mesh data (using centralized ParaView-style method)
    vtkPolyData* polyData = getPolyDataForSelection();
    if (!polyData) {
        CVLog::Warning(
                "[cvSelectionManipulationTool] No polyData available - "
                "falling back to simplified algorithm");

        // Fallback: simple ID-based heuristic
        vtkIdTypeArray* result = vtkIdTypeArray::New();
        QSet<vtkIdType> selectedSet, boundarySet;
        for (vtkIdType i = 0; i < currentSelection->GetNumberOfTuples(); ++i) {
            selectedSet.insert(currentSelection->GetValue(i));
        }
        for (vtkIdType id : selectedSet) {
            bool isBoundary = false;
            for (int j = -1; j <= 1; ++j) {
                if (j != 0 && !selectedSet.contains(id + j)) {
                    isBoundary = true;
                    break;
                }
            }
            if (!isBoundary) {
                result->InsertNextValue(id);
            }
        }
        return result;
    }

    // TOPOLOGY-BASED IMPLEMENTATION with ParaView Advanced Parameters
    // Shrink by removing boundary layers

    CVLog::Print(
            QString("[cvSelectionManipulationTool] Shrinking selection using "
                    "topology - "
                    "fieldAssociation=%1 (0=cells, 1=points), removeSeed=%2, "
                    "removeIntermediateLayers=%3, numberOfLayers=%4")
                    .arg(fieldAssociation)
                    .arg(m_removeSeed)
                    .arg(m_removeIntermediateLayers)
                    .arg(m_numberOfLayers));

    // Determine actual number of layers (0 means single layer, i.e. 1)
    int effectiveLayers = (m_numberOfLayers > 0) ? m_numberOfLayers : 1;

    // Keep track of all layers (outer to inner)
    QVector<QSet<vtkIdType>> layers;  // layers[0] = outermost boundary,
                                      // layers[1] = next inner, etc.

    // Current remaining selection
    QSet<vtkIdType> remaining;
    for (vtkIdType i = 0; i < currentSelection->GetNumberOfTuples(); ++i) {
        remaining.insert(currentSelection->GetValue(i));
    }

    // Peel off boundary layers
    for (int layer = 0; layer < effectiveLayers; ++layer) {
        if (remaining.isEmpty()) {
            CVLog::Print(QString("[cvSelectionManipulationTool] No more "
                                 "elements at layer %1, stopping")
                                 .arg(layer));
            break;
        }

        // Convert remaining to vtkIdTypeArray
        vtkIdTypeArray* remainingArray = vtkIdTypeArray::New();
        for (vtkIdType id : remaining) {
            remainingArray->InsertNextValue(id);
        }

        // Find boundary of current remaining selection
        vtkSmartPointer<vtkIdTypeArray> boundary =
                findBoundary(remainingArray, fieldAssociation);
        remainingArray->Delete();  // Still need to delete remainingArray
                                   // (created with New())

        if (!boundary || boundary->GetNumberOfTuples() == 0) {
            CVLog::Print(QString("[cvSelectionManipulationTool] No boundary "
                                 "found at layer %1, stopping")
                                 .arg(layer));
            break;
        }

        // Store this boundary layer
        QSet<vtkIdType> boundarySet;
        for (vtkIdType i = 0; i < boundary->GetNumberOfTuples(); ++i) {
            vtkIdType id = boundary->GetValue(i);
            boundarySet.insert(id);
            remaining.remove(id);  // Remove from remaining
        }
        // Smart pointer handles cleanup automatically

        layers.append(boundarySet);

        CVLog::PrintDebug(QString("[cvSelectionManipulationTool] Removed layer "
                                  "%1: %2 items, %3 remaining")
                                  .arg(layer)
                                  .arg(boundarySet.size())
                                  .arg(remaining.size()));
    }

    // Build result based on advanced parameters
    vtkIdTypeArray* result = vtkIdTypeArray::New();
    QSet<vtkIdType> resultSet;

    if (m_removeIntermediateLayers) {
        // Only keep the innermost core (what remains after peeling all layers)
        resultSet = remaining;
        CVLog::Print(QString("[cvSelectionManipulationTool] Keeping only "
                             "innermost core (%1 items)")
                             .arg(resultSet.size()));
    } else if (m_removeSeed) {
        // Keep only the removed boundary layers (not the inner core)
        for (const QSet<vtkIdType>& layer : layers) {
            resultSet.unite(layer);
        }
        CVLog::Print(QString("[cvSelectionManipulationTool] Keeping only "
                             "removed boundaries (%1 items)")
                             .arg(resultSet.size()));
    } else {
        // Default: keep the inner core (what remains after shrinking)
        resultSet = remaining;
        CVLog::Print(QString("[cvSelectionManipulationTool] Keeping inner "
                             "selection (%1 items)")
                             .arg(resultSet.size()));
    }

    // If result would be empty, keep at least one element (prevent complete
    // disappearance)
    if (resultSet.isEmpty() && currentSelection->GetNumberOfTuples() > 0) {
        CVLog::Warning(
                "[cvSelectionManipulationTool] Result would be empty - keeping "
                "one element");
        resultSet.insert(currentSelection->GetValue(0));
    }

    // Convert QSet to vtkIdTypeArray
    for (vtkIdType id : resultSet) {
        result->InsertNextValue(id);
    }

    CVLog::Print(QString("[cvSelectionManipulationTool] Selection shrunk from "
                         "%1 to %2 items")
                         .arg(currentSelection->GetNumberOfTuples())
                         .arg(result->GetNumberOfTuples()));

    return result;
}

//-----------------------------------------------------------------------------
vtkIdTypeArray* cvSelectionManipulationTool::findNeighbors(
        const vtkSmartPointer<vtkIdTypeArray>& selection,
        int fieldAssociation) {
    if (!selection || selection->GetNumberOfTuples() == 0) {
        CVLog::Warning(
                "[cvSelectionManipulationTool] No selection for finding "
                "neighbors");
        return nullptr;
    }

    if (!m_viewer) {
        CVLog::Error("[cvSelectionManipulationTool] No visualizer available");
        return nullptr;
    }

    // TOPOLOGY-BASED IMPLEMENTATION (ParaView-consistent)
    // Find all elements that are topologically adjacent to the selection
    // but NOT in the selection itself

    // Get polyData using centralized ParaView-style method
    vtkPolyData* polyData = getPolyDataForSelection();

    if (!polyData) {
        CVLog::Error(
                "[cvSelectionManipulationTool] No polyData available for "
                "topology queries");
        vtkIdTypeArray* result = vtkIdTypeArray::New();
        return result;
    }

    // Build topological links
    polyData->BuildLinks();

    QSet<vtkIdType> selectionSet;
    QSet<vtkIdType> neighborSet;

    // Add selection to set
    for (vtkIdType i = 0; i < selection->GetNumberOfTuples(); ++i) {
        selectionSet.insert(selection->GetValue(i));
    }

    // Find neighbors
    if (fieldAssociation == 0) {
        // CELLS: Find neighboring cells that share points
        for (vtkIdType cellId : selectionSet) {
            QSet<vtkIdType> cellNeighbors;
            findCellNeighbors(cellId, polyData, cellNeighbors);

            // Add only neighbors NOT in selection
            for (vtkIdType neighborId : cellNeighbors) {
                if (!selectionSet.contains(neighborId)) {
                    neighborSet.insert(neighborId);
                }
            }
        }
    } else {
        // POINTS: Find neighboring points connected through edges
        for (vtkIdType pointId : selectionSet) {
            QSet<vtkIdType> pointNeighbors;
            findPointNeighbors(pointId, polyData, pointNeighbors);

            // Add only neighbors NOT in selection
            for (vtkIdType neighborId : pointNeighbors) {
                if (!selectionSet.contains(neighborId)) {
                    neighborSet.insert(neighborId);
                }
            }
        }
    }

    // Convert set to array
    vtkIdTypeArray* result = vtkIdTypeArray::New();
    for (vtkIdType id : neighborSet) {
        result->InsertNextValue(id);
    }

    CVLog::Print(QString("[cvSelectionManipulationTool] Found %1 topological "
                         "neighbors for %2 selected items")
                         .arg(result->GetNumberOfTuples())
                         .arg(selection->GetNumberOfTuples()));

    return result;
}

//-----------------------------------------------------------------------------
vtkIdTypeArray* cvSelectionManipulationTool::findBoundary(
        const vtkSmartPointer<vtkIdTypeArray>& selection,
        int fieldAssociation) {
    if (!selection || selection->GetNumberOfTuples() == 0) {
        CVLog::Warning(
                "[cvSelectionManipulationTool] No selection for finding "
                "boundary");
        return nullptr;
    }

    if (!m_viewer) {
        CVLog::Error("[cvSelectionManipulationTool] No visualizer available");
        return nullptr;
    }

    // TOPOLOGY-BASED IMPLEMENTATION (ParaView-consistent)
    // Boundary elements are those in the selection that have at least one
    // topological neighbor NOT in the selection

    // Get polyData using centralized ParaView-style method
    vtkPolyData* polyData = getPolyDataForSelection();

    if (!polyData) {
        CVLog::Error(
                "[cvSelectionManipulationTool] No polyData available for "
                "topology queries");
        vtkIdTypeArray* result = vtkIdTypeArray::New();
        return result;
    }

    // Build topological links
    polyData->BuildLinks();

    QSet<vtkIdType> selectionSet;
    QSet<vtkIdType> boundarySet;

    // Add selection to set for fast lookup
    for (vtkIdType i = 0; i < selection->GetNumberOfTuples(); ++i) {
        selectionSet.insert(selection->GetValue(i));
    }

    // Identify boundary elements
    if (fieldAssociation == 0) {
        // CELLS: A cell is on boundary if it has any neighbor cell not in
        // selection
        for (vtkIdType cellId : selectionSet) {
            QSet<vtkIdType> neighbors;
            findCellNeighbors(cellId, polyData, neighbors);

            // If any neighbor is outside selection, this is a boundary cell
            for (vtkIdType neighborId : neighbors) {
                if (!selectionSet.contains(neighborId)) {
                    boundarySet.insert(cellId);
                    break;  // Already confirmed as boundary
                }
            }
        }
    } else {
        // POINTS: A point is on boundary if it has any neighbor point not in
        // selection
        for (vtkIdType pointId : selectionSet) {
            QSet<vtkIdType> neighbors;
            findPointNeighbors(pointId, polyData, neighbors);

            // If any neighbor is outside selection, this is a boundary point
            for (vtkIdType neighborId : neighbors) {
                if (!selectionSet.contains(neighborId)) {
                    boundarySet.insert(pointId);
                    break;  // Already confirmed as boundary
                }
            }
        }
    }

    // Convert set to array
    vtkIdTypeArray* result = vtkIdTypeArray::New();
    for (vtkIdType id : boundarySet) {
        result->InsertNextValue(id);
    }

    CVLog::Print(QString("[cvSelectionManipulationTool] Found %1 boundary "
                         "elements out of %2 selected items")
                         .arg(result->GetNumberOfTuples())
                         .arg(selection->GetNumberOfTuples()));

    return result;
}

//-----------------------------------------------------------------------------
void cvSelectionManipulationTool::findCellNeighbors(
        vtkIdType cellId, vtkPolyData* polyData, QSet<vtkIdType>& neighbors) {
    if (!polyData || cellId < 0 || cellId >= polyData->GetNumberOfCells()) {
        return;
    }

    // ParaView-style topological neighbor finding for cells:
    // Two cells are neighbors if they share at least one point

    vtkCell* cell = polyData->GetCell(cellId);
    if (!cell) return;

    vtkIdList* pointIds = cell->GetPointIds();

    // For each point in this cell, find all cells that use this point
    for (vtkIdType i = 0; i < pointIds->GetNumberOfIds(); ++i) {
        vtkIdType pointId = pointIds->GetId(i);

        vtkSmartPointer<vtkIdList> cellIds = vtkSmartPointer<vtkIdList>::New();
        polyData->GetPointCells(pointId, cellIds);

        // Add all neighbor cells (except the original cell)
        for (vtkIdType j = 0; j < cellIds->GetNumberOfIds(); ++j) {
            vtkIdType neighborCellId = cellIds->GetId(j);
            if (neighborCellId != cellId) {
                neighbors.insert(neighborCellId);
            }
        }
    }
}

//-----------------------------------------------------------------------------
void cvSelectionManipulationTool::findPointNeighbors(
        vtkIdType pointId, vtkPolyData* polyData, QSet<vtkIdType>& neighbors) {
    if (!polyData || pointId < 0 || pointId >= polyData->GetNumberOfPoints()) {
        return;
    }

    // ParaView-style topological neighbor finding for points:
    // Two points are neighbors if they are connected by an edge (share a cell)

    vtkSmartPointer<vtkIdList> cellIds = vtkSmartPointer<vtkIdList>::New();
    polyData->GetPointCells(pointId, cellIds);

    // For each cell that uses this point, get all other points in that cell
    for (vtkIdType i = 0; i < cellIds->GetNumberOfIds(); ++i) {
        vtkIdType cellId = cellIds->GetId(i);
        vtkCell* cell = polyData->GetCell(cellId);
        if (!cell) continue;

        vtkIdList* cellPointIds = cell->GetPointIds();

        // Add all points in this cell (except the original point)
        for (vtkIdType j = 0; j < cellPointIds->GetNumberOfIds(); ++j) {
            vtkIdType neighborPointId = cellPointIds->GetId(j);
            if (neighborPointId != pointId) {
                neighbors.insert(neighborPointId);
            }
        }
    }
}

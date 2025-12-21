// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionBase.h"

#include "PclUtils/PCLVis.h"
#include "cvViewSelectionManager.h"  // For singleton access

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkActor.h>
#include <vtkDataSet.h>
#include <vtkDataSetMapper.h>
#include <vtkMapper.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPropCollection.h>
#include <vtkRenderer.h>

// Qt
#include <QList>

// ECV
#include <CVLog.h>

//-----------------------------------------------------------------------------
PclUtils::PCLVis* cvSelectionBase::getPCLVis() const {
    if (!m_viewer) {
        return nullptr;
    }
    return reinterpret_cast<PclUtils::PCLVis*>(m_viewer);
}

//-----------------------------------------------------------------------------
bool cvSelectionBase::hasValidPCLVis() const {
    return m_viewer && getPCLVis() != nullptr;
}

//-----------------------------------------------------------------------------
// ParaView-style helper methods
//-----------------------------------------------------------------------------

vtkDataSet* cvSelectionBase::getDataFromActor(vtkActor* actor) {
    if (!actor) {
        return nullptr;
    }

    vtkMapper* mapper = actor->GetMapper();
    if (!mapper) {
        return nullptr;
    }

    vtkDataSet* data = mapper->GetInput();

    if (data) {
        CVLog::PrintDebug(QString("[cvSelectionBase::getDataFromActor] Got "
                                  "data: %1 points, %2 cells")
                                  .arg(data->GetNumberOfPoints())
                                  .arg(data->GetNumberOfCells()));
    }

    return data;
}

//-----------------------------------------------------------------------------
vtkPolyData* cvSelectionBase::getPolyDataForSelection(
        const cvSelectionData* selectionData) {
    vtkPolyData* polyData = nullptr;

    // Priority 1: From provided selection data's actor info (ParaView way!)
    if (selectionData && selectionData->hasActorInfo()) {
        polyData = selectionData->primaryPolyData();
        if (polyData) {
            CVLog::PrintDebug(
                    "[cvSelectionBase] Using polyData from provided selection "
                    "actor info");
            return polyData;
        }
    }

    // Priority 2: From selection manager singleton
    // Note: Only call if this is NOT the manager itself to avoid circular calls
    cvViewSelectionManager* manager = cvViewSelectionManager::instance();
    if (manager && dynamic_cast<cvViewSelectionManager*>(this) != manager) {
        polyData = manager->getPolyData();
        if (polyData) {
            CVLog::PrintDebug(
                    "[cvSelectionBase] Using polyData from selection manager "
                    "singleton");
            return polyData;
        }
    }

    // Priority 3: Fallback to first data actor (for non-selection operations)
    if (!polyData) {
        QList<vtkActor*> actors = getDataActors();
        if (!actors.isEmpty()) {
            vtkDataSet* data = getDataFromActor(actors.first());
            polyData = vtkPolyData::SafeDownCast(data);
            if (polyData) {
                CVLog::PrintDebug(
                        "[cvSelectionBase] Fallback to first data actor");
            }
        }
    }

    if (!polyData) {
        CVLog::Warning("[cvSelectionBase] No polyData available for selection");
    }

    return polyData;
}

//-----------------------------------------------------------------------------
QList<vtkActor*> cvSelectionBase::getDataActors() const {
    QList<vtkActor*> actors;

    PclUtils::PCLVis* pclVis = getPCLVis();
    if (!pclVis) {
        CVLog::Warning(
                "[cvSelectionBase::getDataActors] visualizer is nullptr");
        return actors;
    }

    vtkRenderer* renderer = pclVis->getCurrentRenderer();
    if (!renderer) {
        CVLog::Warning("[cvSelectionBase::getDataActors] renderer is nullptr");
        return actors;
    }

    // Collect all pickable actors with data
    vtkPropCollection* props = renderer->GetViewProps();
    props->InitTraversal();

    while (vtkProp* prop = props->GetNextProp()) {
        vtkActor* actor = vtkActor::SafeDownCast(prop);

        if (actor && actor->GetVisibility() && actor->GetPickable() &&
            actor->GetMapper() && actor->GetMapper()->GetInput()) {
            actors.append(actor);
        }
    }

    CVLog::PrintDebug(
            QString("[cvSelectionBase::getDataActors] Found %1 data actors")
                    .arg(actors.size()));

    return actors;
}

//-----------------------------------------------------------------------------
std::vector<vtkPolyData*> cvSelectionBase::getAllPolyDataFromVisualizer() {
    std::vector<vtkPolyData*> polyDataList;

    PclUtils::PCLVis* pclVis = getPCLVis();
    if (!pclVis) {
        CVLog::Warning(
                "[cvSelectionBase::getAllPolyDataFromVisualizer] visualizer is "
                "nullptr");
        return polyDataList;
    }

    vtkRenderer* renderer = pclVis->getCurrentRenderer();
    if (!renderer) {
        CVLog::Warning(
                "[cvSelectionBase::getAllPolyDataFromVisualizer] renderer is "
                "nullptr");
        return polyDataList;
    }

    // Collect all polydata from actors
    vtkPropCollection* props = renderer->GetViewProps();
    props->InitTraversal();

    int actorCount = 0;
    int polyDataCount = 0;

    while (vtkProp* p = props->GetNextProp()) {
        vtkActor* actor = vtkActor::SafeDownCast(p);
        if (actor && actor->GetMapper() &&
            actor->GetVisibility() &&  // Check visibility
            actor->GetPickable()) {    // Check if pickable (skip helper actors)
            actorCount++;
            vtkMapper* mapper = actor->GetMapper();

            vtkPolyData* polyData = nullptr;

            // Try vtkPolyDataMapper first (most common for textured meshes)
            vtkPolyDataMapper* polyMapper =
                    vtkPolyDataMapper::SafeDownCast(mapper);
            if (polyMapper) {
                polyData = vtkPolyData::SafeDownCast(polyMapper->GetInput());
            } else {
                // Try vtkDataSetMapper (used for non-textured meshes)
                vtkDataSetMapper* datasetMapper =
                        vtkDataSetMapper::SafeDownCast(mapper);
                if (datasetMapper) {
                    vtkDataSet* dataset = datasetMapper->GetInput();
                    if (dataset) {
                        polyData = vtkPolyData::SafeDownCast(dataset);
                    }
                }
            }

            // Add valid polyData to list
            if (polyData && polyData->GetNumberOfCells() > 0) {
                polyDataList.push_back(polyData);
                polyDataCount++;
                CVLog::PrintDebug(QString("[cvSelectionBase::"
                                          "getAllPolyDataFromVisualizer] Found "
                                          "PolyData #%1: "
                                          "%2 cells, %3 points")
                                          .arg(polyDataCount)
                                          .arg(polyData->GetNumberOfCells())
                                          .arg(polyData->GetNumberOfPoints()));
            }
        }
    }

    CVLog::PrintDebug(QString("[cvSelectionBase::getAllPolyDataFromVisualizer] "
                              "Found %1 polyData(s) "
                              "from %2 actor(s)")
                              .arg(polyDataCount)
                              .arg(actorCount));

    return polyDataList;
}

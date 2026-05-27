// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VtkLODHelper.h"

#include <VTKExtensions/Views/vtkPVLODActor.h>

#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkDataSetMapper.h>
#include <vtkMaskPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkQuadricClustering.h>
#include <vtkRenderer.h>

#include <algorithm>
#include <cmath>

namespace VtkRendering {

bool BuildAndAttachLODMapper(vtkPVLODActor* actor, vtkDataSet* data) {
    if (!actor || !data) return false;

    vtkIdType numPoints = data->GetNumberOfPoints();
    if (numPoints < kLODPointThreshold) return false;

    auto* poly = vtkPolyData::SafeDownCast(data);
    bool isMesh = poly && poly->GetNumberOfCells() > 0 &&
                  poly->GetNumberOfCells() != poly->GetNumberOfVerts();

    vtkSmartPointer<vtkMapper> lodMapper;

    if (isMesh) {
        auto decimator = vtkSmartPointer<vtkQuadricClustering>::New();
        decimator->SetInputData(poly);
        decimator->SetUseInputPoints(true);
        decimator->SetCopyCellData(true);
        double bounds[6];
        poly->GetBounds(bounds);
        double maxDim = std::max({bounds[1] - bounds[0],
                                  bounds[3] - bounds[2],
                                  bounds[5] - bounds[4]});
        int divs = static_cast<int>(
                std::cbrt(static_cast<double>(numPoints) * kLODTargetRatio));
        divs = std::max(divs, 10);
        decimator->SetNumberOfXDivisions(divs);
        decimator->SetNumberOfYDivisions(divs);
        decimator->SetNumberOfZDivisions(divs);
        decimator->Update();

        auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputConnection(decimator->GetOutputPort());
        auto* origMapper = actor->GetMapper();
        if (origMapper) {
            mapper->SetScalarVisibility(origMapper->GetScalarVisibility());
            mapper->SetScalarMode(origMapper->GetScalarMode());
            if (origMapper->GetLookupTable())
                mapper->SetLookupTable(origMapper->GetLookupTable());
            double range[2];
            origMapper->GetScalarRange(range);
            mapper->SetScalarRange(range);
        }
        lodMapper = mapper;
    } else {
        auto mask = vtkSmartPointer<vtkMaskPoints>::New();
        mask->SetInputData(data);
        vtkIdType targetPts = std::min(
                static_cast<vtkIdType>(numPoints * kLODTargetRatio),
                kLODMaxPoints);
        int ratio = std::max(static_cast<int>(numPoints / targetPts), 2);
        mask->SetOnRatio(ratio);
        mask->SetRandomModeType(0);
        mask->Update();

        auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputConnection(mask->GetOutputPort());
        auto* origMapper = actor->GetMapper();
        if (origMapper) {
            mapper->SetScalarVisibility(origMapper->GetScalarVisibility());
            mapper->SetScalarMode(origMapper->GetScalarMode());
            if (origMapper->GetLookupTable())
                mapper->SetLookupTable(origMapper->GetLookupTable());
            double range[2];
            origMapper->GetScalarRange(range);
            mapper->SetScalarRange(range);
        }
        lodMapper = mapper;
    }

    actor->SetLODMapper(lodMapper);
    return true;
}

void SetLODEnabledForRenderer(vtkRenderer* renderer, bool enable) {
    if (!renderer) return;
    auto* actors = renderer->GetActors();
    if (!actors) return;
    actors->InitTraversal();
    vtkActor* a = nullptr;
    while ((a = actors->GetNextActor())) {
        auto* pvlod = vtkPVLODActor::SafeDownCast(a);
        if (pvlod && pvlod->GetLODMapper()) {
            pvlod->SetEnableLOD(enable ? 1 : 0);
        }
    }
}

}  // namespace VtkRendering

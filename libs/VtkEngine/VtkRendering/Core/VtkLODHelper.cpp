// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VtkLODHelper.h"

#include <VTKExtensions/Views/vtkPVLODActor.h>
#include <Visualization/VtkVis.h>
#include <Visualization/vtkGLView.h>
#include <ecvDisplayTools.h>
#include <ecvGenericVisualizer3D.h>
#include <ecvGuiParameters.h>
#include <ecvViewManager.h>
#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkDataObject.h>
#include <vtkDataSetMapper.h>
#include <vtkLookupTable.h>
#include <vtkMapper.h>
#include <vtkPointGaussianMapper.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkQuadricClustering.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkScalarsToColors.h>

#include <QCoreApplication>
#include <QtConcurrent/QtConcurrentRun>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <unordered_map>

namespace VtkRendering {

namespace {

struct MapperScalarState {
    int scalarVisibility = 0;
    int scalarMode = 0;
    vtkSmartPointer<vtkScalarsToColors> lookupTable;
    double range[2] = {0.0, 0.0};
};

MapperScalarState CaptureMapperScalars(vtkMapper* mapper) {
    MapperScalarState state;
    if (!mapper) return state;
    state.scalarVisibility = mapper->GetScalarVisibility();
    state.scalarMode = mapper->GetScalarMode();
    state.lookupTable = mapper->GetLookupTable();
    mapper->GetScalarRange(state.range);
    return state;
}

void ApplyMapperScalars(vtkMapper* dest, const MapperScalarState& state) {
    if (!dest) return;
    dest->SetScalarVisibility(state.scalarVisibility);
    dest->SetScalarMode(state.scalarMode);
    if (state.lookupTable) {
        dest->SetLookupTable(state.lookupTable);
    }
    dest->SetScalarRange(state.range);
}

vtkSmartPointer<vtkPolyDataMapper> BuildQuadricLODMapper(
        vtkDataSet* data,
        const MapperScalarState& scalars,
        double lodResolution) {
    if (!data || data->GetNumberOfPoints() == 0) return nullptr;

    auto decimator = vtkSmartPointer<vtkQuadricClustering>::New();
    decimator->SetInputData(data);
    ConfigureParaViewQuadricClustering(decimator, lodResolution);
    decimator->Update();

    auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(decimator->GetOutputPort());
    ApplyMapperScalars(mapper, scalars);
    mapper->SetStatic(1);
    return mapper;
}

void SetDesiredUpdateRateForRenderer(vtkRenderer* renderer, double rate) {
    if (!renderer) return;
    vtkRenderWindow* rw = renderer->GetRenderWindow();
    if (!rw) return;
    vtkRenderWindowInteractor* iren = rw->GetInteractor();
    if (iren) {
        iren->SetDesiredUpdateRate(rate);
    }
}

std::mutex g_lodBuildMutex;
std::unordered_map<vtkPVLODActor*, uint64_t> g_lodBuildGeneration;

uint64_t NextLODBuildGeneration(vtkPVLODActor* actor) {
    std::lock_guard<std::mutex> lock(g_lodBuildMutex);
    return ++g_lodBuildGeneration[actor];
}

uint64_t CurrentLODBuildGeneration(vtkPVLODActor* actor) {
    std::lock_guard<std::mutex> lock(g_lodBuildMutex);
    auto it = g_lodBuildGeneration.find(actor);
    return it != g_lodBuildGeneration.end() ? it->second : 0;
}

void AttachLODMapperOnMainThread(
        const vtkWeakPointer<vtkPVLODActor>& weakActor,
        const vtkSmartPointer<vtkPolyDataMapper>& lodMapper,
        vtkDataObject* expectedSource,
        uint64_t buildGeneration) {
    vtkPVLODActor* actor = weakActor.Get();
    if (!actor || !lodMapper || !expectedSource) return;

    if (CurrentLODBuildGeneration(actor) != buildGeneration) return;
    if (actor->GetFullResolutionInput() != expectedSource) return;

    actor->SetLODMapper(lodMapper);
}

/// Zero-copy async build: keep @a data alive via ref-count; safe when updates
/// replace mapper input with a new vtkDataObject instead of mutating in place.
void ScheduleLODMapperBuild(vtkPVLODActor* actor,
                            vtkDataSet* data,
                            vtkMapper* origMapper,
                            double lodResolution) {
    if (!actor || !data || data->GetNumberOfPoints() == 0) return;
    if (actor->GetLODMapper()) return;

    vtkSmartPointer<vtkDataSet> frozenInput = data;
    vtkDataObject* sourceToken = frozenInput.Get();
    const uint64_t buildGeneration = NextLODBuildGeneration(actor);

    vtkWeakPointer<vtkPVLODActor> weakActor = actor;
    const MapperScalarState scalars = CaptureMapperScalars(origMapper);

    (void)QtConcurrent::run([weakActor, frozenInput, sourceToken, scalars,
                             lodResolution, buildGeneration]() {
        vtkSmartPointer<vtkPolyDataMapper> lodMapper =
                BuildQuadricLODMapper(frozenInput, scalars, lodResolution);
        if (!lodMapper) return;

        QCoreApplication* app = QCoreApplication::instance();
        if (!app) return;

        QMetaObject::invokeMethod(
                app,
                [weakActor, lodMapper, sourceToken, buildGeneration]() {
                    AttachLODMapperOnMainThread(weakActor, lodMapper,
                                                sourceToken, buildGeneration);
                },
                Qt::QueuedConnection);
    });
}

bool ShouldBuildLODForActor(vtkPVLODActor* actor) {
    if (!actor || actor->GetLODMapper()) return false;
    if (GetSkipPointGaussianLOD() &&
        vtkPointGaussianMapper::SafeDownCast(
                actor->GetFullResolutionMapper())) {
        return false;
    }
    vtkDataObject* input = actor->GetFullResolutionInput();
    auto* data = vtkDataSet::SafeDownCast(input);
    return data && data->GetNumberOfPoints() > 0;
}

}  // namespace

double GetLODRenderingThresholdMB() {
    return std::max(0.0, ecvGui::Parameters().lodRenderingThresholdMB);
}

double GetLODResolution() {
    return std::clamp(ecvGui::Parameters().lodResolution, 0.0, 1.0);
}

double GetInteractiveUpdateRate() {
    const double rate = ecvGui::Parameters().lodInteractiveUpdateRate;
    return rate > 0.0 ? rate : kDefaultInteractiveUpdateRate;
}

double GetStillUpdateRate() {
    const double rate = ecvGui::Parameters().lodStillUpdateRate;
    return rate > 0.0 ? rate : kDefaultStillUpdateRate;
}

bool GetSkipPointGaussianLOD() {
    return ecvGui::Parameters().lodSkipPointGaussian;
}

int LODResolutionToQuadricDivisions(double lodResolution) {
    lodResolution = std::clamp(lodResolution, 0.0, 1.0);
    return static_cast<int>(150.0 * lodResolution) + 10;
}

void ConfigureParaViewQuadricClustering(vtkQuadricClustering* decimator,
                                        double lodResolution) {
    if (!decimator) return;
    const int divs = LODResolutionToQuadricDivisions(lodResolution);
    decimator->SetNumberOfXDivisions(divs);
    decimator->SetNumberOfYDivisions(divs);
    decimator->SetNumberOfZDivisions(divs);
    decimator->SetUseInputPoints(1);
    decimator->SetCopyCellData(1);
    decimator->SetUseInternalTriangles(0);
}

bool ShouldUseLODRendering(double geometrySizeMB) {
    return GetLODRenderingThresholdMB() <= geometrySizeMB;
}

double ComputeRendererGeometrySizeMB(vtkRenderer* renderer) {
    if (!renderer) return 0.0;

    vtkIdType sizeKB = 0;
    vtkActorCollection* actors = renderer->GetActors();
    if (!actors) return 0.0;

    actors->InitTraversal();
    while (vtkActor* actor = actors->GetNextActor()) {
        if (!actor->GetVisibility()) continue;

        vtkMapper* mapper = actor->GetMapper();
        if (!mapper) continue;

        vtkDataObject* input = mapper->GetInputDataObject(0, 0);
        if (input) {
            sizeKB += input->GetActualMemorySize();
        }
    }

    return static_cast<double>(sizeKB) / 1024.0;
}

double ComputeVisibleDataSizeMB(const CloudActorMap& actors) {
    vtkIdType sizeKB = 0;
    for (const auto& kv : actors) {
        vtkPVLODActor* actor = kv.second.actor;
        if (!actor || !actor->GetVisibility()) continue;

        vtkDataObject* input = actor->GetFullResolutionInput();
        if (input) {
            sizeKB += input->GetActualMemorySize();
        }
    }
    return static_cast<double>(sizeKB) / 1024.0;
}

void InvalidateLODMapper(vtkPVLODActor* actor) {
    if (!actor) return;
    actor->SetLODMapper(nullptr);
    NextLODBuildGeneration(actor);
}

bool BuildAndAttachLODMapper(vtkPVLODActor* actor, vtkDataSet* data) {
    return BuildAndAttachLODMapper(actor, data, GetLODResolution());
}

bool BuildAndAttachLODMapper(vtkPVLODActor* actor,
                             vtkDataSet* data,
                             double lodResolution) {
    if (!actor || !data || data->GetNumberOfPoints() == 0) return false;
    // Lazy: defer quadric build until interaction
    // (EnsureLODMappersForCloudActors).
    (void)lodResolution;
    return true;
}

bool BuildAndAttachLODMapperSync(vtkPVLODActor* actor,
                                 vtkDataSet* data,
                                 double lodResolution) {
    if (!actor || !data || data->GetNumberOfPoints() == 0) return false;

    vtkSmartPointer<vtkPolyDataMapper> lodMapper = BuildQuadricLODMapper(
            data, CaptureMapperScalars(actor->GetFullResolutionMapper()),
            lodResolution);
    if (!lodMapper) return false;

    actor->SetLODMapper(lodMapper);
    return true;
}

void EnsureLODMappersForCloudActors(const CloudActorMap& actors) {
    const double geometrySizeMB = ComputeVisibleDataSizeMB(actors);
    if (!ShouldUseLODRendering(geometrySizeMB)) return;

    const double lodResolution = GetLODResolution();
    for (const auto& kv : actors) {
        vtkPVLODActor* actor = kv.second.actor;
        if (!ShouldBuildLODForActor(actor)) continue;

        vtkDataSet* data =
                vtkDataSet::SafeDownCast(actor->GetFullResolutionInput());
        ScheduleLODMapperBuild(actor, data, actor->GetFullResolutionMapper(),
                               lodResolution);
    }
}

void SetLODEnabledForRenderer(vtkRenderer* renderer, bool enable) {
    if (!renderer) return;
    vtkActorCollection* actors = renderer->GetActors();
    if (!actors) return;

    actors->InitTraversal();
    while (vtkActor* actor = actors->GetNextActor()) {
        auto* pvlod = vtkPVLODActor::SafeDownCast(actor);
        if (!pvlod || !pvlod->GetLODMapper()) continue;

        if (enable && GetSkipPointGaussianLOD() &&
            vtkPointGaussianMapper::SafeDownCast(
                    pvlod->GetFullResolutionMapper())) {
            continue;
        }

        pvlod->SetEnableLOD(enable ? 1 : 0);
    }
}

void BeginInteractiveLOD(vtkRenderer* renderer, double geometrySizeMB) {
    SetDesiredUpdateRateForRenderer(renderer, GetInteractiveUpdateRate());
    if (ShouldUseLODRendering(geometrySizeMB)) {
        SetLODEnabledForRenderer(renderer, true);
    } else {
        SetLODEnabledForRenderer(renderer, false);
    }
}

void BeginInteractiveLOD(vtkRenderer* renderer) {
    BeginInteractiveLOD(renderer, ComputeRendererGeometrySizeMB(renderer));
}

void EndInteractiveLOD(vtkRenderer* renderer) {
    SetDesiredUpdateRateForRenderer(renderer, GetStillUpdateRate());
    SetLODEnabledForRenderer(renderer, false);
}

void RebuildAllInteractiveLOD() {
    auto rebuildVis = [](ecvGenericVisualizer3D* viz3d) {
        auto* vis = dynamic_cast<Visualization::VtkVis*>(viz3d);
        if (vis) vis->rebuildAllLODMappers();
    };

    for (auto* view : ecvViewManager::instance().getAllViews()) {
        if (!view) continue;
        if (auto* glView = dynamic_cast<vtkGLView*>(view)) {
            rebuildVis(glView->getVisualizer3D());
        }
    }

    if (auto* dt = ecvViewManager::instance().displayTools()) {
        rebuildVis(dt->getVisualizer3D());
    }
}

void SyncStillUpdateRatesForAllViews() {
    const double rate = GetStillUpdateRate();
    auto syncVis = [rate](ecvGenericVisualizer3D* viz3d) {
        auto* vis = dynamic_cast<Visualization::VtkVis*>(viz3d);
        if (!vis) return;
        if (auto iren = vis->getRenderWindowInteractor()) {
            iren->SetDesiredUpdateRate(rate);
        }
    };

    for (auto* view : ecvViewManager::instance().getAllViews()) {
        if (!view) continue;
        if (auto* glView = dynamic_cast<vtkGLView*>(view)) {
            syncVis(glView->getVisualizer3D());
        }
    }

    if (auto* dt = ecvViewManager::instance().displayTools()) {
        syncVis(dt->getVisualizer3D());
    }
}

}  // namespace VtkRendering

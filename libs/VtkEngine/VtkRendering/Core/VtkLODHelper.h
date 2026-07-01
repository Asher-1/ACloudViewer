// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <VtkRendering/Core/ActorMap.h>
#include <vtkSmartPointer.h>

#include "qVTK.h"

class vtkMapper;
class vtkPVLODActor;
class vtkPolyData;
class vtkDataSet;
class vtkRenderer;
class vtkQuadricClustering;

namespace VtkRendering {

/// Default values aligned with ParaView RenderViewSettings.
constexpr double kDefaultLODRenderingThresholdMB = 20.0;
constexpr double kDefaultLODResolution = 0.5;
constexpr double kDefaultStillUpdateRate = 0.002;
constexpr double kDefaultInteractiveUpdateRate = 5.0;

/// Current LOD threshold (MB) from Display Settings / ecvGui parameters.
QVTK_ENGINE_LIB_API double GetLODRenderingThresholdMB();

/// Current LOD resolution [0, 1] from Display Settings.
QVTK_ENGINE_LIB_API double GetLODResolution();

/// Current interactive desired update rate from Display Settings.
QVTK_ENGINE_LIB_API double GetInteractiveUpdateRate();

/// Current still desired update rate from Display Settings.
QVTK_ENGINE_LIB_API double GetStillUpdateRate();

/// Whether Point Gaussian actors skip LOD during interaction.
QVTK_ENGINE_LIB_API bool GetSkipPointGaussianLOD();

/// Map ParaView LODResolution to vtkQuadricClustering grid divisions.
/// Matches vtkGeometryRepresentationInternal.h (non-TBB path):
/// factor 0.0 -> 10, 0.5 -> 85, 1.0 -> 160.
QVTK_ENGINE_LIB_API int LODResolutionToQuadricDivisions(double lodResolution);

/// Configure vtkQuadricClustering like ParaView's DecimationFilterType.
QVTK_ENGINE_LIB_API void ConfigureParaViewQuadricClustering(
        vtkQuadricClustering* decimator, double lodResolution);

/// Returns true when interactive LOD should be enabled for @a geometrySizeMB.
/// Mirrors vtkPVRenderView::ShouldUseLODRendering().
QVTK_ENGINE_LIB_API bool ShouldUseLODRendering(double geometrySizeMB);

/// Sum GetActualMemorySize() (KB) of all visible renderer actors, converted to
/// MB. Includes overlays (axes, scale bar, etc.) — use ComputeVisibleDataSizeMB
/// when matching ParaView's visible-representation threshold.
QVTK_ENGINE_LIB_API double ComputeRendererGeometrySizeMB(vtkRenderer* renderer);

/// Sum GetActualMemorySize() (KB) of visible cloud/mesh actors only.
/// Mirrors ParaView GetDeliveryManager()->GetVisibleDataSize(false) / 1024.0.
QVTK_ENGINE_LIB_API double ComputeVisibleDataSizeMB(
        const CloudActorMap& actors);

/// Lazy/async LOD build: no work at load time; builds on first interactive
/// render.
/// @return true if a build was scheduled or is already present.
QVTK_ENGINE_LIB_API bool BuildAndAttachLODMapper(vtkPVLODActor* actor,
                                                 vtkDataSet* data);
QVTK_ENGINE_LIB_API bool BuildAndAttachLODMapper(vtkPVLODActor* actor,
                                                 vtkDataSet* data,
                                                 double lodResolution);

/// Drop cached LOD after geometry changes; rebuilt lazily on next interaction.
QVTK_ENGINE_LIB_API void InvalidateLODMapper(vtkPVLODActor* actor);

/// Build missing LOD mappers for visible scene actors (called on interaction
/// start).
QVTK_ENGINE_LIB_API void EnsureLODMappersForCloudActors(
        const CloudActorMap& actors);

/// Synchronous LOD build (settings rebuild, small overlays).
QVTK_ENGINE_LIB_API bool BuildAndAttachLODMapperSync(vtkPVLODActor* actor,
                                                     vtkDataSet* data,
                                                     double lodResolution);

/// Enable or disable LOD on vtkPVLODActor instances that have a LOD mapper.
QVTK_ENGINE_LIB_API void SetLODEnabledForRenderer(vtkRenderer* renderer,
                                                  bool enable);

/// ParaView interactive render begin: update rate + conditional LOD enable.
QVTK_ENGINE_LIB_API void BeginInteractiveLOD(vtkRenderer* renderer);
QVTK_ENGINE_LIB_API void BeginInteractiveLOD(vtkRenderer* renderer,
                                             double geometrySizeMB);

/// ParaView still render end: update rate + disable LOD.
QVTK_ENGINE_LIB_API void EndInteractiveLOD(vtkRenderer* renderer);

/// Rebuild LOD mappers on all views (after LOD resolution change in settings).
QVTK_ENGINE_LIB_API void RebuildAllInteractiveLOD();

/// Apply still update rate to all render window interactors.
QVTK_ENGINE_LIB_API void SyncStillUpdateRatesForAllViews();

}  // namespace VtkRendering

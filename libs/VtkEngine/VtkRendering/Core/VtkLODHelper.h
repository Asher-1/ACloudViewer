// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vtkSmartPointer.h>

#include "qVTK.h"

class vtkMapper;
class vtkPVLODActor;
class vtkPolyData;
class vtkDataSet;
class vtkRenderer;

namespace VtkRendering {

/// Threshold (number of points) above which LOD is generated.
constexpr vtkIdType kLODPointThreshold = 500000;

/// Target fraction of points for the LOD representation (0.0–1.0).
constexpr double kLODTargetRatio = 0.1;

/// Maximum number of points in the LOD representation.
constexpr vtkIdType kLODMaxPoints = 100000;

/// Build a LOD mapper for the given input data and attach it to the actor.
/// For meshes (cells > verts), uses vtkQuadricClustering for fast decimation.
/// For point clouds, uses vtkMaskPoints for uniform downsampling.
/// Only generates LOD when the input exceeds kLODPointThreshold.
/// Returns true if a LOD mapper was successfully created and attached.
QVTK_ENGINE_LIB_API bool BuildAndAttachLODMapper(vtkPVLODActor* actor,
                                                 vtkDataSet* data);

/// Enable or disable LOD on all vtkPVLODActor instances in a renderer.
/// Typically called at interaction start (enable=true) and end (enable=false).
QVTK_ENGINE_LIB_API void SetLODEnabledForRenderer(vtkRenderer* renderer,
                                                  bool enable);

}  // namespace VtkRendering

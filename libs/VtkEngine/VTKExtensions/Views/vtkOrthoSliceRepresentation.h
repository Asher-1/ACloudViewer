// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vtkSmartPointer.h>

#include "qVTK.h"

class vtkActor;
class vtkPolyData;
class vtkRenderer;

class QVTK_ENGINE_LIB_API vtkOrthoSliceRepresentation {
public:
    enum PlaneIndex { PLANE_TOP = 0, PLANE_SIDE = 1, PLANE_FRONT = 2 };

    vtkOrthoSliceRepresentation();
    ~vtkOrthoSliceRepresentation();

    void setInputPolyData(vtkPolyData* input);
    vtkPolyData* inputPolyData() const;

    void setSliceOrigin(double x, double y, double z);
    void update();

    void addToRenderers(vtkRenderer* orthoRenderers[3],
                        vtkRenderer* perspectiveRenderer);
    void removeFromRenderers(vtkRenderer* orthoRenderers[3],
                             vtkRenderer* perspectiveRenderer);

    vtkActor* sliceActor2D(PlaneIndex plane) const;
    vtkActor* sliceActor3D(PlaneIndex plane) const;
    vtkActor* surfaceActor3D() const;
    vtkActor* outlineActor3D() const;

    bool hasValidSlice(PlaneIndex plane) const;

private:
    void ensureSurfaceActor();
    void ensureOutlineActor();

    struct Impl;
    Impl* d;
};

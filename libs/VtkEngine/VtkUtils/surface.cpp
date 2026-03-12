// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "surface.h"

#include <vtkDelaunay2D.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>

#include "vtkutils.h"

namespace VtkUtils {

class SurfacePrivate {
public:
    SurfacePrivate(Surface* q) : q_ptr(q) {}

    Surface* q_ptr;
    vtkActor* surfaceActor = nullptr;
    QList<Point3F> points;
};

Surface::Surface(QWidget* parent) : VtkWidget(parent) {
    d_ptr = new SurfacePrivate(this);
}

Surface::~Surface() { delete d_ptr; }

void Surface::setPoints(const QList<Point3F>& points) {
    d_ptr->points = points;
    renderSurface();
}

void Surface::renderSurface() {
    if (d_ptr->points.isEmpty()) return;

    VTK_CREATE(vtkPoints, vtkpoints);

    foreach (const Point3F& pot, d_ptr->points)
        vtkpoints->InsertNextPoint(pot.x, pot.y, pot.z);

    VTK_CREATE(vtkPolyData, polydata);
    polydata->SetPoints(vtkpoints);

    VTK_CREATE(vtkDelaunay2D, del);
    del->SetInputData(polydata);
    del->Update();

    VTK_CREATE(vtkPolyDataMapper, mapper);
    mapper->SetInputConnection(del->GetOutputPort());
    mapper->Update();

    surfaceActor()->SetMapper(mapper);
    defaultRenderer()->AddActor(surfaceActor());

    update();
}

vtkActor* Surface::surfaceActor() const {
    VtkUtils::vtkInitOnce(&d_ptr->surfaceActor);
    return d_ptr->surfaceActor;
}

}  // namespace VtkUtils

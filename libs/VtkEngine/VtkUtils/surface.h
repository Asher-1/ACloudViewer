// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file surface.h
/// @brief VTK widget for rendering surface from point cloud.

#include "point3f.h"
#include "qVTK.h"
#include "vtkwidget.h"

class vtkActor;
namespace VtkUtils {

class SurfacePrivate;
/// @class Surface
/// @brief Renders a surface mesh from a list of 3D points using VTK.
class QVTK_ENGINE_LIB_API Surface : public VtkWidget {
    Q_OBJECT
public:
    explicit Surface(QWidget* parent = nullptr);
    virtual ~Surface();

    /// @param points List of 3D points for surface generation
    void setPoints(const QList<Point3F>& points);

protected:
    virtual void renderSurface();
    vtkActor* surfaceActor() const;

private:
    SurfacePrivate* d_ptr;
    Q_DISABLE_COPY(Surface)
};

}  // namespace VtkUtils

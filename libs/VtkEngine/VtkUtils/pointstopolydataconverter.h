// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file pointstopolydataconverter.h
/// @brief Converts Point3F array and optional vertices to vtkPolyData.

#include "qVTK.h"

// CV_CORE_LIB
#include <CVGeom.h>
#include <vtkSmartPointer.h>

#include <QVector>

#include "point3f.h"
#include "signalledrunable.h"

class vtkPolyData;
namespace VtkUtils {
/// @class PointsToPolyDataConverter
/// @brief Converts points and triangle vertices to vtkPolyData; runs as
/// SignalledRunnable.
class QVTK_ENGINE_LIB_API PointsToPolyDataConverter : public SignalledRunnable {
    Q_OBJECT
public:
    /// @param points 3D points
    /// @param vertices Triangle vertex indices (optional; for mesh)
    explicit PointsToPolyDataConverter(
            const QVector<Point3F>& points,
            const QVector<Tuple3ui>& vertices = QVector<Tuple3ui>());

    void run();

    /// @return Converted vtkPolyData (after run completes)
    vtkPolyData* polyData() const;

private:
    QVector<Point3F> m_points;
    QVector<Tuple3ui> m_vertices;
    vtkSmartPointer<vtkPolyData> m_polyData;
};

}  // namespace VtkUtils

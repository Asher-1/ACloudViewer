// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pointstopolydataconverter.h"

#include <vtkDoubleArray.h>
#include <vtkPoints.h>

#include "vtkutils.h"

namespace VtkUtils {

PointsToPolyDataConverter::PointsToPolyDataConverter(
        const QVector<Point3F>& points, const QVector<Tuple3ui>& vertices)
    : m_points(points), m_vertices(vertices) {}

void PointsToPolyDataConverter::run() {
    if (m_points.isEmpty()) {
        emit finished();
        return;
    }

    VtkUtils::vtkInitOnce(m_polyData);

    VTK_CREATE(vtkPoints, vtkpoints);
    VTK_CREATE(vtkDoubleArray, scalarArray);
    VTK_CREATE(vtkCellArray, cell_array);
    scalarArray->SetName("scalar");

    int pointNum = static_cast<int>(m_points.size());
    vtkpoints->SetNumberOfPoints(pointNum);

    for (int i = 0; i < pointNum; ++i) {
        const Point3F& p3f = m_points.at(i);
        vtkpoints->InsertPoint(i, p3f.x, p3f.y, p3f.z);
        scalarArray->InsertTuple1(i, p3f.z);  // z value as scalar
    }

    for (int i = 0; i < m_vertices.size(); ++i) {
        cell_array->InsertNextCell(3);
        for (size_t j = 0; j < 3; j++)
            cell_array->InsertCellPoint(m_vertices.at(i).u[j]);
    }
    if (!m_vertices.isEmpty()) {
        m_polyData->SetPolys(cell_array);
    }

    m_polyData->SetPoints(vtkpoints);
    m_polyData->GetPointData()->SetScalars(scalarArray);

    emit finished();
}

vtkPolyData* PointsToPolyDataConverter::polyData() const { return m_polyData; }

}  // namespace VtkUtils

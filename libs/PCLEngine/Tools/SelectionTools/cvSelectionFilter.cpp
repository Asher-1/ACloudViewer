// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionFilter.h"

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkIdTypeArray.h>
#include <vtkMath.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkTriangle.h>

// Qt
#include <QSet>

//-----------------------------------------------------------------------------
cvSelectionFilter::cvSelectionFilter(QObject* parent) : QObject(parent) {
    CVLog::PrintDebug("[cvSelectionFilter] Initialized");
}

//-----------------------------------------------------------------------------
cvSelectionFilter::~cvSelectionFilter() {}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::filterByAttributeRange(
        vtkPolyData* polyData,
        const cvSelectionData& input,
        const QString& attributeName,
        double minValue,
        double maxValue) {
    if (!polyData || input.isEmpty() || attributeName.isEmpty()) {
        return cvSelectionData();
    }

    // Get appropriate data arrays
    vtkDataArray* array = nullptr;
    if (input.fieldAssociation() == cvSelectionData::POINTS) {
        array = polyData->GetPointData()->GetArray(
                attributeName.toUtf8().constData());
    } else {
        array = polyData->GetCellData()->GetArray(
                attributeName.toUtf8().constData());
    }

    if (!array) {
        CVLog::Warning(QString("[cvSelectionFilter] Attribute '%1' not found")
                               .arg(attributeName));
        return cvSelectionData();
    }

    QVector<qint64> filteredIds;
    QVector<qint64> inputIds = input.ids();

    for (qint64 id : inputIds) {
        if (id < 0 || id >= array->GetNumberOfTuples()) {
            continue;
        }

        // Get attribute value (use first component for multi-component arrays)
        double value = array->GetComponent(id, 0);

        if (value >= minValue && value <= maxValue) {
            filteredIds.append(id);
        }
    }

    CVLog::Print(QString("[cvSelectionFilter] Attribute range filter: %1 -> %2 "
                         "items")
                         .arg(inputIds.size())
                         .arg(filteredIds.size()));

    return cvSelectionData(filteredIds, input.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::filterByAttributeComparison(
        vtkPolyData* polyData,
        const cvSelectionData& input,
        const QString& attributeName,
        ComparisonOp op,
        double value) {
    if (!polyData || input.isEmpty() || attributeName.isEmpty()) {
        return cvSelectionData();
    }

    vtkDataArray* array = nullptr;
    if (input.fieldAssociation() == cvSelectionData::POINTS) {
        array = polyData->GetPointData()->GetArray(
                attributeName.toUtf8().constData());
    } else {
        array = polyData->GetCellData()->GetArray(
                attributeName.toUtf8().constData());
    }

    if (!array) {
        CVLog::Warning(QString("[cvSelectionFilter] Attribute '%1' not found")
                               .arg(attributeName));
        return cvSelectionData();
    }

    QVector<qint64> filteredIds;
    QVector<qint64> inputIds = input.ids();

    for (qint64 id : inputIds) {
        if (id < 0 || id >= array->GetNumberOfTuples()) {
            continue;
        }

        double attrValue = array->GetComponent(id, 0);
        bool pass = false;

        switch (op) {
            case EQUAL:
                pass = (qAbs(attrValue - value) < 1e-6);
                break;
            case NOT_EQUAL:
                pass = (qAbs(attrValue - value) >= 1e-6);
                break;
            case LESS_THAN:
                pass = (attrValue < value);
                break;
            case LESS_EQUAL:
                pass = (attrValue <= value);
                break;
            case GREATER_THAN:
                pass = (attrValue > value);
                break;
            case GREATER_EQUAL:
                pass = (attrValue >= value);
                break;
            default:
                break;
        }

        if (pass) {
            filteredIds.append(id);
        }
    }

    CVLog::Print(QString("[cvSelectionFilter] Attribute comparison filter: %1 "
                         "-> %2 items")
                         .arg(inputIds.size())
                         .arg(filteredIds.size()));

    return cvSelectionData(filteredIds, input.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::filterByArea(vtkPolyData* polyData,
                                                const cvSelectionData& input,
                                                double minArea,
                                                double maxArea) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    if (input.fieldAssociation() != cvSelectionData::CELLS) {
        CVLog::Warning(
                "[cvSelectionFilter] Area filter only works with cell "
                "selection");
        return cvSelectionData();
    }

    QVector<qint64> filteredIds;
    QVector<qint64> inputIds = input.ids();

    for (qint64 id : inputIds) {
        if (id < 0 || id >= polyData->GetNumberOfCells()) {
            continue;
        }

        double area = computeCellArea(polyData, id);
        if (area >= minArea && area <= maxArea) {
            filteredIds.append(id);
        }
    }

    CVLog::Print(QString("[cvSelectionFilter] Area filter: %1 -> %2 cells")
                         .arg(inputIds.size())
                         .arg(filteredIds.size()));

    return cvSelectionData(filteredIds, cvSelectionData::CELLS);
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::filterByNormalAngle(
        vtkPolyData* polyData,
        const cvSelectionData& input,
        double refX,
        double refY,
        double refZ,
        double minAngleDeg,
        double maxAngleDeg) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    if (input.fieldAssociation() != cvSelectionData::CELLS) {
        CVLog::Warning(
                "[cvSelectionFilter] Normal angle filter only works with cell "
                "selection");
        return cvSelectionData();
    }

    vtkDataArray* normals = polyData->GetCellData()->GetNormals();
    if (!normals) {
        CVLog::Warning("[cvSelectionFilter] No cell normals available");
        return cvSelectionData();
    }

    double refNormal[3] = {refX, refY, refZ};
    vtkMath::Normalize(refNormal);

    QVector<qint64> filteredIds;
    QVector<qint64> inputIds = input.ids();

    for (qint64 id : inputIds) {
        if (id < 0 || id >= polyData->GetNumberOfCells()) {
            continue;
        }

        double* normal = normals->GetTuple3(id);
        double angleDeg = computeAngleBetweenNormals(normal, refNormal);

        if (angleDeg >= minAngleDeg && angleDeg <= maxAngleDeg) {
            filteredIds.append(id);
        }
    }

    CVLog::Print(
            QString("[cvSelectionFilter] Normal angle filter: %1 -> %2 cells")
                    .arg(inputIds.size())
                    .arg(filteredIds.size()));

    return cvSelectionData(filteredIds, cvSelectionData::CELLS);
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::filterByBoundingBox(
        vtkPolyData* polyData,
        const cvSelectionData& input,
        const double bounds[6]) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    QVector<qint64> filteredIds;
    QVector<qint64> inputIds = input.ids();

    if (input.fieldAssociation() == cvSelectionData::POINTS) {
        // Filter points
        for (qint64 id : inputIds) {
            if (id < 0 || id >= polyData->GetNumberOfPoints()) {
                continue;
            }

            double point[3];
            polyData->GetPoint(id, point);

            if (isPointInBounds(point, bounds)) {
                filteredIds.append(id);
            }
        }
    } else {
        // Filter cells - check if cell center is in bounds
        for (qint64 id : inputIds) {
            if (id < 0 || id >= polyData->GetNumberOfCells()) {
                continue;
            }

            vtkCell* cell = polyData->GetCell(id);
            if (!cell) continue;

            // Compute cell center
            double center[3] = {0, 0, 0};
            vtkIdType npts = cell->GetNumberOfPoints();
            for (vtkIdType i = 0; i < npts; ++i) {
                double pt[3];
                polyData->GetPoint(cell->GetPointId(i), pt);
                center[0] += pt[0];
                center[1] += pt[1];
                center[2] += pt[2];
            }
            if (npts > 0) {
                center[0] /= npts;
                center[1] /= npts;
                center[2] /= npts;
            }

            if (isPointInBounds(center, bounds)) {
                filteredIds.append(id);
            }
        }
    }

    CVLog::Print(
            QString("[cvSelectionFilter] Bounding box filter: %1 -> %2 items")
                    .arg(inputIds.size())
                    .arg(filteredIds.size()));

    return cvSelectionData(filteredIds, input.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::filterByDistanceFromPoint(
        vtkPolyData* polyData,
        const cvSelectionData& input,
        double x,
        double y,
        double z,
        double minDistance,
        double maxDistance) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    double refPoint[3] = {x, y, z};
    QVector<qint64> filteredIds;
    QVector<qint64> inputIds = input.ids();

    if (input.fieldAssociation() == cvSelectionData::POINTS) {
        for (qint64 id : inputIds) {
            if (id < 0 || id >= polyData->GetNumberOfPoints()) {
                continue;
            }

            double point[3];
            polyData->GetPoint(id, point);

            double dist = computeDistance(point, refPoint);
            if (dist >= minDistance && dist <= maxDistance) {
                filteredIds.append(id);
            }
        }
    } else {
        for (qint64 id : inputIds) {
            if (id < 0 || id >= polyData->GetNumberOfCells()) {
                continue;
            }

            vtkCell* cell = polyData->GetCell(id);
            if (!cell) continue;

            // Compute cell center
            double center[3] = {0, 0, 0};
            vtkIdType npts = cell->GetNumberOfPoints();
            for (vtkIdType i = 0; i < npts; ++i) {
                double pt[3];
                polyData->GetPoint(cell->GetPointId(i), pt);
                center[0] += pt[0];
                center[1] += pt[1];
                center[2] += pt[2];
            }
            if (npts > 0) {
                center[0] /= npts;
                center[1] /= npts;
                center[2] /= npts;
            }

            double dist = computeDistance(center, refPoint);
            if (dist >= minDistance && dist <= maxDistance) {
                filteredIds.append(id);
            }
        }
    }

    CVLog::Print(QString("[cvSelectionFilter] Distance filter: %1 -> %2 items")
                         .arg(inputIds.size())
                         .arg(filteredIds.size()));

    return cvSelectionData(filteredIds, input.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::filterByNeighborCount(
        vtkPolyData* polyData,
        const cvSelectionData& input,
        int minNeighbors,
        int maxNeighbors) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    if (input.fieldAssociation() != cvSelectionData::CELLS) {
        CVLog::Warning(
                "[cvSelectionFilter] Neighbor count filter only works with "
                "cell selection");
        return cvSelectionData();
    }

    QVector<qint64> filteredIds;
    QVector<qint64> inputIds = input.ids();

    for (qint64 id : inputIds) {
        if (id < 0 || id >= polyData->GetNumberOfCells()) {
            continue;
        }

        int neighborCount = countCellNeighbors(polyData, id);
        if (neighborCount >= minNeighbors && neighborCount <= maxNeighbors) {
            filteredIds.append(id);
        }
    }

    CVLog::Print(
            QString("[cvSelectionFilter] Neighbor count filter: %1 -> %2 cells")
                    .arg(inputIds.size())
                    .arg(filteredIds.size()));

    return cvSelectionData(filteredIds, cvSelectionData::CELLS);
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::combineAND(const cvSelectionData& a,
                                              const cvSelectionData& b) {
    if (a.isEmpty() || b.isEmpty()) {
        return cvSelectionData();
    }

    if (a.fieldAssociation() != b.fieldAssociation()) {
        CVLog::Warning(
                "[cvSelectionFilter] Cannot combine selections with different "
                "field associations");
        return cvSelectionData();
    }

    // Intersection of IDs
    QSet<qint64> setA = QSet<qint64>(a.ids().begin(), a.ids().end());
    QSet<qint64> setB = QSet<qint64>(b.ids().begin(), b.ids().end());
    QSet<qint64> result = setA.intersect(setB);

    QVector<qint64> resultIds = QVector<qint64>(result.begin(), result.end());

    CVLog::Print(QString("[cvSelectionFilter] AND: %1 & %2 = %3")
                         .arg(a.count())
                         .arg(b.count())
                         .arg(resultIds.size()));

    return cvSelectionData(resultIds, a.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::combineOR(const cvSelectionData& a,
                                             const cvSelectionData& b) {
    if (a.isEmpty()) return b;
    if (b.isEmpty()) return a;

    if (a.fieldAssociation() != b.fieldAssociation()) {
        CVLog::Warning(
                "[cvSelectionFilter] Cannot combine selections with different "
                "field associations");
        return cvSelectionData();
    }

    // Union of IDs
    QSet<qint64> setA = QSet<qint64>(a.ids().begin(), a.ids().end());
    QSet<qint64> setB = QSet<qint64>(b.ids().begin(), b.ids().end());
    QSet<qint64> result = setA.unite(setB);

    QVector<qint64> resultIds = QVector<qint64>(result.begin(), result.end());

    CVLog::Print(QString("[cvSelectionFilter] OR: %1 U %2 = %3")
                         .arg(a.count())
                         .arg(b.count())
                         .arg(resultIds.size()));

    return cvSelectionData(resultIds, a.fieldAssociation());
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionFilter::invert(vtkPolyData* polyData,
                                          const cvSelectionData& input) {
    if (!polyData || input.isEmpty()) {
        return cvSelectionData();
    }

    QSet<qint64> selectedSet =
            QSet<qint64>(input.ids().begin(), input.ids().end());
    QVector<qint64> invertedIds;

    vtkIdType totalCount = (input.fieldAssociation() == cvSelectionData::POINTS)
                                   ? polyData->GetNumberOfPoints()
                                   : polyData->GetNumberOfCells();

    for (vtkIdType i = 0; i < totalCount; ++i) {
        if (!selectedSet.contains(i)) {
            invertedIds.append(i);
        }
    }

    CVLog::Print(QString("[cvSelectionFilter] Invert: %1 -> %2")
                         .arg(input.count())
                         .arg(invertedIds.size()));

    return cvSelectionData(invertedIds, input.fieldAssociation());
}

//-----------------------------------------------------------------------------
QStringList cvSelectionFilter::getAttributeNames(vtkPolyData* polyData,
                                                 bool pointData) {
    QStringList names;

    if (!polyData) {
        return names;
    }

    vtkFieldData* data = pointData ? (vtkFieldData*)polyData->GetPointData()
                                   : (vtkFieldData*)polyData->GetCellData();

    if (!data) {
        return names;
    }

    int numArrays = data->GetNumberOfArrays();
    for (int i = 0; i < numArrays; ++i) {
        vtkDataArray* array = data->GetArray(i);
        if (array && array->GetName()) {
            names.append(QString::fromUtf8(array->GetName()));
        }
    }

    return names;
}

//-----------------------------------------------------------------------------
// Private helper methods
//-----------------------------------------------------------------------------

double cvSelectionFilter::computeCellArea(vtkPolyData* polyData,
                                          vtkIdType cellId) {
    vtkCell* cell = polyData->GetCell(cellId);
    if (!cell) return 0.0;

    // Only compute area for triangles
    if (cell->GetCellType() == VTK_TRIANGLE) {
        double p0[3], p1[3], p2[3];
        polyData->GetPoint(cell->GetPointId(0), p0);
        polyData->GetPoint(cell->GetPointId(1), p1);
        polyData->GetPoint(cell->GetPointId(2), p2);

        return vtkTriangle::TriangleArea(p0, p1, p2);
    }

    // For other cell types, approximate with polygon area
    vtkIdType npts = cell->GetNumberOfPoints();
    if (npts < 3) return 0.0;

    double area = 0.0;
    double p0[3], p1[3], p2[3];
    polyData->GetPoint(cell->GetPointId(0), p0);

    for (vtkIdType i = 1; i < npts - 1; ++i) {
        polyData->GetPoint(cell->GetPointId(i), p1);
        polyData->GetPoint(cell->GetPointId(i + 1), p2);
        area += vtkTriangle::TriangleArea(p0, p1, p2);
    }

    return area;
}

//-----------------------------------------------------------------------------
double cvSelectionFilter::computeAngleBetweenNormals(const double n1[3],
                                                     const double n2[3]) {
    double dot = vtkMath::Dot(n1, n2);
    dot = qBound(-1.0, dot, 1.0);  // Clamp to [-1, 1]
    double angleRad = std::acos(dot);
    return vtkMath::DegreesFromRadians(angleRad);
}

//-----------------------------------------------------------------------------
bool cvSelectionFilter::isPointInBounds(const double point[3],
                                        const double bounds[6]) {
    return (point[0] >= bounds[0] && point[0] <= bounds[1] &&
            point[1] >= bounds[2] && point[1] <= bounds[3] &&
            point[2] >= bounds[4] && point[2] <= bounds[5]);
}

//-----------------------------------------------------------------------------
double cvSelectionFilter::computeDistance(const double p1[3],
                                          const double p2[3]) {
    double dx = p1[0] - p2[0];
    double dy = p1[1] - p2[1];
    double dz = p1[2] - p2[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

//-----------------------------------------------------------------------------
int cvSelectionFilter::countCellNeighbors(vtkPolyData* polyData,
                                          vtkIdType cellId) {
    vtkCell* cell = polyData->GetCell(cellId);
    if (!cell) return 0;

    QSet<vtkIdType> neighbors;

    // For each edge of the cell, find cells sharing that edge
    vtkIdType npts = cell->GetNumberOfPoints();
    for (vtkIdType i = 0; i < npts; ++i) {
        vtkIdType p1 = cell->GetPointId(i);
        vtkIdType p2 = cell->GetPointId((i + 1) % npts);

        vtkSmartPointer<vtkIdList> cellIds = vtkSmartPointer<vtkIdList>::New();
        polyData->GetCellEdgeNeighbors(cellId, p1, p2, cellIds);

        for (vtkIdType j = 0; j < cellIds->GetNumberOfIds(); ++j) {
            neighbors.insert(cellIds->GetId(j));
        }
    }

    return neighbors.size();
}

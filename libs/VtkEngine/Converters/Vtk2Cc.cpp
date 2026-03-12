// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file Vtk2Cc.cpp
 * @brief Implementation of VTK to CloudViewer data conversion.
 */

#include <Converters/Vtk2Cc.h>

#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif

// CV_CORE_LIB
#include <CVGeom.h>
#include <CVLog.h>

// CV_DB_LIB
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvScalarField.h>

// VTK
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkFieldData.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>

// Support for VTK 7.1 upwards
#ifdef vtkGenericDataArray_h
#define SetTupleValue SetTypedTuple
#define InsertNextTupleValue InsertNextTypedTuple
#define GetTupleValue GetTypedTuple
#endif

namespace Converters {

ccPointCloud* Vtk2Cc::ConvertToPointCloud(vtkPolyData* polydata, bool silent) {
    if (!polydata) {
        if (!silent) {
            CVLog::Warning("[Vtk2Cc::ConvertToPointCloud] polydata is nullptr");
        }
        return nullptr;
    }

    vtkIdType pointCount = polydata->GetNumberOfPoints();
    if (pointCount == 0) {
        if (!silent) {
            CVLog::Warning(
                    "[Vtk2Cc::ConvertToPointCloud] polydata has 0 points");
        }
        return nullptr;
    }

    vtkPointData* pointData = polydata->GetPointData();
    vtkFieldData* fieldData = polydata->GetFieldData();

    vtkUnsignedCharArray* colors = nullptr;
    bool hasSourceRGBFlag = false;

    if (fieldData) {
        vtkIntArray* hasRGBArray =
                vtkIntArray::SafeDownCast(fieldData->GetArray("HasSourceRGB"));
        if (hasRGBArray && hasRGBArray->GetValue(0) == 1) {
            hasSourceRGBFlag = true;
        }
    }

    if (pointData) {
        if (hasSourceRGBFlag) {
            const char* colorArrayNames[] = {"RGB", "Colors", "rgba", "rgb"};
            for (const char* name : colorArrayNames) {
                vtkDataArray* arr = pointData->GetArray(name);
                if (arr) {
                    if (arr->GetNumberOfComponents() == 3 ||
                        arr->GetNumberOfComponents() == 4) {
                        colors = vtkUnsignedCharArray::SafeDownCast(arr);
                        if (colors) break;
                    }
                }
            }
        }
    }

    vtkFloatArray* normals = nullptr;
    if (pointData) {
        normals = vtkFloatArray::SafeDownCast(pointData->GetNormals());
        if (!normals) {
            vtkDataArray* arr = pointData->GetArray("Normals");
            if (arr && arr->GetNumberOfComponents() == 3) {
                normals = vtkFloatArray::SafeDownCast(arr);
            }
        }
    }

    ccPointCloud* cloud = new ccPointCloud("vertices");

    if (!cloud->resize(static_cast<unsigned>(pointCount))) {
        if (!silent) {
            CVLog::Warning("[Vtk2Cc::ConvertToPointCloud] not enough memory!");
        }
        delete cloud;
        return nullptr;
    }

    if (normals && !cloud->reserveTheNormsTable()) {
        if (!silent) {
            CVLog::Warning("[Vtk2Cc] not enough memory for normals!");
        }
        delete cloud;
        return nullptr;
    }

    if (colors && !cloud->reserveTheRGBTable()) {
        if (!silent) {
            CVLog::Warning("[Vtk2Cc] not enough memory for colors!");
        }
        delete cloud;
        return nullptr;
    }

    for (vtkIdType i = 0; i < pointCount; ++i) {
        double coordinate[3];
        polydata->GetPoint(i, coordinate);
        cloud->setPoint(static_cast<std::size_t>(i),
                        CCVector3::fromArray(coordinate));
        if (normals) {
            float normal[3];
            normals->GetTupleValue(i, normal);
            CCVector3 N(static_cast<PointCoordinateType>(normal[0]),
                        static_cast<PointCoordinateType>(normal[1]),
                        static_cast<PointCoordinateType>(normal[2]));
            cloud->addNorm(N);
        }
        if (colors) {
            unsigned char color[3];
            colors->GetTupleValue(i, color);
            ecvColor::Rgb C(static_cast<ColorCompType>(color[0]),
                            static_cast<ColorCompType>(color[1]),
                            static_cast<ColorCompType>(color[2]));
            cloud->addRGBColor(C);
        }
    }

    if (normals) {
        cloud->showNormals(true);
    }
    if (colors) {
        cloud->showColors(true);
    }

    if (pointData && pointCount > 0) {
        int numArrays = pointData->GetNumberOfArrays();
        for (int i = 0; i < numArrays; ++i) {
            vtkDataArray* dataArray = pointData->GetArray(i);
            if (!dataArray || dataArray->GetNumberOfComponents() != 1) {
                continue;
            }
            if (dataArray->GetNumberOfTuples() < pointCount) {
                continue;
            }
            if (dataArray == reinterpret_cast<vtkDataArray*>(colors) ||
                dataArray == reinterpret_cast<vtkDataArray*>(normals)) {
                continue;
            }

            const char* arrayName = dataArray->GetName();
            if (!arrayName || strlen(arrayName) == 0) {
                continue;
            }

            ccScalarField* scalarField = new ccScalarField(arrayName);
            if (!scalarField->resizeSafe(static_cast<unsigned>(pointCount))) {
                scalarField->release();
                continue;
            }

            for (vtkIdType j = 0; j < pointCount; ++j) {
                double value = dataArray->GetTuple1(j);
                scalarField->setValue(static_cast<unsigned>(j),
                                      static_cast<ScalarType>(value));
            }

            scalarField->computeMinAndMax();
            int sfIdx = cloud->addScalarField(scalarField);
            if (sfIdx < 0) {
                scalarField->release();
            } else {
                QString fieldName = QString(arrayName).toLower();
                if (fieldName.contains("label") ||
                    fieldName.contains("class") ||
                    fieldName.contains("segment") ||
                    fieldName.contains("cluster")) {
                    cloud->setCurrentDisplayedScalarField(sfIdx);
                    cloud->showSF(true);
                }
            }
        }
    }

    return cloud;
}

ccMesh* Vtk2Cc::ConvertToMesh(vtkPolyData* polydata, bool silent) {
    if (!polydata) {
        return nullptr;
    }

    vtkSmartPointer<vtkPoints> mesh_points = polydata->GetPoints();
    if (!mesh_points) {
        if (!silent) {
            CVLog::Warning("[Vtk2Cc::ConvertToMesh] polydata has no points!");
        }
        return nullptr;
    }

    unsigned nr_points =
            static_cast<unsigned>(mesh_points->GetNumberOfPoints());
    unsigned nr_polygons = static_cast<unsigned>(polydata->GetNumberOfPolys());
    if (nr_points == 0) {
        if (!silent) {
            CVLog::Warning("[Vtk2Cc::ConvertToMesh] cannot find points data!");
        }
        return nullptr;
    }

    ccPointCloud* vertices = ConvertToPointCloud(polydata, silent);
    if (!vertices) {
        return nullptr;
    }
    vertices->setEnabled(false);
    vertices->setLocked(false);

    ccMesh* mesh = new ccMesh(vertices);
    mesh->setName("Mesh");
    mesh->addChild(vertices);

    if (!mesh->reserve(nr_polygons)) {
        if (!silent) {
            CVLog::Warning("[Vtk2Cc::ConvertToMesh] not enough memory!");
        }
        delete mesh;
        return nullptr;
    }

#ifdef VTK_CELL_ARRAY_V2
    const vtkIdType* cell_points;
#else
    vtkIdType* cell_points;
#endif
    vtkIdType nr_cell_points;
    vtkCellArray* mesh_polygons = polydata->GetPolys();
    mesh_polygons->InitTraversal();
    unsigned int validTriangles = 0;
    unsigned int skippedCells = 0;

    while (mesh_polygons->GetNextCell(nr_cell_points, cell_points)) {
        if (nr_cell_points != 3) {
            ++skippedCells;
            continue;
        }
        mesh->addTriangle(static_cast<unsigned>(cell_points[0]),
                          static_cast<unsigned>(cell_points[1]),
                          static_cast<unsigned>(cell_points[2]));
        ++validTriangles;
    }

    if (validTriangles == 0) {
        if (!silent) {
            CVLog::Warning(
                    "[Vtk2Cc::ConvertToMesh] No triangles found in polydata");
        }
        delete mesh;
        return nullptr;
    }

    if (skippedCells > 0 && !silent) {
        CVLog::Warning(
                QString("[Vtk2Cc::ConvertToMesh] Skipped %1 non-triangle "
                        "cell(s), added %2 triangle(s)")
                        .arg(skippedCells)
                        .arg(validTriangles));
    }

    vertices->shrinkToFit();
    mesh->shrinkToFit();
    NormsIndexesTableType* norms = mesh->getTriNormsTable();
    if (norms) {
        norms->shrink_to_fit();
    }

    return mesh;
}

ccPolyline* Vtk2Cc::ConvertToPolyline(vtkPolyData* polydata, bool silent) {
    if (!polydata) return nullptr;

    ccPointCloud* obj = ConvertToPointCloud(polydata, silent);
    if (!obj) {
        CVLog::Error(
                "[Vtk2Cc::ConvertToPolyline] failed to convert vtkPolyData "
                "to ccPointCloud");
        return nullptr;
    }

    if (obj->size() == 0) {
        CVLog::Warning(
                "[Vtk2Cc::ConvertToPolyline] polyline vertices is empty!");
        delete obj;
        return nullptr;
    }

    return ConvertToPolyline(obj);
}

ccPolyline* Vtk2Cc::ConvertToPolyline(ccPointCloud* vertices) {
    if (!vertices || !vertices->isKindOf(CV_TYPES::POINT_CLOUD)) {
        return nullptr;
    }

    ccPointCloud* polyVertices = ccHObjectCaster::ToPointCloud(vertices);
    if (!polyVertices) {
        return nullptr;
    }

    ccPolyline* curvePoly = new ccPolyline(polyVertices);
    if (!curvePoly) {
        return nullptr;
    }

    unsigned verticesCount = polyVertices->size();
    if (curvePoly->reserve(verticesCount)) {
        curvePoly->addPointIndex(0, verticesCount);
        curvePoly->setVisible(true);

        bool closed = false;
        CCVector3 start = CCVector3::fromArray(polyVertices->getPoint(0)->u);
        CCVector3 end = CCVector3::fromArray(
                polyVertices->getPoint(verticesCount - 1)->u);
        if (cloudViewer::LessThanEpsilon((end - start).norm())) {
            closed = true;
        }

        curvePoly->setClosed(closed);
        curvePoly->setName("polyline");
        curvePoly->addChild(polyVertices);
        curvePoly->showColors(true);
        curvePoly->setTempColor(ecvColor::green);
        curvePoly->set2DMode(false);
    } else {
        delete curvePoly;
        curvePoly = nullptr;
    }

    return curvePoly;
}

ccHObject::Container Vtk2Cc::ConvertToMultiPolylines(
        vtkPolyData* polydata, QString baseName, const ecvColor::Rgb& color) {
    ccHObject::Container container;

    vtkIdType iCells = polydata->GetNumberOfCells();
    for (vtkIdType i = 0; i < iCells; i++) {
        ccPointCloud* vertices = nullptr;
        vtkCell* cell = polydata->GetCell(i);
        vtkIdType ptsCount = cell->GetNumberOfPoints();
        if (ptsCount > 1) {
            vertices = new ccPointCloud("vertices");
            if (!vertices->reserve(static_cast<unsigned>(ptsCount))) {
                CVLog::Error("not enough memory to allocate vertices...");
                return container;
            }

            for (vtkIdType iPt = 0; iPt < ptsCount; ++iPt) {
                CCVector3 P =
                        CCVector3::fromArray(cell->GetPoints()->GetPoint(iPt));
                vertices->addPoint(P);
            }
        }

        if (vertices && vertices->size() == 0) {
            delete vertices;
            vertices = nullptr;
        }

        if (vertices) {
            vertices->setName("vertices");
            vertices->setEnabled(false);
            vertices->setPointSize(4);
            vertices->showColors(true);
            vertices->setTempColor(ecvColor::red);
            if (vertices->hasNormals()) vertices->showNormals(true);
            if (vertices->hasScalarFields()) {
                vertices->setCurrentDisplayedScalarField(0);
                vertices->showSF(true);
            }

            ccPolyline* poly = ConvertToPolyline(vertices);
            if (!poly) {
                delete vertices;
                continue;
            }

            QString contourName = baseName;
            if (poly->size() > 1) {
                contourName += QString(" (part %1)").arg(i + 1);
            }
            poly->setName(contourName);
            poly->showColors(true);
            poly->setTempColor(color);
            poly->set2DMode(false);

            container.push_back(poly);
        }
    }
    return container;
}

}  // namespace Converters

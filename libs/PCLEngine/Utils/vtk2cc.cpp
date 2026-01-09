// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Utils/vtk2cc.h>

#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif

// Local
#include <Utils/PCLCloud.h>
#include <Utils/PCLConv.h>
#include <Utils/cc2sm.h>
#include <Utils/my_point_types.h>
#include <Utils/sm2cc.h>

// PCL
#include <pcl/common/io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>

#include <pcl/io/impl/vtk_lib_io.hpp>

// CV_CORE_LIB
#include <CVGeom.h>

// ECV_DB_LIB
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvScalarField.h>

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkFloatArray.h>
#include <vtkPolyData.h>

// Support for VTK 7.1 upwards
#ifdef vtkGenericDataArray_h
#define SetTupleValue SetTypedTuple
#define InsertNextTupleValue InsertNextTypedTuple
#define GetTupleValue GetTypedTuple
#endif

ccPointCloud* vtk2cc::ConvertToPointCloud(vtkPolyData* polydata, bool silent) {
    if (!polydata) {
        if (!silent) {
            CVLog::Warning("[vtk2cc::ConvertToPointCloud] polydata is nullptr");
        }
        return nullptr;
    }

    vtkIdType pointCount = polydata->GetNumberOfPoints();

    // Check for empty polydata
    if (pointCount == 0) {
        if (!silent) {
            CVLog::Warning(
                    "[vtk2cc::ConvertToPointCloud] polydata has 0 points");
        }
        return nullptr;
    }

    vtkPointData* pointData = polydata->GetPointData();
    vtkFieldData* fieldData = polydata->GetFieldData();

    // Get colors - ONLY use colors if:
    // 1. "HasSourceRGB" flag is set in FieldData (indicating actual source RGB
    // data)
    // 2. OR we find a specifically named "RGB" or "Colors" array with unsigned
    // char data This prevents scalar field data from being mistakenly treated
    // as colors
    vtkUnsignedCharArray* colors = nullptr;
    bool hasSourceRGBFlag = false;

    // Check for HasSourceRGB flag first
    if (fieldData) {
        vtkIntArray* hasRGBArray =
                vtkIntArray::SafeDownCast(fieldData->GetArray("HasSourceRGB"));
        if (hasRGBArray && hasRGBArray->GetValue(0) == 1) {
            hasSourceRGBFlag = true;
        }
    }

    if (pointData) {
        // STRICT: Only use colors if HasSourceRGB flag is explicitly set
        // This prevents scalar fields (like curvature shown as color) from
        // being mistakenly treated as actual RGB color data
        if (hasSourceRGBFlag) {
            // Look for explicitly named RGB arrays WITH 3 or 4 components
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
        // Note: We intentionally do NOT fall back to searching by array name
        // without the HasSourceRGB flag. This prevents scalar fields
        // (curvature, intensity, etc.) from being confused with actual RGB
        // colors.
    }

    // Get normals - first try active normals, then fallback to named arrays
    vtkFloatArray* normals = nullptr;
    if (pointData) {
        // First try active normals
        normals = vtkFloatArray::SafeDownCast(pointData->GetNormals());

        // If no active normals, look for arrays named "Normals"
        if (!normals) {
            vtkDataArray* arr = pointData->GetArray("Normals");
            if (arr && arr->GetNumberOfComponents() == 3) {
                normals = vtkFloatArray::SafeDownCast(arr);
            }
        }
    }

    // create cloud
    ccPointCloud* cloud = new ccPointCloud("vertices");

    if (!cloud->resize(static_cast<unsigned>(pointCount))) {
        if (!silent) {
            CVLog::Warning(QString(
                    "[vtk2cc::ConvertToPointCloud] not enough memory!"));
        }
        delete cloud;
        cloud = nullptr;
        return nullptr;
    }

    if (normals && !cloud->reserveTheNormsTable()) {
        if (!silent) {
            CVLog::Warning(
                    QString("[getPointCloudFromPolyData] not enough memory!"));
        }
        delete cloud;
        cloud = nullptr;
        return nullptr;
    }

    if (colors && !cloud->reserveTheRGBTable()) {
        if (!silent) {
            CVLog::Warning(
                    QString("[getPointCloudFromPolyData] not enough memory!"));
        }
        delete cloud;
        cloud = nullptr;
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

    // Copy scalar fields (labels, intensity, etc.) from point data
    if (pointData && pointCount > 0) {
        int numArrays = pointData->GetNumberOfArrays();

        for (int i = 0; i < numArrays; ++i) {
            vtkDataArray* dataArray = pointData->GetArray(i);

            // Only handle single-component (scalar) arrays
            if (!dataArray || dataArray->GetNumberOfComponents() != 1) {
                continue;
            }

            // Check if the array has valid data
            if (dataArray->GetNumberOfTuples() < pointCount) {
                if (!silent) {
                    CVLog::Warning(
                            QString("[vtk2cc] Scalar array %1 has only %2 "
                                    "tuples but pointCount is %3")
                                    .arg(dataArray->GetName()
                                                 ? dataArray->GetName()
                                                 : "(unnamed)")
                                    .arg(dataArray->GetNumberOfTuples())
                                    .arg(pointCount));
                }
                continue;
            }

            // Skip arrays already handled as colors/normals
            if (dataArray == colors || dataArray == normals) {
                continue;
            }

            const char* arrayName = dataArray->GetName();
            if (!arrayName || strlen(arrayName) == 0) {
                continue;  // Skip unnamed arrays
            }

            // Create new scalar field
            ccScalarField* scalarField = new ccScalarField(arrayName);
            if (!scalarField->reserveSafe(static_cast<unsigned>(pointCount))) {
                if (!silent) {
                    CVLog::Warning(QString("[vtk2cc] Failed to allocate scalar "
                                           "field: %1")
                                           .arg(arrayName));
                }
                scalarField->release();
                continue;
            }

            // Resize to ensure space is allocated
            if (!scalarField->resizeSafe(static_cast<unsigned>(pointCount))) {
                if (!silent) {
                    CVLog::Warning(QString("[vtk2cc] Failed to resize scalar "
                                           "field: %1")
                                           .arg(arrayName));
                }
                scalarField->release();
                continue;
            }

            // Copy data
            for (vtkIdType j = 0; j < pointCount; ++j) {
                double value = dataArray->GetTuple1(j);
                scalarField->setValue(static_cast<unsigned>(j),
                                      static_cast<ScalarType>(value));
            }

            scalarField->computeMinAndMax();

            // Add to point cloud
            int sfIdx = cloud->addScalarField(scalarField);
            if (sfIdx < 0) {
                if (!silent) {
                    CVLog::Warning(
                            QString("[vtk2cc] Failed to add scalar field: %1")
                                    .arg(arrayName));
                }
                scalarField->release();
            } else {
                // Auto-detect and display label fields
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

ccMesh* vtk2cc::ConvertToMesh(vtkPolyData* polydata, bool silent) {
    if (!polydata) {
        return nullptr;
    }

    vtkSmartPointer<vtkPoints> mesh_points = polydata->GetPoints();
    if (!mesh_points) {
        if (!silent) {
            CVLog::Warning(
                    QString("[getMeshFromPolyData] polydata has no points!"));
        }
        return nullptr;
    }

    unsigned nr_points =
            static_cast<unsigned>(mesh_points->GetNumberOfPoints());
    unsigned nr_polygons = static_cast<unsigned>(polydata->GetNumberOfPolys());
    if (nr_points == 0) {
        if (!silent) {
            CVLog::Warning(
                    QString("[getMeshFromPolyData] cannot find points data!"));
        }
        return nullptr;
    }

    ccPointCloud* vertices = ConvertToPointCloud(polydata, silent);
    if (!vertices) {
        return nullptr;
    }
    vertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    vertices->setLocked(false);

    // mesh
    ccMesh* mesh = new ccMesh(vertices);
    mesh->setName("Mesh");
    mesh->addChild(vertices);

    if (!mesh->reserve(nr_polygons)) {
        if (!silent) {
            CVLog::Warning(QString("[getMeshFromPolyData] not enough memory!"));
        }
        delete mesh;
        return nullptr;
    }

#ifdef VTK_CELL_ARRAY_V2
    const vtkIdType* cell_points;
#else   // VTK_CELL_ARRAY_V2
    vtkIdType* cell_points;
#endif  // VTK_CELL_ARRAY_V2
    vtkIdType nr_cell_points;
    vtkCellArray* mesh_polygons = polydata->GetPolys();
    mesh_polygons->InitTraversal();
    unsigned int validTriangles = 0;
    unsigned int skippedCells = 0;

    while (mesh_polygons->GetNextCell(nr_cell_points, cell_points)) {
        if (nr_cell_points != 3) {
            // Skip non-triangle cells but continue processing
            ++skippedCells;
            continue;
        }

        mesh->addTriangle(static_cast<unsigned>(cell_points[0]),
                          static_cast<unsigned>(cell_points[1]),
                          static_cast<unsigned>(cell_points[2]));
        ++validTriangles;
    }

    // Check if we have any valid triangles
    if (validTriangles == 0) {
        if (!silent) {
            CVLog::Warning(QString(
                    "[getMeshFromPolyData] No triangles found in polydata"));
        }
        delete mesh;
        return nullptr;
    }

    // Log skipped cells if any
    if (skippedCells > 0 && !silent) {
        CVLog::Warning(QString("[getMeshFromPolyData] Skipped %1 non-triangle "
                               "cell(s), "
                               "added %2 triangle(s)")
                               .arg(skippedCells)
                               .arg(validTriangles));
    }

    // do some cleaning
    {
        vertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType* normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }

    return mesh;
}

ccPolyline* vtk2cc::ConvertToPolyline(vtkPolyData* polydata, bool silent) {
    if (!polydata) return nullptr;

    ccPointCloud* obj = ConvertToPointCloud(polydata, silent);
    if (!obj) {
        CVLog::Error(
                QString("[getPolylineFromPolyData] failed to convert "
                        "vtkPolyData to ccPointCloud"));
        return nullptr;
    }

    if (obj->size() == 0) {
        CVLog::Warning(QString(
                "[getPolylineFromPolyData] polyline vertices is empty!"));
        return nullptr;
    }

    return ConvertToPolyline(obj);
}

ccPolyline* vtk2cc::ConvertToPolyline(ccPointCloud* vertices) {
    if (!vertices || !vertices->isKindOf(CV_TYPES::POINT_CLOUD)) {
        return nullptr;
    }

    ccPointCloud* polyVertices = ccHObjectCaster::ToPointCloud(vertices);
    if (!polyVertices) {
        return nullptr;
    }

    ccPolyline* curvePoly = new ccPolyline(polyVertices);
    {
        if (!curvePoly) {
            return nullptr;
        }

        unsigned verticesCount = polyVertices->size();
        if (curvePoly->reserve(verticesCount)) {
            curvePoly->addPointIndex(0, verticesCount);
            curvePoly->setVisible(true);

            bool closed = false;
            CCVector3 start =
                    CCVector3::fromArray(polyVertices->getPoint(0)->u);
            CCVector3 end = CCVector3::fromArray(
                    polyVertices->getPoint(verticesCount - 1)->u);
            if (cloudViewer::LessThanEpsilon((end - start).norm())) {
                closed = true;
            } else {
                closed = false;
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
    }

    return curvePoly;
}

ccHObject::Container vtk2cc::ConvertToMultiPolylines(
        vtkPolyData* polydata, QString baseName, const ecvColor::Rgb& color) {
    // initialize output
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
            // end POINTS
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
                vertices = nullptr;
                continue;
            }

            // update global scale and shift by m_entity
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

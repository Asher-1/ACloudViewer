// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionExporter.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <ReferenceCloud.h>

// ECV_DB_LIB
#include <ecvGenericMesh.h>
#include <ecvMaterialSet.h>
#include <ecvMesh.h>
#include <ecvObject.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>
#include <ecvSerializableObject.h>

// ECV_IO_LIB - Use existing I/O infrastructure
#include <AutoIO.h>
#include <FileIO.h>
#include <FileIOFilter.h>

// PCL Utils - Use enhanced vtk2cc for conversion
#include <Utils/vtk2cc.h>

// VTK
#include <vtkCell.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkExtractSelection.h>
#include <vtkFieldData.h>
#include <vtkFloatArray.h>
#include <vtkGeometryFilter.h>
#include <vtkIdTypeArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>
#include <vtkSmartPointer.h>
#include <vtkTriangle.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>

// Qt
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QMap>
#include <QSettings>
#include <QWidget>

//-----------------------------------------------------------------------------
ccMesh* cvSelectionExporter::exportToMesh(vtkPolyData* polyData,
                                          const cvSelectionData& selectionData,
                                          const ExportOptions& options) {
    if (!polyData) {
        CVLog::Error("[cvSelectionExporter] polyData is nullptr");
        return nullptr;
    }

    if (selectionData.isEmpty()) {
        CVLog::Error("[cvSelectionExporter] Selection is empty");
        return nullptr;
    }

    if (selectionData.fieldAssociation() != cvSelectionData::CELLS) {
        CVLog::Error(
                "[cvSelectionExporter] Selection must be CELLS for mesh "
                "export");
        return nullptr;
    }

    // Extract selected geometry
    vtkSmartPointer<vtkPolyData> extracted =
            extractSelection(polyData, selectionData);
    if (!extracted) {
        CVLog::Error("[cvSelectionExporter] Failed to extract selection");
        return nullptr;
    }

    // Convert to ccMesh
    QString meshName =
            options.name.isEmpty()
                    ? QString("Selection_%1_cells").arg(selectionData.count())
                    : options.name;
    ccMesh* mesh = vtkPolyDataToCCMesh(extracted, meshName);

    if (!mesh) {
        return nullptr;
    }

    CVLog::Print(QString("[cvSelectionExporter] Created mesh '%1' with %2 "
                         "triangles")
                         .arg(meshName)
                         .arg(mesh->size()));

    // Save to file if requested
    if (options.saveToFile && !options.filename.isEmpty()) {
        if (!saveObjectToFile(mesh, options.filename, options.writeAscii,
                              options.compressed)) {
            CVLog::Warning(QString("[cvSelectionExporter] Failed to save mesh "
                                   "to file: %1")
                                   .arg(options.filename));
        }
    }
    return mesh;
}

//-----------------------------------------------------------------------------
ccPointCloud* cvSelectionExporter::exportToPointCloud(
        vtkPolyData* polyData,
        const cvSelectionData& selectionData,
        const ExportOptions& options) {
    if (!polyData) {
        CVLog::Error("[cvSelectionExporter] polyData is nullptr");
        return nullptr;
    }

    if (selectionData.isEmpty()) {
        CVLog::Error("[cvSelectionExporter] Selection is empty");
        return nullptr;
    }

    if (selectionData.fieldAssociation() != cvSelectionData::POINTS) {
        CVLog::Error(
                "[cvSelectionExporter] Selection must be POINTS for point "
                "cloud export");
        return nullptr;
    }

    // Extract selected geometry
    vtkSmartPointer<vtkPolyData> extracted =
            extractSelection(polyData, selectionData);
    if (!extracted) {
        CVLog::Error("[cvSelectionExporter] Failed to extract selection");
        return nullptr;
    }

    // Convert to ccPointCloud
    QString cloudName =
            options.name.isEmpty()
                    ? QString("Selection_%1_points").arg(selectionData.count())
                    : options.name;
    ccPointCloud* cloud = vtkPolyDataToCCPointCloud(extracted, cloudName);

    // Smart pointer handles cleanup automatically

    if (!cloud) {
        return nullptr;
    }

    CVLog::Print(QString("[cvSelectionExporter] Created point cloud '%1' "
                         "with %2 points")
                         .arg(cloudName)
                         .arg(cloud->size()));

    // Save to file if requested
    if (options.saveToFile && !options.filename.isEmpty()) {
        if (!saveObjectToFile(cloud, options.filename, options.writeAscii,
                              options.compressed)) {
            CVLog::Warning(QString("[cvSelectionExporter] Failed to save point "
                                   "cloud to file: %1")
                                   .arg(options.filename));
        }
    }
    return cloud;
}

//-----------------------------------------------------------------------------
ccPointCloud* cvSelectionExporter::exportFromSourceCloud(
        ccPointCloud* sourceCloud,
        const cvSelectionData& selectionData,
        const ExportOptions& options) {
    if (!sourceCloud) {
        CVLog::Error("[cvSelectionExporter] sourceCloud is nullptr");
        return nullptr;
    }

    if (selectionData.isEmpty()) {
        CVLog::Error("[cvSelectionExporter] Selection is empty");
        return nullptr;
    }

    if (selectionData.fieldAssociation() != cvSelectionData::POINTS) {
        CVLog::Error(
                "[cvSelectionExporter] Selection must be POINTS for point "
                "cloud export");
        return nullptr;
    }

    // Get selected point IDs
    const QVector<vtkIdType>& selectedIds = selectionData.ids();
    if (selectedIds.isEmpty()) {
        CVLog::Error("[cvSelectionExporter] No point IDs in selection");
        return nullptr;
    }

    // Validate IDs and create ReferenceCloud
    unsigned int cloudSize = sourceCloud->size();
    cloudViewer::ReferenceCloud refCloud(sourceCloud);
    if (!refCloud.reserve(static_cast<unsigned>(selectedIds.size()))) {
        CVLog::Error(
                "[cvSelectionExporter] Failed to reserve memory for "
                "reference cloud");
        return nullptr;
    }

    int validCount = 0;
    int invalidCount = 0;
    for (vtkIdType id : selectedIds) {
        if (id >= 0 && static_cast<unsigned int>(id) < cloudSize) {
            refCloud.addPointIndex(static_cast<unsigned>(id));
            ++validCount;
        } else {
            ++invalidCount;
        }
    }

    if (invalidCount > 0) {
        CVLog::Warning(
                QString("[cvSelectionExporter] Filtered %1 invalid point IDs "
                        "(out of range [0, %2))")
                        .arg(invalidCount)
                        .arg(cloudSize));
    }

    if (validCount == 0) {
        CVLog::Error(
                "[cvSelectionExporter] No valid point IDs after filtering");
        return nullptr;
    }

    // Use partialClone to extract selected points with all attributes
    int warnings = 0;
    ccPointCloud* result = sourceCloud->partialClone(&refCloud, &warnings);

    if (!result) {
        CVLog::Error("[cvSelectionExporter] partialClone failed");
        return nullptr;
    }

    // Log any warnings from partialClone
    if (warnings & ccPointCloud::WRN_OUT_OF_MEM_FOR_COLORS) {
        CVLog::Warning(
                "[cvSelectionExporter] partialClone: out of memory for "
                "colors");
    }
    if (warnings & ccPointCloud::WRN_OUT_OF_MEM_FOR_NORMALS) {
        CVLog::Warning(
                "[cvSelectionExporter] partialClone: out of memory for "
                "normals");
    }
    if (warnings & ccPointCloud::WRN_OUT_OF_MEM_FOR_SFS) {
        CVLog::Warning(
                "[cvSelectionExporter] partialClone: out of memory for "
                "scalar fields");
    }
    if (warnings & ccPointCloud::WRN_OUT_OF_MEM_FOR_FWF) {
        CVLog::Warning(
                "[cvSelectionExporter] partialClone: out of memory for "
                "full waveform data");
    }

    // Set name
    QString cloudName = options.name.isEmpty()
                                ? QString("Selection_%1_points").arg(validCount)
                                : options.name;
    result->setName(cloudName);

    CVLog::PrintDebug(
            QString("[cvSelectionExporter] SUCCESS: Created point cloud "
                    "'%1' with %2 points, %3 scalar fields, "
                    "hasColors=%4, hasNormals=%5")
                    .arg(cloudName)
                    .arg(result->size())
                    .arg(result->getNumberOfScalarFields())
                    .arg(result->hasColors() ? "yes" : "no")
                    .arg(result->hasNormals() ? "yes" : "no"));

    // Save to file if requested
    if (options.saveToFile && !options.filename.isEmpty()) {
        if (!saveObjectToFile(result, options.filename, options.writeAscii,
                              options.compressed)) {
            CVLog::Warning(QString("[cvSelectionExporter] Failed to save point "
                                   "cloud to file: %1")
                                   .arg(options.filename));
        }
    }

    return result;
}

//-----------------------------------------------------------------------------
ccMesh* cvSelectionExporter::exportFromSourceMesh(
        ccMesh* sourceMesh,
        const cvSelectionData& selectionData,
        const ExportOptions& options) {
    CVLog::Print(
            "[cvSelectionExporter::exportFromSourceMesh] START - Direct "
            "extraction from source ccMesh using partialClone");

    if (!sourceMesh) {
        CVLog::Error("[cvSelectionExporter] sourceMesh is nullptr");
        return nullptr;
    }

    if (selectionData.isEmpty()) {
        CVLog::Error("[cvSelectionExporter] Selection is empty");
        return nullptr;
    }

    if (selectionData.fieldAssociation() != cvSelectionData::CELLS) {
        CVLog::Error(
                "[cvSelectionExporter] Selection must be CELLS for mesh "
                "export");
        return nullptr;
    }

    // Get selected triangle IDs
    const QVector<vtkIdType>& selectedIds = selectionData.ids();
    if (selectedIds.isEmpty()) {
        CVLog::Error("[cvSelectionExporter] No triangle IDs in selection");
        return nullptr;
    }

    // Convert QVector<vtkIdType> to std::vector<unsigned>
    unsigned int meshSize = sourceMesh->size();
    std::vector<unsigned> triangleIndices;
    triangleIndices.reserve(selectedIds.size());

    for (vtkIdType id : selectedIds) {
        if (id >= 0 && static_cast<unsigned int>(id) < meshSize) {
            triangleIndices.push_back(static_cast<unsigned>(id));
        }
    }

    if (triangleIndices.empty()) {
        CVLog::Error(
                "[cvSelectionExporter] No valid triangle IDs after filtering");
        return nullptr;
    }
    // Use ccMesh::partialClone to create the extracted mesh
    // This handles all the vertex mapping, normals, materials, and texture
    // coordinates
    int warnings = 0;
    ccMesh* result = sourceMesh->partialClone(triangleIndices, &warnings);
    // Report warnings if any
    if (warnings != 0) {
        CVLog::Warning(QString("[cvSelectionExporter] partialClone completed "
                               "with warnings: %1")
                               .arg(warnings));
    }

    if (!result) {
        CVLog::Error("[cvSelectionExporter] partialClone failed");
        return nullptr;
    }

    // Set the mesh name
    QString meshName =
            options.name.isEmpty()
                    ? QString("Selection_%1_triangles").arg(result->size())
                    : options.name;
    result->setName(meshName);

    // Copy display properties from source mesh
    result->showColors(sourceMesh->colorsShown());
    result->showNormals(sourceMesh->normalsShown());
    result->showSF(sourceMesh->sfShown());
    result->showMaterials(sourceMesh->materialsShown());

    CVLog::PrintDebug(
            QString("[cvSelectionExporter] SUCCESS: Created mesh '%1' "
                    "with %2 triangles, %3 vertices")
                    .arg(meshName)
                    .arg(result->size())
                    .arg(result->getAssociatedCloud()
                                 ? result->getAssociatedCloud()->size()
                                 : 0));

    // Save to file if requested
    if (options.saveToFile && !options.filename.isEmpty()) {
        if (!saveObjectToFile(result, options.filename, options.writeAscii,
                              options.compressed)) {
            CVLog::Warning(QString("[cvSelectionExporter] Failed to save mesh "
                                   "to file: %1")
                                   .arg(options.filename));
        }
    }

    return result;
}

//-----------------------------------------------------------------------------
bool cvSelectionExporter::exportToFile(vtkPolyData* polyData,
                                       const cvSelectionData& selectionData,
                                       const QString& filename,
                                       bool writeAscii,
                                       bool compressed) {
    if (!polyData || selectionData.isEmpty() || filename.isEmpty()) {
        CVLog::Error("[cvSelectionExporter] Invalid parameters");
        return false;
    }

    // Extract selection
    vtkSmartPointer<vtkPolyData> extracted =
            extractSelection(polyData, selectionData);
    if (!extracted) {
        CVLog::Error("[cvSelectionExporter] Failed to extract selection");
        return false;
    }

    // Convert to ccHObject
    QFileInfo fileInfo(filename);
    ccHObject* object = nullptr;

    if (selectionData.fieldAssociation() == cvSelectionData::CELLS) {
        object = vtkPolyDataToCCMesh(extracted, fileInfo.baseName());
    } else {
        object = vtkPolyDataToCCPointCloud(extracted, fileInfo.baseName());
    }

    // Smart pointer handles cleanup automatically

    if (!object) {
        CVLog::Error("[cvSelectionExporter] Failed to convert selection");
        return false;
    }

    // Save using eCV_io module
    bool success = saveObjectToFile(object, filename, writeAscii, compressed);
    delete object;

    if (success) {
        CVLog::Print(QString("[cvSelectionExporter] Exported selection to: %1")
                             .arg(filename));
    } else {
        CVLog::Error(QString("[cvSelectionExporter] Failed to export to: %1")
                             .arg(filename));
    }

    return success;
}

//-----------------------------------------------------------------------------
vtkPolyData* cvSelectionExporter::extractSelection(
        vtkPolyData* polyData, const cvSelectionData& selectionData) {
    CVLog::Print("[cvSelectionExporter::extractSelection] START");

    if (!polyData || selectionData.isEmpty()) {
        CVLog::Error("[cvSelectionExporter] extractSelection: Invalid input");
        return nullptr;
    }

    // Get and validate VTK array
    vtkSmartPointer<vtkIdTypeArray> vtkArray = selectionData.vtkArray();
    if (!vtkArray || vtkArray->GetNumberOfTuples() == 0) {
        CVLog::Error(
                "[cvSelectionExporter] extractSelection: Selection array is "
                "null or empty");
        return nullptr;
    }

    bool isPointSelection =
            (selectionData.fieldAssociation() == cvSelectionData::POINTS);

    // Validate selection IDs against polyData bounds
    vtkIdType maxValidId = isPointSelection ? polyData->GetNumberOfPoints()
                                            : polyData->GetNumberOfCells();

    // Filter out invalid IDs to prevent crashes
    vtkSmartPointer<vtkIdTypeArray> validArray =
            vtkSmartPointer<vtkIdTypeArray>::New();
    for (vtkIdType i = 0; i < vtkArray->GetNumberOfTuples(); ++i) {
        vtkIdType id = vtkArray->GetValue(i);
        if (id >= 0 && id < maxValidId) {
            validArray->InsertNextValue(id);
        }
    }

    if (validArray->GetNumberOfTuples() == 0) {
        CVLog::Error(
                QString("[cvSelectionExporter] extractSelection: No valid IDs "
                        "(all %1 IDs are out of range [0, %2))")
                        .arg(vtkArray->GetNumberOfTuples())
                        .arg(maxValidId));
        return nullptr;
    }

    if (validArray->GetNumberOfTuples() < vtkArray->GetNumberOfTuples()) {
        CVLog::Warning(
                QString("[cvSelectionExporter] extractSelection: Filtered "
                        "%1 invalid IDs (kept %2 of %3)")
                        .arg(vtkArray->GetNumberOfTuples() -
                             validArray->GetNumberOfTuples())
                        .arg(validArray->GetNumberOfTuples())
                        .arg(vtkArray->GetNumberOfTuples()));
    }

    vtkSmartPointer<vtkPolyData> result = vtkSmartPointer<vtkPolyData>::New();

    // For POINT selections, directly copy selected points to new polydata
    // vtkGeometryFilter doesn't handle point-only extractions well
    if (isPointSelection) {
        vtkIdType numSelectedPoints = validArray->GetNumberOfTuples();
        // Create new points array
        vtkSmartPointer<vtkPoints> newPoints =
                vtkSmartPointer<vtkPoints>::New();
        newPoints->SetNumberOfPoints(numSelectedPoints);

        // Create vertex cells for each point (so they can be rendered)
        vtkSmartPointer<vtkCellArray> vertices =
                vtkSmartPointer<vtkCellArray>::New();

        // Copy point data arrays
        vtkPointData* srcPointData = polyData->GetPointData();
        vtkPointData* dstPointData = result->GetPointData();

        // Log source point data info
        if (srcPointData) {
            CVLog::Print(
                    QString("[cvSelectionExporter] Source has %1 arrays, "
                            "normals=%2, scalars=%3, tcoords=%4")
                            .arg(srcPointData->GetNumberOfArrays())
                            .arg(srcPointData->GetNormals() ? "yes" : "no")
                            .arg(srcPointData->GetScalars() ? "yes" : "no")
                            .arg(srcPointData->GetTCoords() ? "yes" : "no"));

            // Log details of each array for debugging
            for (int a = 0; a < srcPointData->GetNumberOfArrays(); ++a) {
                vtkDataArray* arr = srcPointData->GetArray(a);
                if (arr) {
                    CVLog::PrintDebug(
                            QString("[cvSelectionExporter]   Array[%1]: "
                                    "name='%2', components=%3, tuples=%4, "
                                    "type=%5")
                                    .arg(a)
                                    .arg(arr->GetName() ? arr->GetName()
                                                        : "(unnamed)")
                                    .arg(arr->GetNumberOfComponents())
                                    .arg(arr->GetNumberOfTuples())
                                    .arg(arr->GetClassName()));
                }
            }
        }

        // Copy selected points and their data
        // CVLog::Print("[cvSelectionExporter] Copying points...");
        for (vtkIdType i = 0; i < numSelectedPoints; ++i) {
            vtkIdType srcId = validArray->GetValue(i);
            double pt[3];
            polyData->GetPoint(srcId, pt);
            newPoints->SetPoint(i, pt);

            // Add vertex cell
            vertices->InsertNextCell(1);
            vertices->InsertCellPoint(i);
        }

        // Set points first
        result->SetPoints(newPoints);
        result->SetVerts(vertices);

        // Now copy point data arrays AFTER setting points
        if (srcPointData && numSelectedPoints > 0) {
            // Copy each named array
            for (int a = 0; a < srcPointData->GetNumberOfArrays(); ++a) {
                vtkDataArray* srcArray = srcPointData->GetArray(a);
                if (!srcArray) continue;

                // Skip arrays without names (coordinate arrays)
                const char* arrName = srcArray->GetName();
                if (!arrName || strlen(arrName) == 0) {
                    CVLog::PrintDebug(QString("[cvSelectionExporter] Skipping "
                                              "unnamed array[%1]")
                                              .arg(a));
                    continue;
                }

                vtkSmartPointer<vtkDataArray> dstArray;
                dstArray.TakeReference(srcArray->NewInstance());
                dstArray->SetName(arrName);
                dstArray->SetNumberOfComponents(
                        srcArray->GetNumberOfComponents());
                dstArray->SetNumberOfTuples(numSelectedPoints);

                int numComp = srcArray->GetNumberOfComponents();

                // Debug: Check source array values for first few points
                if (numSelectedPoints > 0 && numComp == 1) {
                    vtkIdType firstSrcId = validArray->GetValue(0);
                    double firstVal = srcArray->GetTuple1(firstSrcId);
                    CVLog::PrintDebug(QString("[cvSelectionExporter] Array "
                                              "'%1': srcId[0]=%2 -> value=%3")
                                              .arg(arrName)
                                              .arg(firstSrcId)
                                              .arg(firstVal));
                }

                // Copy data for each selected point using appropriate method
                for (vtkIdType i = 0; i < numSelectedPoints; ++i) {
                    vtkIdType srcId = validArray->GetValue(i);

                    // Bounds check
                    if (srcId < 0 || srcId >= srcArray->GetNumberOfTuples()) {
                        CVLog::Warning(
                                QString("[cvSelectionExporter] Invalid srcId "
                                        "%1 for array '%2' (max=%3)")
                                        .arg(srcId)
                                        .arg(arrName)
                                        .arg(srcArray->GetNumberOfTuples()));
                        continue;
                    }

                    // Use GetTuple/SetTuple for all component types
                    double* tuple = srcArray->GetTuple(srcId);
                    dstArray->SetTuple(i, tuple);
                }

                // Debug: Verify copied values
                if (numSelectedPoints > 0 && numComp == 1) {
                    double copiedVal = dstArray->GetTuple1(0);
                    CVLog::PrintDebug(QString("[cvSelectionExporter] Array "
                                              "'%1': copied[0] = %2")
                                              .arg(arrName)
                                              .arg(copiedVal));
                }

                dstPointData->AddArray(dstArray);
            }

            // Set active arrays (normals, scalars, tcoords) from copied arrays
            // The arrays were already copied in the loop above - we just need
            // to set them as "active" in the destination point data

            // Set active normals if source has them
            vtkDataArray* srcNormals = srcPointData->GetNormals();
            if (srcNormals && srcNormals->GetName()) {
                vtkDataArray* dstNormals =
                        dstPointData->GetArray(srcNormals->GetName());
                if (dstNormals) {
                    dstPointData->SetNormals(dstNormals);
                }
            }

            // Set active scalars (colors) if source has them
            vtkDataArray* srcScalars = srcPointData->GetScalars();
            if (srcScalars && srcScalars->GetName()) {
                vtkDataArray* dstScalars =
                        dstPointData->GetArray(srcScalars->GetName());
                if (dstScalars) {
                    dstPointData->SetScalars(dstScalars);
                }
            }

            // Set active TCoords if source has them
            vtkDataArray* srcTCoords = srcPointData->GetTCoords();
            if (srcTCoords && srcTCoords->GetName()) {
                vtkDataArray* dstTCoords =
                        dstPointData->GetArray(srcTCoords->GetName());
                if (dstTCoords) {
                    dstPointData->SetTCoords(dstTCoords);
                }
            }
        }
    } else {
        // For CELL selections, use vtkExtractSelection + vtkGeometryFilter
        vtkSmartPointer<vtkSelectionNode> selectionNode =
                vtkSmartPointer<vtkSelectionNode>::New();
        selectionNode->SetContentType(vtkSelectionNode::INDICES);
        selectionNode->SetFieldType(vtkSelectionNode::CELL);
        selectionNode->SetSelectionList(validArray);

        vtkSmartPointer<vtkSelection> selection =
                vtkSmartPointer<vtkSelection>::New();
        selection->AddNode(selectionNode);

        vtkSmartPointer<vtkExtractSelection> extractor =
                vtkSmartPointer<vtkExtractSelection>::New();
        extractor->SetInputData(0, polyData);
        extractor->SetInputData(1, selection);
        extractor->Update();

        vtkUnstructuredGrid* extracted =
                vtkUnstructuredGrid::SafeDownCast(extractor->GetOutput());

        if (!extracted || extracted->GetNumberOfPoints() == 0) {
            CVLog::Error("[cvSelectionExporter] Cell extraction failed");
            return nullptr;
        }

        CVLog::PrintDebug(QString("[cvSelectionExporter] Extracted %1 points, "
                                  "%2 cells (cell selection)")
                                  .arg(extracted->GetNumberOfPoints())
                                  .arg(extracted->GetNumberOfCells()));

        // Convert to polydata
        vtkSmartPointer<vtkGeometryFilter> geometryFilter =
                vtkSmartPointer<vtkGeometryFilter>::New();
        geometryFilter->SetInputData(extracted);
        geometryFilter->Update();

        vtkPolyData* filteredOutput = geometryFilter->GetOutput();
        if (!filteredOutput || filteredOutput->GetNumberOfPoints() == 0) {
            CVLog::Warning(
                    "[cvSelectionExporter] Geometry filter produced no output");
            return nullptr;
        }

        result->DeepCopy(filteredOutput);
    }

    if (result->GetNumberOfPoints() == 0) {
        CVLog::Warning("[cvSelectionExporter] Extraction produced 0 points");
        return nullptr;
    }

    // Copy field data (metadata like DatasetName) from source to result
    vtkFieldData* srcFieldData = polyData->GetFieldData();
    if (srcFieldData && srcFieldData->GetNumberOfArrays() > 0) {
        vtkFieldData* dstFieldData = result->GetFieldData();
        for (int i = 0; i < srcFieldData->GetNumberOfArrays(); ++i) {
            vtkAbstractArray* arr = srcFieldData->GetAbstractArray(i);
            if (arr) {
                dstFieldData->AddArray(arr);
            }
        }
        CVLog::PrintDebug(
                QString("[cvSelectionExporter] Copied %1 field data arrays")
                        .arg(srcFieldData->GetNumberOfArrays()));
    }

    // Return raw pointer (caller must manage)
    result->Register(nullptr);
    return result.Get();
}

//-----------------------------------------------------------------------------
ccMesh* cvSelectionExporter::vtkPolyDataToCCMesh(vtkPolyData* polyData,
                                                 const QString& name) {
    if (!polyData) {
        return nullptr;
    }

    // Use enhanced vtk2cc with full ScalarField support
    ccMesh* mesh = vtk2cc::ConvertToMesh(polyData, false);
    if (mesh) {
        mesh->setName(name);
    }
    return mesh;
}

//-----------------------------------------------------------------------------
ccPointCloud* cvSelectionExporter::vtkPolyDataToCCPointCloud(
        vtkPolyData* polyData, const QString& name) {
    if (!polyData) {
        return nullptr;
    }

    // Use enhanced vtk2cc with full ScalarField support
    ccPointCloud* cloud = vtk2cc::ConvertToPointCloud(polyData, false);
    if (cloud) {
        cloud->setName(name);
    }
    return cloud;
}

//-----------------------------------------------------------------------------
bool cvSelectionExporter::saveObjectToFile(ccHObject* object,
                                           const QString& filename,
                                           bool writeAscii,
                                           bool compressed) {
    if (!object || filename.isEmpty()) {
        CVLog::Error(
                "[cvSelectionExporter] Invalid parameters for file export");
        return false;
    }

    // Use eCV_io module for saving
    // This supports all formats: BIN, OBJ, PLY, STL, PCD, etc.
    std::string filenameStr = filename.toStdString();

    try {
        // Try using AutoIO module first (supports most formats)
        bool success = cloudViewer::io::WriteEntity(filenameStr, *object,
                                                    writeAscii, compressed,
                                                    false  // print_progress
        );

        if (success) {
            CVLog::Print(
                    QString("[cvSelectionExporter] Successfully saved to: %1")
                            .arg(filename));
            return true;
        }

        // Fallback to FileIOFilter for formats not supported by AutoIO
        FileIOFilter::SaveParameters params;
        params.alwaysDisplaySaveDialog = false;
        params.parentWidget = nullptr;

        QFileInfo fileInfo(filename);
        QString ext = fileInfo.suffix().toLower();

        // Try to find appropriate filter
        FileIOFilter::Shared filter =
                FileIOFilter::FindBestFilterForExtension(ext);

        if (!filter) {
            CVLog::Error(QString("[cvSelectionExporter] No filter found for "
                                 "extension: %1")
                                 .arg(ext));
            return false;
        }

        CC_FILE_ERROR result =
                FileIOFilter::SaveToFile(object, filename, params, filter);

        if (result != CC_FERR_NO_ERROR) {
            FileIOFilter::DisplayErrorMessage(result, "saving", filename);
            return false;
        }

        CVLog::Print(QString("[cvSelectionExporter] Successfully saved to: %1")
                             .arg(filename));
        return true;

    } catch (const std::exception& e) {
        CVLog::Error(QString("[cvSelectionExporter] Exception while saving: %1")
                             .arg(e.what()));
        return false;
    }
}

//-----------------------------------------------------------------------------
bool cvSelectionExporter::saveObjectToFileWithDialog(ccHObject* object,
                                                     bool isMesh,
                                                     QWidget* parent) {
    if (!object) {
        CVLog::Error("[cvSelectionExporter] Object is nullptr");
        return false;
    }

    // Load last used path from settings
    QSettings settings;
    settings.beginGroup("SelectionExport");
    QString currentPath =
            settings.value("LastPath", QDir::homePath()).toString();
    QString defaultName = object->getName();

    // Load last used filter for this type
    QString settingsKey = isMesh ? "LastFilterMesh" : "LastFilterCloud";
    QString lastFilter = settings.value(settingsKey).toString();

    // Build file filters based on object type
    QStringList fileFilters;
    QString selectedFilter = lastFilter;

    for (const FileIOFilter::Shared& filter : FileIOFilter::GetFilters()) {
        bool canSave = false;

        if (isMesh) {
            bool isExclusive = true;
            bool multiple = false;
            canSave = filter->canSave(CV_TYPES::MESH, multiple, isExclusive);
        } else {
            bool isExclusive = true;
            bool multiple = false;
            canSave = filter->canSave(CV_TYPES::POINT_CLOUD, multiple,
                                      isExclusive);
        }

        if (canSave) {
            // getFileFilters returns a QStringList
            QStringList filterStrs = filter->getFileFilters(false);
            fileFilters << filterStrs;

            // If no last filter, use the first one
            if (selectedFilter.isEmpty() && !filterStrs.isEmpty()) {
                selectedFilter = filterStrs.first();
            }
        }
    }

    if (fileFilters.isEmpty()) {
        CVLog::Error("[cvSelectionExporter] No suitable file filter found");
        settings.endGroup();
        return false;
    }

    // Show file save dialog
    QString fullPath = currentPath + "/" + defaultName;
    QString selectedFilename = QFileDialog::getSaveFileName(
            parent, QObject::tr("Export Selection to File"), fullPath,
            fileFilters.join(";;"), &selectedFilter);

    if (selectedFilename.isEmpty()) {
        settings.endGroup();
        return false;  // User cancelled
    }

    // Save using FileIOFilter
    FileIOFilter::SaveParameters parameters;
    parameters.alwaysDisplaySaveDialog = true;
    parameters.parentWidget = parent;

    CC_FILE_ERROR result = FileIOFilter::SaveToFile(object, selectedFilename,
                                                    parameters, selectedFilter);

    if (result != CC_FERR_NO_ERROR) {
        CVLog::Error(QString("[cvSelectionExporter] Failed to save file: %1")
                             .arg(selectedFilename));
        settings.endGroup();
        return false;
    }

    // Save settings for next time
    currentPath = QFileInfo(selectedFilename).absolutePath();
    settings.setValue("LastPath", currentPath);
    settings.setValue(settingsKey, selectedFilter);
    settings.endGroup();

    CVLog::Print(QString("[cvSelectionExporter] Selection exported to: %1")
                         .arg(selectedFilename));

    return true;
}

//=============================================================================
// Batch Export Implementation (merged from cvSelectionExporterBatch.cpp)
//=============================================================================

//-----------------------------------------------------------------------------
QList<ccMesh*> cvSelectionExporter::batchExportToMeshes(
        vtkPolyData* polyData,
        const QList<cvSelectionData>& selections,
        const QString& baseName) {
    QList<ccMesh*> meshes;

    if (!polyData || selections.isEmpty()) {
        CVLog::Error(
                "[cvSelectionExporter] Invalid parameters for batch export");
        return meshes;
    }

    int index = 1;
    for (const cvSelectionData& selection : selections) {
        if (selection.isEmpty()) {
            CVLog::Warning(
                    QString("[cvSelectionExporter] Skipping empty selection %1")
                            .arg(index));
            ++index;
            continue;
        }

        if (selection.fieldAssociation() != cvSelectionData::CELLS) {
            CVLog::Warning(QString("[cvSelectionExporter] Skipping non-cell "
                                   "selection %1")
                                   .arg(index));
            ++index;
            continue;
        }

        QString name =
                QString("%1_%2").arg(baseName).arg(index, 3, 10, QChar('0'));
        cvSelectionExporter::ExportOptions opts;
        opts.name = name;
        ccMesh* mesh = exportToMesh(polyData, selection, opts);

        if (mesh) {
            meshes.append(mesh);
            CVLog::Print(QString("[cvSelectionExporter] Batch exported mesh "
                                 "%1/%2: %3")
                                 .arg(index)
                                 .arg(selections.size())
                                 .arg(name));
        } else {
            CVLog::Error(
                    QString("[cvSelectionExporter] Failed to export mesh %1")
                            .arg(index));
        }

        ++index;
    }

    CVLog::Print(
            QString("[cvSelectionExporter] Batch export complete: %1/%2 meshes")
                    .arg(meshes.size())
                    .arg(selections.size()));

    return meshes;
}

//-----------------------------------------------------------------------------
QList<ccPointCloud*> cvSelectionExporter::batchExportToPointClouds(
        vtkPolyData* polyData,
        const QList<cvSelectionData>& selections,
        const QString& baseName) {
    QList<ccPointCloud*> clouds;

    if (!polyData || selections.isEmpty()) {
        CVLog::Error(
                "[cvSelectionExporter] Invalid parameters for batch export");
        return clouds;
    }

    int index = 1;
    for (const cvSelectionData& selection : selections) {
        if (selection.isEmpty()) {
            CVLog::Warning(
                    QString("[cvSelectionExporter] Skipping empty selection %1")
                            .arg(index));
            ++index;
            continue;
        }

        if (selection.fieldAssociation() != cvSelectionData::POINTS) {
            CVLog::Warning(QString("[cvSelectionExporter] Skipping non-point "
                                   "selection %1")
                                   .arg(index));
            ++index;
            continue;
        }

        QString name =
                QString("%1_%2").arg(baseName).arg(index, 3, 10, QChar('0'));
        cvSelectionExporter::ExportOptions opts;
        opts.name = name;
        ccPointCloud* cloud = exportToPointCloud(polyData, selection, opts);

        if (cloud) {
            clouds.append(cloud);
            CVLog::Print(QString("[cvSelectionExporter] Batch exported cloud "
                                 "%1/%2: %3")
                                 .arg(index)
                                 .arg(selections.size())
                                 .arg(name));
        } else {
            CVLog::Error(
                    QString("[cvSelectionExporter] Failed to export cloud %1")
                            .arg(index));
        }

        ++index;
    }

    CVLog::Print(
            QString("[cvSelectionExporter] Batch export complete: %1/%2 clouds")
                    .arg(clouds.size())
                    .arg(selections.size()));

    return clouds;
}

//-----------------------------------------------------------------------------
int cvSelectionExporter::batchExportToFiles(
        vtkPolyData* polyData,
        const QList<cvSelectionData>& selections,
        const QString& outputDir,
        const QString& format,
        const QString& baseName,
        std::function<void(int)> progressCallback) {
    if (!polyData || selections.isEmpty() || outputDir.isEmpty()) {
        CVLog::Error(
                "[cvSelectionExporter] Invalid parameters for batch export to "
                "files");
        return 0;
    }

    // Create output directory if it doesn't exist
    QDir dir;
    if (!dir.exists(outputDir)) {
        if (!dir.mkpath(outputDir)) {
            CVLog::Error(QString("[cvSelectionExporter] Failed to create "
                                 "output directory: %1")
                                 .arg(outputDir));
            return 0;
        }
    }

    int successCount = 0;
    int totalCount = selections.size();

    for (int i = 0; i < totalCount; ++i) {
        const cvSelectionData& selection = selections[i];
        int index = i + 1;

        if (selection.isEmpty()) {
            CVLog::Warning(
                    QString("[cvSelectionExporter] Skipping empty selection %1")
                            .arg(index));
            if (progressCallback) {
                progressCallback((index * 100) / totalCount);
            }
            continue;
        }

        // Build filename
        QString filename = QString("%1/%2_%3.%4")
                                   .arg(outputDir)
                                   .arg(baseName)
                                   .arg(index, 3, 10, QChar('0'))
                                   .arg(format.toLower());

        // Export directly to file
        bool success = cvSelectionExporter::exportToFile(
                polyData, selection, filename, false, false);

        if (success) {
            ++successCount;
            CVLog::Print(QString("[cvSelectionExporter] Exported %1/%2: %3")
                                 .arg(index)
                                 .arg(totalCount)
                                 .arg(filename));
        } else {
            CVLog::Error(QString("[cvSelectionExporter] Failed to export %1/%2")
                                 .arg(index)
                                 .arg(totalCount));
        }

        // Progress callback
        if (progressCallback) {
            progressCallback((index * 100) / totalCount);
        }
    }

    CVLog::Print(QString("[cvSelectionExporter] Batch export complete: %1/%2 "
                         "files exported to %3")
                         .arg(successCount)
                         .arg(totalCount)
                         .arg(outputDir));

    return successCount;
}

//-----------------------------------------------------------------------------
bool cvSelectionExporter::exportNumbered(vtkPolyData* polyData,
                                         const cvSelectionData& selection,
                                         const QString& outputPath,
                                         int number) {
    if (!polyData || selection.isEmpty() || outputPath.isEmpty()) {
        return false;
    }

    // Replace %1 with number
    QString filename = outputPath.arg(number, 3, 10, QChar('0'));

    // Determine format from extension
    QFileInfo fileInfo(filename);
    QString format = fileInfo.suffix().toLower();

    // Export directly to file
    return cvSelectionExporter::exportToFile(polyData, selection, filename,
                                             false, false);
}

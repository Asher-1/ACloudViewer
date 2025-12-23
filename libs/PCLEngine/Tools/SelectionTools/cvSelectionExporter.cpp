// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionExporter.h"

// CV_CORE_LIB
#include <CVLog.h>

// ECV_DB_LIB
#include <ecvGenericMesh.h>
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
#include <vtkIdTypeArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSelection.h>
#include <vtkSelectionNode.h>
#include <vtkSmartPointer.h>
#include <vtkTriangle.h>
#include <vtkUnstructuredGrid.h>
#include <vtkGeometryFilter.h>

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

    // Smart pointer handles cleanup automatically

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

    // Note: If addToScene is true, the caller must handle adding to scene
    // using MainWindow::addToDB() or similar

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

    // Note: If addToScene is true, the caller must handle adding to scene
    // using MainWindow::addToDB() or similar

    return cloud;
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
    if (!polyData || selectionData.isEmpty()) {
        return nullptr;
    }

    // Create selection node
    vtkSmartPointer<vtkSelectionNode> selectionNode =
            vtkSmartPointer<vtkSelectionNode>::New();
    selectionNode->SetContentType(vtkSelectionNode::INDICES);

    if (selectionData.fieldAssociation() == cvSelectionData::CELLS) {
        selectionNode->SetFieldType(vtkSelectionNode::CELL);
    } else {
        selectionNode->SetFieldType(vtkSelectionNode::POINT);
    }

    selectionNode->SetSelectionList(selectionData.vtkArray());

    // Create selection
    vtkSmartPointer<vtkSelection> selection =
            vtkSmartPointer<vtkSelection>::New();
    selection->AddNode(selectionNode);

    // Extract selection
    vtkSmartPointer<vtkExtractSelection> extractor =
            vtkSmartPointer<vtkExtractSelection>::New();
    extractor->SetInputData(0, polyData);
    extractor->SetInputData(1, selection);
    extractor->Update();

    vtkUnstructuredGrid* extracted =
            vtkUnstructuredGrid::SafeDownCast(extractor->GetOutput());

    if (!extracted) {
        CVLog::Error("[cvSelectionExporter] Extraction failed");
        return nullptr;
    }

    // Validate extracted data
    vtkIdType numPoints = extracted->GetNumberOfPoints();
    vtkIdType numCells = extracted->GetNumberOfCells();
    
    if (numPoints == 0) {
        CVLog::Warning("[cvSelectionExporter] Extraction produced 0 points");
        return nullptr;
    }
    
    CVLog::PrintDebug(QString("[cvSelectionExporter] Extracted %1 points, %2 cells")
                              .arg(numPoints)
                              .arg(numCells));

    // Convert vtkUnstructuredGrid to vtkPolyData using vtkGeometryFilter
    // Note: ShallowCopy from vtkUnstructuredGrid to vtkPolyData doesn't work
    // correctly for all cases, especially for point-only selections
    vtkSmartPointer<vtkGeometryFilter> geometryFilter =
            vtkSmartPointer<vtkGeometryFilter>::New();
    geometryFilter->SetInputData(extracted);
    geometryFilter->Update();

    vtkPolyData* filteredOutput = geometryFilter->GetOutput();
    if (!filteredOutput || filteredOutput->GetNumberOfPoints() == 0) {
        CVLog::Warning("[cvSelectionExporter] Geometry filter produced no output");
        return nullptr;
    }

    // Create result and deep copy to ensure proper memory management
    vtkSmartPointer<vtkPolyData> result = vtkSmartPointer<vtkPolyData>::New();
    result->DeepCopy(filteredOutput);

    // Return raw pointer (caller must manage) - for backward compatibility
    // Note: Caller is responsible for calling Delete() on the returned pointer
    result->Register(nullptr);  // Increment ref count for caller
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

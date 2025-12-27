// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file cvSelectionExporterBatch.cpp
 * @brief Batch export functionality for cvSelectionExporter
 *
 * This file contains the batch export implementation, separated for clarity.
 */

#include "cvSelectionExporter.h"

// CV_CORE_LIB
#include <CVLog.h>

// Qt
#include <QDir>
#include <QFileInfo>

//=============================================================================
// Batch Export Implementation
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

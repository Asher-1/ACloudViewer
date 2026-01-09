// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"

// LOCAL
#include "cvSelectionData.h"

// Qt
#include <QList>
#include <QString>

// STL
#include <functional>

// Forward declarations
class ccHObject;
class ccMesh;
class ccPointCloud;
class vtkPolyData;
class QWidget;

/**
 * @brief Helper class for exporting selections to CloudViewer objects or files
 *
 * Provides functionality to convert VTK selections to:
 * - ccMesh (from selected cells)
 * - ccPointCloud (from selected points)
 * - Various file formats (OBJ, PLY, STL, BIN, etc.) - uses eCV_io module
 * - Direct import to scene/rendering window
 *
 * Based on ParaView's selection extraction filters:
 * - vtkExtractSelection
 * - pqExtractSelectionsReaction
 */
class QPCL_ENGINE_LIB_API cvSelectionExporter {
public:
    /**
     * @brief Options for exporting selections
     */
    struct ExportOptions {
        QString name;           //< Name for the created object
        bool addToScene;        //< Whether to add to scene directly
        bool saveToFile;        //< Whether to save to file
        QString filename;       //< Output filename (if saveToFile=true)
        bool writeAscii;        //< Use ASCII format (if supported)
        bool compressed;        //< Use compression (if supported)
        QWidget* parentWidget;  //< Parent widget for dialogs

        // Constructor with default values
        ExportOptions()
            : addToScene(false),
              saveToFile(false),
              writeAscii(false),
              compressed(false),
              parentWidget(nullptr) {}
    };

    /**
     * @brief Export selected cells to ccMesh
     * @param polyData Source mesh data
     * @param selectionData Selection data (must be CELLS)
     * @param options Export options
     * @return Pointer to created ccMesh, or nullptr on failure
     * @note Caller takes ownership of returned object if not added to scene
     */
    static ccMesh* exportToMesh(vtkPolyData* polyData,
                                const cvSelectionData& selectionData,
                                const ExportOptions& options = ExportOptions());

    /**
     * @brief Export selected points to ccPointCloud
     * @param polyData Source mesh data
     * @param selectionData Selection data (must be POINTS)
     * @param options Export options
     * @return Pointer to created ccPointCloud, or nullptr on failure
     * @note Caller takes ownership of returned object if not added to scene
     */
    static ccPointCloud* exportToPointCloud(
            vtkPolyData* polyData,
            const cvSelectionData& selectionData,
            const ExportOptions& options = ExportOptions());

    //=========================================================================
    // Direct Extraction from ccPointCloud/ccMesh (bypasses VTK conversion)
    //=========================================================================

    /**
     * @brief Export selected points directly from source ccPointCloud
     *
     * This method bypasses VTK→ccPointCloud conversion by using the source
     * ccPointCloud's partialClone method directly with selected indices.
     * This preserves all scalar fields, normals, RGB colors, etc.
     *
     * @param sourceCloud Source point cloud (original data)
     * @param selectionData Selection data (must be POINTS)
     * @param options Export options
     * @return Pointer to created ccPointCloud, or nullptr on failure
     * @note Caller takes ownership of returned object if not added to scene
     */
    static ccPointCloud* exportFromSourceCloud(
            ccPointCloud* sourceCloud,
            const cvSelectionData& selectionData,
            const ExportOptions& options = ExportOptions());

    /**
     * @brief Export selected cells directly from source ccMesh
     *
     * This method bypasses VTK→ccMesh conversion by extracting triangles
     * directly from the source ccMesh using selected cell indices.
     *
     * @param sourceMesh Source mesh (original data)
     * @param selectionData Selection data (must be CELLS)
     * @param options Export options
     * @return Pointer to created ccMesh, or nullptr on failure
     * @note Caller takes ownership of returned object if not added to scene
     */
    static ccMesh* exportFromSourceMesh(
            ccMesh* sourceMesh,
            const cvSelectionData& selectionData,
            const ExportOptions& options = ExportOptions());

    /**
     * @brief Export selection to file (uses eCV_io module)
     * @param polyData Source mesh data
     * @param selectionData Selection data
     * @param filename Output filename (extension determines format)
     * @param writeAscii Use ASCII format if supported
     * @param compressed Use compression if supported
     * @return True on success
     */
    static bool exportToFile(vtkPolyData* polyData,
                             const cvSelectionData& selectionData,
                             const QString& filename,
                             bool writeAscii = false,
                             bool compressed = false);

    /**
     * @brief Extract selected geometry as new vtkPolyData
     * @param polyData Source mesh data
     * @param selectionData Selection data
     * @return Extracted vtkPolyData (caller must delete), or nullptr on failure
     */
    static vtkPolyData* extractSelection(vtkPolyData* polyData,
                                         const cvSelectionData& selectionData);

    //=========================================================================
    // Batch Export Functionality
    //=========================================================================

    /**
     * @brief Export multiple selections to meshes
     * @param polyData Source mesh data
     * @param selections List of selections (all must be CELLS)
     * @param baseName Base name for meshes (e.g., "Selection")
     * @return List of created meshes
     */
    static QList<ccMesh*> batchExportToMeshes(
            vtkPolyData* polyData,
            const QList<cvSelectionData>& selections,
            const QString& baseName = "Selection");

    /**
     * @brief Export multiple selections to point clouds
     * @param polyData Source mesh data
     * @param selections List of selections (all must be POINTS)
     * @param baseName Base name for clouds
     * @return List of created point clouds
     */
    static QList<ccPointCloud*> batchExportToPointClouds(
            vtkPolyData* polyData,
            const QList<cvSelectionData>& selections,
            const QString& baseName = "Selection");

    /**
     * @brief Export multiple selections to files
     * @param polyData Source mesh data
     * @param selections List of selections
     * @param outputDir Output directory
     * @param format File format ("obj", "ply", "stl", "bin")
     * @param baseName Base name for files
     * @param progressCallback Optional callback for progress updates (0-100)
     * @return Number of successfully exported files
     */
    static int batchExportToFiles(
            vtkPolyData* polyData,
            const QList<cvSelectionData>& selections,
            const QString& outputDir,
            const QString& format,
            const QString& baseName = "selection",
            std::function<void(int)> progressCallback = nullptr);

    /**
     * @brief Export selection with numbered naming
     * @param polyData Source mesh data
     * @param selection Selection data
     * @param outputPath Full output path with number placeholder (%1)
     * @param number File number
     * @return True on success
     */
    static bool exportNumbered(vtkPolyData* polyData,
                               const cvSelectionData& selection,
                               const QString& outputPath,
                               int number);

    /**
     * @brief Save a ccHObject to file with file dialog
     * Uses QFileDialog to let user choose filename and format
     * Remembers last used path and filter using QSettings
     *
     * @param object Object to save
     * @param isMesh True if object is a mesh, false if point cloud
     * @param parent Parent widget for dialog (can be nullptr)
     * @return True if save succeeded, false if cancelled or failed
     */
    static bool saveObjectToFileWithDialog(ccHObject* object,
                                           bool isMesh,
                                           QWidget* parent = nullptr);

private:
    /**
     * @brief Convert vtkPolyData to ccMesh
     * @note Now uses enhanced vtk2cc with ScalarField support
     */
    static ccMesh* vtkPolyDataToCCMesh(vtkPolyData* polyData,
                                       const QString& name);

    /**
     * @brief Convert vtkPolyData to ccPointCloud
     * @note Now uses enhanced vtk2cc with ScalarField support
     */
    static ccPointCloud* vtkPolyDataToCCPointCloud(vtkPolyData* polyData,
                                                   const QString& name);

    /**
     * @brief Save ccHObject to file using eCV_io module
     * @param object Object to save
     * @param filename Output filename
     * @param writeAscii Use ASCII format if supported
     * @param compressed Use compression if supported
     * @return True on success
     */
    static bool saveObjectToFile(ccHObject* object,
                                 const QString& filename,
                                 bool writeAscii = false,
                                 bool compressed = false);
};

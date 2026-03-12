// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file Vtk2Cc.h
 * @brief VTK to CloudViewer data structure converter.
 *
 * Provides utilities for converting VTK poly data representations
 * back into native CloudViewer entities (point clouds, meshes, polylines).
 * This is the reverse direction of Cc2Vtk.
 */

#include "qVTK.h"

// CV_DB_LIB
#include <ecvColorTypes.h>

class ccHObject;
class ccMesh;
class ccPointCloud;
class ccPolyline;
class vtkPolyData;

namespace Converters {

/**
 * @class Vtk2Cc
 * @brief Converts VTK poly data to CloudViewer (CV_db) entities.
 *
 * Static utility class that converts vtkPolyData objects into
 * CloudViewer-native types such as ccPointCloud, ccMesh, and ccPolyline.
 * Handles point coordinates, normals, RGB colors, and scalar fields
 * during conversion.
 *
 * @see Cc2Vtk for the reverse conversion direction.
 */
class QVTK_ENGINE_LIB_API Vtk2Cc {
public:
    /**
     * @brief Convert vtkPolyData to a ccPointCloud.
     *
     * Extracts points, normals, RGB colors, and scalar fields from
     * the VTK poly data. Scalar fields with names containing "label",
     * "class", "segment", or "cluster" are automatically set as the
     * displayed scalar field.
     *
     * @param polydata Source VTK poly data. Must contain at least one point.
     * @param silent   If true, suppresses warning messages on failure.
     * @return New ccPointCloud instance (caller owns), or nullptr on failure.
     */
    static ccPointCloud* ConvertToPointCloud(vtkPolyData* polydata,
                                             bool silent = false);

    /**
     * @brief Convert vtkPolyData to a ccMesh.
     *
     * Converts polygon cells (triangles only) from VTK poly data into
     * a CloudViewer mesh. Non-triangle cells are skipped with a warning.
     * The vertices are created via ConvertToPointCloud() and added as
     * a child of the returned mesh.
     *
     * @param polydata Source VTK poly data containing polygon cells.
     * @param silent   If true, suppresses warning messages on failure.
     * @return New ccMesh instance (caller owns), or nullptr on failure.
     */
    static ccMesh* ConvertToMesh(vtkPolyData* polydata, bool silent = false);

    /**
     * @brief Convert vtkPolyData to a ccPolyline.
     *
     * First converts the data to a point cloud, then constructs a polyline
     * from the resulting vertices. The polyline is automatically closed if
     * the first and last points coincide.
     *
     * @param polydata Source VTK poly data.
     * @param silent   If true, suppresses warning messages on failure.
     * @return New ccPolyline instance (caller owns), or nullptr on failure.
     */
    static ccPolyline* ConvertToPolyline(vtkPolyData* polydata,
                                         bool silent = false);

    /**
     * @brief Construct a ccPolyline from an existing ccPointCloud.
     *
     * Creates a polyline using the given vertices. Auto-detects closure
     * by comparing the first and last points.
     *
     * @param vertices Point cloud providing the polyline vertices.
     * @return New ccPolyline instance (caller owns), or nullptr on failure.
     */
    static ccPolyline* ConvertToPolyline(ccPointCloud* vertices);

    /**
     * @brief Extract multiple polylines from a multi-cell vtkPolyData.
     *
     * Each cell in the poly data is converted into a separate polyline.
     * Cells with fewer than 2 points are skipped.
     *
     * @param polydata Source VTK poly data with multiple cells.
     * @param baseName Base name prefix for generated polylines.
     * @param color    Display color applied to all generated polylines.
     * @return Vector of ccHObject pointers (caller owns each element).
     */
    static std::vector<ccHObject*> ConvertToMultiPolylines(
            vtkPolyData* polydata,
            QString baseName = "Slice",
            const ecvColor::Rgb& color = ecvColor::green);

private:
    Vtk2Cc() = delete;
};

}  // namespace Converters

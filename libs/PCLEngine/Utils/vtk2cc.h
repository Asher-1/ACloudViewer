// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "qPCL.h"

// CV_DB_LIB
#include <ecvColorTypes.h>

class ccHObject;
class ccMesh;
class ccPointCloud;
class ccPolyline;
class vtkPolyData;

/**
 * @brief VTK to CloudViewer converter
 * 
 * Provides static methods for converting VTK polydata to CloudViewer's
 * native data structures (ccPointCloud, ccMesh, ccPolyline).
 * 
 * Supports conversion from VTK polydata to:
 * - Point clouds
 * - Triangle meshes
 * - Single polylines
 * - Multiple polylines (e.g., from slicing operations)
 */
class QPCL_ENGINE_LIB_API vtk2cc {
public:
    /**
     * @brief Convert VTK polydata to CloudViewer point cloud
     * @param polydata Input VTK polydata
     * @param silent Suppress warning messages (default: false)
     * @return CloudViewer point cloud, or nullptr on failure
     * @static
     * 
     * Extracts points from VTK polydata and creates a ccPointCloud.
     * Preserves colors and other point data if available.
     */
    static ccPointCloud* ConvertToPointCloud(vtkPolyData* polydata,
                                             bool silent = false);
    
    /**
     * @brief Convert VTK polydata to CloudViewer mesh
     * @param polydata Input VTK polydata with polygon cells
     * @param silent Suppress warning messages (default: false)
     * @return CloudViewer mesh, or nullptr on failure
     * @static
     * 
     * Converts VTK polydata containing triangles to ccMesh.
     * Non-triangular polygons will be triangulated.
     */
    static ccMesh* ConvertToMesh(vtkPolyData* polydata, bool silent = false);
    
    /**
     * @brief Convert VTK polydata to CloudViewer polyline
     * @param polydata Input VTK polydata with line cells
     * @param silent Suppress warning messages (default: false)
     * @return CloudViewer polyline, or nullptr on failure
     * @static
     * 
     * Extracts line cells from VTK polydata and creates a single ccPolyline.
     */
    static ccPolyline* ConvertToPolyline(vtkPolyData* polydata,
                                         bool silent = false);

    /**
     * @brief Convert point cloud to polyline
     * @param vertices Point cloud containing polyline vertices
     * @return CloudViewer polyline connecting all points, or nullptr on failure
     * @static
     * 
     * Creates a polyline by connecting all points in order.
     * Useful for creating contours or paths.
     */
    static ccPolyline* ConvertToPolyline(ccPointCloud* vertices);
    
    /**
     * @brief Convert VTK polydata to multiple polylines
     * @param polydata Input VTK polydata with multiple line cells
     * @param baseName Base name for generated polylines (default: "Slice")
     * @param color Color to assign to polylines (default: green)
     * @return Vector of CloudViewer polylines (as ccHObjects)
     * @static
     * 
     * Extracts all line cells as separate polylines.
     * Useful for processing slicing results or contour extraction.
     * Each polyline is named with baseName + index.
     */
    static std::vector<ccHObject*> ConvertToMultiPolylines(
            vtkPolyData* polydata,
            QString baseName = "Slice",
            const ecvColor::Rgb& color = ecvColor::green);
};

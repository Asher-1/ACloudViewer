// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file Cc2Vtk.h
 * @brief CloudViewer (CV_db) to VTK data structure converter.
 *
 * Provides direct conversion from CloudViewer entities (point clouds,
 * meshes, polylines, line sets) to vtkPolyData for VTK rendering,
 * bypassing PCL data structures entirely.
 */

#include <vtkSmartPointer.h>

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "qVTK.h"

class vtkPolyData;
class vtkMatrix4x4;
class vtkDataArray;
class ccPointCloud;
class ccGenericMesh;
class ccPolyline;

namespace cloudViewer {
namespace geometry {
class LineSet;
}
}  // namespace cloudViewer

namespace Converters {

/// Direct CV_db to VTK converter. Bypasses PCL data structures entirely.
/// Each method converts a CV_db entity into a vtkPolyData ready for rendering.
class QVTK_ENGINE_LIB_API Cc2Vtk {
public:
    /// Convert a ccPointCloud to a vtkPolyData with vertex cells.
    /// The result has vtkPoints (float), optional RGB scalars,
    /// optional normals, and one vertex cell per point.
    /// @param cloud          Source point cloud.
    /// @param include_colors If true and colors exist, adds RGB scalars.
    /// @param include_normals If true and normals exist, adds normal array.
    /// @param include_sf     If true and a scalar field is displayed, colors
    ///                       from the active scalar field are used instead of
    ///                       RGB.
    /// @param show_mode      If true, respects visibility filtering.
    static vtkSmartPointer<vtkPolyData> PointCloudToPolyData(
            const ccPointCloud* cloud,
            bool include_colors = true,
            bool include_normals = true,
            bool include_sf = false,
            bool show_mode = true);

    /// Obtain color data as a vtkDataArray (3-component unsigned char RGB).
    /// Uses scalar-field-derived colors when @p sf_colors is true,
    /// otherwise uses the cloud's per-point RGB.
    static bool GetVtkScalars(const ccPointCloud* cloud,
                              vtkSmartPointer<vtkDataArray>& scalars,
                              bool sf_colors,
                              bool show_mode = true);

    /// Convert a ccGenericMesh (non-textured) to vtkPolyData.
    /// Uses per-triangle vertex layout (pointIndex = triIdx * 3 + vertIdx).
    static vtkSmartPointer<vtkPolyData> MeshToPolyData(
            const ccPointCloud* vertex_cloud, ccGenericMesh* mesh);

    /// Convert a ccGenericMesh with textures to vtkPolyData.
    /// Also outputs per-material texture coordinates and a transformation
    /// matrix.
    static bool TextureMeshToPolyData(
            const ccPointCloud* vertex_cloud,
            ccGenericMesh* mesh,
            vtkSmartPointer<vtkPolyData>& polydata,
            vtkSmartPointer<vtkMatrix4x4>& transformation,
            std::vector<std::vector<Eigen::Vector2f>>& tex_coordinates);

    /// Convert a ccPolyline to vtkPolyData with line cells.
    static vtkSmartPointer<vtkPolyData> PolylineToPolyData(
            const ccPolyline* polyline);

    /// Convert a LineSet to vtkPolyData with line cells and per-line colors.
    static vtkSmartPointer<vtkPolyData> LineSetToPolyData(
            const cloudViewer::geometry::LineSet* lineset);

    /// Add a single scalar field from ccPointCloud to an existing vtkPolyData.
    /// Used to sync scalar field data for selection/extraction.
    static void AddScalarFieldToPolyData(vtkPolyData* polydata,
                                         const ccPointCloud* cloud,
                                         int scalar_field_index);

    /// Simplified scalar field name (spaces replaced with underscores).
    static std::string GetSimplifiedSFName(const std::string& cc_sf_name);

private:
    Cc2Vtk() = delete;
};

}  // namespace Converters

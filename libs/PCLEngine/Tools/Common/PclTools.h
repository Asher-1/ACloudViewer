// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include <Utils/PCLCloud.h>
#include <Utils/PCLConv.h>

#include "qPCL.h"

// CV_CORE_LIB
#include <CVLog.h>

// CV_DB_LIB
#include <ecvDrawContext.h>

// PCL_SURFACING
#include <pcl/surface/texture_mapping.h>

// PCL COMMON
#include <pcl/ModelCoefficients.h>
#include <pcl/point_cloud.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

class vtkDataSet;
class vtkActor;
class vtkLODActor;
class vtkPoints;
class vtkPropAssembly;
class vtkAbstractWidget;
class vtkUnstructuredGrid;

namespace cloudViewer {
namespace geometry {
class LineSet;
}
namespace camera {
class PinholeCameraTrajectory;
}
}  // namespace cloudViewer

class ccMesh;
class ccGBLSensor;
class ccCameraSensor;

/**
 * @namespace PclTools
 * @brief Utility functions for PCL/VTK visualization and geometry manipulation
 *
 * Provides helper functions for:
 * - Texture mesh creation and mapping
 * - VTK actor creation from datasets
 * - Geometric primitive generation (lines, cubes, planes)
 * - Sensor visualization
 * - Coordinate system creation
 * - Point cloud and line set conversions
 */
namespace PclTools {

// =====================================================================
// Texture Mapping
// =====================================================================

/**
 * @brief Create textured mesh from OBJ file
 * @param filePath Path to OBJ file
 * @param show_cameras Whether to visualize camera positions (default: false)
 * @param verbose Enable verbose output (default: false)
 * @return PCL texture mesh with mapped textures
 *
 * Loads mesh and applies texture mapping using camera parameters.
 */
QPCL_ENGINE_LIB_API PCLTextureMesh::Ptr CreateTexturingMesh(
        const std::string& filePath,
        bool show_cameras = false,
        bool verbose = false);

/**
 * @brief Create textured mesh from OBJ file with camera trajectory
 * @param filePath Path to OBJ file
 * @param cameraTrajectory Camera trajectory for texture projection
 * @param show_cameras Whether to visualize camera positions (default: false)
 * @param verbose Enable verbose output (default: false)
 * @return PCL texture mesh with mapped textures
 */
QPCL_ENGINE_LIB_API PCLTextureMesh::Ptr CreateTexturingMesh(
        const std::string& filePath,
        const cloudViewer::camera::PinholeCameraTrajectory& cameraTrajectory,
        bool show_cameras = false,
        bool verbose = false);

/**
 * @brief Create textured mesh from triangle mesh and cameras
 * @param triangles Input triangle mesh
 * @param cameras Camera vector for texture projection
 * @param show_cameras Whether to visualize camera positions (default: false)
 * @param verbose Enable verbose output (default: false)
 * @return PCL texture mesh with mapped textures
 */
PCLTextureMesh::Ptr CreateTexturingMesh(
        const PCLMesh::ConstPtr& triangles,
        const pcl::texture_mapping::CameraVector& cameras,
        bool show_cameras = false,
        bool verbose = false);

// =====================================================================
// VTK Actor Creation
// =====================================================================

/**
 * @brief Get default scalar interpolation setting for VTK dataset
 * @param data VTK dataset to check
 * @return 0 if interpolation should be off, 1 if on
 *
 * Returns 0 (off) for polydata with only vertices, 1 (on) otherwise.
 * Used by CreateActorFromVTKDataSet methods.
 */
inline int GetDefaultScalarInterpolationForDataSet(vtkDataSet* data) {
    vtkPolyData* polyData = vtkPolyData::SafeDownCast(data);
    return (polyData &&
            polyData->GetNumberOfCells() != polyData->GetNumberOfVerts());
}

/**
 * @brief Create VTK actor from VTK dataset
 * @param data Input VTK dataset (polydata, unstructured grid, etc.)
 * @param actor Output VTK actor
 * @param use_scalars Use scalar data for coloring (default: true)
 * @param use_vbos Use vertex buffer objects (default: false)
 *
 * Creates a renderable actor from a VTK dataset with appropriate mapper
 * settings.
 */
void CreateActorFromVTKDataSet(const vtkSmartPointer<vtkDataSet>& data,
                               vtkSmartPointer<vtkActor>& actor,
                               bool use_scalars = true,
                               bool use_vbos = false);

/**
 * @brief Create VTK LOD actor from VTK dataset
 * @param data Input VTK dataset
 * @param actor Output VTK LOD actor
 * @param use_scalars Use scalar data for coloring (default: true)
 * @param use_vbos Use vertex buffer objects (default: false)
 *
 * Creates a level-of-detail actor for improved rendering performance.
 */
void CreateActorFromVTKDataSet(const vtkSmartPointer<vtkDataSet>& data,
                               vtkSmartPointer<vtkLODActor>& actor,
                               bool use_scalars = true,
                               bool use_vbos = false);

/**
 * @brief Allocate VTK unstructured grid
 * @param polydata Output unstructured grid smart pointer
 *
 * Helper function to allocate and initialize an unstructured grid.
 */
void AllocVtkUnstructuredGrid(vtkSmartPointer<vtkUnstructuredGrid>& polydata);

// =====================================================================
// Scalar Bar and Transformation
// =====================================================================

/**
 * @brief Update scalar bar widget
 * @param widget Scalar bar widget to update
 * @param CONTEXT Draw context with scalar field information
 * @return true on success
 */
bool UpdateScalarBar(vtkAbstractWidget* widget, const CC_DRAW_CONTEXT& CONTEXT);

/**
 * @brief Transform VTK polydata
 * @param polyData Polydata to transform
 * @param trans Transformation matrix
 * @return true on success
 */
bool TransformPolyData(vtkSmartPointer<vtkPolyData> polyData,
                       const ccGLMatrixd& trans);

/**
 * @brief Transform VTK points
 * @param points Points to transform
 * @param trans Transformation matrix
 * @return true on success
 */
bool TransformVtkPoints(vtkSmartPointer<vtkPoints> points,
                        const ccGLMatrixd& trans);

// =====================================================================
// Line Set Conversion
// =====================================================================

/**
 * @brief Extract VTK points from line set
 * @param lineset Input line set
 * @return VTK points
 */
vtkSmartPointer<vtkPoints> GetVtkPointsFromLineSet(
        const cloudViewer::geometry::LineSet& lineset);

/**
 * @brief Extract points, lines, and colors from line set
 * @param lineset Input line set
 * @param points Output VTK points
 * @param lines Output VTK line cells
 * @param colors Output VTK color array
 * @return true on success
 */
bool GetVtkPointsAndLinesFromLineSet(
        const cloudViewer::geometry::LineSet& lineset,
        vtkSmartPointer<vtkPoints> points,
        vtkSmartPointer<vtkCellArray> lines,
        vtkSmartPointer<vtkUnsignedCharArray> colors);

/**
 * @brief Create coordinate system from line set
 * @param lineset Input line set defining axes
 * @return VTK polydata representing coordinate system
 */
vtkSmartPointer<vtkPolyData> CreateCoordinateFromLineSet(
        const cloudViewer::geometry::LineSet& lineset);

/**
 * @brief Create VTK polydata from line set
 * @param lineset Input line set
 * @param useLineSource Use vtkLineSource for rendering (default: true)
 * @return VTK polydata with line geometry
 */
vtkSmartPointer<vtkPolyData> CreatePolyDataFromLineSet(
        const cloudViewer::geometry::LineSet& lineset,
        bool useLineSource = true);

// =====================================================================
// Geometric Primitive Creation
// =====================================================================

/**
 * @brief Set uniform color for polydata
 * @param polyData Polydata to color
 * @param color RGB color
 * @param is_cell Apply to cells instead of points (default: false)
 */
void SetPolyDataColor(vtkSmartPointer<vtkPolyData> polyData,
                      const ecvColor::Rgb& color,
                      bool is_cell = false);

/**
 * @brief Add vertex cells to polydata
 * @param polyData Polydata to modify
 */
void AddPolyDataCell(vtkSmartPointer<vtkPolyData> polyData);

/**
 * @brief Create line polydata from points
 * @param points Line vertices
 * @return VTK polydata with line cells
 */
vtkSmartPointer<vtkPolyData> CreateLine(vtkSmartPointer<vtkPoints> points);

/**
 * @brief Create colored line polydata
 * @param points Line vertices
 * @param lines Line connectivity
 * @param colors Per-vertex colors
 * @return VTK polydata with colored lines
 */
vtkSmartPointer<vtkPolyData> CreateLine(
        vtkSmartPointer<vtkPoints> points,
        vtkSmartPointer<vtkCellArray> lines,
        vtkSmartPointer<vtkUnsignedCharArray> colors);

/**
 * @brief Create transformed cube
 * @param width Cube width
 * @param height Cube height
 * @param depth Cube depth
 * @param trans Transformation matrix
 * @return VTK polydata representing cube
 */
vtkSmartPointer<vtkPolyData> CreateCube(double width,
                                        double height,
                                        double depth,
                                        const ccGLMatrixd& trans);

/**
 * @brief Create axis-aligned cube
 * @param width Cube width
 * @param height Cube height
 * @param depth Cube depth
 * @return VTK polydata representing cube
 */
vtkSmartPointer<vtkPolyData> CreateCube(double width,
                                        double height,
                                        double depth);

// =====================================================================
// Sensor Visualization
// =====================================================================

/**
 * @brief Create visualization for GBL sensor
 * @param gBLSensor Input GBL sensor
 * @return VTK polydata representing sensor
 */
vtkSmartPointer<vtkPolyData> CreateGBLSensor(const ccGBLSensor* gBLSensor);

/**
 * @brief Create visualization for camera sensor
 * @param cameraSensor Input camera sensor
 * @param lineColor Color for frustum lines
 * @param planeColor Color for frustum planes
 * @return VTK polydata representing camera frustum
 */
vtkSmartPointer<vtkPolyData> CreateCameraSensor(
        const ccCameraSensor* cameraSensor,
        const ecvColor::Rgb& lineColor,
        const ecvColor::Rgb& planeColor);

/**
 * @brief Create plane from model coefficients
 * @param coefficients Plane equation coefficients [a, b, c, d]
 * @param x Center X coordinate
 * @param y Center Y coordinate
 * @param z Center Z coordinate
 * @param scale Plane size scale factor (default: 1)
 * @return VTK polydata representing plane
 */
vtkSmartPointer<vtkPolyData> CreatePlane(
        const pcl::ModelCoefficients& coefficients,
        double x,
        double y,
        double z,
        double scale = 1);

/**
 * @brief Create plane from model coefficients (auto-positioned)
 * @param coefficients Plane equation coefficients
 * @return VTK polydata representing plane
 */
vtkSmartPointer<vtkPolyData> CreatePlane(
        const pcl::ModelCoefficients& coefficients);

/**
 * @brief Create labeled coordinate system
 * @param axesLength Length of axes (default: 1.5)
 * @param xLabel X axis label (default: "X")
 * @param yLabel Y axis label (default: "Y")
 * @param zLabel Z axis label (default: "Z")
 * @param xPlus Positive X direction label (default: "+X")
 * @param xMinus Negative X direction label (default: "-X")
 * @param yPlus Positive Y direction label (default: "+Y")
 * @param yMinus Negative Y direction label (default: "-Y")
 * @param zPlus Positive Z direction label (default: "+Z")
 * @param zMinus Negative Z direction label (default: "-Z")
 * @return VTK prop assembly with axes and labels
 */
vtkSmartPointer<vtkPropAssembly> CreateCoordinate(
        double axesLength = 1.5,
        const std::string& xLabel = "X",
        const std::string& yLabel = "Y",
        const std::string& zLabel = "Z",
        const std::string& xPlus = "+X",
        const std::string& xMinus = "-X",
        const std::string& yPlus = "+Y",
        const std::string& yMinus = "-Y",
        const std::string& zPlus = "+Z",
        const std::string& zMinus = "-Z");
};  // namespace PclTools

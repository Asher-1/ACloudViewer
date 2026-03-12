// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** @file VtkRenderingUtils.h
 *  @brief VTK actor creation, geometric primitives, and poly data utilities
 */

#include <vtkCellArray.h>
#include <vtkDataSet.h>
#include <vtkDataSetMapper.h>
#include <vtkLODActor.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>

#include <string>

#include "qVTK.h"

// CV_DB_LIB
#include <ecvDrawContext.h>

// Forward declarations
class vtkAbstractWidget;
class vtkPoints;
class vtkPropAssembly;
class ccGBLSensor;
class ccCameraSensor;
class ccGLMatrixd;

namespace cloudViewer {
namespace geometry {
class LineSet;
}
}  // namespace cloudViewer

namespace VtkRendering {

// =====================================================================
// VTK Actor Creation
// =====================================================================

/// @param data Input VTK dataset
/// @param actor Output LOD actor (created)
/// @param use_scalars Use scalar colors if present
QVTK_ENGINE_LIB_API void CreateActorFromVTKDataSet(
        const vtkSmartPointer<vtkDataSet>& data,
        vtkSmartPointer<vtkLODActor>& actor,
        bool use_scalars = true);

/// @param data Input VTK dataset
/// @param actor Output actor (created)
/// @param use_scalars Use scalar colors if present
QVTK_ENGINE_LIB_API void CreateActorFromVTKDataSet(
        const vtkSmartPointer<vtkDataSet>& data,
        vtkSmartPointer<vtkActor>& actor,
        bool use_scalars = true);

// =====================================================================
// Scalar Bar
// =====================================================================

/// @param widget Scalar bar widget to update
/// @param context Draw context
/// @return true on success
QVTK_ENGINE_LIB_API bool UpdateScalarBar(vtkAbstractWidget* widget,
                                         const CC_DRAW_CONTEXT& context);

// =====================================================================
// Geometric Primitive Creation
// =====================================================================

/// @param width Cube width
/// @param height Cube height
/// @param depth Cube depth
/// @param trans Transformation matrix
/// @return vtkPolyData for cube
QVTK_ENGINE_LIB_API vtkSmartPointer<vtkPolyData> CreateCube(
        double width, double height, double depth, const ccGLMatrixd& trans);

/// @param width Cube width
/// @param height Cube height
/// @param depth Cube depth
/// @return vtkPolyData for cube
QVTK_ENGINE_LIB_API vtkSmartPointer<vtkPolyData> CreateCube(double width,
                                                            double height,
                                                            double depth);

// =====================================================================
// Sensor Visualization
// =====================================================================

/// @param gBLSensor GBL sensor to visualize
/// @return vtkPolyData for sensor geometry
QVTK_ENGINE_LIB_API vtkSmartPointer<vtkPolyData> CreateGBLSensor(
        const ccGBLSensor* gBLSensor);

/// @param cameraSensor Camera sensor to visualize
/// @param lineColor Line color
/// @param planeColor Plane color
/// @return vtkPolyData for camera geometry
QVTK_ENGINE_LIB_API vtkSmartPointer<vtkPolyData> CreateCameraSensor(
        const ccCameraSensor* cameraSensor,
        const ecvColor::Rgb& lineColor,
        const ecvColor::Rgb& planeColor);

// =====================================================================
// Line Set Conversion
// =====================================================================

/// @param lineset Line set geometry
/// @param useLineSource Use vtkLineSource for each segment
/// @return vtkPolyData for line set
QVTK_ENGINE_LIB_API vtkSmartPointer<vtkPolyData> CreatePolyDataFromLineSet(
        const cloudViewer::geometry::LineSet& lineset,
        bool useLineSource = true);

// =====================================================================
// Coordinate System
// =====================================================================

/// @param axesLength Length of each axis
/// @param xLabel X axis label
/// @param yLabel Y axis label
/// @param zLabel Z axis label
/// @param xPlus +X label
/// @param xMinus -X label
/// @param yPlus +Y label
/// @param yMinus -Y label
/// @param zPlus +Z label
/// @param zMinus -Z label
/// @return vtkPropAssembly for coordinate axes
QVTK_ENGINE_LIB_API vtkSmartPointer<vtkPropAssembly> CreateCoordinate(
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

// =====================================================================
// Internal Helpers (used by functions above, also available externally)
// =====================================================================

/// @param polyData Poly data to color
/// @param color RGB color
/// @param is_cell Apply to cells (true) or points (false)
QVTK_ENGINE_LIB_API void SetPolyDataColor(vtkSmartPointer<vtkPolyData> polyData,
                                          const ecvColor::Rgb& color,
                                          bool is_cell = false);

/// @param polyData Poly data to add cell to
QVTK_ENGINE_LIB_API void AddPolyDataCell(vtkSmartPointer<vtkPolyData> polyData);

/// @param polyData Poly data to transform
/// @param trans Transformation matrix
/// @return true on success
QVTK_ENGINE_LIB_API bool TransformPolyData(
        vtkSmartPointer<vtkPolyData> polyData, const ccGLMatrixd& trans);

/// @param points VTK points to transform
/// @param trans Transformation matrix
/// @return true on success
QVTK_ENGINE_LIB_API bool TransformVtkPoints(vtkSmartPointer<vtkPoints> points,
                                            const ccGLMatrixd& trans);

}  // namespace VtkRendering

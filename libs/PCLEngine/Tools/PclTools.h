//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#                    COPYRIGHT: CLOUDVIEWER  project                     #
//#                                                                        #
//##########################################################################

#ifndef QPCL_PCLTOOLS_HEADER
#define QPCL_PCLTOOLS_HEADER

// LOCAL
#include "qPCL.h"
#include "PclUtils/PCLConv.h"
#include "PclUtils/PCLCloud.h"

// CV_CORE_LIB
#include <CVLog.h>

// ECV_DB_LIB
#include <ecvDrawContext.h>

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
}

class ccGBLSensor;
class ccCameraSensor;
namespace PclTools
{

    PCLTextureMesh::Ptr CreateTexturingMesh(const std::string& filePath, bool save = false);
    PCLTextureMesh::Ptr CreateTexturingMesh(const PCLMesh::ConstPtr triangles, const std::string& filePath, bool save = false);

	// Helper function called by createActorFromVTKDataSet () methods.
	// This function determines the default setting of vtkMapper::InterpolateScalarsBeforeMapping.
	// Return 0, interpolation off, if data is a vtkPolyData that contains only vertices.
	// Return 1, interpolation on, for anything else.
	inline int GetDefaultScalarInterpolationForDataSet(vtkDataSet* data)
	{
        vtkPolyData* polyData = vtkPolyData::SafeDownCast(data); // Check that polyData != nullptr in case of segfault
		return (polyData && polyData->GetNumberOfCells() != polyData->GetNumberOfVerts());
	}

	/** \brief Internal method. Creates a vtk actor from a vtk polydata object.
	  * \param[in] data the vtk polydata object to create an actor for
	  * \param[out] actor the resultant vtk actor object
	  * \param[in] use_scalars set scalar properties to the mapper if it exists in the data. Default: true.
	  */
	void CreateActorFromVTKDataSet(const vtkSmartPointer<vtkDataSet> &data,
		vtkSmartPointer<vtkActor> &actor,
		bool use_scalars = true, bool use_vbos = false);

	
	/** \brief Internal method. Creates a vtk actor from a vtk polydata object.
		* \param[in] data the vtk polydata object to create an actor for
		* \param[out] actor the resultant vtk actor object
		* \param[in] use_scalars set scalar properties to the mapper if it exists in the data. Default: true.
		*/
	void CreateActorFromVTKDataSet (const vtkSmartPointer<vtkDataSet> &data,
									vtkSmartPointer<vtkLODActor> &actor,
									bool use_scalars = true,
									bool use_vbos = false);

	/** \brief Allocate a new unstructured grid smartpointer. For internal use only.
	  * \param[out] polydata the resultant unstructured grid.
	  */
	void AllocVtkUnstructuredGrid(vtkSmartPointer<vtkUnstructuredGrid> &polydata);

    bool UpdateScalarBar(vtkAbstractWidget* widget, const CC_DRAW_CONTEXT& CONTEXT);

    bool TransformPolyData(vtkSmartPointer<vtkPolyData> polyData, const ccGLMatrixd& trans);
    bool TransformVtkPoints(vtkSmartPointer<vtkPoints> points, const ccGLMatrixd& trans);


    vtkSmartPointer<vtkPoints> GetVtkPointsFromLineSet(const cloudViewer::geometry::LineSet& lineset);
    bool GetVtkPointsAndLinesFromLineSet(const cloudViewer::geometry::LineSet &lineset,
                                         vtkSmartPointer<vtkPoints> points,
                                         vtkSmartPointer<vtkCellArray> lines,
                                         vtkSmartPointer<vtkUnsignedCharArray> colors);

    vtkSmartPointer<vtkPolyData> CreateCoordinateFromLineSet(const cloudViewer::geometry::LineSet& lineset);
    vtkSmartPointer<vtkPolyData> CreatePolyDataFromLineSet(const cloudViewer::geometry::LineSet& lineset, bool useLineSource = true);

    void SetPolyDataColor(vtkSmartPointer<vtkPolyData> polyData,
                          const ecvColor::Rgb& color, bool is_cell = false);
    void AddPolyDataCell(vtkSmartPointer<vtkPolyData> polyData);
    vtkSmartPointer<vtkPolyData> CreateLine(vtkSmartPointer<vtkPoints> points);
    vtkSmartPointer<vtkPolyData> CreateLine(vtkSmartPointer<vtkPoints> points,
                                            vtkSmartPointer<vtkCellArray> lines,
                                            vtkSmartPointer<vtkUnsignedCharArray> colors);
    vtkSmartPointer<vtkPolyData> CreateCube(double width, double height, double depth, const ccGLMatrixd &trans);
    vtkSmartPointer<vtkPolyData> CreateCube(double width, double height, double depth);
    vtkSmartPointer<vtkPolyData> CreateGBLSensor(const ccGBLSensor *gBLSensor);
    vtkSmartPointer<vtkPolyData> CreateCameraSensor(const ccCameraSensor *cameraSensor,
                                               const ecvColor::Rgb& lineColor,
                                               const ecvColor::Rgb& planeColor);
    vtkSmartPointer<vtkPolyData> CreatePlane(const pcl::ModelCoefficients &coefficients,
                                             double x, double y, double z, double scale = 1);
    vtkSmartPointer<vtkPolyData> CreatePlane(const pcl::ModelCoefficients &coefficients);
    vtkSmartPointer<vtkPropAssembly> CreateCoordinate(double axesLength = 1.5,
                                                      const std::string& xLabel = "x",
                                                      const std::string& yLabel = "y",
                                                      const std::string& zLabel = "z",
                                                      const std::string& xPlus = "R",
                                                      const std::string& xMinus = "L",
                                                      const std::string& yPlus = "A",
                                                      const std::string& yMinus = "P",
                                                      const std::string& zPlus = "I",
                                                      const std::string& zMinus = "S");
};

#endif // QPCL_PCLTOOLS_HEADER

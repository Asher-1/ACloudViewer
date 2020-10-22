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
#include "../qPCL.h"
#include "../PclUtils/PCLConv.h"
#include "../PclUtils/PCLCloud.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <vector>

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
class vtkAbstractWidget;
class vtkUnstructuredGrid;

namespace PclTools
{
	// Helper function called by createActorFromVTKDataSet () methods.
	// This function determines the default setting of vtkMapper::InterpolateScalarsBeforeMapping.
	// Return 0, interpolation off, if data is a vtkPolyData that contains only vertices.
	// Return 1, interpolation on, for anything else.
	inline int GetDefaultScalarInterpolationForDataSet(vtkDataSet* data)
	{
		vtkPolyData* polyData = vtkPolyData::SafeDownCast(data); // Check that polyData != NULL in case of segfault
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

	vtkSmartPointer<vtkDataSet> CreateLine(vtkSmartPointer<vtkPoints> points);

	bool UpdateScalarBar(vtkAbstractWidget* widget, const CC_DRAW_CONTEXT& CONTEXT);
};

#endif // QPCL_PCLTOOLS_HEADER
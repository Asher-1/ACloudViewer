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

#include "PclTools.h"

#include "PclUtils/cc2sm.h"
#include "PclUtils/sm2cc.h"

#include <vtkProperty.h>
#include <vtkActor.h>
#include <vtkLineSource.h>
#include <vtkLODActor.h>
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPolygon.h>
#include <vtkDataSet.h>
#include <vtkPointData.h>
#include <vtkDataSetMapper.h>
#include <vtkUnstructuredGrid.h>
#include <pcl/visualization/vtk/vtkVertexBufferObjectMapper.h>

#if defined(_WIN32)
  // Remove macros defined in Windows.h
#undef near
#undef far
#endif


/////////////////////////////////////////////////////////////////////////////////////////////
void PclTools::CreateActorFromVTKDataSet(
	const vtkSmartPointer<vtkDataSet> &data,
	vtkSmartPointer<vtkLODActor> &actor,
	bool use_scalars,
	bool use_vbos)
{
	// If actor is not initialized, initialize it here
	if (!actor)
		actor = vtkSmartPointer<vtkLODActor>::New();

#if VTK_RENDERING_BACKEND_OPENGL_VERSION < 2
	if (use_vbos)
	{
		vtkSmartPointer<vtkVertexBufferObjectMapper> mapper = vtkSmartPointer<vtkVertexBufferObjectMapper>::New();

		mapper->SetInput(data);

		if (use_scalars)
		{
			vtkSmartPointer<vtkDataArray> scalars = data->GetPointData()->GetScalars();
			double minmax[2];
			if (scalars)
			{
				scalars->GetRange(minmax);
				mapper->SetScalarRange(minmax);

				mapper->SetScalarModeToUsePointData();
				mapper->SetInterpolateScalarsBeforeMapping(GetDefaultScalarInterpolationForDataSet(data));
				mapper->ScalarVisibilityOn();
			}
		}

		actor->SetNumberOfCloudPoints(int(std::max<vtkIdType>(1, data->GetNumberOfPoints() / 10)));
		actor->GetProperty()->SetInterpolationToFlat();

		/// FIXME disabling backface culling due to known VTK bug: vtkTextActors are not
		/// shown when there is a vtkActor with backface culling on present in the scene
		/// Please see VTK bug tracker for more details: http://www.vtk.org/Bug/view.php?id=12588
		// actor->GetProperty ()->BackfaceCullingOn ();

		actor->SetMapper(mapper);
	}
	else
#endif
	{
		vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
#if VTK_MAJOR_VERSION < 6
		mapper->SetInput(data);
#else
		mapper->SetInputData(data);
#endif

		if (use_scalars)
		{
			vtkSmartPointer<vtkDataArray> scalars = data->GetPointData()->GetScalars();
			double minmax[2];
			if (scalars)
			{
				scalars->GetRange(minmax);
				mapper->SetScalarRange(minmax);

				mapper->SetScalarModeToUsePointData();
				mapper->SetInterpolateScalarsBeforeMapping(GetDefaultScalarInterpolationForDataSet(data));
				mapper->ScalarVisibilityOn();
			}
		}
#if VTK_RENDERING_BACKEND_OPENGL_VERSION < 2
		mapper->ImmediateModeRenderingOff();
#endif

		actor->SetNumberOfCloudPoints(int(std::max<vtkIdType>(1, data->GetNumberOfPoints() / 10)));
		actor->GetProperty()->SetInterpolationToFlat();

		/// FIXME disabling backface culling due to known VTK bug: vtkTextActors are not
		/// shown when there is a vtkActor with backface culling on present in the scene
		/// Please see VTK bug tracker for more details: http://www.vtk.org/Bug/view.php?id=12588
		// actor->GetProperty ()->BackfaceCullingOn ();

		actor->SetMapper(mapper);
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////
void PclTools::CreateActorFromVTKDataSet(const vtkSmartPointer<vtkDataSet> &data,
	vtkSmartPointer<vtkActor> &actor, bool use_scalars, bool use_vbos)
{
	// If actor is not initialized, initialize it here
	if (!actor)
		actor = vtkSmartPointer<vtkActor>::New();

#if VTK_RENDERING_BACKEND_OPENGL_VERSION < 2
	if (use_vbos)
	{
		vtkSmartPointer<vtkVertexBufferObjectMapper> mapper = vtkSmartPointer<vtkVertexBufferObjectMapper>::New();

		mapper->SetInput(data);

		if (use_scalars)
		{
			vtkSmartPointer<vtkDataArray> scalars = data->GetPointData()->GetScalars();
			double minmax[2];
			if (scalars)
			{
				scalars->GetRange(minmax);
				mapper->SetScalarRange(minmax);

				mapper->SetScalarModeToUsePointData();
				mapper->SetInterpolateScalarsBeforeMapping(GetDefaultScalarInterpolationForDataSet(data));
				mapper->ScalarVisibilityOn();
			}
		}

		//actor->SetNumberOfCloudPoints (int (std::max<vtkIdType> (1, data->GetNumberOfPoints () / 10)));
		actor->GetProperty()->SetInterpolationToFlat();

		/// FIXME disabling backface culling due to known VTK bug: vtkTextActors are not
		/// shown when there is a vtkActor with backface culling on present in the scene
		/// Please see VTK bug tracker for more details: http://www.vtk.org/Bug/view.php?id=12588
		// actor->GetProperty ()->BackfaceCullingOn ();

		actor->SetMapper(mapper);
	}
	else
#endif
	{
		vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
#if VTK_MAJOR_VERSION < 6
		mapper->SetInput(data);
#else
		mapper->SetInputData(data);
#endif

		if (use_scalars)
		{
			vtkSmartPointer<vtkDataArray> scalars = data->GetPointData()->GetScalars();
			double minmax[2];
			if (scalars)
			{
				scalars->GetRange(minmax);
				mapper->SetScalarRange(minmax);

				mapper->SetScalarModeToUsePointData();
				mapper->SetInterpolateScalarsBeforeMapping(GetDefaultScalarInterpolationForDataSet(data));
				mapper->ScalarVisibilityOn();
			}
		}
#if VTK_RENDERING_BACKEND_OPENGL_VERSION < 2
		mapper->ImmediateModeRenderingOff();
#endif

		//actor->SetNumberOfCloudPoints (int (std::max<vtkIdType> (1, data->GetNumberOfPoints () / 10)));
		actor->GetProperty()->SetInterpolationToFlat();

		/// FIXME disabling backface culling due to known VTK bug: vtkTextActors are not
		/// shown when there is a vtkActor with backface culling on present in the scene
		/// Please see VTK bug tracker for more details: http://www.vtk.org/Bug/view.php?id=12588
		// actor->GetProperty ()->BackfaceCullingOn ();

		actor->SetMapper(mapper);
	}

	//actor->SetNumberOfCloudPoints (std::max<vtkIdType> (1, data->GetNumberOfPoints () / 10));
	actor->GetProperty()->SetInterpolationToFlat();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void PclTools::AllocVtkUnstructuredGrid(vtkSmartPointer<vtkUnstructuredGrid> &polydata)
{
	polydata = vtkSmartPointer<vtkUnstructuredGrid>::New();
}

vtkSmartPointer<vtkDataSet>
PclTools::CreateLine(vtkSmartPointer<vtkPoints> points)
{
	vtkSmartPointer<vtkLineSource> lineSource = vtkSmartPointer<vtkLineSource>::New();
	lineSource->SetPoints(points);
	lineSource->Update();
	return (lineSource->GetOutput());
}
/*=========================================================================

  Program:   ParaView
  Module:    $RCSfile$

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.paraview.org/HTML/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   vtkPVCenterAxesActor
 *
 * vtkPVCenterAxesActor is an actor for the center-axes used in ParaView. It
 * merely uses vtkAxes as the poly data source.
*/

#ifndef ECV_VTK_CENTER_AXES_ACTOR_H
#define ECV_VTK_CENTER_AXES_ACTOR_H

#include "vtkOpenGLActor.h"
#include "qPCL.h" // needed for export macro

class vtkAxes;
class vtkPolyDataMapper;

namespace VTKExtensions
{
	class QPCL_ENGINE_LIB_API vtkPVCenterAxesActor : public vtkOpenGLActor
	{
	public:
		static vtkPVCenterAxesActor* New();
		vtkTypeMacro(vtkPVCenterAxesActor, vtkOpenGLActor);
		void PrintSelf(ostream& os, vtkIndent indent) override;

		/**
		 * If Symmetric is on, the the axis continue to negative values.
		 */
		void SetSymmetric(int);

		/**
		 * Option for computing normals.  By default they are computed.
		 */
		void SetComputeNormals(int);

	protected:
		vtkPVCenterAxesActor();
		~vtkPVCenterAxesActor() override;

		vtkAxes* Axes;
		vtkPolyDataMapper* Mapper;

	private:
		vtkPVCenterAxesActor(const vtkPVCenterAxesActor&) = delete;
		void operator=(const vtkPVCenterAxesActor&) = delete;
	};

}

#endif // ECV_VTK_CENTER_AXES_ACTOR_H

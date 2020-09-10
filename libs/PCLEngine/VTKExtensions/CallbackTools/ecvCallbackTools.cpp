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

#include "ecvCallbackTools.h"

// LOCAL
#include "VTKExtensions/Widgets/CustomVtkBoxWidget.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <CVGeom.h>

// ECV_DB_LIB
#include <ecvColorTypes.h>

// VTK
#include <vtkAssembly.h>
#include <vtkRenderWindow.h>
#include <vtkCamera.h>
#include <vtkRenderer.h>
#include <vtkTransform.h>
#include <vtkRendererCollection.h>
#include <vtkProperty2D.h>
#include <vtkRenderWindow.h>
#include <vtkLookupTable.h>
#include <vtkPNGReader.h>
#include <vtkImageData.h>
#include <vtkLogoRepresentation.h>
#include <vtkLogoWidget.h>
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkScalarBarWidget.h>
#include <vtkScalarBarActor.h>
#include <vtkScalarBarRepresentation.h>
#include <vtkColorTransferFunction.h>
#include <vtkPlane.h>
#include <vtkImplicitPlaneRepresentation.h>
#include <vtkImplicitPlaneWidget2.h>
#include <vtkBoxWidget2.h>
#include <vtkBoxRepresentation.h>
#include <vtkAngleWidget.h>
#include <vtkAngleRepresentation.h>
#include <vtkTextWidget.h>
#include <vtkTextActor.h>

namespace CallbackTools
{

	/******************************** vtkIPWCallback *********************************/
	vtkIPWCallback::vtkIPWCallback() 
		: Plane(nullptr), 
		  Actor(nullptr) 
	{}

	void vtkIPWCallback::Execute(vtkObject *caller, unsigned long, void*)
	{
		vtkImplicitPlaneWidget2 *planeWidget =
			reinterpret_cast<vtkImplicitPlaneWidget2*>(caller);
		vtkImplicitPlaneRepresentation *rep =
			reinterpret_cast<vtkImplicitPlaneRepresentation*>(planeWidget->GetRepresentation());
		rep->GetPlane(this->Plane);
	}


	/******************************** vtkBoxCallback *********************************/
	vtkBoxCallback::vtkBoxCallback()
	{
		m_actors.clear();
	}

	void vtkBoxCallback::SetActors(const std::vector< vtkActor* > actors)
	{
		m_actors = actors;
	}

	void vtkBoxCallback::Execute(vtkObject*caller, unsigned long, void*)
	{
		//将调用该回调函数的调用者caller指针，转换为vtkBoxWidget2类型对象指针
		//vtkSmartPointer<vtkBoxWidget2> boxWidget = vtkBoxWidget2::SafeDownCast(caller);
		vtkSmartPointer<CustomVtkBoxWidget> boxWidget = CustomVtkBoxWidget::SafeDownCast(caller);
		// vtkSmartPointer<vtkBoxWidget2> boxWidget=reinterpret_cast<vtkBoxWidget2>(caller);
		vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New();
		//将boxWidget中的变换矩阵保存在t中
		boxWidget->GetTransform(t);
		for (vtkActor* actor : this->m_actors)
		{
			if (actor)
			{
				actor->SetUserTransform(t);
			}
		}
		
		//emit uerTransform(t->GetMatrix()->GetData());
	}

	/******************************** vtkBoxCallback2 *********************************/
	vtkBoxCallback2::vtkBoxCallback2()
	{
	}

	void vtkBoxCallback2::SetActor(vtkSmartPointer<vtkActor> actor)
	{
		m_actor = actor;
	}

	void vtkBoxCallback2::Execute(vtkObject*caller, unsigned long, void*)
	{
		//将调用该回调函数的调用者caller指针，转换为vtkBoxWidget2类型对象指针
		vtkSmartPointer<vtkBoxWidget2> boxWidget = vtkBoxWidget2::SafeDownCast(caller);
		// vtkSmartPointer<vtkBoxWidget2> boxWidget=reinterpret_cast<vtkBoxWidget2>(caller);这样转换不可以，vtkBoxWidget可以
		vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New();
		//将boxWidget中的变换矩阵保存在t中
		vtkBoxRepresentation::SafeDownCast(boxWidget->GetRepresentation())->GetTransform(t);
		this->m_actor->SetUserTransform(t);
	}

	/******************************** vtkAngleCallBack *********************************/
	vtkAngleCallBack::vtkAngleCallBack()
		: m_angle(nullptr),
		  m_text(nullptr)
	{
	}

	void vtkAngleCallBack::Execute(vtkObject *caller, unsigned long eventId, void *callData)
	{
		if (!m_text)
			return;

		if (eventId == vtkCommand::StartInteractionEvent)
			m_text->On();
		if (eventId == vtkCommand::InteractionEvent) {
			char text[200];
			sprintf(text, "Angle: %f", m_angle->GetAngleRepresentation()->GetAngle());
			m_text->GetTextActor()->SetInput(text);
		}
	}

}

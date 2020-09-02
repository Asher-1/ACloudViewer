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

#ifndef QPCL_CALLBACK_TOOLS_HEADER
#define QPCL_CALLBACK_TOOLS_HEADER

// LOCAL
#include "VtkUtils/abstractwidgetobserver.h"

// VTK
#include <vtkSmartPointer.h>

class vtkActor;
class vtkAssembly;
class vtkPlane;
class vtkObject;
class vtkAngleWidget;
class vtkTextWidget;
class vtkTransform;

// Implicit Plane callback class
namespace CallbackTools
{
	class vtkIPWCallback : public VtkUtils::AbstractWidgetObserver
	{
	public:
		static vtkIPWCallback *New()
		{
			return new vtkIPWCallback;
		}
		virtual void Execute(vtkObject *caller, unsigned long, void*) override;

		vtkIPWCallback();
		vtkPlane *Plane;
		vtkActor *Actor;

	};

	// box callback class
	class vtkBoxCallback :public VtkUtils::AbstractWidgetObserver
	{
		Q_OBJECT
	public:
		static vtkBoxCallback* New()
		{
			return new vtkBoxCallback;
		}

		virtual void Execute(vtkObject*caller, unsigned long, void*) override;
		void SetActors(const std::vector< vtkActor* > actors);
	signals:
		void uerTransform(double * trans);

	protected:
		vtkBoxCallback();
		~vtkBoxCallback() = default;
		std::vector< vtkActor* > m_actors;
	};

	// box callback2 class
	class vtkBoxCallback2 :public VtkUtils::AbstractWidgetObserver
	{
	public:
		static vtkBoxCallback2* New()
		{
			return new vtkBoxCallback2;
		}

		void SetActor(vtkSmartPointer<vtkActor> actor);

		virtual void Execute(vtkObject*caller, unsigned long, void*) override;
	protected:
		vtkBoxCallback2();
        ~vtkBoxCallback2() override = default;
	public:
		vtkSmartPointer<vtkActor> m_actor;
	};


	// angle callback class
	class vtkAngleCallBack : public VtkUtils::AbstractWidgetObserver
	{
	public:
		static vtkAngleCallBack *New()
		{
			return new vtkAngleCallBack;
		}

		virtual void Execute(vtkObject *caller, unsigned long eventId, void *callData) override;

		vtkAngleCallBack();
        vtkSmartPointer<vtkAngleWidget> m_angle;
        vtkSmartPointer<vtkTextWidget> m_text;
	};

}

#endif // QPCL_CALLBACK_TOOLS_HEADER

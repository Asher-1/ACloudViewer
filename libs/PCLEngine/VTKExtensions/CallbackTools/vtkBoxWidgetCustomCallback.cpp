#include "vtkBoxWidgetCustomCallback.h"
#include <vtkActor.h>
#include <vtkTransform.h>
#include <vtkBoxWidget.h>
#include <vtkBoxRepresentation.h>

vtkBoxWidgetCustomCallback *vtkBoxWidgetCustomCallback::New()
{
    return new vtkBoxWidgetCustomCallback;
}

void vtkBoxWidgetCustomCallback::SetActor(vtkSmartPointer<vtkActor> actor)
{
	m_actor = actor;
}

void vtkBoxWidgetCustomCallback::Execute(vtkObject *caller, unsigned long, void *)
{
	if (m_preview)
	{
		//将调用该回调函数的调用者caller指针，转换为vtkBoxWidget2类型对象指针
		vtkSmartPointer<vtkBoxWidget> boxWidget = vtkBoxWidget::SafeDownCast(caller);
		// vtkSmartPointer<vtkBoxWidget2> boxWidget=reinterpret_cast<vtkBoxWidget2>(caller);这样转换不可以，vtkBoxWidget可以
		vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New();
		//将boxWidget中的变换矩阵保存在t中
		//vtkBoxRepresentation::SafeDownCast(boxWidget->GetRepresentation())->GetTransform(t);
		boxWidget->GetTransform(t);
		//this->m_actor->SetUserTransform(t);
	}
}



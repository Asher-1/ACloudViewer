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
		//�����øûص������ĵ�����callerָ�룬ת��ΪvtkBoxWidget2���Ͷ���ָ��
		vtkSmartPointer<vtkBoxWidget> boxWidget = vtkBoxWidget::SafeDownCast(caller);
		// vtkSmartPointer<vtkBoxWidget2> boxWidget=reinterpret_cast<vtkBoxWidget2>(caller);����ת�������ԣ�vtkBoxWidget����
		vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New();
		//��boxWidget�еı任���󱣴���t��
		//vtkBoxRepresentation::SafeDownCast(boxWidget->GetRepresentation())->GetTransform(t);
		boxWidget->GetTransform(t);
		//this->m_actor->SetUserTransform(t);
	}
}



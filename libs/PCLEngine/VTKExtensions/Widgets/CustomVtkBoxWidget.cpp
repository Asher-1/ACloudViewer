#include "CustomVtkBoxWidget.h"

#include <vtkMath.h>
#include <vtkPoints.h>
#include <vtkRenderer.h>
#include <vtkTransform.h>
#include <vtkBoxWidget2.h>
#include <vtkDoubleArray.h>
#include <vtkRenderWindowInteractor.h>

vtkStandardNewMacro(CustomVtkBoxWidget);


/* ƽ�ƣ�	ѭ���������еĵ㼯���ǵ㡢���㣩����ƽ��ÿ����              */
// ���룺
// double *p1	:����ƶ�֮ǰ�ĵ������
// double *p2	:����ƶ�֮��ĵ������
/************************************************************************/
void CustomVtkBoxWidget::Translate(double * p1, double * p2)
{
	if (!m_translateX && !m_translateY && !m_translateZ)
	{
		return;
	}

	double *pts =
		static_cast<vtkDoubleArray *>(this->Points->GetData())->GetPointer(0);
	double v[3];

	v[0] = m_translateX ? p2[0] - p1[0] : 0;
	v[1] = m_translateY ? p2[1] - p1[1] : 0;
	v[2] = m_translateZ ? p2[2] - p1[2] : 0;

	// Move the corners
	for (int i = 0; i < 8; i++)
	{
		*pts++ += v[0];
		*pts++ += v[1];
		*pts++ += v[2];
	}
	this->PositionHandles();
}

/************************************************************************/
/* ����                                                                 */
// ���룺
// double *p1	:����ƶ�֮ǰ�ĵ������
// double *p2	:����ƶ�֮��ĵ������
/************************************************************************/
void CustomVtkBoxWidget::Scale(double * p1, double * p2, int X, int Y)
{
	if (!m_scale)
	{
		return;
	}

	double *pts =
		static_cast<vtkDoubleArray *>(this->Points->GetData())->GetPointer(0);
	double *center
		= static_cast<vtkDoubleArray *>(this->Points->GetData())->GetPointer(3 * 14);
	double sf;

	if (Y > this->Interactor->GetLastEventPosition()[1])
	{
		sf = 1.03;
	}
	else
	{
		sf = 0.97;
	}

	// Move the corners
	for (int i = 0; i < 8; i++, pts += 3)
	{
		pts[0] = sf * (pts[0] - center[0]) + center[0];
		pts[1] = sf * (pts[1] - center[1]) + center[1];
		pts[2] = sf * (pts[2] - center[2]) + center[2];
	}
	this->PositionHandles();
}

/************************************************************************/
/* ��ת                                                                 */
// ����
// int X		:��굱ǰ������꣺x
// int Y		:��굱ǰ�������: y
// double *p1	:����ƶ�֮ǰ�ĵ������
// double *p2	:����ƶ�֮��ĵ������
// double *vpn	:������
/************************************************************************/
void CustomVtkBoxWidget::Rotate(int X, int Y, double *p1, double *p2, double *vpn)
{
	if (!m_rotateX && !m_rotateY && !m_rotateZ)
	{
		return;
	}

	double *pts =
		static_cast<vtkDoubleArray *>(this->Points->GetData())->GetPointer(0);
	double *center =
		static_cast<vtkDoubleArray *>(this->Points->GetData())->GetPointer(3 * 14);
	double v[3];		// vector of motion
	double axis[3];		// axis of rotation
	double theta;		// rotation angle
	int i;

	v[0] = m_rotateX ? p2[0] - p1[0] : 0;
	v[1] = m_rotateY ? p2[1] - p1[1] : 0;
	v[2] = m_rotateZ ? p2[2] - p1[2] : 0;

	// Create axis of rotation and angle of rotation
	vtkMath::Cross(vpn, v, axis);
	if (vtkMath::Normalize(axis) == 0.0)
	{
		return;
	}
	int *size = this->CurrentRenderer->GetSize();
	double l2 = (X - this->Interactor->GetLastEventPosition()[0])*(X - this->Interactor->GetLastEventPosition()[0])
		+ (Y - this->Interactor->GetLastEventPosition()[1])*(Y - this->Interactor->GetLastEventPosition()[1]);
	theta = 360.0 * sqrt(l2 / (size[0] * size[0] + size[1] * size[1]));
	// vtkTransform	:describes linear transformations via a 4x4 matrix 
	//Manipulate the transform to reflect the rotation
	this->Transform->Identity();
	this->Transform->Translate(center[0], center[1], center[2]);
	this->Transform->RotateWXYZ(theta, axis);
	this->Transform->Translate(-center[0], -center[1], -center[2]);

	//Set the corners
	vtkPoints *newPts = vtkPoints::New(VTK_DOUBLE);
	this->Transform->TransformPoints(this->Points, newPts);

	for (i = 0; i < 8; i++, pts += 3)
	{// �任8���������
		this->Points->SetPoint(i, newPts->GetPoint(i));
	}

	newPts->Delete();
	this->PositionHandles(); // ���¼����������꣬������ص�����
}

#ifndef QVTK_TRANSFORM_VTKBOXWIDGET_HEADER
#define QVTK_TRANSFORM_VTKBOXWIDGET_HEADER

#include "qPCL.h"

/**
 * @brief The CustomVtkBoxWidget class
 * CustomVtkBoxWidget restricts the transformation
 */

#include <vtkBoxWidget.h>

class QPCL_ENGINE_LIB_API CustomVtkBoxWidget : public vtkBoxWidget {
public:
	static CustomVtkBoxWidget *New();

	vtkTypeMacro(CustomVtkBoxWidget, vtkBoxWidget);

	virtual void Translate(double *p1, double *p2) override;
	virtual void Scale(double *p1, double *p2, int X, int Y) override;
	virtual void Rotate(int X, int Y, double *p1, double *p2, double *vpn) override;

	void SetTranslateXEnabled(bool state) { m_translateX = state; }
	void SetTranslateYEnabled(bool state) { m_translateY = state; }
	void SetTranslateZEnabled(bool state) { m_translateZ = state; }
	void SetRotateXEnabled(bool state) { m_rotateX = state; }
	void SetRotateYEnabled(bool state) { m_rotateY = state; }
	void SetRotateZEnabled(bool state) { m_rotateZ = state; }
	void SetScaleEnabled(bool state) { m_scale = state; }

private:
	bool m_translateX = true;
	bool m_translateY = true;
	bool m_translateZ = true;
	bool m_rotateX = true;
	bool m_rotateY = true;
	bool m_rotateZ = true;
	bool m_scale = true;

};
#endif // QVTK_TRANSFORM_VTKBOXWIDGET_HEADER

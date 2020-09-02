#ifndef TOOLS_CUT_FILTER_H
#define TOOLS_CUT_FILTER_H

#include "cvGenericFilter.h"
#include <vtkSmartPointer.h>

namespace Ui
{
	class CutFilterDlg;
}

class vtkPlanes;
class vtk3DWidget;
class vtkBoxWidget;
class vtkSphereWidget;
class vtkImplicitPlaneWidget;
class vtkBoxWidgetCustomCallback;

class cvCutFilter : public cvGenericFilter
{
	Q_OBJECT
protected:
	enum CutType { Box, Plane, Sphere };

	explicit cvCutFilter(QWidget* parent = nullptr);

	void setNormal(double normal[3]);
	void setOrigin(double origin[3]);
	void setRadius(double radius);
	void updateCutWidget();

	virtual void initFilter() override;

	void setCutType(CutType type);
	CutType cutType() const;

public:
	virtual void showInteractor(bool state) override;
	virtual void getInteractorBounds(ccBBox& bbox) override;
	virtual void getInteractorTransformation(ccGLMatrixd& trans) override;
	virtual void shift(const CCVector3d& v) override;

protected slots:
	void onOriginChanged(double* orgin);
	void onNormalChanged(double* normal);
	void onCenterChanged(double* center);
	void onRadiusChanged(double radius);
	void onPlanesChanged(vtkPlanes* planes);
	void showContourLines(bool show = true);
	void onPreview(bool dummy);

private slots:
	void on_cutTypeCombo_currentIndexChanged(int index);
	void on_displayEffectCombo_currentIndexChanged(int index);
	void on_radiusSpinBox_valueChanged(double arg1);
	void on_originXSpinBox_valueChanged(double arg1);
	void on_originYSpinBox_valueChanged(double arg1);
	void on_originZSpinBox_valueChanged(double arg1);
	void on_normalXSpinBox_valueChanged(double arg1);
	void on_normalYSpinBox_valueChanged(double arg1);
	void on_normalZSpinBox_valueChanged(double arg1);
	void on_centerXSpinBox_valueChanged(double arg1);
	void on_centerYSpinBox_valueChanged(double arg1);
	void on_centerZSpinBox_valueChanged(double arg1);
	void on_showPlaneCheckBox_toggled(bool checked);
	void on_showContourLinesCheckBox_toggled(bool checked);
	void on_negativeCheckBox_toggled(bool checked);

protected:
	virtual void clearAllActor() override;
	virtual void modelReady() override;
	virtual void dataChanged() override;
	virtual void updateUi() override;

	void createUi();

	void resetPlaneWidget();
	void resetSphereWidget();
	void resetBoxWidget();

protected:
	double m_normal[3];
	double m_origin[3];
	double m_center[3];
	double m_radius;

	vtkSmartPointer<vtkBoxWidget> m_boxWidget;
	vtkSmartPointer<vtkSphereWidget> m_sphereWidget;
	vtkSmartPointer<vtkImplicitPlaneWidget> m_planeWidget;

	Ui::CutFilterDlg* m_configUi = nullptr;

	CutType m_cutType = CutType::Box;
	vtkSmartPointer<vtkPlanes> m_planes;
	vtkSmartPointer<vtkActor> m_contourLinesActor;
};

#endif // TOOLS_CUT_FILTER_H

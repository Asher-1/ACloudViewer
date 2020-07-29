#ifndef TOOLS_ISOSURFACE_FILTER_H
#define TOOLS_ISOSURFACE_FILTER_H

#include "cvGenericFilter.h"

namespace Ui
{
	class cvIsoSurfaceFilterDlg;
}

class vtkContourFilter;
class cvIsoSurfaceFilter : public cvGenericFilter
{
	Q_OBJECT

public:
	explicit cvIsoSurfaceFilter(QWidget *parent = 0);
	~cvIsoSurfaceFilter();

	virtual void apply() override;

	virtual ccHObject* getOutput() override;

	virtual void clearAllActor() override;

protected:
	virtual void createUi() override;
	virtual void modelReady() override;
	virtual void colorsChanged() override;

	virtual void initFilter() override;
	virtual void dataChanged() override;

protected slots:
	void onDoubleSpinBoxValueChanged(double value);
	void onSpinBoxValueChanged(int value);
	void onComboBoxIndexChanged(int index);
	void on_gradientCombo_activated(int index);

protected:
	Ui::cvIsoSurfaceFilterDlg* m_configUi = nullptr;
	double m_minScalar = .0;
	double m_maxScalar = .0;
	int m_numOfContours = 10;
	QString m_currentScalarName;

	vtkSmartPointer<vtkContourFilter> m_contourFilter;
};

#endif // TOOLS_ISOSURFACE_FILTER_H

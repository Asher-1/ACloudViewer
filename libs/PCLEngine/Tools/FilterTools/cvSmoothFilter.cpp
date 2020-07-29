#include "cvSmoothFilter.h"
#include "ui_cvSmoothFilterDlg.h"
#include "ui_cvGenericFilterDlg.h"

#include <VtkUtils/vtkutils.h>

#include <vtkDecimatePro.h>
#include <vtkRenderer.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkLODActor.h>
#include <vtkSmoothPolyDataFilter.h>

#include <CVLog.h>

cvSmoothFilter::cvSmoothFilter(QWidget* parent) : cvGenericFilter(parent)
{
	setWindowTitle(tr("Smooth"));
	createUi();
}

void cvSmoothFilter::createUi()
{
	cvGenericFilter::createUi();

	m_configUi = new Ui::cvSmoothFilterDlg;
	setupConfigWidget(m_configUi);
}

void cvSmoothFilter::apply()
{
	if (!m_dataObject) {
		CVLog::Error(QString("cvSmoothFilter::apply: null data object."));
		return;
	}

	if (!m_meshMode) {
		CVLog::Error(QString("cvSmoothFilter::apply mesh supported only!"));
		return;
	}

	if (m_keepMode)
	{
		return;
	}

	VtkUtils::vtkInitOnce(m_smoothFilter);
	m_smoothFilter->SetInputData(m_dataObject);
	m_smoothFilter->SetNumberOfIterations(m_numberOfIterations);
	m_smoothFilter->SetFeatureEdgeSmoothing(m_featureEdgeSmoothing);
	m_smoothFilter->SetBoundarySmoothing(m_boundarySmoothing);
	m_smoothFilter->SetConvergence(m_convergence);
	m_smoothFilter->SetFeatureAngle(m_featureAngle);
	m_smoothFilter->SetEdgeAngle(m_edgeAngle);

	m_smoothFilter->Update();

	if (!m_filterActor)
	{
		VTK_CREATE(vtkPolyDataMapper, mapper);
		mapper->SetInputConnection(m_smoothFilter->GetOutputPort());

		VtkUtils::vtkInitOnce(m_filterActor);
		m_filterActor->SetMapper(mapper);
		addActor(m_filterActor);
	}
	
	update();
}

void cvSmoothFilter::initFilter()
{
	if (m_modelActor)
	{
		// hide origin actor
		m_modelActor->SetVisibility(0);
	}
}

void cvSmoothFilter::dataChanged()
{
	m_keepMode = true;
	m_configUi->convergenceSpinBox->setValue(0.3);
	m_configUi->numOfIterationsSpinBox->setValue(20);
	m_configUi->featureEdgeSmoothingCheckBox->setChecked(true);
	m_configUi->boundarySmoothingCheckBox->setChecked(true);
	m_configUi->featureAngleSpinBox->setValue(20.0);
	m_configUi->edgeAngleSpinBox->setValue(20.0);
	m_keepMode = false;
	apply();
}

ccHObject * cvSmoothFilter::getOutput()
{
	if (!m_smoothFilter) return nullptr;

	// set exported polydata
	m_smoothFilter->Update();
	setResultData(m_smoothFilter->GetOutput());
	return cvGenericFilter::getOutput();
}

void cvSmoothFilter::clearAllActor()
{
	cvGenericFilter::clearAllActor();
}

void cvSmoothFilter::on_convergenceSpinBox_valueChanged(double arg1)
{
	m_convergence = arg1;
	apply();
}

void cvSmoothFilter::on_numOfIterationsSpinBox_valueChanged(int arg1)
{
	m_numberOfIterations = arg1;
	apply();
}

void cvSmoothFilter::on_featureEdgeSmoothingCheckBox_toggled(bool checked)
{
	m_featureEdgeSmoothing = checked;
	apply();
}

void cvSmoothFilter::on_boundarySmoothingCheckBox_toggled(bool checked)
{
	m_boundarySmoothing = checked;
	apply();
}

void cvSmoothFilter::on_featureAngleSpinBox_valueChanged(double arg1)
{
	m_featureAngle = arg1;
	apply();
}

void cvSmoothFilter::on_edgeAngleSpinBox_valueChanged(double arg1)
{
	m_edgeAngle = arg1;
	apply();
}

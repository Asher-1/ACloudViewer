#include "cvGlyphFilter.h"
#include "ui_cvGlyphFilterDlg.h"
#include "ui_cvGenericFilterDlg.h"

#include <VtkUtils/utils.h>
#include <VtkUtils/colorpushbutton.h>
#include <VtkUtils/signalblocker.h>

#include <VtkUtils/vtkutils.h>

#include <vtkGlyph3D.h>
#include <vtkArrowSource.h>
#include <vtkConeSource.h>
#include <vtkLineSource.h>
#include <vtkCylinderSource.h>
#include <vtkPointSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkProperty.h>
#include <vtkLODActor.h>
#include <vtkLabeledDataMapper.h>
#include <vtkActor2D.h>
#include <vtkProperty2D.h>
#include <vtkSphereSource.h>
#include <vtkSmartPointer.h>
#include <vtkPlaneSource.h>

#include <QWidget>

// CV_CORE_LIB
#include <CVLog.h>

cvGlyphFilter::cvGlyphFilter(QWidget* parent) : cvGenericFilter(parent)
{
	setWindowTitle(tr("Glyph"));
	setObjectName("cvGlyphFilter");
	createUi();

	VtkUtils::SignalBlocker sb(m_configUi->sizeSpinBox);
	sb.addObject(m_configUi->glyphColorButton);
	sb.addObject(m_configUi->labelColorButton);
	sb.addObject(m_configUi->modeCombo);

	m_configUi->sizeSpinBox->setValue(m_size);
	m_configUi->glyphColorButton->setCurrentColor(m_glyphColor);
	m_configUi->labelColorButton->setCurrentColor(m_labelColor);
	m_configUi->modeCombo->setCurrentIndex(m_labelMode);

	connect(m_configUi->glyphColorButton, SIGNAL(colorChanged(QColor)), this, SLOT(onColorChanged(QColor)));
	connect(m_configUi->labelColorButton, SIGNAL(colorChanged(QColor)), this, SLOT(onColorChanged(QColor)));
}

void cvGlyphFilter::apply()
{
	if (!m_dataObject)
		return;

	VtkUtils::vtkInitOnce(m_glyph3d);
	m_glyph3d->SetInputData(m_dataObject);
	m_glyph3d->SetScaleModeToDataScalingOff();
	m_glyph3d->SetScaling(0);

	switch (m_shape) {
	case Arrow:
	{
		VTK_CREATE(vtkArrowSource, as);
		m_glyph3d->SetSourceConnection(as->GetOutputPort());
	}
		break;

	case Cone:
	{
		VTK_CREATE(vtkConeSource, cs);
		m_glyph3d->SetSourceConnection(cs->GetOutputPort());
	}
		break;

	case Line:
	{
		VTK_CREATE(vtkLineSource, ls);
		m_glyph3d->SetSourceConnection(ls->GetOutputPort());
	}
		break;

	case Cylinder:
	{
		VTK_CREATE(vtkCylinderSource, cs);
		m_glyph3d->SetSourceConnection(cs->GetOutputPort());
	}
		break;

	case Sphere:
	{
		VTK_CREATE(vtkSphereSource, ss);
		m_glyph3d->SetSourceConnection(ss->GetOutputPort());
	}
		break;

	case Point:
	{
		VTK_CREATE(vtkPointSource, ps);
		ps->SetNumberOfPoints(1);
		m_glyph3d->SetSourceConnection(ps->GetOutputPort());
	}
	 break;
	}

	// Create a mapper and actor
	VTK_CREATE(vtkPolyDataMapper, mapper);
	mapper->SetInputConnection(m_glyph3d->GetOutputPort());
	mapper->SetScalarVisibility(!m_glyphColor.isValid());

	if (!m_filterActor)
	{
		VtkUtils::vtkInitOnce(m_filterActor);
		m_filterActor->SetMapper(mapper);
		addActor(m_filterActor);
	}

	double vtkClr[3];
	Utils::vtkColor(m_glyphColor, vtkClr);
	m_filterActor->GetProperty()->SetColor(vtkClr);
	m_filterActor->GetProperty()->SetPointSize(m_size);
	showLabels(m_labelVisible);

	applyDisplayEffect();
}

ccHObject * cvGlyphFilter::getOutput()
{
	//if (!m_glyph3d) return nullptr;
	//setResultData(m_glyph3d->GetOutput());
	//return cvGenericFilter::getOutput();

	// this filter module has no meaningful output and just for showing
	return nullptr;
}

void cvGlyphFilter::showInteractor(bool state)
{
	if (m_labelActor)
	{
		m_labelActor->SetVisibility(state);
	}
	if (m_filterActor)
	{
		m_filterActor->SetVisibility(state);
	}
}

void cvGlyphFilter::clearAllActor()
{
	if (m_labelActor)
	{
		removeActor(m_labelActor);
	}
	cvGenericFilter::clearAllActor();
}

void cvGlyphFilter::modelReady()
{
	cvGenericFilter::modelReady();
	apply();
}

void cvGlyphFilter::createUi()
{
	cvGenericFilter::createUi();

	m_configUi = new Ui::cvGlyphFilterDlg;
	setupConfigWidget(m_configUi);
}

void cvGlyphFilter::initFilter()
{
	cvGenericFilter::initFilter();
	if (m_configUi)
	{
		m_configUi->displayEffectCombo->setCurrentIndex(DisplayEffect::Transparent);
	}
}

void cvGlyphFilter::dataChanged()
{
	apply();
}

void cvGlyphFilter::on_sizeSpinBox_valueChanged(double arg1)
{
	m_size = arg1;
	apply();
}

void cvGlyphFilter::on_shapeCombo_currentIndexChanged(int index)
{
	m_shape = static_cast<Shape>(index);
	apply();
}

void cvGlyphFilter::onColorChanged(const QColor& clr)
{
	Widgets::ColorPushButton* button = qobject_cast<Widgets::ColorPushButton*>(sender());

	if (button == m_configUi->glyphColorButton)
		m_glyphColor = clr;
	else if (button == m_configUi->labelColorButton) {
		m_labelColor = clr;
		setLabelsColor(clr);
	}

	apply();
}

void cvGlyphFilter::on_displayEffectCombo_currentIndexChanged(int index)
{
	setDisplayEffect(static_cast<DisplayEffect>(index));
}

void cvGlyphFilter::on_labelGroupBox_toggled(bool arg1)
{
	m_labelVisible = arg1;
	showLabels(arg1);
}

void cvGlyphFilter::showLabels(bool show)
{
	if (!show && !m_labelActor)
		return;

	if (m_labelActor && !show) {
		m_labelActor->SetVisibility(show);
		update();
		return;
	}

	if (!m_labelActor)
	{
		VtkUtils::vtkInitOnce(m_labelActor);
		addActor(m_labelActor);
		CVLog::Warning(QString("showing labels may reduce efficiency!"));
	}

	VTK_CREATE(vtkLabeledDataMapper, labelMapper);
	labelMapper->SetFieldDataName("Isovalues");
	labelMapper->SetInputData(m_dataObject);
	labelMapper->SetLabelMode(m_labelMode);
	labelMapper->SetLabelFormat("%6.2f");

	m_labelActor->SetMapper(labelMapper);
	m_labelActor->SetVisibility(show);
	update();
}

void cvGlyphFilter::setLabelsColor(const QColor& clr)
{
	if (!m_labelActor)
		return;

	double vtkClr[3];
	Utils::vtkColor(clr, vtkClr);
	m_labelActor->GetProperty()->SetColor(vtkClr);
	update();
}

void cvGlyphFilter::on_modeCombo_currentIndexChanged(int index)
{
	m_labelMode = index;
	showLabels();
}

#include "cvClipFilter.h"

#include "VtkUtils/utils.h"
#include <VtkUtils/vtkutils.h>
#include <vtkClipPolyData.h>
#include <vtkClipDataSet.h>
#include <vtkLODActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkDataSetMapper.h>
#include <vtkRenderer.h>
#include <vtkBox.h>
#include <vtkSphere.h>
#include <vtkPlane.h>
#include <vtkPlanes.h>
#include <vtkLookupTable.h>

cvClipFilter::cvClipFilter(QWidget *parent) : cvCutFilter(parent)
{
	setWindowTitle(tr("Clip"));
}

cvClipFilter::~cvClipFilter()
{
}

void cvClipFilter::clearAllActor()
{
	cvCutFilter::clearAllActor();
}

void cvClipFilter::apply()
{
	if (!m_dataObject || m_keepMode)
		return;

	if (isValidPolyData()) {

		VtkUtils::vtkInitOnce(m_PolyClip);
		m_PolyClip->SetInputData(m_dataObject);
		
		switch (cutType()) {
		case cvCutFilter::Plane:
		{
			VTK_CREATE(vtkPlane, plane);
			plane->SetOrigin(m_origin);
			plane->SetNormal(m_normal);
			m_PolyClip->SetClipFunction(plane);
		}
		break;

		case cvCutFilter::Box:
		{
			m_PolyClip->SetClipFunction(m_planes);
		}
		break;

		case cvCutFilter::Sphere:
		{
			VTK_CREATE(vtkSphere, sphere);
			sphere->SetCenter(m_center);
			sphere->SetRadius(m_radius);
			m_PolyClip->SetClipFunction(sphere);
		}
		break;
		}

		m_negative ? m_PolyClip->InsideOutOn() : m_PolyClip->InsideOutOff();
		m_PolyClip->Update();

		if (!m_filterActor)
		{
			VtkUtils::vtkInitOnce(m_filterActor);
			VTK_CREATE(vtkPolyDataMapper, mapper);
			mapper->SetInputConnection(m_PolyClip->GetOutputPort());
			m_filterActor->SetMapper(mapper);
			addActor(m_filterActor);
		}
	}
	else if (isValidDataSet()) {

		VtkUtils::vtkInitOnce(m_DataSetClip);
		m_DataSetClip->SetInputData(m_dataObject);

		switch (cutType()) {
		case cvCutFilter::Plane:
		{
			VTK_CREATE(vtkPlane, plane);
			plane->SetOrigin(m_origin);
			plane->SetNormal(m_normal);
			m_DataSetClip->SetClipFunction(plane);
		}
		break;

		case cvCutFilter::Box:
		{
			VTK_CREATE(vtkBox, box);
		}
		break;

		case cvCutFilter::Sphere:
		{
			VTK_CREATE(vtkSphere, sphere);
			sphere->SetCenter(m_center);
			sphere->SetRadius(m_radius);
			m_DataSetClip->SetClipFunction(sphere);
		}
		break;
		}

		m_DataSetClip->Update();

		if (!m_filterActor)
		{
			VtkUtils::vtkInitOnce(m_filterActor);
			VTK_CREATE(vtkDataSetMapper, mapper);
			mapper->SetInputConnection(m_DataSetClip->GetOutputPort());
			m_filterActor->SetMapper(mapper);
			addActor(m_filterActor);
		}
	}
	
	applyDisplayEffect();
}

ccHObject * cvClipFilter::getOutput()
{
	if (isValidPolyData() && m_PolyClip)
	{
		// set exported polydata
		setResultData(m_PolyClip->GetOutput());

		// enable Clipped Output
		m_PolyClip->GenerateClippedOutputOn();

		// update remaining part
		m_negative ? m_PolyClip->InsideOutOn() : m_PolyClip->InsideOutOff();
		m_PolyClip->Update();
		m_dataObject->DeepCopy(m_PolyClip->GetClippedOutput());

		// disable Clipped Output
		m_PolyClip->GenerateClippedOutputOff();

	}
	else if (isValidDataSet() && m_DataSetClip)
	{
		// set exported DataSet
		setResultData((vtkDataObject*)m_DataSetClip->GetOutput());

		// enable Clipped Output
		m_DataSetClip->GenerateClippedOutputOn();

		// update remaining part
		m_negative ? m_DataSetClip->InsideOutOn() : m_DataSetClip->InsideOutOff();
		m_DataSetClip->Update();
		m_dataObject->DeepCopy((vtkDataObject*)m_DataSetClip->GetClippedOutput());

		// disable Clipped Output
		m_DataSetClip->GenerateClippedOutputOff();
	}

	// export clipping data
	return cvGenericFilter::getOutput();
}

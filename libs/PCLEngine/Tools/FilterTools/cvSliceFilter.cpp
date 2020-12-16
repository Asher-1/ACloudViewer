#include "cvSliceFilter.h"

#include "ui_cvCutFilterDlg.h"
#include "ui_cvGenericFilterDlg.h"

#include "PclUtils/vtk2cc.h"
#include <VtkUtils/utils.h>
#include <VtkUtils/vtkutils.h>

#include <vtkLODActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkDataSetMapper.h>
#include <vtkRenderer.h>
#include <vtkBox.h>
#include <vtkSphere.h>
#include <vtkCutter.h>
#include <vtkPlane.h>
#include <vtkPlanes.h>
#include <vtkLookupTable.h>
#include <vtkStripper.h>

// CV_CORE_LIB
#include <CVLog.h>

// ECV_DB_LIB
#include <ecvMesh.h>
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

cvSliceFilter::cvSliceFilter(QWidget* parent) : cvCutFilter(parent)
{
	setWindowTitle(tr("Slice"));
}

void cvSliceFilter::clearAllActor()
{
	cvCutFilter::clearAllActor();
}

void cvSliceFilter::apply()
{
	if (!m_dataObject || m_keepMode)
		return;

	VtkUtils::vtkInitOnce(m_cutter);
	m_cutter->SetInputData(m_dataObject);

	switch (cutType()) {
	case cvCutFilter::Plane:
	{
		VTK_CREATE(vtkPlane, plane);
		plane->SetOrigin(m_origin);
		plane->SetNormal(m_normal);
		m_cutter->SetCutFunction(plane);
	}
		break;

	case cvCutFilter::Box:
	{
		m_cutter->SetCutFunction(m_planes);
	}
		break;

	case cvCutFilter::Sphere:
	{
		VTK_CREATE(vtkSphere, sphere);
		sphere->SetCenter(m_center);
		sphere->SetRadius(m_radius);
		m_cutter->SetCutFunction(sphere);
	}
		break;
	}

	if (!m_filterActor)
	{
		VtkUtils::vtkInitOnce(m_filterActor);
		vtkSmartPointer<vtkLookupTable> lut = createLookupTable(scalarMin(), scalarMax());
		VTK_CREATE(vtkPolyDataMapper, mapper);
		mapper->SetInputConnection(m_cutter->GetOutputPort());
		mapper->SetLookupTable(lut);
		m_filterActor->SetMapper(mapper);
		addActor(m_filterActor);
	}
	
	// update contour lines
	if (m_contourLinesActor && m_contourLinesActor->GetVisibility())
		showContourLines();

	applyDisplayEffect();
}

ccHObject * cvSliceFilter::getOutput()
{
	if (!m_cutter) return nullptr;

	// update Cutter
	m_cutter->GenerateCutScalarsOff();
	m_cutter->Update();

	// get cut strips polyData
	VtkUtils::vtkInitOnce(m_cutStrips);
	m_cutStrips->SetInputData(m_cutter->GetOutput());
	m_cutStrips->Update();
	vtkPolyData * polydata = m_cutStrips->GetOutput();
    if (nullptr == polydata) return nullptr;
	ccHObject* result = new ccHObject();
	ccHObject::Container container = vtk2ccConverter().getMultiPolylinesFromPolyData(polydata, "Slice", ecvColor::green);
	
	if (!container.empty() && m_entity)
	{
		for (auto & obj : container)
		{
			if (!obj)
			{
				continue;
			}
			ccPolyline* poly = ccHObjectCaster::ToPolyline(obj);
			if (!poly) continue;

			if (m_entity->isKindOf(CV_TYPES::POINT_CLOUD))
			{
				ccPointCloud* ccCloud = ccHObjectCaster::ToPointCloud(m_entity);
				poly->setGlobalScale(ccCloud->getGlobalScale());
				poly->setGlobalShift(ccCloud->getGlobalShift());
			}
			else if (m_entity->isKindOf(CV_TYPES::MESH))
			{
				ccMesh* mesh = ccHObjectCaster::ToMesh(m_entity);
				poly->setGlobalScale(mesh->getAssociatedCloud()->getGlobalScale());
				poly->setGlobalShift(mesh->getAssociatedCloud()->getGlobalShift());
			}

			poly->setName(m_entity->getName() + "-" + poly->getName());
			result->addChild(poly);
		}
	}
	else
	{
		delete result;
		result = nullptr;
	}
	return result;
}

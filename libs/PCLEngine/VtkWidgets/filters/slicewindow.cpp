#include "slicewindow.h"

#include "ui_generalfilterwindow.h"
#include "ui_cutconfig.h"

#include <VtkUtils/utils.h>

#include <VtkUtils/vtkutils.h>
#include <VtkUtils/vtkwidget.h>

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

#include <QDebug>
#include <ecvHObject.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

SliceWindow::SliceWindow(QWidget* parent) : CutWindow(parent)
{
	setWindowTitle(QString("Slice"));
}

void SliceWindow::apply()
{
	if (!m_dataObject)
		return;

	VTK_CREATE(vtkCutter, cutter);
	cutter->SetInputData(m_dataObject);

	switch (cutType()) {
	case CutWindow::Plane:
	{
		VTK_CREATE(vtkPlane, plane);
		plane->SetOrigin(m_origin);
		plane->SetNormal(m_normal);
		cutter->SetCutFunction(plane);
	}
		break;

	case CutWindow::Box:
	{
		cutter->SetCutFunction(m_planes);
	}
		break;

	case CutWindow::Sphere:
	{
		VTK_CREATE(vtkSphere, sphere);
		sphere->SetCenter(m_center);
		sphere->SetRadius(m_radius);
		cutter->SetCutFunction(sphere);
	}
		break;
	}

	VtkUtils::vtkInitOnce(m_filterActor);
	if (m_outputMode)
	{
		cutter->GenerateCutScalarsOff();
		cutter->Update();
		VtkUtils::vtkInitOnce(m_cutStrips);
		m_cutStrips->SetInputData(cutter->GetOutput());
		m_cutStrips->Update();
		
		setResultData(m_cutStrips->GetOutput());
		setOutputMode(false);
	}

	vtkSmartPointer<vtkLookupTable> lut = createLookupTable(scalarMin(), scalarMax());

	VTK_CREATE(vtkPolyDataMapper, mapper);
	mapper->SetInputConnection(cutter->GetOutputPort());
	mapper->SetLookupTable(lut);
	m_filterActor->SetMapper(mapper);

	m_vtkWidget->defaultRenderer()->AddActor(m_filterActor);
	m_vtkWidget->update();

	// update contour lines
	if (m_contourLinesActor && m_contourLinesActor->GetVisibility())
		showContourLines();

	applyDisplayEffect();
}

ccHObject * SliceWindow::getOutput() const
{
	vtkDataObject* vtkData = resultData();
	if (!vtkData)
	{
		return nullptr;
	}
	vtkPolyData * polydata = vtkPolyData::SafeDownCast(vtkData);
	if (NULL == polydata)
	{
		return nullptr;
	}

	ccHObject* container = new ccHObject();

	int iCells = polydata->GetNumberOfCells();
	for (int i = 0; i < iCells; i++)
	{
		ccPointCloud* vertices = 0;
		vtkCell* cell = polydata->GetCell(i);
		vtkIdType ptsCount = cell->GetNumberOfPoints();
		if (ptsCount > 0)
		{
			vertices = new ccPointCloud("vertices");
			if (!vertices->reserve(ptsCount))
			{
				CVLog::Error("not enough memory to allocate vertices...");
				return nullptr;
			}

			for (vtkIdType iPt = 0; iPt < ptsCount; ++iPt)
			{
                CCVector3 P = CCVector3::fromArray(cell->GetPoints()->GetPoint(iPt));
				vertices->addPoint(P);
			}
			// end POINTS
		}
		if (vertices && vertices->size() == 0)
		{
			delete vertices;
			vertices = 0;
		}

		if (vertices)
		{
			container->addChild(vertices);
			vertices->setVisible(true);
			if (vertices->hasNormals())
				vertices->showNormals(true);
			if (vertices->hasScalarFields())
			{
				vertices->setCurrentDisplayedScalarField(0);
				vertices->showSF(true);
			}
			if (vertices->hasColors())
				vertices->showColors(true);
		}
	}

	return container;
}


#include "cvGenericFilter.h"

#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif

// LOCAL
#include "PclUtils/vtk2cc.h"
#include "PclUtils/PCLVis.h"
#include "VtkUtils/vtkutils.h"

// CV_CORE_LIB
#include <CVLog.h>

// ECV_DB_LIB
#include <ecvBBox.h>
#include <ecvHObject.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvDisplayTools.h>

// VTK
#include <vtkRenderWindowInteractor.h>
#include <vtkDataSetMapper.h>
#include <vtkPolyDataMapper.h>
#include <vtkDataArray.h>
#include <vtkActor.h>
#include <vtkProp.h>
#include <vtkDataSet.h>
#include <vtk3DWidget.h>
#include <vtkUnstructuredGrid.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkAxesActor.h>
#include <vtkScalarBarActor.h>
#include <vtkLookupTable.h>
#include <vtkTextProperty.h>
#include <vtkLODActor.h>
#include <vtkOutlineFilter.h>

cvGenericFilter::cvGenericFilter(QWidget *parent) 
	: QWidget(parent)
	, m_ui(new Ui::GenericFilterDlg)
	, m_keepMode(false)
	, m_negative(false)
	, m_meshMode(false)
	, m_preview(true)
{
	setWindowTitle(tr("GenericFilter"));
	connect(ecvDisplayTools::TheInstance(), &ecvDisplayTools::doubleButtonClicked,
			this, &cvGenericFilter::onDoubleClick);
}

cvGenericFilter::~cvGenericFilter()
{
	VtkUtils::vtkSafeDelete(m_dataObject);
	delete m_ui;
}

void cvGenericFilter::onDoubleClick(int x, int y)
{
	applyDisplayEffect();
}

////////////////////Initialization///////////////////////////
void cvGenericFilter::start()
{
	// call child function
	modelReady();

	// update data
	dataChanged();

	// according to data type
	initFilter();

	// update screen
	update();
}

bool cvGenericFilter::setInput(ccHObject * obj)
{
	m_entity = obj;
	m_id = QString::number(m_entity->getUniqueID()).toStdString();

	if (m_entity->isKindOf(CV_TYPES::POINT_CLOUD))
	{
		m_meshMode = false;
	}
	else if (m_entity->isKindOf(CV_TYPES::MESH))
	{
		m_meshMode = true;
	}
	else
	{
		CVLog::Error("only mesh and point cloud are supported!");
		return false;
	}

	if (!initModel())
	{
		return false;
	}

	updateUi();
	return true;
}

bool cvGenericFilter::initModel()
{
	if (m_dataObject) {
		m_dataObject->Delete();
		m_dataObject = nullptr;
	}
	assert(m_viewer);
	m_modelActor = m_viewer->getActorById(m_id);
	if (!m_modelActor)
	{
		return false;
	}

	vtkPolyData* polydata = reinterpret_cast<vtkPolyDataMapper*>(m_modelActor->GetMapper())->GetInput();
	if (!polydata)
	{
		return false;
	}

	int npoints = static_cast<int>(polydata->GetNumberOfPoints());
	if (npoints > MAX_PREVIEW_NUMBER)
	{
		m_preview = false;
	}
	else
	{
		m_preview = true;
	}

	if (!m_dataObject)
	{
		m_dataObject = vtkPolyData::New();
	}
	m_dataObject->DeepCopy(polydata);

	if (!m_resultData)
	{
		m_resultData = vtkPolyData::New();
	}

	return true;
}

ccHObject * cvGenericFilter::getOutput()
{
	vtkDataObject* vtkData = resultData();
	if (!vtkData)
	{
		return nullptr;
	}

	vtkPolyData * polydata = vtkPolyData::SafeDownCast(vtkData);
    if (!polydata)
	{
		return nullptr;
	}

	ccHObject* result;
	if (m_meshMode)
	{
		result = vtk2ccConverter().getMeshFromPolyData(polydata);
		if (!result)
		{
			CVLog::Warning(QString("try to save in cloud format"));
			result = vtk2ccConverter().getPointCloudFromPolyData(polydata);
		}
	}
	else
	{
		result = vtk2ccConverter().getPointCloudFromPolyData(polydata);
	}

	return result;
}

void cvGenericFilter::getOutput(
	std::vector<ccHObject*>& outputSlices,
	std::vector<ccPolyline*>& outputContours)
{
    outputContours.clear();
	ccHObject* slices = getOutput();
	if (slices)
	{
		outputSlices.push_back(slices);
	}
}

void cvGenericFilter::modelReady()
{
	showOutline(false);
	ecvDisplayTools::UpdateCamera();
}

void cvGenericFilter::setUpViewer(PclUtils::PCLVis* viewer)
{
	if (!viewer) return;
	m_viewer = viewer;
	setInteractor(viewer->getRenderWindowInteractor());
}

void cvGenericFilter::getInteractorInfos(ccBBox & bbox, ccGLMatrixd & trans)
{
	getInteractorBounds(bbox);
	getInteractorTransformation(trans);
}

void cvGenericFilter::colorsChanged()
{
	vtkSmartPointer<vtkLookupTable> lut = createLookupTable(m_scalarMin, m_scalarMax);
	if (m_scalarBar)
		m_scalarBar->SetLookupTable(lut);
	if (m_modelActor)
		m_modelActor->GetMapper()->SetLookupTable(lut);
	if (m_filterActor)
		m_filterActor->GetMapper()->SetLookupTable(lut);
	if (m_viewer)
		update();
}

////////////////////Visualization///////////////////////////
void cvGenericFilter::update()
{
	QWidget::update();
	ecvDisplayTools::UpdateScreen();
}

void cvGenericFilter::reset()
{
	dataChanged();
	update();
}

void cvGenericFilter::restoreOrigin()
{
	if (!m_viewer) return;
	if (initModel())
	{
		start();
	}
}

void cvGenericFilter::setDisplayEffect(cvGenericFilter::DisplayEffect effect)
{
	if (m_displayEffect != effect) {
		m_displayEffect = effect;
		applyDisplayEffect();
	}
}

cvGenericFilter::DisplayEffect cvGenericFilter::displayEffect() const
{
	return m_displayEffect;
}

void cvGenericFilter::updateSize()
{
	adjustSize();
	QWidget* widget = topLevelWidget();
	if (widget)
	{
		widget->adjustSize();
		if (widget->topLevelWidget())
		{
			widget->topLevelWidget()->adjustSize();
		}
	}
}

void cvGenericFilter::UpdateScalarRange()
{
	double scalarRange[2];
	if (isValidPolyData()) {
		vtkPolyData* polyData = vtkPolyData::SafeDownCast(m_dataObject);
		polyData->GetScalarRange(scalarRange);
	} else if (isValidDataSet()) {
		vtkDataSet* dataSet = vtkDataSet::SafeDownCast(m_dataObject);
		dataSet->GetScalarRange(scalarRange);
	}
	setScalarRange(scalarRange[0], scalarRange[1]);
}

void cvGenericFilter::applyDisplayEffect()
{
	if (m_modelActor) {
		switch (m_displayEffect) {
		case Transparent:
			m_modelActor->GetProperty()->SetOpacity(0.3);
			m_modelActor->SetVisibility(1);
			m_modelActor->GetProperty()->SetRepresentationToSurface();
			break;

		case Opaque:
			m_modelActor->SetVisibility(0);
			break;

		case Points:
			m_modelActor->SetVisibility(1);
			m_modelActor->GetProperty()->SetRepresentationToPoints();
			break;

		case Wireframe:
			m_modelActor->SetVisibility(1);
			m_modelActor->GetProperty()->SetRepresentationToWireframe();
			break;
		}

		vtkSmartPointer<vtkLookupTable> lut = createLookupTable(m_scalarMin, m_scalarMax);
		m_modelActor->GetMapper()->SetLookupTable(lut);
		update();
	}
}

void cvGenericFilter::showScalarBar(bool show)
{
	if (!m_scalarBar) {
		vtkSmartPointer<vtkLookupTable> lut = createLookupTable(m_scalarMin, m_scalarMax);
		lut->Build();

		m_scalarBar = vtkScalarBarActor::New();
		m_scalarBar->SetPosition(0.0, 0.0);
		m_scalarBar->SetWidth(.1); // fraction of window width
		m_scalarBar->SetHeight(.8); // fraction of window height
		m_scalarBar->GetTitleTextProperty()->SetFontSize(3);
		m_scalarBar->GetTitleTextProperty()->SetBold(0);
		m_scalarBar->GetTitleTextProperty()->SetItalic(0);
		m_scalarBar->SetOrientationToVertical();
		m_scalarBar->SetTitle("Scalar");
		m_scalarBar->SetLookupTable(lut);
		addActor(m_scalarBar);
	}

	m_scalarBar->SetVisibility(show);
	update();
}

void cvGenericFilter::showOutline(bool show)
{
	if (!m_dataObject) {
		CVLog::Error(QString("cvGenericFilter::showOutline: null data object, quit."));
		return;
	}

	if (!m_outlineActor) 
	{
		VTK_CREATE(vtkOutlineFilter, outline);
		outline->SetInputData(m_dataObject);

		VTK_CREATE(vtkPolyDataMapper, mapper);
		mapper->SetInputConnection(outline->GetOutputPort());

		VtkUtils::vtkInitOnce(m_outlineActor);
		m_outlineActor->SetMapper(mapper);
		addActor(m_outlineActor);
	}

	m_outlineActor->SetVisibility(show);
	update();
}

void cvGenericFilter::setOutlineColor(const QColor &clr)
{
	if (!m_outlineActor)
		return;

	double vtkClr[3];
	Utils::vtkColor(clr, vtkClr);
	m_outlineActor->GetProperty()->SetColor(vtkClr);
	update();
}

void cvGenericFilter::setScalarBarColors(const QColor& clr1, const QColor& clr2)
{
	if (m_color1 == clr1 && m_color2 == clr2)
		return;

	m_color1 = clr1;
	m_color2 = clr2;
	colorsChanged();
}

QColor cvGenericFilter::color1() const
{
	return m_color1;
}

QColor cvGenericFilter::color2() const
{
	return m_color2;
}

void cvGenericFilter::setScalarRange(double min, double max)
{
	m_scalarMin = qMin(min, max);
	m_scalarMax = qMax(min, max);
}

double cvGenericFilter::scalarMin() const
{
	return m_scalarMin;
}

double cvGenericFilter::scalarMax() const
{
	return m_scalarMax;
}

vtkSmartPointer<vtkDataArray> cvGenericFilter::getActorScalars(vtkSmartPointer<vtkActor> actor)
{
	if (!actor) return nullptr;

	return actor->GetMapper()->GetInput()->GetPointData()->GetScalars();
}

int cvGenericFilter::getDefaultScalarInterpolationForDataSet(vtkDataSet * data)
{
    vtkPolyData* polyData = vtkPolyData::SafeDownCast(data); // Check that polyData != nullptr in case of segfault
	return (polyData && polyData->GetNumberOfCells() != polyData->GetNumberOfVerts());
}

vtkSmartPointer<vtkLookupTable> cvGenericFilter::createLookupTable(double min, double max)
{
	double hsv1[3];
	double hsv2[3];
	Utils::qColor2HSV(m_color1, hsv1);
	Utils::qColor2HSV(m_color2, hsv2);

	VTK_CREATE(vtkLookupTable, lut);
	lut->SetHueRange(hsv1[0], hsv2[0]);
	lut->SetSaturationRange(hsv1[1], hsv2[1]);
	lut->SetValueRange(hsv1[2], hsv2[2]);
	lut->SetTableRange(min, max);
	lut->Build();

	return lut;
}

void cvGenericFilter::addActor(const vtkSmartPointer<vtkProp> actor)
{
	if (!m_viewer) return;
	m_viewer->addActorToRenderer(actor);
}

void cvGenericFilter::removeActor(const vtkSmartPointer<vtkProp> actor)
{
	if (!m_viewer) return;
	m_viewer->removeActorFromRenderer(actor);
}

void cvGenericFilter::clearAllActor()
{
	if (!m_viewer) return;

	if (m_modelActor)
	{
		m_modelActor->GetProperty()->SetOpacity(1.0);
		m_modelActor->SetVisibility(1);
		if (m_meshMode)
		{
			m_modelActor->GetProperty()->SetRepresentationToSurface();
		}
	}

	if (m_outlineActor)
	{
		removeActor(m_outlineActor);
	}

	if (m_scalarBar)
	{
		removeActor(m_scalarBar);
	}

	if (m_filterActor)
	{
		removeActor(m_filterActor);
	}
}


////////////////////Util Function///////////////////////////
template <class DataObject, class Mapper>
void cvGenericFilter::createActorFromData(vtkDataObject* dataObj)
{
	if (!dataObj)
		return;

	m_dataObject = dataObj;
	DataObject* data = DataObject::SafeDownCast(dataObj);

	if (!data) {
		CVLog::Error(QString("ansys parsing: null data set."));
		return;
	}

	VTK_CREATE(Mapper, mapper);
	mapper->SetInputData(data);
	mapper->Update();

//	VtkUtils::vtkInitOnce(m_modelActor);
    m_modelActor = vtkSmartPointer<vtkLODActor>::New();
	m_modelActor->SetMapper(mapper);

    static_cast<vtkSmartPointer<vtkLODActor>>(m_modelActor)->SetNumberOfCloudPoints(int(std::max<vtkIdType>(1, data->GetNumberOfPoints() / 10)));
	m_modelActor->GetProperty()->SetInterpolationToFlat();

	addActor(m_modelActor);
	modelReady();

	update();
	dataChanged();
}

void cvGenericFilter::setInteractor(vtkRenderWindowInteractor * interactor)
{
	m_interactor = interactor;
}

void cvGenericFilter::setResultData(vtkSmartPointer<vtkDataObject> data)
{
	m_resultData->DeepCopy(data);
}

vtkSmartPointer<vtkDataObject> cvGenericFilter::resultData() const
{
	return m_resultData;
}

bool cvGenericFilter::isValidPolyData() const
{
	return vtkPolyData::SafeDownCast(m_dataObject) != nullptr;
}

bool cvGenericFilter::isValidDataSet() const
{
	return vtkDataSet::SafeDownCast(m_dataObject) != nullptr;
}

void cvGenericFilter::safeOff(vtk3DWidget* widget)
{
	if (widget)
		widget->Off();
}

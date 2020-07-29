#include "cvCutFilter.h"
#include "ui_cvCutFilterDlg.h"
#include "ui_cvGenericFilterDlg.h"

#include <VtkUtils/utils.h>
#include <VtkUtils/gradientcombobox.h>
#include <VtkUtils/signalblocker.h>

#include <VtkUtils/vtkutils.h>
#include <VtkUtils/spherewidgetobserver.h>
#include <VtkUtils/implicitplanewidgetobserver.h>
#include <VtkUtils/boxwidgetobserver.h>

#include <vtkBoxWidget.h>
#include <vtkImplicitPlaneWidget.h>
#include <vtkSphereWidget.h>
#include <vtkTransform.h>
#include <vtkProperty.h>
#include <vtkPlanes.h>
#include <vtkPlane.h>
#include <vtkContourFilter.h>
#include <vtkRenderer.h>
#include <vtkCleanPolyData.h>
#include <vtkAppendPolyData.h>
#include <vtkClipPolyData.h>
#include <vtkFloatArray.h>
#include <vtkCellData.h>
#include <vtkLookupTable.h>
#include <vtkLODActor.h>
#include <QWidget>

// CV_CORE_LIB
#include <CVLog.h>

// ECV_DB_LIB
#include <ecvBBox.h>

cvCutFilter::cvCutFilter(QWidget* parent)
	: cvGenericFilter(parent)
{
	setWindowTitle(tr("Cut"));
	createUi();
}

void cvCutFilter::clearAllActor()
{
	if (m_contourLinesActor)
	{
		removeActor(m_contourLinesActor);
	}

	if (m_boxWidget)
	{
		safeOff(m_boxWidget);
	}

	if (m_planeWidget)
	{
		safeOff(m_planeWidget);
	}

	if (m_sphereWidget)
	{
		safeOff(m_sphereWidget);
	}

	cvGenericFilter::clearAllActor();
}

void cvCutFilter::createUi()
{
	cvGenericFilter::createUi();

	m_configUi = new Ui::CutFilterDlg;
	setupConfigWidget(m_configUi);
	m_configUi->planeGroupBox->hide();
	m_configUi->sphereGroupBox->show();
	setNegative(m_configUi->negativeCheckBox->isChecked());
}


void cvCutFilter::showInteractor(bool state)
{
	switch (m_cutType)
	{
	case cvCutFilter::Box:
		if (m_boxWidget)
		{
			state ? m_boxWidget->On() : safeOff(m_boxWidget);
		}
		break;
	case cvCutFilter::Plane:
		if (m_planeWidget)
		{
			state ? m_planeWidget->On() : safeOff(m_planeWidget);
		}
		break;
	case cvCutFilter::Sphere:
		if (m_sphereWidget)
		{
			state ? m_sphereWidget->On() : safeOff(m_sphereWidget);
		}
		break;
	default:
		break;
	}
}

void cvCutFilter::getInteractorBounds(ccBBox & bbox)
{
	double bounds[6];

	bool valid = true;
	switch (m_cutType)
	{
	case cvCutFilter::Box:
		if (m_boxWidget)
		{
			m_boxWidget->GetProp3D()->GetBounds(bounds);
		}
		break;
	case cvCutFilter::Plane:
		if (m_planeWidget)
		{
			m_planeWidget->GetProp3D()->GetBounds(bounds);
		}
		break;
	case cvCutFilter::Sphere:
		if (m_sphereWidget)
		{
			m_sphereWidget->GetProp3D()->GetBounds(bounds);
		}
		break;
	default:
		valid = false;
		break;
	}

	if (!valid)
	{
		bbox.setValidity(valid);
		return;
	}

	CCVector3 minCorner(bounds[0], bounds[2], bounds[4]);
	CCVector3 maxCorner(bounds[1], bounds[3], bounds[5]);
	bbox.minCorner() = minCorner;
	bbox.maxCorner() = maxCorner;
	bbox.setValidity(valid);
}

void cvCutFilter::getInteractorTransformation(ccGLMatrixd & trans)
{
	switch (m_cutType)
	{
	case cvCutFilter::Box:
		if (m_boxWidget)
		{
			m_boxWidget->GetProp3D()->GetMatrix(trans.data());
		}
		break;
	case cvCutFilter::Plane:
		if (m_planeWidget)
		{
			m_planeWidget->GetProp3D()->GetMatrix(trans.data());
		}
		break;
	case cvCutFilter::Sphere:
		if (m_sphereWidget)
		{
			m_sphereWidget->GetProp3D()->GetMatrix(trans.data());
		}
		break;
	default:
		break;
	}
}

void cvCutFilter::shift(const CCVector3d & v)
{
	switch (m_cutType)
	{
	case cvCutFilter::Box:
		if (m_boxWidget)
		{
			VTK_CREATE(vtkTransform, trans);
			m_boxWidget->GetTransform(trans);
			trans->Translate(v.u);
			m_boxWidget->SetTransform(trans);
			apply();
		}
		break;
	case cvCutFilter::Plane:
		if (m_planeWidget)
		{
			// Modify and update planeWidget
		}
		break;
	case cvCutFilter::Sphere:
		if (m_sphereWidget)
		{
			double newPos[3];
			CCVector3d::vadd(v.u, m_center, newPos);
			onCenterChanged(newPos);
		}
		break;
	default:
		break;
	}
}

void cvCutFilter::updateCutWidget()
{
	double bounds[6];
	if (isValidPolyData()) {
		vtkPolyData* polyData = vtkPolyData::SafeDownCast(m_dataObject);
		polyData->GetBounds(bounds);
	}
	else if (isValidDataSet()) {
		vtkDataSet* dataSet = vtkDataSet::SafeDownCast(m_dataObject);
		dataSet->GetBounds(bounds);
	}

	switch (m_cutType) {
	case Plane:
	{
		if (!m_planeWidget) {
			VtkUtils::vtkInitOnce(m_planeWidget);
			m_planeWidget->SetInteractor(getInteractor());
			m_planeWidget->SetPlaceFactor(1);
			m_planeWidget->SetProp3D(m_modelActor);
			m_planeWidget->SetOutlineTranslation(0); // make the outline non-movable
			m_planeWidget->GetPlaneProperty()->SetOpacity(0.7);
			m_planeWidget->GetPlaneProperty()->SetColor(.0, .0, .0);

			VtkUtils::ImplicitPlaneWidgetObserver* observer = new VtkUtils::ImplicitPlaneWidgetObserver(this);
			connect(observer, SIGNAL(originChanged(double*)), this, SLOT(onOriginChanged(double*)));
			connect(observer, SIGNAL(normalChanged(double*)), this, SLOT(onNormalChanged(double*)));
			observer->attach(m_planeWidget);

			resetPlaneWidget();
		}

		m_planeWidget->On();
		m_planeWidget->SetOrigin(m_origin);
		m_planeWidget->SetNormal(m_normal);
		m_planeWidget->PlaceWidget(bounds);

		safeOff(m_sphereWidget);
		safeOff(m_boxWidget);

		m_configUi->planeGroupBox->show();
		m_configUi->sphereGroupBox->hide();
		cvGenericFilter::updateSize();
	}
	break;

	case Sphere:
	{
		if (!m_sphereWidget) {
			VtkUtils::vtkInitOnce(m_sphereWidget);
			m_sphereWidget->SetInteractor(getInteractor());
			m_sphereWidget->SetProp3D(m_modelActor);
			m_sphereWidget->PlaceWidget();

			VtkUtils::SphereWidgetObserver* observer = new VtkUtils::SphereWidgetObserver(this);
			observer->attach(m_sphereWidget);
			connect(observer, SIGNAL(centerChanged(double*)), this, SLOT(onCenterChanged(double*)));
			connect(observer, SIGNAL(radiusChanged(double)), this, SLOT(onRadiusChanged(double)));
			resetSphereWidget();
		}

		m_sphereWidget->On();
		m_sphereWidget->PlaceWidget(bounds);
		m_sphereWidget->SetCenter(m_center);
		m_sphereWidget->SetRadius(m_radius);
		safeOff(m_planeWidget);
		safeOff(m_boxWidget);

		m_configUi->planeGroupBox->hide();
		m_configUi->sphereGroupBox->show();
		cvGenericFilter::updateSize();
	}
	break;

	case Box:
	{
		if (!m_boxWidget) {
			VtkUtils::vtkInitOnce(m_boxWidget);
			m_boxWidget->SetInteractor(getInteractor());
			m_boxWidget->SetPlaceFactor(1.0);

			m_boxWidget->SetProp3D(m_modelActor);
			m_boxWidget->PlaceWidget();

			VtkUtils::BoxWidgetObserver* observer = new VtkUtils::BoxWidgetObserver(this);
			observer->attach(m_boxWidget);
			connect(observer, SIGNAL(planesChanged(vtkPlanes*)), this, SLOT(onPlanesChanged(vtkPlanes*)));
		}

		m_boxWidget->On();
		m_boxWidget->PlaceWidget(bounds);
		safeOff(m_planeWidget);
		safeOff(m_sphereWidget);

		m_configUi->planeGroupBox->hide();
		m_configUi->sphereGroupBox->hide();
		cvGenericFilter::updateSize();
	}
	break;
	}
}

void cvCutFilter::setNormal(double normal[3])
{
	if (Utils::ArrayComparator<double>()(normal, m_normal))
		return;

	Utils::ArrayAssigner<double>()(m_normal, normal);
	apply();
}

void cvCutFilter::setOrigin(double origin[3])
{
	if (Utils::ArrayComparator<double>()(origin, m_normal))
		return;

	Utils::ArrayAssigner<double>()(m_normal, origin);
	apply();
}

void cvCutFilter::setRadius(double radius)
{
	m_radius = radius;
}

void cvCutFilter::setCutType(CutType type)
{
	if (m_cutType != type) {
		m_cutType = type;
		updateCutWidget();
	}
}

cvCutFilter::CutType cvCutFilter::cutType() const
{
	return m_cutType;
}

void cvCutFilter::initFilter()
{
	cvGenericFilter::initFilter();

	if (!m_configUi) return;

	m_meshMode ?  m_configUi->displayEffectCombo->setCurrentIndex(DisplayEffect::Transparent) :
				  m_configUi->displayEffectCombo->setCurrentIndex(DisplayEffect::Opaque);
}

void cvCutFilter::onOriginChanged(double* origin)
{
	if (Utils::ArrayComparator<double>()(m_origin, origin))
		return;

	Utils::ArrayAssigner<double>()(m_origin, origin);

	VtkUtils::SignalBlocker sb(m_configUi->originXSpinBox);
	sb.addObject(m_configUi->originYSpinBox);
	sb.addObject(m_configUi->originZSpinBox);

	m_configUi->originXSpinBox->setValue(origin[0]);
	m_configUi->originYSpinBox->setValue(origin[1]);
	m_configUi->originZSpinBox->setValue(origin[2]);

	apply();
}

void cvCutFilter::onNormalChanged(double* normal)
{
	if (Utils::ArrayComparator<double>()(m_normal, normal))
		return;

	Utils::ArrayAssigner<double>()(m_normal, normal);

	VtkUtils::SignalBlocker sb(m_configUi->normalXSpinBox);
	sb.addObject(m_configUi->normalYSpinBox);
	sb.addObject(m_configUi->normalZSpinBox);

	m_configUi->normalXSpinBox->setValue(normal[0]);
	m_configUi->normalYSpinBox->setValue(normal[1]);
	m_configUi->normalZSpinBox->setValue(normal[2]);

	apply();
}

void cvCutFilter::onCenterChanged(double* center)
{
	if (Utils::ArrayComparator<double>()(m_center, center))
		return;

	Utils::ArrayAssigner<double>()(m_center, center);

	VtkUtils::SignalBlocker sb(m_configUi->centerXSpinBox);
	sb.addObject(m_configUi->centerYSpinBox);
	sb.addObject(m_configUi->centerZSpinBox);

	m_configUi->centerXSpinBox->setValue(center[0]);
	m_configUi->centerYSpinBox->setValue(center[1]);
	m_configUi->centerZSpinBox->setValue(center[2]);

	apply();
}

void cvCutFilter::onRadiusChanged(double radius)
{
	if (m_radius == radius) return;

	setRadius(radius);

	VtkUtils::SignalBlocker sb(m_configUi->radiusSpinBox);

	m_configUi->radiusSpinBox->setValue(radius);
	apply();
}

void cvCutFilter::onPlanesChanged(vtkPlanes* planes)
{
	m_planes = planes;
	apply();
}

void cvCutFilter::showContourLines(bool show)
{
	if (!m_meshMode)
	{
		return;
	}

	if (!show && !m_dataObject)
		return;

	if (!show && m_contourLinesActor) {
		m_contourLinesActor->SetVisibility(show);
		update();
		return;
	}

	vtkPolyData* polyData = vtkPolyData::SafeDownCast(m_dataObject);
	if (!polyData)
		return;

	vtkPointData* pointData = polyData->GetPointData();

	// store contour names
	int numOfArrays = pointData->GetNumberOfArrays();
	if (!numOfArrays)
		return;

	//
	QMap<QString, vtkDataArray*> scalarNameDataMap;
	QString contourName;
	for (int i = 0; i < numOfArrays; ++i) {
		contourName = pointData->GetArrayName(i);
		scalarNameDataMap.insert(contourName, pointData->GetArray(contourName.toStdString().c_str()));
	}

	//
	VTK_CREATE(vtkDoubleArray, scalars);
	double range[2];

	// now extracting contour data
	for (int i = 0; i < numOfArrays; ++i) {
		vtkDataArray* da = pointData->GetArray(contourName.toUtf8().data());
		if (da) {
			int numOfTuple = da->GetNumberOfTuples();
			da->GetRange(range);
			for (int j = 0; j < numOfTuple; ++j) {
				double val = da->GetTuple1(j);
				scalars->InsertNextTuple1(val);
			}
			break;
		}
	}

	/// START
	polyData->GetPointData()->SetScalars(scalars);
	VTK_CREATE(vtkAppendPolyData, appendFilledContours);
	int numberOfContours = 10;

	double delta = (range[1] - range[0]) / static_cast<double> (numberOfContours - 1);

	// Keep the clippers alive
	std::vector<vtkSmartPointer<vtkClipPolyData> > clippersLo;
	std::vector<vtkSmartPointer<vtkClipPolyData> > clippersHi;

	for (int i = 0; i < numberOfContours; i++) {
		double valueLo = range[0] + static_cast<double> (i) * delta;
		double valueHi = range[0] + static_cast<double> (i + 1) * delta;
		clippersLo.push_back(vtkSmartPointer<vtkClipPolyData>::New());
		clippersLo[i]->SetValue(valueLo);
		if (i == 0)
			clippersLo[i]->SetInputData(polyData);
		else
			clippersLo[i]->SetInputConnection(clippersHi[i - 1]->GetOutputPort(1));
		clippersLo[i]->InsideOutOff();
		clippersLo[i]->Update();

		clippersHi.push_back(vtkSmartPointer<vtkClipPolyData>::New());
		clippersHi[i]->SetValue(valueHi);
		clippersHi[i]->SetInputConnection(clippersLo[i]->GetOutputPort());
		clippersHi[i]->GenerateClippedOutputOn();
		clippersHi[i]->InsideOutOn();
		clippersHi[i]->Update();
		if (clippersHi[i]->GetOutput()->GetNumberOfCells() == 0)
			continue;

		VTK_CREATE(vtkFloatArray, cd);
		cd->SetNumberOfComponents(1);
		cd->SetNumberOfTuples(clippersHi[i]->GetOutput()->GetNumberOfCells());
		cd->FillComponent(0, valueLo);

		clippersHi[i]->GetOutput()->GetCellData()->SetScalars(cd);
		appendFilledContours->AddInputConnection(clippersHi[i]->GetOutputPort());
	}

	VTK_CREATE(vtkCleanPolyData, filledContours);
	filledContours->SetInputConnection(appendFilledContours->GetOutputPort());

	setResultData(filledContours->GetOutput());

	VTK_CREATE(vtkLookupTable, lut);
	lut->SetNumberOfColors(numberOfContours);
	lut->SetRange(range[0], range[1]);
	lut->SetHueRange(0.6667, 0);
	lut->Build();

	VTK_CREATE(vtkPolyDataMapper, contourMapper);
	contourMapper->SetInputConnection(filledContours->GetOutputPort());
	contourMapper->SetScalarRange(range[0], range[1]);
	contourMapper->SetScalarModeToUseCellData();
	contourMapper->SetLookupTable(lut);

	if (!m_contourLinesActor)
		m_contourLinesActor = vtkActor::New();
	m_contourLinesActor->SetMapper(contourMapper);
	m_contourLinesActor->SetVisibility(show);

	addActor(m_contourLinesActor);
	update();
}

void cvCutFilter::modelReady()
{
	// initialize vars
	double bounds[6];
	double center[3];
	double scalarRange[2];
	if (isValidPolyData()) {
		vtkPolyData* polyData = vtkPolyData::SafeDownCast(m_dataObject);
		polyData->GetBounds(bounds);
		polyData->GetCenter(center);
		polyData->GetScalarRange(scalarRange);
	} else if (isValidDataSet()) {
		vtkDataSet* dataSet = vtkDataSet::SafeDownCast(m_dataObject);
		dataSet->GetBounds(bounds);
		dataSet->GetCenter(center);
		dataSet->GetScalarRange(scalarRange);
	}

	setScalarRange(scalarRange[0], scalarRange[1]);

	double xRange = bounds[1] - bounds[0];
	double yRange = bounds[3] - bounds[2];
	double zRange = bounds[5] - bounds[4];
	double minRange = qMin(xRange, qMin(yRange, zRange));
	setRadius((double)minRange / 2);

	Utils::ArrayAssigner<double>()(m_center, center);
	Utils::ArrayAssigner<double>()(m_origin, center);

	m_normal[0] = 1;
	m_normal[1] = 0;
	m_normal[2] = 0;

	VtkUtils::SignalBlocker sb(m_configUi->centerXSpinBox);
	sb.addObject(m_configUi->centerYSpinBox);
	sb.addObject(m_configUi->centerZSpinBox);
	sb.addObject(m_configUi->originZSpinBox);
	sb.addObject(m_configUi->originYSpinBox);
	sb.addObject(m_configUi->originZSpinBox);
	sb.addObject(m_configUi->radiusSpinBox);

	m_configUi->centerXSpinBox->setValue(center[0]);
	m_configUi->centerYSpinBox->setValue(center[1]);
	m_configUi->centerZSpinBox->setValue(center[2]);
	m_configUi->originXSpinBox->setValue(center[0]);
	m_configUi->originYSpinBox->setValue(center[1]);
	m_configUi->originZSpinBox->setValue(center[2]);
	m_configUi->radiusSpinBox->setValue(m_radius);

	if (m_meshMode)
	{
		showScalarBar(true);
	}
	
	updateCutWidget();

	cvGenericFilter::modelReady();
}

void cvCutFilter::dataChanged()
{
	switch (m_cutType) {
	case Plane:
	{
		resetPlaneWidget();
		if (m_sphereWidget)
		{
			m_normal[0] = 1;
			m_normal[1] = 0;
			m_normal[2] = 0;
			m_planeWidget->SetNormal(m_normal);
			m_planeWidget->SetOrigin(m_origin);
		}
	}
		break;

	case Sphere:
	{
		resetSphereWidget();
		if (m_sphereWidget)
		{
			m_sphereWidget->SetCenter(m_center);
			m_sphereWidget->SetRadius(m_radius);
		}
	}
		break;

	case Box:
		resetBoxWidget();
		break;
	}

	apply();
}

void cvCutFilter::resetPlaneWidget()
{
	if (!m_dataObject) return;
	if (!m_planeWidget) {
		CVLog::Error(QString("cvCutFilter::resetPlaneWidget: null plane widget."));
		return;
	}

	double bounds[6];
	double origin[3];
	if (isValidPolyData()) {
		vtkPolyData* polyData = vtkPolyData::SafeDownCast(m_dataObject);
		polyData->GetBounds(bounds);
		polyData->GetCenter(origin);
	}
	else if (isValidDataSet()) {
		vtkDataSet* dataSet = vtkDataSet::SafeDownCast(m_dataObject);
		dataSet->GetBounds(bounds);
		dataSet->GetCenter(origin);
	}
	cvGenericFilter::UpdateScalarRange();

	Utils::ArrayAssigner<double>()(m_origin, origin);
	m_planeWidget->PlaceWidget(bounds);

}

void cvCutFilter::resetSphereWidget()
{
	if (!m_dataObject) return;
	if (!m_sphereWidget) {
		CVLog::Error(QString("cvCutFilter::resetSphereWidget: null sphere widget."));
		return;
	}

	double bounds[6];
	double center[3];
	if (isValidPolyData()) {
		vtkPolyData* polyData = vtkPolyData::SafeDownCast(m_dataObject);
		polyData->GetBounds(bounds);
		polyData->GetCenter(center);
	}
	else if (isValidDataSet()) {
		vtkDataSet* dataSet = vtkDataSet::SafeDownCast(m_dataObject);
		dataSet->GetBounds(bounds);
		dataSet->GetCenter(center);
	}

	double xRange = bounds[1] - bounds[0];
	double yRange = bounds[3] - bounds[2];
	double zRange = bounds[5] - bounds[4];
	double minRange = qMin(xRange, qMin(yRange, zRange));
	setRadius((double)minRange / 2);

	Utils::ArrayAssigner<double>()(m_center, center);
	m_sphereWidget->PlaceWidget(bounds);
}

void cvCutFilter::resetBoxWidget()
{
	if (!m_dataObject) return;

	if (!m_boxWidget) {
		CVLog::Error(QString("cvCutFilter::resetBoxWidget: null box widget."));
		return;
	}

	double bounds[6];
	if (isValidPolyData()) {
		vtkPolyData* polyData = vtkPolyData::SafeDownCast(m_dataObject);
		polyData->GetBounds(bounds);

	}
	else if (isValidDataSet()) {
		vtkDataSet* dataSet = vtkDataSet::SafeDownCast(m_dataObject);
		dataSet->GetBounds(bounds);
	}
	cvGenericFilter::UpdateScalarRange();
	m_boxWidget->PlaceWidget(bounds);
	m_boxWidget->GetPlanes(m_planes);
	onPlanesChanged(m_planes);
}

void cvCutFilter::on_cutTypeCombo_currentIndexChanged(int index)
{
	setCutType(static_cast<CutType>(index));
	apply();
}

void cvCutFilter::on_displayEffectCombo_currentIndexChanged(int index)
{
	setDisplayEffect(static_cast<DisplayEffect>(index));
}

void cvCutFilter::on_radiusSpinBox_valueChanged(double arg1)
{
	setRadius(arg1);
	updateCutWidget();
	apply();
}

void cvCutFilter::on_originXSpinBox_valueChanged(double arg1)
{
	m_origin[0] = arg1;
	updateCutWidget();
	apply();
}

void cvCutFilter::on_originYSpinBox_valueChanged(double arg1)
{
	m_origin[1] = arg1;
	updateCutWidget();
	apply();
}

void cvCutFilter::on_originZSpinBox_valueChanged(double arg1)
{
	m_origin[2] = arg1;
	updateCutWidget();
	apply();
}

void cvCutFilter::on_normalXSpinBox_valueChanged(double arg1)
{
	m_normal[0] = arg1;
	updateCutWidget();
	apply();
}

void cvCutFilter::on_normalYSpinBox_valueChanged(double arg1)
{
	m_normal[1] = arg1;
	updateCutWidget();
	apply();
}

void cvCutFilter::on_normalZSpinBox_valueChanged(double arg1)
{
	m_normal[2] = arg1;
	updateCutWidget();
	apply();
}

void cvCutFilter::on_centerXSpinBox_valueChanged(double arg1)
{
	m_center[0] = arg1;
	updateCutWidget();
	apply();
}

void cvCutFilter::on_centerYSpinBox_valueChanged(double arg1)
{
	m_center[1] = arg1;
	updateCutWidget();
	apply();
}

void cvCutFilter::on_centerZSpinBox_valueChanged(double arg1)
{
	m_center[2] = arg1;
	updateCutWidget();
	apply();
}

void cvCutFilter::on_showPlaneCheckBox_toggled(bool checked)
{
	if (m_planeWidget)
		m_planeWidget->SetDrawPlane(checked);
}

void cvCutFilter::on_showContourLinesCheckBox_toggled(bool checked)
{
	showContourLines(checked);
}

void cvCutFilter::on_negativeCheckBox_toggled(bool checked)
{
	setNegative(checked);
	apply();
}

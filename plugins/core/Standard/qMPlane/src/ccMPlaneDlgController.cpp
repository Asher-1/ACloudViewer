//##########################################################################
//#                                                                        #
//#                    CLOUDVIEWER PLUGIN: qMPlane                         #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 of the License.               #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#         COPYRIGHT: AIT Austrian Institute of Technology GmbH           #
//#                                                                        #
//##########################################################################

// Local dependencies
#include "ccMPlaneDlgController.h"

// eCV
#include <ecvPickingHub.h>

// Qt
#include "qfiledialog.h"

constexpr char *MPLANE_PLANE_NAME = "Fitting-plane";


ccMPlaneDlgController::ccMPlaneDlgController(ecvMainAppInterface * app) : QObject(), m_app(app)
{
	m_device = std::make_unique<ccMeasurementDevice>(m_app);
	m_dialog = new ccMPlaneDlg((QWidget*)m_app->getMainWindow());
	m_dialog->linkWith(m_app->getActiveWindow());

	connect(m_dialog, &ccMPlaneDlg::signalCloseButtonPressed, this, &ccMPlaneDlgController::onCloseButtonPressed);
	connect(m_dialog, &ccMPlaneDlg::signalTabChanged, this, &ccMPlaneDlgController::onNewTab);
	connect(m_dialog, &ccMPlaneDlg::signalMeasureNameChanged, this, &ccMPlaneDlgController::onMeasureNameChanged);
	connect(m_dialog, &ccMPlaneDlg::signalMeasurementModeChanged, this, &ccMPlaneDlgController::onMeasurementModeChanged);
	connect(m_dialog, &ccMPlaneDlg::signalFittingPointDelete, this, &ccMPlaneDlgController::onFittingPointDelete);
	connect(m_dialog, &ccMPlaneDlg::signalShowNormalCheckBoxClicked, this, &ccMPlaneDlgController::onNormalCheckBoxClicked);
	connect(m_dialog, &ccMPlaneDlg::signalSaveButtonClicked, this, &ccMPlaneDlgController::onSaveButtonClicked);
}

void ccMPlaneDlgController::openDialog(ccPointCloud * selectedCloud)
{
	m_app->freezeUI(true);
	m_selectedCloud = selectedCloud;

	registerDialog();

	loadDataFromSelectedCloud();
	startPicking();
}

void ccMPlaneDlgController::onCloseButtonPressed()
{
	m_app->freezeUI(false);

	m_dialog->clearFittingPoints();
	m_dialog->clearMeasurementPoints();

	m_dialog->stop(true);
	m_app->unregisterOverlayDialog(m_dialog);
	m_app->updateOverlayDialogsPlacement();
	//m_app->getActiveWindow()->redraw(true, false);
	m_app->refreshSelected(true);

	stopPicking();
}

void ccMPlaneDlgController::onNewTab(int tabIndex)
{
	m_mode = tabIndex == 0 ? CC_Mode::CC_POINT_SELECTION : CC_Mode::CC_MEASUREMENT;
}

void ccMPlaneDlgController::onMeasureNameChanged(QTableWidgetItem *item)
{
	if (!m_data->renameMeasurement(item->text(), item->row())) {
		m_app->dispToConsole(QString("[MPlane] Measurement point %1 already exists!").arg(item->text()), ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
		m_dialog->renameMeasurement(m_data->getMeasurementPoints()[item->row()].getName(), item->row());
	}
}

void ccMPlaneDlgController::onMeasurementModeChanged()
{
	m_signedMeasurement = m_dialog->isSignedMeasurement();
	updateScalarfield();
	updateMeasurements();
}

void ccMPlaneDlgController::onFittingPointDelete(int index)
{
	m_data->deleteFittingPoint(index);
	
	updateFittingPoints();
	updatAllMeasurementEntities();
	
	m_dialog->selectFittingPoint(m_data->getActualFittingPointIndex());
	m_app->refreshAll();
	m_app->updateUI();
	//m_selectedCloud->refreshDisplay();
}

void ccMPlaneDlgController::onNormalCheckBoxClicked(bool checked)
{
	m_showNormal = checked;
	m_data->getPlane()->showNormalVector(checked);
	//m_selectedCloud->prepareDisplayForRefresh();
	//m_selectedCloud->refreshDisplay();
}

void ccMPlaneDlgController::onSaveButtonClicked()
{
	QString fileName = QFileDialog::getSaveFileName(m_dialog,
		tr("Save Measurements"), "",
		tr("Comma-separated values (*.csv);;All Files (*)"));
	QFile file(fileName);
	if (file.open(QIODevice::ReadWrite | QIODevice::Truncate)) {
		QTextStream stream(&file);
		stream << "measurement,x-coord,y-coord,z-coord,distance" << endl;
		for (ccMPlanePoint point : m_data->getMeasurementPoints()) {
			stream
				<< point.getName() << ","
				<< point.getCoordinates().x << ","
				<< point.getCoordinates().y << ","
				<< point.getCoordinates().z << ","
				<< point.getDistance() << endl;
		}
	}
}

void ccMPlaneDlgController::onItemPicked(const ccPickingListener::PickedItem & item)
{
	ccHObject *entity = item.entity;
	if (entity && entity->isKindOf(CV_TYPES::POINT_CLOUD))
	{
		ccPointCloud* sourceCloud = static_cast<ccPointCloud*>(entity);
		if (sourceCloud != m_selectedCloud) {
			m_app->dispToConsole("[MPlane] Can only pick points from selected cloud.", ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
		}
		else {
			if (m_mode == CC_Mode::CC_POINT_SELECTION) {
				pickFittingPoint(item);
				updatAllMeasurementEntities();
				m_dialog->selectFittingPoint(m_data->getActualFittingPointIndex());
			}
			else {
				pickMeasurementPoint(item);
			}
		}
	}
}

void ccMPlaneDlgController::loadDataFromSelectedCloud()
{
	m_data = std::make_unique<ccMeasurementRecorder>(m_selectedCloud, m_app);
	m_data->loadDataFromSelectedCloud();

	updateFittingPoints();
	updatAllMeasurementEntities();

	m_dialog->selectFittingPoint(m_data->getActualFittingPointIndex());
}

void ccMPlaneDlgController::registerDialog()
{
	m_app->registerOverlayDialog(m_dialog, Qt::Corner::TopLeftCorner);
	m_dialog->start();
	m_app->updateOverlayDialogsPlacement();
	//m_app->getActiveGLWindow()->redraw(true, false);
	m_app->refreshSelected(true);
}

void ccMPlaneDlgController::startPicking()
{
	if (!m_app->pickingHub())
	{
		m_app->dispToConsole("[MPlane] Could not retrieve valid picking hub. Measurement aborted.", ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
	}

	if (!m_app->pickingHub()->addListener(this, true, true))
	{
		m_app->dispToConsole("[MPlane] Another tool is already using the picking mechanism. Stop it first", ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
	}

	m_app->getActiveWindow()->installEventFilter(this);
}

void ccMPlaneDlgController::stopPicking()
{
	m_app->pickingHub()->removeListener(this);
	m_app->getActiveWindow()->removeEventFilter(this);
}

void ccMPlaneDlgController::pickFittingPoint(const ccPickingListener::PickedItem & item)
{
	m_data->addFittingPoint(item);

	// Display fitting point in table
	m_dialog->addFittingPoint(m_data->getActualFittingPointIndex() - 1, item.P3D);

	// Update display: Necessary for Stereo-build
	//m_selectedCloud->prepareDisplayForRefresh();
	m_app->refreshSelected();
}

void ccMPlaneDlgController::pickMeasurementPoint(const ccPickingListener::PickedItem & item)
{
	float distance = m_device->measurePointToPlaneDistance(m_data->getPlane(), item.P3D, m_signedMeasurement);
	m_data->addMeasurementPoint(item, distance);
	const auto &point = m_data->getMeasurementPoints().back();
	m_dialog->addMeasurementPoint(point.getName(), point.getDistance());

	// Update display: Necessary for Stereo-build
	//m_selectedCloud->prepareDisplayForRefresh();
	//m_app->refreshAll();
	m_app->refreshSelected();
}

void ccMPlaneDlgController::updatePlane()
{
	ccPlane *plane = m_device->fitPlaneToPoints(m_data->getFittingPoints(), MPLANE_PLANE_NAME);
	plane->showNormalVector(m_showNormal);
	m_data->setPlane(plane);
}

void ccMPlaneDlgController::updateScalarfield()
{
	m_device->createScalarFieldForCloud(m_data->getPlane(), m_selectedCloud, m_signedMeasurement);
}

void ccMPlaneDlgController::updateMeasurements()
{
	m_dialog->clearMeasurementPoints();
	const std::vector<ccMPlanePoint>& points = m_data->getMeasurementPoints();
	for (auto i = 0; i < points.size(); ++i) {
		float distance = m_device->measurePointToPlaneDistance(m_data->getPlane(), points[i].getCoordinates(), m_signedMeasurement);
		m_data->updateMeasurement(distance, i);
		m_dialog->addMeasurementPoint(points[i].getName(), distance);
	}
}

void ccMPlaneDlgController::updateFittingPoints()
{
	m_dialog->clearFittingPoints();
	const std::vector<ccMPlanePoint>& points = m_data->getFittingPoints();
	for (auto i = 0; i < points.size(); ++i) {
		m_data->renameFittingPoint(QString("Point %1").arg(i), i);
		m_dialog->addFittingPoint(i, points[i].getCoordinates());
	}
	//m_selectedCloud->prepareDisplayForRefresh();
	//m_app->refreshAll();
	m_app->refreshSelected();
}

void ccMPlaneDlgController::updatAllMeasurementEntities()
{
	if (m_data->getFittingPointAmount() >= 3) {
		m_dialog->enableMeasurementTab(true);
		updatePlane();
		updateScalarfield();
		updateMeasurements();
	}
	else {
		m_dialog->enableMeasurementTab(false);
		m_data->deletePlane();
		m_device->deleteScalarFieldFromCloud(m_selectedCloud);
	}
}

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

#ifndef Q_MPLANE_DIALOG_CONTROLLER
#define Q_MPLANE_DIALOG_CONTROLLER

// Std
#include <memory>

// Qt
#include <QtGui>

// CC
#include "ecvMainAppInterface.h"
#include <ecvPickingListener.h>

// Local dependencies
#include "ccMPlaneDlg.h"
#include "ccMeasurementRecorder.h"
#include "ccMeasurementDevice.h"

class ccMPlaneDlgController : public QObject, public ccPickingListener {
	Q_OBJECT

public:
	explicit ccMPlaneDlgController(ecvMainAppInterface *app);
	void openDialog(ccPointCloud *selectedCloud);

protected slots:
	void onCloseButtonPressed();
	void onNewTab(int tabIndex);
	void onMeasureNameChanged(QTableWidgetItem *);
	void onMeasurementModeChanged();
	void onFittingPointDelete(int index);
	void onNormalCheckBoxClicked(bool checked);
	void onSaveButtonClicked();
	virtual void onItemPicked(const ccPickingListener::PickedItem& pi) override;

private:
	void loadDataFromSelectedCloud();
	void registerDialog();
	void startPicking();
	void stopPicking();
	void pickFittingPoint(const ccPickingListener::PickedItem & item);
	void pickMeasurementPoint(const ccPickingListener::PickedItem & item);
	void updatePlane();
	void updateScalarfield();
	void updateMeasurements();
	void updateFittingPoints();
	void updatAllMeasurementEntities();

private:
	ecvMainAppInterface *m_app;
	ccMPlaneDlg *m_dialog;
	std::unique_ptr<ccMeasurementRecorder> m_data;
	std::unique_ptr<ccMeasurementDevice> m_device;
	ccPointCloud *m_selectedCloud = nullptr;

	enum CC_Mode { CC_POINT_SELECTION, CC_MEASUREMENT };
	CC_Mode m_mode = CC_POINT_SELECTION;

	bool m_signedMeasurement = false;
	bool m_showNormal = false;
};


#endif
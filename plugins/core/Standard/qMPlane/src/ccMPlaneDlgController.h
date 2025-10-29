// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Std
#include <memory>

// Qt
#include <QtGui>

// CC
#include <ecvPickingListener.h>

#include "ecvMainAppInterface.h"

// Local dependencies
#include "ccMPlaneDlg.h"
#include "ccMeasurementDevice.h"
#include "ccMeasurementRecorder.h"

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
    virtual void onItemPicked(const ccPickingListener::PickedItem &pi) override;

private:
    void loadDataFromSelectedCloud();
    void registerDialog();
    void startPicking();
    void stopPicking();
    void pickFittingPoint(const ccPickingListener::PickedItem &item);
    void pickMeasurementPoint(const ccPickingListener::PickedItem &item);
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

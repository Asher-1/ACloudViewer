// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_camSensorProjectDlg.h>

class ccCameraSensor;

//! Camera sensor parameters dialog
class ccCamSensorProjectionDlg : public QDialog,
                                 public Ui::CamSensorProjectDialog {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccCamSensorProjectionDlg(QWidget* parent = 0);

    void initWithCamSensor(const ccCameraSensor* sensor);
    void updateCamSensor(ccCameraSensor* sensor);
};

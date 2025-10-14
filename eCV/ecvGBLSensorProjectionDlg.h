// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_gblSensorProjectDlg.h>

class ccGBLSensor;

//! Ground-based (lidar) sensor parameters dialog
class ccGBLSensorProjectionDlg : public QDialog,
                                 public Ui::GBLSensorProjectDialog {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccGBLSensorProjectionDlg(QWidget* parent = 0);

    void initWithGBLSensor(const ccGBLSensor* sensor);
    void updateGBLSensor(ccGBLSensor* sensor);
};

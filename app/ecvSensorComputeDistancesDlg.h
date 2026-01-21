// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_sensorComputeDistancesDlg.h>

//! Dialog for sensor range computation
class ccSensorComputeDistancesDlg : public QDialog,
                                    public Ui::sensorComputeDistancesDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccSensorComputeDistancesDlg(QWidget* parent = 0);

    //! Returns whether computed distances should be squared or not
    bool computeSquaredDistances() const;
};

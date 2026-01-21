// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_sensorComputeScatteringAnglesDlg.h>

//! Dialog for scattering angles computation
class ccSensorComputeScatteringAnglesDlg
    : public QDialog,
      public Ui::sensorComputeScatteringAnglesDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccSensorComputeScatteringAnglesDlg(QWidget* parent = 0);

    //! Returns whether angles should be converted to degrees
    bool anglesInDegrees() const;
};

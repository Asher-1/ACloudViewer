// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvSensorComputeScatteringAnglesDlg.h"

ccSensorComputeScatteringAnglesDlg::ccSensorComputeScatteringAnglesDlg(
        QWidget* parent /*=0*/)
    : QDialog(parent, Qt::Tool), Ui::sensorComputeScatteringAnglesDlg() {
    setupUi(this);
}

bool ccSensorComputeScatteringAnglesDlg::anglesInDegrees() const {
    return anglesToDegCheckbox->checkState() == Qt::Checked;
}
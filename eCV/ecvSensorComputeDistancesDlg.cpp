// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvSensorComputeDistancesDlg.h"

ccSensorComputeDistancesDlg::ccSensorComputeDistancesDlg(QWidget* parent /*=0*/)
    : QDialog(parent, Qt::Tool), Ui::sensorComputeDistancesDlg() {
    setupUi(this);
}

bool ccSensorComputeDistancesDlg::computeSquaredDistances() const {
    return checkSquaredDistance->checkState() == Qt::Checked;
}
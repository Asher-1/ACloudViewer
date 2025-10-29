// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvNoiseFilterDlg.h"

ecvNoiseFilterDlg::ecvNoiseFilterDlg(QWidget* parent /*=0*/)
    : QDialog(parent, Qt::Tool), Ui::NoiseFilterDialog() {
    setupUi(this);
}

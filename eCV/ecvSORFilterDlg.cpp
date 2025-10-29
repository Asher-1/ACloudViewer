// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvSORFilterDlg.h"

ecvSORFilterDlg::ecvSORFilterDlg(QWidget* parent /*=0*/)
    : QDialog(parent, Qt::Tool), Ui::SorFilterDialog() {
    setupUi(this);
}

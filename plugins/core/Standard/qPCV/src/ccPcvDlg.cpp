// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccPcvDlg.h"

ccPcvDlg::ccPcvDlg(QWidget* parent)
    : QDialog(parent, Qt::Tool), Ui::PCVDialog() {
    setupUi(this);
}

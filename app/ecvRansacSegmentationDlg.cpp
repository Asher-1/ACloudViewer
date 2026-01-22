// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvRansacSegmentationDlg.h"

ecvRansacSegmentationDlg::ecvRansacSegmentationDlg(QWidget* parent)
    : QDialog(parent, Qt::Tool), Ui::RansacSegmentationDlg() {
    setupUi(this);
}
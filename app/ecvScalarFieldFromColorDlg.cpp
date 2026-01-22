// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvScalarFieldFromColorDlg.h"

// Qt
#include <QPushButton>

// CV_DB_LIB
#include <assert.h>
#include <ecvPointCloud.h>
#ifdef _MSC_VER
#include <windows.h>
#endif

ccScalarFieldFromColorDlg::ccScalarFieldFromColorDlg(QWidget* parent /*=0*/)
    : QDialog(parent, Qt::Tool), Ui::scalarFieldFromColorDlg() {
    setupUi(this);
}

bool ccScalarFieldFromColorDlg::getRStatus() {
    return this->checkBoxR->isChecked();
}

bool ccScalarFieldFromColorDlg::getGStatus() {
    return this->checkBoxG->isChecked();
}

bool ccScalarFieldFromColorDlg::getBStatus() {
    return this->checkBoxB->isChecked();
}

bool ccScalarFieldFromColorDlg::getCompositeStatus() {
    return this->checkBoxComposite->isChecked();
}

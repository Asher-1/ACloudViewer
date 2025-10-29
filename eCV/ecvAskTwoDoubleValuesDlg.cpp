// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvAskTwoDoubleValuesDlg.h"

ccAskTwoDoubleValuesDlg::ccAskTwoDoubleValuesDlg(const char* vName1,
                                                 const char* vName2,
                                                 double minVal,
                                                 double maxVal,
                                                 double defaultVal1,
                                                 double defaultVal2,
                                                 int precision /*=6*/,
                                                 const char* windowTitle /*=0*/,
                                                 QWidget* parent /*=0*/)
    : QDialog(parent, Qt::Tool), Ui::AskTwoDoubleValuesDialog() {
    setupUi(this);

    label1->setText(vName1);
    label2->setText(vName2);
    doubleSpinBox1->setDecimals(precision);
    doubleSpinBox2->setDecimals(precision);
    doubleSpinBox1->setRange(minVal, maxVal);
    doubleSpinBox2->setRange(minVal, maxVal);
    doubleSpinBox1->setValue(defaultVal1);
    doubleSpinBox2->setValue(defaultVal2);

    if (windowTitle) setWindowTitle(windowTitle);
}

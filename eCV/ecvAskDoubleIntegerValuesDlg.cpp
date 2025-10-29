// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvAskDoubleIntegerValuesDlg.h"

ecvAskDoubleIntegerValuesDlg::ecvAskDoubleIntegerValuesDlg(
        const char* vName1,
        const char* vName2,
        double minVal,
        double maxVal,
        int minInt,
        int maxInt,
        double defaultVal1,
        int defaultVal2,
        int precision /*=6*/,
        const char* windowTitle /*=0*/,
        QWidget* parent /*=0*/)
    : QDialog(parent, Qt::Tool), Ui::AskDoubleIntegerValuesDialog() {
    setupUi(this);

    label1->setText(vName1);
    label2->setText(vName2);
    doubleSpinBox->setDecimals(precision);
    doubleSpinBox->setRange(minVal, maxVal);
    integerSpinBox->setRange(minInt, maxInt);
    doubleSpinBox->setValue(defaultVal1);
    integerSpinBox->setValue(defaultVal2);

    if (windowTitle) setWindowTitle(windowTitle);
}

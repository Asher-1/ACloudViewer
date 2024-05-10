// ##########################################################################
// #                                                                        #
// #                              CLOUDVIEWER                               #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #          COPYRIGHT: EDF R&D / DAHAI LU                                 #
// #                                                                        #
// ##########################################################################

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

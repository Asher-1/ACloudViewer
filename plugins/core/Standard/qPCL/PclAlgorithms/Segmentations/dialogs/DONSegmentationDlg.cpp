// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "DONSegmentationDlg.h"

DONSegmentationDlg::DONSegmentationDlg(QWidget* parent)
    : QDialog(parent, Qt::Tool), Ui::DONSegmentationDlg() {
    setupUi(this);
    buttonGroup->setExclusive(true);
    buttonGroup->setId(curvatureRadioButton, 0);
    buttonGroup->setId(xRadioButton, 1);
    buttonGroup->setId(yRadioButton, 2);
    buttonGroup->setId(zRadioButton, 3);
}

const QString DONSegmentationDlg::getComparisonField() {
    return buttonGroup->checkedButton()->text();
}

void DONSegmentationDlg::getComparisonTypes(QStringList& types) {
    types.clear();
    if (equalCheckBox->isChecked()) {
        if (greaterCheckBox->isChecked()) {
            types << "GE";
        }

        if (lessThanCheckBox->isChecked()) {
            types << "LE";
        }

        if (!greaterCheckBox->isChecked() && !lessThanCheckBox->isChecked()) {
            types << "EQ";
        }
    } else {
        if (greaterCheckBox->isChecked()) {
            types << "GT";
        }

        if (lessThanCheckBox->isChecked()) {
            types << "LT";
        }
    }
}

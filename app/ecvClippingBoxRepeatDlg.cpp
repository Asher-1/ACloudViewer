// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvClippingBoxRepeatDlg.h"

// Qt
#include <QPushButton>

// system
#include <assert.h>

ccClippingBoxRepeatDlg::ccClippingBoxRepeatDlg(
        bool singleContourMode /*=false*/, QWidget* parent /*=0*/)
    : QDialog(parent) {
    setupUi(this);

    if (!singleContourMode) {
        connect(xRepeatCheckBox, &QAbstractButton::toggled, this,
                &ccClippingBoxRepeatDlg::onDimChecked);
        connect(yRepeatCheckBox, &QAbstractButton::toggled, this,
                &ccClippingBoxRepeatDlg::onDimChecked);
        connect(zRepeatCheckBox, &QAbstractButton::toggled, this,
                &ccClippingBoxRepeatDlg::onDimChecked);
    } else {
        // single contour extraction mode!
        repeatDimGroupBox->setTitle("Flat dimension");

        connect(xRepeatCheckBox, &QAbstractButton::toggled, this,
                &ccClippingBoxRepeatDlg::onDimXChecked);
        connect(yRepeatCheckBox, &QAbstractButton::toggled, this,
                &ccClippingBoxRepeatDlg::onDimYChecked);
        connect(zRepeatCheckBox, &QAbstractButton::toggled, this,
                &ccClippingBoxRepeatDlg::onDimZChecked);
        setFlatDim(0);

        extractContoursGroupBox->setChecked(true);
        extractContoursGroupBox->setCheckable(false);
        projectOnBestFitCheckBox->setVisible(true);
        projectOnBestFitCheckBox->setChecked(false);

        randomColorCheckBox->setChecked(false);
        otherOptionsGroupBox->setVisible(false);
    }
}

void ccClippingBoxRepeatDlg::setRepeatDim(unsigned char dim) {
    assert(dim < 3);
    QCheckBox* boxes[3] = {xRepeatCheckBox, yRepeatCheckBox, zRepeatCheckBox};

    for (unsigned char d = 0; d < 3; ++d) {
        boxes[d]->setChecked(d == dim);
    }
}

void ccClippingBoxRepeatDlg::onDimXChecked(bool state) {
    assert(state);
    setFlatDim(0);
}
void ccClippingBoxRepeatDlg::onDimYChecked(bool state) {
    assert(state);
    setFlatDim(1);
}
void ccClippingBoxRepeatDlg::onDimZChecked(bool state) {
    assert(state);
    setFlatDim(2);
}

void ccClippingBoxRepeatDlg::setFlatDim(unsigned char dim) {
    assert(dim < 3);
    QCheckBox* boxes[3] = {xRepeatCheckBox, yRepeatCheckBox, zRepeatCheckBox};

    for (unsigned char d = 0; d < 3; ++d) {
        boxes[d]->blockSignals(true);
        // disable the current dimension
        // and uncheck the other dimensions
        boxes[d]->setChecked(d == dim);
        boxes[d]->setEnabled(d != dim);
        boxes[d]->blockSignals(false);
    }
}

void ccClippingBoxRepeatDlg::onDimChecked(bool) {
    // if only one dimension is checked, then the user can choose to project
    // the points along this dimension
    int sum = static_cast<int>(xRepeatCheckBox->isChecked()) +
              static_cast<int>(yRepeatCheckBox->isChecked()) +
              static_cast<int>(zRepeatCheckBox->isChecked());

    if (sum == 1) {
        if (!projectOnBestFitCheckBox->isVisible())
            projectOnBestFitCheckBox->setChecked(false);
        projectOnBestFitCheckBox->setVisible(true);
    } else {
        projectOnBestFitCheckBox->setVisible(false);
        projectOnBestFitCheckBox->setChecked(true);
    }

    buttonBox->button(QDialogButtonBox::Ok)->setEnabled(sum != 0);
}

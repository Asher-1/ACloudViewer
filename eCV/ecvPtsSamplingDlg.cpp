// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvPtsSamplingDlg.h"

ccPtsSamplingDlg::ccPtsSamplingDlg(QWidget* parent /*=0*/)
    : QDialog(parent, Qt::Tool), Ui::PointsSamplingDialog() {
    setupUi(this);
}

bool ccPtsSamplingDlg::generateNormals() const {
    return normalsCheckBox->isChecked();
}

void ccPtsSamplingDlg::setGenerateNormals(bool state) {
    normalsCheckBox->setChecked(state);
}

bool ccPtsSamplingDlg::interpolateRGB() const {
    return colorsCheckBox->isChecked();
}

bool ccPtsSamplingDlg::interpolateTexture() const {
    return textureCheckBox->isChecked();
}

bool ccPtsSamplingDlg::useDensity() const { return dRadioButton->isChecked(); }

void ccPtsSamplingDlg::setUseDensity(bool state) {
    dRadioButton->setChecked(state);
}

double ccPtsSamplingDlg::getDensityValue() const {
    return dDoubleSpinBox->value();
}

void ccPtsSamplingDlg::setDensityValue(double density) {
    dDoubleSpinBox->setValue(density);
}

unsigned ccPtsSamplingDlg::getPointsNumber() const {
    return pnSpinBox->value();
}

void ccPtsSamplingDlg::setPointsNumber(int count) {
    pnSpinBox->setValue(count);
}

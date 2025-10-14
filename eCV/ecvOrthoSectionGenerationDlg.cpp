// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvOrthoSectionGenerationDlg.h"

// system
#include <cmath>

ccOrthoSectionGenerationDlg::ccOrthoSectionGenerationDlg(QWidget* parent /*=0*/)
    : QDialog(parent, Qt::Tool),
      Ui::OrthoSectionGenerationDlg(),
      m_pathLength(0) {
    setupUi(this);

    connect(stepDoubleSpinBox,
            static_cast<void (QDoubleSpinBox::*)(double)>(
                    &QDoubleSpinBox::valueChanged),
            this, &ccOrthoSectionGenerationDlg::onStepChanged);
}

void ccOrthoSectionGenerationDlg::setPathLength(double l) {
    m_pathLength = l;
    pathLengthLineEdit->setText(QString::number(l));
    stepDoubleSpinBox->setValue(l / 9);
    widthDoubleSpinBox->setValue(l / 5);
}

void ccOrthoSectionGenerationDlg::setAutoSaveAndRemove(bool state) {
    autoSaveAndRemoveCheckBox->setChecked(state);
}

bool ccOrthoSectionGenerationDlg::autoSaveAndRemove() const {
    return autoSaveAndRemoveCheckBox->isChecked();
}

void ccOrthoSectionGenerationDlg::setGenerationStep(double s) {
    stepDoubleSpinBox->setValue(s);
}

void ccOrthoSectionGenerationDlg::setSectionsWidth(double w) {
    widthDoubleSpinBox->setValue(w);
}

double ccOrthoSectionGenerationDlg::getGenerationStep() const {
    return stepDoubleSpinBox->value();
}

double ccOrthoSectionGenerationDlg::getSectionsWidth() const {
    return widthDoubleSpinBox->value();
}

void ccOrthoSectionGenerationDlg::onStepChanged(double step) {
    if (step < 0) return;

    unsigned count = step < 1.0e-6 ? 1
                                   : 1 + static_cast<unsigned>(std::floor(
                                                 m_pathLength / step));
    sectionCountLineEdit->setText(QString::number(count));
}

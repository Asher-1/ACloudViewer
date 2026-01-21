// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvSmoothPolylineDlg.h"

// ui
#include <ui_smoothPolylineDlg.h>

// Qt
#include <QDialog>

ccSmoothPolylineDialog::ccSmoothPolylineDialog(QWidget* parent /*=nullptr*/)
    : QDialog(parent, Qt::Tool), m_ui(new Ui_SmoothPolylineDialog) {
    m_ui->setupUi(this);
}

ccSmoothPolylineDialog::~ccSmoothPolylineDialog() {
    if (m_ui) {
        delete m_ui;
        m_ui = nullptr;
    }
}

void ccSmoothPolylineDialog::setIerationCount(int count) {
    m_ui->iterationSpinBox->setValue(count);
}

void ccSmoothPolylineDialog::setRatio(double ratio) {
    m_ui->ratioDoubleSpinBox->setValue(ratio);
}

int ccSmoothPolylineDialog::getIerationCount() const {
    return m_ui->iterationSpinBox->value();
}

double ccSmoothPolylineDialog::getRatio() const {
    return m_ui->ratioDoubleSpinBox->value();
}

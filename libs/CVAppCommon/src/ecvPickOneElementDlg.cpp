// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvPickOneElementDlg.h"

// UI file
#include <ui_pickOneElementDlg.h>

ccPickOneElementDlg::ccPickOneElementDlg(
        const QString &label,
        const QString &windowTitle /*=QString()*/,
        QWidget *parent /*=0*/)
    : QDialog(parent, Qt::Tool), m_ui(new Ui_PickOneElementDialog) {
    m_ui->setupUi(this);

    if (!windowTitle.isNull()) {
        setWindowTitle(windowTitle);
    }

    m_ui->comboLabel->setText(label);
}

ccPickOneElementDlg::~ccPickOneElementDlg() {
    if (m_ui) {
        delete m_ui;
        m_ui = nullptr;
    }
}

void ccPickOneElementDlg::addElement(const QString &elementName) {
    m_ui->comboBox->addItem(elementName);
}

void ccPickOneElementDlg::setDefaultIndex(int index) {
    m_ui->comboBox->setCurrentIndex(index);
}

int ccPickOneElementDlg::getSelectedIndex() {
    return m_ui->comboBox->currentIndex();
}

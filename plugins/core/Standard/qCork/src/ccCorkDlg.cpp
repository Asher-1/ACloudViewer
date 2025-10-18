// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccCorkDlg.h"

ccCorkDlg::ccCorkDlg(QWidget* parent)
    : QDialog(parent, Qt::Tool),
      Ui::CorkDialog(),
      m_selectedOperation(UNION),
      m_isSwapped(false) {
    setupUi(this);

    connect(unionPushButton, SIGNAL(clicked()), this, SLOT(unionSelected()));
    connect(interPushButton, SIGNAL(clicked()), this,
            SLOT(intersectSelected()));
    connect(diffPushButton, SIGNAL(clicked()), this, SLOT(diffSelected()));
    connect(symDiffPushButton, SIGNAL(clicked()), this,
            SLOT(symDiffSelected()));
    connect(swapToolButton, SIGNAL(clicked()), this, SLOT(swap()));
}

void ccCorkDlg::setNames(QString A, QString B) {
    meshALineEdit->setText(A);
    meshBLineEdit->setText(B);
}

void ccCorkDlg::unionSelected() {
    m_selectedOperation = UNION;
    accept();
}

void ccCorkDlg::intersectSelected() {
    m_selectedOperation = INTERSECT;
    accept();
}

void ccCorkDlg::diffSelected() {
    m_selectedOperation = DIFF;
    accept();
}

void ccCorkDlg::symDiffSelected() {
    m_selectedOperation = SYM_DIFF;
    accept();
}

void ccCorkDlg::swap() {
    m_isSwapped = !m_isSwapped;

    QString A = meshALineEdit->text();
    QString B = meshBLineEdit->text();
    setNames(B, A);
}

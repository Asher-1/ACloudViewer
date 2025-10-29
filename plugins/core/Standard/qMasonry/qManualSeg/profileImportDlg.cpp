// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "profileImportDlg.h"

// Qt
#include <QFileDialog>

// System
#include <assert.h>

ProfileImportDlg::ProfileImportDlg(QWidget* parent)
    : QDialog(parent, Qt::Tool), Ui::ProfileImportDlg() {
    setupUi(this);

    connect(browseToolButton, &QAbstractButton::clicked, this,
            &ProfileImportDlg::browseFile);
}

int ProfileImportDlg::getAxisDimension() const {
    return axisDimComboBox->currentIndex();
}

void ProfileImportDlg::browseFile() {
    QString filter("2D profile (*.txt)");

    // open file loading dialog
    QString filename = QFileDialog::getOpenFileName(
            0, "Select profile file", getFilename(), filter
#if defined(Q_OS_WIN) && defined(_DEBUG)
            ,
            0, QFileDialog::DontUseNativeDialog
#endif
    );

    if (filename.isEmpty()) return;

    setDefaultFilename(filename);
}

void ProfileImportDlg::setDefaultFilename(QString filename) {
    inputFileLineEdit->setText(filename);
}

QString ProfileImportDlg::getFilename() const {
    return inputFileLineEdit->text();
}

bool ProfileImportDlg::absoluteHeightValues() const {
    return absoluteHeightValuesCheckBox->isChecked();
}

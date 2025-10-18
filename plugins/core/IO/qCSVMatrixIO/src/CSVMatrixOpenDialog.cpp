// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CSVMatrixOpenDialog.h"

#include "ecvFileUtils.h"

// Qt
#include <QFileDialog>
#include <QFileInfo>
#include <QSettings>

CSVMatrixOpenDialog::CSVMatrixOpenDialog(QWidget* parent /*=0*/)
    : QDialog(parent), Ui::CSVMatrixOpenDlg() {
    setupUi(this);

    connect(browseToolButton, &QAbstractButton::clicked, this,
            &CSVMatrixOpenDialog::browseTextureFile);

    // persistent settings
    QSettings settings;
    settings.beginGroup("LoadFile");
    QString currentPath =
            settings.value("currentPath", ecvFileUtils::defaultDocPath())
                    .toString();

    textureFilenameLineEdit->setText(currentPath);
}

void CSVMatrixOpenDialog::browseTextureFile() {
    QString inputFilename = QFileDialog::getOpenFileName(
            this, "Texture file", textureFilenameLineEdit->text(), "*.*");
    if (inputFilename.isEmpty()) return;

    textureFilenameLineEdit->setText(inputFilename);

    // save last loading location
    QSettings settings;
    settings.beginGroup("LoadFile");
    settings.setValue("currentPath", QFileInfo(inputFilename).absolutePath());
    settings.endGroup();
}

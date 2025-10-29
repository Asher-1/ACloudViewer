// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RDBOpenDialog.h"

// Qt
#include <QFileDialog>
#include <QFileInfo>
#include <QSettings>

RDBOpenDialog::RDBOpenDialog(QWidget* parent /*=0*/)
    : QDialog(parent), Ui::RDBOpenDlg() {
    setupUi(this);
}

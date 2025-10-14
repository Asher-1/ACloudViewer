// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Qt
#include <QDialog>

// GUI
#include "ui_openCSVMatrixDlg.h"

//! CSV Matrix Open dialog
class CSVMatrixOpenDialog : public QDialog, public Ui::CSVMatrixOpenDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit CSVMatrixOpenDialog(QWidget* parent = 0);

protected slots:

    //! Bowse texture file
    void browseTextureFile();
};

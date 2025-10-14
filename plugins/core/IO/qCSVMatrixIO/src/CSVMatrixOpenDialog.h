// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CSV_MATRIX_OPEN_DIALOG_HEADER
#define CSV_MATRIX_OPEN_DIALOG_HEADER

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

#endif  // CSV_MATRIX_OPEN_DIALOG_HEADER

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ui_ransacSDDlg.h"

//! Dialog for qRansacSD plugin
class ccRansacSDDlg : public QDialog, public Ui::RansacSDDialog {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccRansacSDDlg(QWidget* parent = 0);

protected slots:

    //! Saves (temporarily) the dialog parameters on acceptation
    void saveSettings();
};

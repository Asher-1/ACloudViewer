// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ui_CSFDlg.h"

//! Dialog for qCSF plugin
class ccCSFDlg : public QDialog, public Ui::CSFDialog {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccCSFDlg(QWidget* parent = 0);

protected slots:

    //! Saves (temporarily) the dialog parameters on acceptation
    void saveSettings();
};

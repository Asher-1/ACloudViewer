// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_CSF_DLG_HEADER
#define ECV_CSF_DLG_HEADER

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

#endif  // ECV_CSF_DLG_HEADER

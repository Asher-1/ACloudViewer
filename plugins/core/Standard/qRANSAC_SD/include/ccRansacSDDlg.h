// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_RANSAC_SD_DLG_HEADER
#define ECV_RANSAC_SD_DLG_HEADER

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

#endif  // ECV_RANSAC_SD_DLG_HEADER

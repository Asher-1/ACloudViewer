// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ui_profileImportDlg.h"

//! Dialog for setting the input parameters required for the automatic
//! segmentation of masonry walls
class ProfileImportDlg : public QDialog, public Ui::ProfileImportDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit ProfileImportDlg(QWidget* parent = 0);

protected slots:

    //! Save settings
    void saveSettings();
};

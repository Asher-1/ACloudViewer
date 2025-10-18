// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_pcvDlg.h>

//! Dialog for the PCV plugin
class ccPcvDlg : public QDialog, public Ui::PCVDialog {
public:
    explicit ccPcvDlg(QWidget* parent = 0);
};

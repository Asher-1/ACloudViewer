// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_PCV_DLG_HEADER
#define ECV_PCV_DLG_HEADER

#include <ui_pcvDlg.h>

//! Dialog for the PCV plugin
class ccPcvDlg : public QDialog, public Ui::PCVDialog {
public:
    explicit ccPcvDlg(QWidget* parent = 0);
};

#endif  // ECV_PCV_DLG_HEADER

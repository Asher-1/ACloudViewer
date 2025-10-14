// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_SOR_FILTER_DLG_HEADER
#define ECV_SOR_FILTER_DLG_HEADER

#include <ui_sorFilterDlg.h>

//! Dialog to choose which dimension(s) (X, Y or Z) should be exported as SF(s)
class ecvSORFilterDlg : public QDialog, public Ui::SorFilterDialog {
    Q_OBJECT

public:
    //! Default constructor
    explicit ecvSORFilterDlg(QWidget* parent = 0);
};

#endif  // ECV_SOR_FILTER_DLG_HEADER

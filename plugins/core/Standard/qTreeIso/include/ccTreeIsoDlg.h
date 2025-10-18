// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
// A Matlab version shared via:
// https://github.com/truebelief/artemis_treeiso

#ifndef CC_TREEISO_DLG_HEADER
#define CC_TREEISO_DLG_HEADER

#include "ui_TreeIsoDlg.h"

//! Dialog for qTreeIso plugin
class ccTreeIsoDlg : public QDialog, public Ui::TreeIsoDialog {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccTreeIsoDlg(QWidget* parent = nullptr);

protected:
};

#endif

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_NurbsCurveFittingDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class NurbsCurveFittingDlg : public QDialog, public Ui::NurbsCurveFittingDlg {
public:
    explicit NurbsCurveFittingDlg(QWidget* parent = 0);
};

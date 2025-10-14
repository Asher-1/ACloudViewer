// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_NurbsSurfaceDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class NurbsSurfaceDlg : public QDialog, public Ui::NurbsSurfaceDlg {
public:
    explicit NurbsSurfaceDlg(QWidget* parent = 0);
};

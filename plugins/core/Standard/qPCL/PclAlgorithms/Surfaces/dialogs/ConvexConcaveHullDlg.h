// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_ConvexConcaveHullDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class ConvexConcaveHullDlg : public QDialog, public Ui::ConvexConcaveHullDlg {
public:
    explicit ConvexConcaveHullDlg(QWidget* parent = 0);
};

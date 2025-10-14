// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_CONVEXCONCAVE_DLG_HEADER
#define Q_PCL_PLUGIN_CONVEXCONCAVE_DLG_HEADER

#include <ui_ConvexConcaveHullDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class ConvexConcaveHullDlg : public QDialog, public Ui::ConvexConcaveHullDlg {
public:
    explicit ConvexConcaveHullDlg(QWidget* parent = 0);
};

#endif  // Q_PCL_PLUGIN_CONVEXCONCAVE_DLG_HEADER

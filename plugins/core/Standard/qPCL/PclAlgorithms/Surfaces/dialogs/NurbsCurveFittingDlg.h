// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_NURBSCURVE_DLG_HEADER
#define Q_PCL_PLUGIN_NURBSCURVE_DLG_HEADER

#include <ui_NurbsCurveFittingDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class NurbsCurveFittingDlg : public QDialog, public Ui::NurbsCurveFittingDlg {
public:
    explicit NurbsCurveFittingDlg(QWidget* parent = 0);
};

#endif  // Q_PCL_PLUGIN_NURBSCURVE_DLG_HEADER

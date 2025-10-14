// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_NURBSSURFACE_DLG_HEADER
#define Q_PCL_PLUGIN_NURBSSURFACE_DLG_HEADER

#include <ui_NurbsSurfaceDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class NurbsSurfaceDlg : public QDialog, public Ui::NurbsSurfaceDlg {
public:
    explicit NurbsSurfaceDlg(QWidget* parent = 0);
};

#endif  // Q_PCL_PLUGIN_NURBSSURFACE_DLG_HEADER

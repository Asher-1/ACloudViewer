// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_MARCHINGCUBE_DLG_HEADER
#define Q_PCL_PLUGIN_MARCHINGCUBE_DLG_HEADER

#include <ui_MarchingCubeDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class MarchingCubeDlg : public QDialog, public Ui::MarchingCubeDlg {
public:
    explicit MarchingCubeDlg(QWidget* parent = 0);
};

#endif  // Q_PCL_PLUGIN_MARCHINGCUBE_DLG_HEADER

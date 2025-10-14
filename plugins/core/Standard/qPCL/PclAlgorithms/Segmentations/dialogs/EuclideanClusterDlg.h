// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_EUCLIDEANCLUSTER_DLG_HEADER
#define Q_PCL_PLUGIN_EUCLIDEANCLUSTER_DLG_HEADER

#include <ui_EuclideanClusterDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class EuclideanClusterDlg : public QDialog, public Ui::EuclideanClusterDlg {
public:
    explicit EuclideanClusterDlg(QWidget* parent = 0);
};

#endif  // Q_PCL_PLUGIN_EUCLIDEANCLUSTER_DLG_HEADER

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_PROJECTIONFILTER_DIALOG_HEADER
#define Q_PCL_PLUGIN_PROJECTIONFILTER_DIALOG_HEADER

#include <ui_ProjectionFilterDlg.h>

// Qt
#include <QDialog>

class ProjectionFilterDlg : public QDialog, public Ui::ProjectionFilterDlg {
public:
    explicit ProjectionFilterDlg(QWidget* parent = nullptr);
};

#endif  // Q_PCL_PLUGIN_PROJECTIONFILTER_DIALOG_HEADER

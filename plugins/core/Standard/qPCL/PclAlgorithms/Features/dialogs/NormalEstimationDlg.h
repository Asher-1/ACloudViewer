// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_NORMAL_ESTIMATION_DIALOG_HEADER
#define Q_PCL_PLUGIN_NORMAL_ESTIMATION_DIALOG_HEADER

#include <ui_NormalEstimationDlg.h>

// Qt
#include <QDialog>

class NormalEstimationDialog : public QDialog,
                               public Ui::NormalEstimationDialog {
public:
    explicit NormalEstimationDialog(QWidget* parent = nullptr);
};

#endif  // Q_PCL_PLUGIN_NORMAL_ESTIMATION_DIALOG_HEADER

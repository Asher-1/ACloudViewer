// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_STATISTICAL_OUTLIERS_REMOVER_DIALOG_HEADER
#define Q_PCL_PLUGIN_STATISTICAL_OUTLIERS_REMOVER_DIALOG_HEADER

#include <ui_StatisticalOutliersRemoverDlg.h>

// Qt
#include <QDialog>

class SORDialog : public QDialog, public Ui::StatisticalOutliersRemoverDlg {
public:
    explicit SORDialog(QWidget* parent = nullptr);
};

#endif  // Q_PCL_PLUGIN_STATISTICAL_OUTLIERS_REMOVER_DIALOG_HEADER

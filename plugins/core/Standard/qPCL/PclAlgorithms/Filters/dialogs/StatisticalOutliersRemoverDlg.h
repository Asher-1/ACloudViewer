// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_StatisticalOutliersRemoverDlg.h>

// Qt
#include <QDialog>

class SORDialog : public QDialog, public Ui::StatisticalOutliersRemoverDlg {
public:
    explicit SORDialog(QWidget* parent = nullptr);
};

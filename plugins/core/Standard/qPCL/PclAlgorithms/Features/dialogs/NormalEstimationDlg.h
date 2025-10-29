// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_NormalEstimationDlg.h>

// Qt
#include <QDialog>

class NormalEstimationDialog : public QDialog,
                               public Ui::NormalEstimationDialog {
public:
    explicit NormalEstimationDialog(QWidget* parent = nullptr);
};

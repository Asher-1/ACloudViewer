// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_ProjectionFilterDlg.h>

// Qt
#include <QDialog>

class ProjectionFilterDlg : public QDialog, public Ui::ProjectionFilterDlg {
public:
    explicit ProjectionFilterDlg(QWidget* parent = nullptr);
};

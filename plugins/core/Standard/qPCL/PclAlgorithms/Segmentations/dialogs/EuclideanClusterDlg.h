// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_EuclideanClusterDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class EuclideanClusterDlg : public QDialog, public Ui::EuclideanClusterDlg {
public:
    explicit EuclideanClusterDlg(QWidget* parent = 0);
};

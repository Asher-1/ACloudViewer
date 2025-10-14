// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_GreedyTriangulationDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class GreedyTriangulationDlg : public QDialog,
                               public Ui::GreedyTriangulationDlg {
public:
    explicit GreedyTriangulationDlg(QWidget* parent = 0);
};

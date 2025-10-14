// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_GREEDYTRIANGULATION_DLG_HEADER
#define Q_PCL_PLUGIN_GREEDYTRIANGULATION_DLG_HEADER

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

#endif  // Q_PCL_PLUGIN_GREEDYTRIANGULATION_DLG_HEADER

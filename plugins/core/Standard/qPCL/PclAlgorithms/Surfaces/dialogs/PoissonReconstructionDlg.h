// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_POISSONRECONSTRUCTION_DLG_HEADER
#define Q_PCL_PLUGIN_POISSONRECONSTRUCTION_DLG_HEADER

#include <ui_PoissonReconstructionDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class PoissonReconstructionDlg : public QDialog,
                                 public Ui::PoissonReconstructionDlg {
public:
    explicit PoissonReconstructionDlg(QWidget* parent = 0);
};

#endif  // Q_PCL_PLUGIN_POISSONRECONSTRUCTION_DLG_HEADER

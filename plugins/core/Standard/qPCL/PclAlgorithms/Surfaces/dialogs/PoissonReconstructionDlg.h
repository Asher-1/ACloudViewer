// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

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

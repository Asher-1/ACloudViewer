// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ui_cellsFusionDlg.h"

//! Dialog for the extraction of facets based on a cell-fusion strategy (qFacets
//! plugin)
class CellsFusionDlg : public QDialog, public Ui::CellsFusionDlg {
public:
    //! Cell fusion algorithm
    enum Algorithm { ALGO_KD_TREE, ALGO_FAST_MARCHING };

    //! Default constructor
    CellsFusionDlg(Algorithm algo, QWidget* parent = 0)
        : QDialog(parent, Qt::Tool), Ui::CellsFusionDlg() {
        setupUi(this);

        switch (algo) {
            case ALGO_KD_TREE:
                algoComboBox->setCurrentIndex(0);
                octreeLevelSpinBox->setEnabled(false);
                kdTreeCellFusionGroupBox->setVisible(true);
                fmCellFusionGroupBox->setVisible(false);
                break;
            case ALGO_FAST_MARCHING:
                algoComboBox->setCurrentIndex(1);
                octreeLevelSpinBox->setEnabled(true);
                kdTreeCellFusionGroupBox->setVisible(false);
                fmCellFusionGroupBox->setVisible(true);
                break;
        }
    }
};

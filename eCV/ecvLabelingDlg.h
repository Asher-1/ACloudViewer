// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_labelingDlg.h>

//! Dialog to define connected components labelinng parameters
class ccLabelingDlg : public QDialog, public Ui::LabelingDialog {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccLabelingDlg(QWidget* parent = 0);

    //! Returns octree level (defines grid step)
    int getOctreeLevel();

    //! Returns min number of points per extracted CC
    int getMinPointsNb();

    //! Specifies whether each extracted CC should get a random color
    bool randomColors();
};

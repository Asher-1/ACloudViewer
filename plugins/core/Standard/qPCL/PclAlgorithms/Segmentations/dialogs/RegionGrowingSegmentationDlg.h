// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_RegionGrowingSegmentationDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class RegionGrowingSegmentationDlg : public QDialog,
                                     public Ui::RegionGrowingSegmentationDlg {
public:
    explicit RegionGrowingSegmentationDlg(QWidget* parent = 0);
};

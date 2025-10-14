// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_REGIONGROWING_DLG_HEADER
#define Q_PCL_PLUGIN_REGIONGROWING_DLG_HEADER

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

#endif  // Q_PCL_PLUGIN_REGIONGROWING_DLG_HEADER

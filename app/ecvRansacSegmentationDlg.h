// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_ransacSegmentationDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class ecvRansacSegmentationDlg : public QDialog,
                                 public Ui::RansacSegmentationDlg {
public:
    explicit ecvRansacSegmentationDlg(QWidget* parent = 0);
};

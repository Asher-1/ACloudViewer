// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_MarchingCubeDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class MarchingCubeDlg : public QDialog, public Ui::MarchingCubeDlg {
public:
    explicit MarchingCubeDlg(QWidget* parent = 0);
};

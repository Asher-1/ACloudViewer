// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "isosurfacewindow.h"

class ThresholdWindow : public IsosurfaceWindow {
    Q_OBJECT
public:
    explicit ThresholdWindow(QWidget* parent = nullptr);
    ~ThresholdWindow();

    void apply();
};

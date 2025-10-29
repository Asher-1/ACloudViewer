// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore>
#include <QtWidgets>

#include "base/reconstruction.h"
#include "ui/options_widget.h"
#include "ui/thread_control_widget.h"
#include "util/option_manager.h"

namespace colmap {

class MainWindow;

class BundleAdjustmentWidget : public OptionsWidget {
public:
    BundleAdjustmentWidget(MainWindow* main_window, OptionManager* options);

    void Show(Reconstruction* reconstruction);

private:
    void Run();
    void Render();

    MainWindow* main_window_;
    OptionManager* options_;
    Reconstruction* reconstruction_;
    ThreadControlWidget* thread_control_widget_;
    QAction* render_action_;
};

}  // namespace colmap

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "controllers/automatic_reconstruction.h"
#include "ui/options_widget.h"
#include "ui/thread_control_widget.h"

namespace colmap {

class MainWindow;

class AutomaticReconstructionWidget : public OptionsWidget {
public:
    AutomaticReconstructionWidget(MainWindow* main_window);

    void Run();

private:
    void RenderResult();

    MainWindow* main_window_;
    AutomaticReconstructionController::Options options_;
    ThreadControlWidget* thread_control_widget_;
    QComboBox* data_type_cb_;
    QComboBox* quality_cb_;
    QComboBox* mesher_cb_;
    QAction* render_result_;
};

}  // namespace colmap

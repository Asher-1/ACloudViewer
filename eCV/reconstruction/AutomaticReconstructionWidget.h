// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "OptionsWidget.h"
#include "controllers/AutomaticReconstructionController.h"

namespace cloudViewer {
class ReconstructionWidget;
class ThreadControlWidget;

class AutomaticReconstructionWidget : public OptionsWidget {
public:
    AutomaticReconstructionWidget(ReconstructionWidget* main_window);

    void Run();

private:
    void RenderResult();

    ReconstructionWidget* main_window_;
    AutomaticReconstructionController::Options options_;
    ThreadControlWidget* thread_control_widget_;
    QComboBox* data_type_cb_;
    QComboBox* quality_cb_;
    QComboBox* mesher_cb_;
    QAction* render_result_;

    std::vector<std::string> meshing_paths_;
    std::vector<std::vector<colmap::PlyPoint>> fused_points_;
};

}  // namespace cloudViewer

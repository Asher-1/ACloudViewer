// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "controllers/AutomaticReconstructionController.h"
#include "ui/options_widget.h"

#include <QtWidgets>

namespace cloudViewer {
class ReconstructionWidget;
class ThreadControlWidget;

class AutomaticReconstructionWidget : public colmap::OptionsWidget {
public:
    AutomaticReconstructionWidget(ReconstructionWidget* main_window);

    void Run();

protected:
    void showEvent(QShowEvent* event) override;

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
    std::vector<std::string> textured_paths_;
    std::vector<std::vector<colmap::PlyPoint>> fused_points_;
    bool texturing_success_ = false;
};

}  // namespace cloudViewer

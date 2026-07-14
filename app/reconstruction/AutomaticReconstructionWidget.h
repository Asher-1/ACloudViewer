// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtWidgets>

#include "controllers/AutomaticReconstructionController.h"
#include "ui/da3_reconstruction_ui_bindings.h"
#include "ui/options_widget.h"

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

    QComboBox* sparse_mode_cb_;
    QComboBox* stereo_mode_cb_;
    QComboBox* da3_model_cb_;
    QComboBox* da3_quant_cb_;
    QCheckBox* dense_cb_;
    colmap::DA3ReconstructionUiControls da3_ui_controls_;

    QAction* render_result_;

    std::vector<std::string> meshing_paths_;
    std::vector<std::string> textured_paths_;
    std::vector<std::vector<colmap::PlyPoint>> fused_points_;
    bool texturing_success_ = false;
};

}  // namespace cloudViewer

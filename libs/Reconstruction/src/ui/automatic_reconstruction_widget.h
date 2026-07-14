// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QShowEvent>

#include "controllers/automatic_reconstruction.h"
#include "ui/da3_reconstruction_ui_bindings.h"
#include "ui/options_widget.h"
#include "ui/thread_control_widget.h"

namespace colmap {

class MainWindow;

class AutomaticReconstructionWidget : public OptionsWidget {
public:
    AutomaticReconstructionWidget(MainWindow* main_window);

    void Run();

protected:
    void showEvent(QShowEvent* event) override;

private:
    void RenderResult();

    MainWindow* main_window_;
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
    DA3ReconstructionUiControls da3_ui_controls_;
    QAction* render_result_;
};

}  // namespace colmap

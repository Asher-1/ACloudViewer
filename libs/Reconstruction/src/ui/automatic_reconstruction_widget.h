// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDoubleSpinBox>
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
    QLabel* da3_hybrid_hint_label_;
    QComboBox* da3_sparse_model_cb_;
    QComboBox* da3_sparse_quant_cb_;
    QComboBox* da3_stereo_model_cb_;
    QComboBox* da3_stereo_quant_cb_;
    QLabel* da3_sparse_model_label_;
    QLabel* da3_sparse_quant_label_;
    QLabel* da3_stereo_model_label_;
    QLabel* da3_stereo_quant_label_;
    QCheckBox* dense_cb_;
    QCheckBox* meshing_cb_;
    QCheckBox* texturing_cb_;
    QCheckBox* fused_point_filter_cb_;
    QLabel* fused_point_filter_label_;
    QLabel* fused_voxel_size_label_;
    QDoubleSpinBox* fused_voxel_size_spin_;
    DA3ReconstructionUiControls da3_ui_controls_;
    QAction* render_result_;
};

}  // namespace colmap

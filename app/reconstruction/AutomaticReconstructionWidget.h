// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QFutureWatcher>
#include <QtWidgets>
#include <atomic>
#include <functional>
#include <memory>

#include "controllers/AutomaticReconstructionController.h"
#include "ui/da3_reconstruction_ui_bindings.h"
#include "ui/options_widget.h"

namespace cloudViewer {
class ReconstructionWidget;
class ThreadControlWidget;

class AutomaticReconstructionWidget : public colmap::OptionsWidget {
public:
    AutomaticReconstructionWidget(ReconstructionWidget* main_window);
    ~AutomaticReconstructionWidget() override;

    void Run();

protected:
    void showEvent(QShowEvent* event) override;

private:
    struct DownloadResult {
        bool canceled = false;
        QString error;
    };
    using DownloadProgress = std::function<void(int64_t, int64_t)>;
    using DownloadOperation = std::function<std::string(
            std::atomic_bool&, const DownloadProgress&)>;

    void startBackgroundDownload(const QString& title,
                                 const QString& initial_label,
                                 DownloadOperation operation);
    void RenderResult();
    void applyAICoreUiAvailability();

    ReconstructionWidget* main_window_;
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
    QComboBox* da3_device_cb_;
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
    colmap::DA3ReconstructionUiControls da3_ui_controls_;

    QAction* render_result_;
    QFutureWatcher<DownloadResult>* download_watcher_ = nullptr;
    std::shared_ptr<std::atomic_bool> download_cancelled_;

    std::vector<std::string> meshing_paths_;
    std::vector<std::string> textured_paths_;
    std::vector<std::vector<colmap::PlyPoint>> fused_points_;
    bool texturing_success_ = false;
    colmap::DA3VramCapWarning da3_vram_warning_;
};

}  // namespace cloudViewer

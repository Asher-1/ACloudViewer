// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "ui/automatic_reconstruction_widget.h"

#include <QMessageBox>
#include <QProgressDialog>
#include <QShowEvent>

#include <filesystem>
#include <set>

#include "controllers/da3_depth_controller.h"
#include "ui/da3_reconstruction_ui_bindings.h"
#include "ui/main_window.h"
#ifdef COLMAP_DOWNLOAD_ENABLED
#include "util/download.h"
#endif

namespace colmap {

AutomaticReconstructionWidget::AutomaticReconstructionWidget(
    MainWindow* main_window)
    : OptionsWidget(main_window),
      main_window_(main_window),
      thread_control_widget_(new ThreadControlWidget(this)) {
  setWindowTitle("Automatic reconstruction");

  AddOptionDirPath(&options_.workspace_path, "Workspace folder");
  AddSpacer();
  AddOptionDirPath(&options_.image_path, "Image folder");
  AddSpacer();
  AddOptionDirPath(&options_.mask_path, "Mask folder");
  AddSpacer();
  AddOptionFilePath(&options_.vocab_tree_path, "Vocabulary tree<br>(optional)");

  AddSpacer();

  QLabel* data_type_label = new QLabel(tr("Data type"), this);
  data_type_label->setFont(font());
  data_type_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(data_type_label, grid_layout_->rowCount(), 0);

  data_type_cb_ = new QComboBox(this);
  data_type_cb_->addItem("Individual images");
  data_type_cb_->addItem("Video frames");
  data_type_cb_->addItem("Internet images");
  grid_layout_->addWidget(data_type_cb_, grid_layout_->rowCount() - 1, 1);

  QLabel* quality_label = new QLabel(tr("Quality"), this);
  quality_label->setFont(font());
  quality_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(quality_label, grid_layout_->rowCount(), 0);

  quality_cb_ = new QComboBox(this);
  quality_cb_->addItem("Low");
  quality_cb_->addItem("Medium");
  quality_cb_->addItem("High");
  quality_cb_->addItem("Extreme");
  quality_cb_->setCurrentIndex(2);
  grid_layout_->addWidget(quality_cb_, grid_layout_->rowCount() - 1, 1);

  AddSpacer();

  AddOptionBool(&options_.single_camera, "Shared intrinsics");
  AddOptionBool(&options_.sparse, "Sparse model");
  dense_cb_ = AddOptionBool(&options_.dense, "Dense model");
  meshing_cb_ = AddOptionBool(&options_.meshing, "Surface meshing");
  texturing_cb_ = AddOptionBool(&options_.texturing, "Mesh texturing");

  QLabel* mesher_label = new QLabel(tr("Mesher"), this);
  mesher_label->setFont(font());
  mesher_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(mesher_label, grid_layout_->rowCount(), 0);

  mesher_cb_ = new QComboBox(this);
  mesher_cb_->addItem("Delaunay");
  mesher_cb_->addItem("Poisson");
  mesher_cb_->setCurrentIndex(0);
  grid_layout_->addWidget(mesher_cb_, grid_layout_->rowCount() - 1, 1);

  connect(meshing_cb_, &QCheckBox::toggled, mesher_cb_, &QWidget::setEnabled);
  connect(meshing_cb_, &QCheckBox::toggled, texturing_cb_,
          [this](bool meshing_enabled) {
            texturing_cb_->setEnabled(meshing_enabled);
            if (!meshing_enabled) {
              texturing_cb_->blockSignals(true);
              texturing_cb_->setChecked(false);
              texturing_cb_->blockSignals(false);
            }
          });
  mesher_cb_->setEnabled(meshing_cb_->isChecked());
  texturing_cb_->setEnabled(meshing_cb_->isChecked());

  AddSpacer();

  // --- DA3 (Depth Anything V3) options ---
  QLabel* sparse_mode_label = new QLabel(tr("Sparse mode"), this);
  sparse_mode_label->setFont(font());
  sparse_mode_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(sparse_mode_label, grid_layout_->rowCount(), 0);

  sparse_mode_cb_ = new QComboBox(this);
  DA3ReconstructionUiBindings::InitSparseModeComboBox(sparse_mode_cb_);
  grid_layout_->addWidget(sparse_mode_cb_, grid_layout_->rowCount() - 1, 1);

  da3_hybrid_hint_label_ = new QLabel(this);
  da3_hybrid_hint_label_->setWordWrap(true);
  da3_hybrid_hint_label_->setStyleSheet("color: palette(mid); font-size: 11px;");
  da3_hybrid_hint_label_->hide();
  grid_layout_->addWidget(da3_hybrid_hint_label_, grid_layout_->rowCount(), 1);

  QLabel* stereo_mode_label = new QLabel(tr("Stereo mode"), this);
  stereo_mode_label->setFont(font());
  stereo_mode_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(stereo_mode_label, grid_layout_->rowCount(), 0);

  stereo_mode_cb_ = new QComboBox(this);
  DA3ReconstructionUiBindings::InitStereoModeComboBox(stereo_mode_cb_);
  grid_layout_->addWidget(stereo_mode_cb_, grid_layout_->rowCount() - 1, 1);

  da3_sparse_model_label_ = new QLabel(tr("DA3 sparse model"), this);
  da3_sparse_model_label_->setFont(font());
  da3_sparse_model_label_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(da3_sparse_model_label_, grid_layout_->rowCount(), 0);

  da3_sparse_model_cb_ = new QComboBox(this);
  DA3ReconstructionUiBindings::InitSparseModelComboBox(da3_sparse_model_cb_);
  grid_layout_->addWidget(da3_sparse_model_cb_, grid_layout_->rowCount() - 1, 1);

  da3_sparse_quant_label_ = new QLabel(tr("DA3 sparse quant"), this);
  da3_sparse_quant_label_->setFont(font());
  da3_sparse_quant_label_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(da3_sparse_quant_label_, grid_layout_->rowCount(), 0);

  da3_sparse_quant_cb_ = new QComboBox(this);
  grid_layout_->addWidget(da3_sparse_quant_cb_, grid_layout_->rowCount() - 1, 1);

  da3_stereo_model_label_ = new QLabel(tr("DA3 stereo model"), this);
  da3_stereo_model_label_->setFont(font());
  da3_stereo_model_label_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(da3_stereo_model_label_, grid_layout_->rowCount(), 0);

  da3_stereo_model_cb_ = new QComboBox(this);
  DA3ReconstructionUiBindings::InitStereoModelComboBox(da3_stereo_model_cb_);
  grid_layout_->addWidget(da3_stereo_model_cb_, grid_layout_->rowCount() - 1, 1);

  da3_stereo_quant_label_ = new QLabel(tr("DA3 stereo quant"), this);
  da3_stereo_quant_label_->setFont(font());
  da3_stereo_quant_label_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(da3_stereo_quant_label_, grid_layout_->rowCount(), 0);

  da3_stereo_quant_cb_ = new QComboBox(this);
  grid_layout_->addWidget(da3_stereo_quant_cb_, grid_layout_->rowCount() - 1, 1);

  da3_ui_controls_.da3_force_recompute_label =
      new QLabel(tr("DA3 force recompute"), this);
  da3_ui_controls_.da3_force_recompute_label->setFont(font());
  da3_ui_controls_.da3_force_recompute_label->setAlignment(
      Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(da3_ui_controls_.da3_force_recompute_label,
                          grid_layout_->rowCount(), 0);

  da3_ui_controls_.da3_force_recompute_cb =
      new QCheckBox(tr("Ignore cached sparse/stereo outputs"), this);
  grid_layout_->addWidget(da3_ui_controls_.da3_force_recompute_cb,
                          grid_layout_->rowCount() - 1, 1);

  da3_ui_controls_.da3_skip_geometric_refine_label =
      new QLabel(tr("Skip geometric refine"), this);
  da3_ui_controls_.da3_skip_geometric_refine_label->setFont(font());
  da3_ui_controls_.da3_skip_geometric_refine_label->setAlignment(
      Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(da3_ui_controls_.da3_skip_geometric_refine_label,
                          grid_layout_->rowCount(), 0);

  da3_ui_controls_.da3_skip_geometric_refine_cb = new QCheckBox(
      tr("Fuse DA3 priors directly (auto-fallback if sparse)"), this);
  grid_layout_->addWidget(da3_ui_controls_.da3_skip_geometric_refine_cb,
                          grid_layout_->rowCount() - 1, 1);

  da3_ui_controls_.sparse_mode_cb = sparse_mode_cb_;
  da3_ui_controls_.stereo_mode_cb = stereo_mode_cb_;
  da3_ui_controls_.da3_sparse_model_cb = da3_sparse_model_cb_;
  da3_ui_controls_.da3_sparse_quant_cb = da3_sparse_quant_cb_;
  da3_ui_controls_.da3_stereo_model_cb = da3_stereo_model_cb_;
  da3_ui_controls_.da3_stereo_quant_cb = da3_stereo_quant_cb_;
  da3_ui_controls_.da3_sparse_model_label = da3_sparse_model_label_;
  da3_ui_controls_.da3_sparse_quant_label = da3_sparse_quant_label_;
  da3_ui_controls_.da3_stereo_model_label = da3_stereo_model_label_;
  da3_ui_controls_.da3_stereo_quant_label = da3_stereo_quant_label_;
  da3_ui_controls_.da3_hybrid_hint_label = da3_hybrid_hint_label_;
  da3_ui_controls_.dense_cb = dense_cb_;
  DA3ReconstructionUiBindings::Install(da3_ui_controls_, this);

  AddOptionFilePath(&options_.da3_sparse_model_path,
                    "DA3 sparse model path<br>(optional, auto-download)");
  AddOptionFilePath(&options_.da3_stereo_model_path,
                    "DA3 stereo model path<br>(optional, auto-download)");

  AddSpacer();

  AddOptionInt(&options_.num_threads, "num_threads", -1);
  AddOptionBool(&options_.use_gpu, "GPU");
  AddOptionText(&options_.gpu_index, "gpu_index");

  fused_point_filter_label_ = new QLabel(tr("Fused point filter"), this);
  fused_point_filter_label_->setFont(font());
  fused_point_filter_label_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(fused_point_filter_label_, grid_layout_->rowCount(), 0);

  fused_point_filter_cb_ =
      new QCheckBox(tr("Voxel downsample + statistical outlier removal"), this);
  grid_layout_->addWidget(fused_point_filter_cb_, grid_layout_->rowCount() - 1,
                          1);

  fused_voxel_size_label_ = new QLabel(tr("Fused voxel size (m)"), this);
  fused_voxel_size_label_->setFont(font());
  fused_voxel_size_label_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(fused_voxel_size_label_, grid_layout_->rowCount(), 0);

  fused_voxel_size_spin_ = new QDoubleSpinBox(this);
  fused_voxel_size_spin_->setDecimals(3);
  fused_voxel_size_spin_->setRange(0.001, 1.0);
  fused_voxel_size_spin_->setSingleStep(0.005);
  fused_voxel_size_spin_->setValue(0.02);
  grid_layout_->addWidget(fused_voxel_size_spin_, grid_layout_->rowCount() - 1,
                          1);

  connect(fused_point_filter_cb_, &QCheckBox::toggled, fused_voxel_size_spin_,
          &QWidget::setEnabled);
  connect(fused_point_filter_cb_, &QCheckBox::toggled, fused_voxel_size_label_,
          &QWidget::setEnabled);
  fused_voxel_size_spin_->setEnabled(false);
  fused_voxel_size_label_->setEnabled(false);

  da3_ui_controls_.fused_point_filter_cb = fused_point_filter_cb_;
  da3_ui_controls_.fused_voxel_size_spin = fused_voxel_size_spin_;
  da3_ui_controls_.fused_voxel_size_label = fused_voxel_size_label_;
  DA3ReconstructionUiBindings::ApplyDa3FusedPointFilterDefaults(da3_ui_controls_);

#ifdef AICore_ENABLED
  DA3ReconstructionUiBindings::SetAICoreAvailable(da3_ui_controls_, true);
#else
  DA3ReconstructionUiBindings::SetAICoreAvailable(da3_ui_controls_, false);
  HideOption(&options_.da3_sparse_model_path);
  HideOption(&options_.da3_stereo_model_path);
#endif

  AddSpacer();

  QPushButton* run_button = new QPushButton(tr("Run"), this);
  grid_layout_->addWidget(run_button, grid_layout_->rowCount(), 1);
  connect(run_button, &QPushButton::released, this,
          &AutomaticReconstructionWidget::Run);

  render_result_ = new QAction(this);
  connect(render_result_, &QAction::triggered, this,
          &AutomaticReconstructionWidget::RenderResult, Qt::QueuedConnection);
}

void AutomaticReconstructionWidget::showEvent(QShowEvent* event) {
  OptionsWidget::showEvent(event);
  DA3ReconstructionUiBindings::ApplyHybridDenseTooltips(da3_ui_controls_,
                                                        options_.image_path);
  DA3ReconstructionUiBindings::Sync(da3_ui_controls_);
}

void AutomaticReconstructionWidget::Run() {
  WriteOptions();

  if (!ExistsDir(options_.workspace_path)) {
    QMessageBox::critical(this, "", tr("Invalid workspace folder"));
    return;
  }

  if (!ExistsDir(options_.image_path)) {
    QMessageBox::critical(this, "", tr("Invalid image folder"));
    return;
  }

  DA3ReconstructionUiBindings::ApplyHybridDenseTooltips(da3_ui_controls_,
                                                        options_.image_path);

  switch (data_type_cb_->currentIndex()) {
    case 0:
      options_.data_type =
          AutomaticReconstructionController::DataType::INDIVIDUAL;
      break;
    case 1:
      options_.data_type = AutomaticReconstructionController::DataType::VIDEO;
      break;
    case 2:
      options_.data_type =
          AutomaticReconstructionController::DataType::INTERNET;
      break;
    default:
      options_.data_type =
          AutomaticReconstructionController::DataType::INDIVIDUAL;
      break;
  }

  switch (quality_cb_->currentIndex()) {
    case 0:
      options_.quality = AutomaticReconstructionController::Quality::LOW;
      break;
    case 1:
      options_.quality = AutomaticReconstructionController::Quality::MEDIUM;
      break;
    case 2:
      options_.quality = AutomaticReconstructionController::Quality::HIGH;
      break;
    case 3:
      options_.quality = AutomaticReconstructionController::Quality::EXTREME;
      break;
    default:
      options_.quality = AutomaticReconstructionController::Quality::HIGH;
      break;
  }

  switch (mesher_cb_->currentIndex()) {
    case 0:
        options_.mesher =
        AutomaticReconstructionController::Mesher::DELAUNAY;
        break;
    case 1:
        options_.mesher =
        AutomaticReconstructionController::Mesher::POISSON;
        break;
    default:
        options_.mesher =
                AutomaticReconstructionController::Mesher::DELAUNAY;
        break;
}

  // DA3 sparse model mode
  switch (sparse_mode_cb_->currentIndex()) {
    case 0:
      options_.sparse_mode = SparseModelMode::COLMAP_NATIVE;
      break;
    case 1:
      options_.sparse_mode = SparseModelMode::DA3_DEPTH_POSE;
      break;
    default:
      options_.sparse_mode = SparseModelMode::COLMAP_NATIVE;
      break;
  }

  // DA3 stereo pipeline mode
  switch (stereo_mode_cb_->currentIndex()) {
    case 0:
      options_.stereo_mode = StereoPipelineMode::COLMAP_PATCH_MATCH;
      break;
    case 1:
      options_.stereo_mode = StereoPipelineMode::DA3_DEPTH_INFERENCE;
      break;
    default:
      options_.stereo_mode = StereoPipelineMode::COLMAP_PATCH_MATCH;
      break;
  }

  options_.da3_sparse_model_type =
      DA3ReconstructionUiBindings::SparseModelTypeFromIndex(
          da3_sparse_model_cb_->currentIndex());
  options_.da3_sparse_quant_type =
      DA3ReconstructionUiBindings::QuantTypeFromComboText(
          da3_sparse_quant_cb_->currentText());

  options_.da3_stereo_model_type =
      DA3ReconstructionUiBindings::StereoModelTypeFromIndex(
          da3_stereo_model_cb_->currentIndex());
  options_.da3_stereo_quant_type =
      DA3ReconstructionUiBindings::QuantTypeFromComboText(
          da3_stereo_quant_cb_->currentText());

  options_.da3_force_recompute =
      da3_ui_controls_.da3_force_recompute_cb &&
      da3_ui_controls_.da3_force_recompute_cb->isChecked();
  options_.da3_skip_geometric_refine =
      da3_ui_controls_.da3_skip_geometric_refine_cb &&
      da3_ui_controls_.da3_skip_geometric_refine_cb->isChecked();

  options_.fused_point_filter.enabled =
      fused_point_filter_cb_ && fused_point_filter_cb_->isChecked();
  options_.fused_point_filter.voxel_size =
      fused_voxel_size_spin_ ? fused_voxel_size_spin_->value() : 0.02;
  options_.fused_point_filter.sor_enabled = true;

  if (options_.stereo_mode == StereoPipelineMode::DA3_DEPTH_INFERENCE &&
      options_.sparse_mode != SparseModelMode::DA3_DEPTH_POSE) {
    QMessageBox::information(
        this, tr("DA3 stereo"),
        tr("DA3 depth inference works best with DA3 (depth+pose) sparse mode "
           "for consistent camera poses. Continuing with the current "
           "sparse/stereo model selection."));
  }

  const bool uses_da3_sparse =
      options_.sparse_mode == SparseModelMode::DA3_DEPTH_POSE;
  const bool uses_da3_stereo =
      options_.stereo_mode == StereoPipelineMode::DA3_DEPTH_INFERENCE;
  if (uses_da3_sparse || uses_da3_stereo) {
    const std::string cache_dir = DA3ModelCacheDir();
    std::filesystem::create_directories(cache_dir);

    std::vector<DA3ModelCacheNeed> needed;
    if (uses_da3_sparse) {
      CollectDA3ModelCacheNeeds(
          cache_dir, options_.da3_sparse_model_type,
          options_.da3_sparse_quant_type, options_.da3_sparse_model_path,
          options_.da3_sparse_metric_model_path, needed);
    }
    if (uses_da3_stereo) {
      CollectDA3ModelCacheNeeds(
          cache_dir, options_.da3_stereo_model_type,
          options_.da3_stereo_quant_type, options_.da3_stereo_model_path,
          options_.da3_stereo_metric_model_path, needed);
    }

    if (!needed.empty()) {
#ifdef COLMAP_DOWNLOAD_ENABLED
      QStringList names;
      std::set<std::string> seen_names;
      for (const auto& m : needed) {
        if (seen_names.insert(m.filename).second) {
          names << QString::fromStdString(m.filename);
        }
      }
      auto answer = QMessageBox::question(
          this, tr("Download DA3 Model(s)"),
          tr("The following DA3 model(s) are not cached locally:\n\n"
             "  %1\n\nDownload them now?").arg(names.join("\n  ")),
          QMessageBox::Yes | QMessageBox::No);
      if (answer != QMessageBox::Yes) return;

      for (auto& m : needed) {
        const auto target = std::filesystem::path(cache_dir) / m.filename;
        if (std::filesystem::exists(target)) {
          if (m.dest_path) *m.dest_path = target.string();
          continue;
        }

        QProgressDialog progress_dialog(
            tr("Downloading %1...").arg(QString::fromStdString(m.filename)),
            tr("Cancel"), 0, 100, this);
        progress_dialog.setWindowModality(Qt::ApplicationModal);
        progress_dialog.setWindowTitle(tr("Downloading DA3 Model"));
        progress_dialog.setAutoClose(false);
        progress_dialog.setAutoReset(false);
        progress_dialog.setMinimumDuration(0);
        progress_dialog.show();
        QApplication::processEvents();

        bool download_canceled = false;
        DownloadProgressCallback progress_callback =
            [&progress_dialog, &download_canceled](
                int64_t downloaded, int64_t total) {
              QApplication::processEvents();
              if (progress_dialog.wasCanceled()) {
                download_canceled = true;
                return;
              }
              if (total > 0) {
                int percent = static_cast<int>((downloaded * 100) / total);
                progress_dialog.setValue(percent);
                double dl_mb =
                    static_cast<double>(downloaded) / (1024.0 * 1024.0);
                double tot_mb =
                    static_cast<double>(total) / (1024.0 * 1024.0);
                progress_dialog.setLabelText(
                    tr("Downloading...\n%1 MB / %2 MB (%3%)")
                        .arg(dl_mb, 0, 'f', 1)
                        .arg(tot_mb, 0, 'f', 1)
                        .arg(percent));
              }
              QApplication::processEvents();
            };

        try {
          std::string downloaded_path =
              DownloadAndCacheFile(m.url, progress_callback);
          progress_dialog.close();

          if (download_canceled || downloaded_path.empty()) {
            QMessageBox::warning(this, tr("Download Canceled"),
                                 tr("DA3 model download was canceled."));
            return;
          }
          auto target = std::filesystem::path(cache_dir) / m.filename;
          if (downloaded_path != target.string()) {
            std::error_code ec;
            std::filesystem::rename(downloaded_path, target, ec);
            if (ec) {
              std::filesystem::copy_file(
                  downloaded_path, target,
                  std::filesystem::copy_options::overwrite_existing, ec);
              if (!ec) std::filesystem::remove(downloaded_path, ec);
            }
          }
          if (m.dest_path) {
            *m.dest_path = target.string();
          }
          for (auto& other : needed) {
            if (other.filename == m.filename && other.dest_path &&
                other.dest_path != m.dest_path) {
              *other.dest_path = target.string();
            }
          }
        } catch (const std::exception& e) {
          progress_dialog.close();
          QMessageBox::critical(
              this, tr("Download Failed"),
              tr("Failed to download DA3 model '%1': %2")
                  .arg(QString::fromStdString(m.filename))
                  .arg(e.what()));
          return;
        }
      }
#else
      QStringList names;
      for (const auto& m : needed)
        names << QString::fromStdString(m.filename);
      QMessageBox::warning(
          this, tr("DA3 Model Not Found"),
          tr("DA3 model(s) not found and download support is disabled:\n"
             "  %1\nPlease provide the model path manually.")
              .arg(names.join("\n  ")));
      return;
#endif
    }
  }

  main_window_->reconstruction_manager_.Clear();
  main_window_->reconstruction_manager_widget_->Update();
  main_window_->RenderClear();
  main_window_->RenderNow();

  AutomaticReconstructionController* controller =
      new AutomaticReconstructionController(
          options_, &main_window_->reconstruction_manager_);

  controller->AddCallback(Thread::FINISHED_CALLBACK,
                          [this]() { render_result_->trigger(); });

  thread_control_widget_->StartThread("Reconstructing...", true, controller);
}

void AutomaticReconstructionWidget::RenderResult() {
  if (main_window_->reconstruction_manager_.Size() > 0) {
    main_window_->reconstruction_manager_widget_->Update();
    main_window_->RenderClear();
    main_window_->RenderNow();
  }

  if (options_.sparse) {
    QMessageBox::information(
        this, "",
        tr("Imported the reconstructed sparse models for visualization. The "
           "models were also exported to the <i>sparse</i> sub-folder in the "
           "workspace."));
  }

  if (options_.dense) {
    const std::string vram_warning = DA3VramCapWarningMessage();
    if (!vram_warning.empty()) {
      QMessageBox::warning(
          this, tr("DA3 GPU memory"),
          tr("%1").arg(QString::fromStdString(vram_warning)));
    }
    QMessageBox::information(
        this, "",
        tr("To visualize the reconstructed dense point cloud, navigate to the "
           "<i>dense</i> sub-folder in your workspace with <i>File > Import "
           "model from...</i>. To visualize the meshed model, you must use an "
           "external viewer such as Meshlab."));
  }
}

}  // namespace colmap

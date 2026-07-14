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
  AddOptionBool(&options_.texturing, "Mesh texturing");

  QLabel* mesher_label = new QLabel(tr("Mesher"), this);
  mesher_label->setFont(font());
  mesher_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(mesher_label, grid_layout_->rowCount(), 0);

  mesher_cb_ = new QComboBox(this);
  mesher_cb_->addItem("Delaunay");
  mesher_cb_->addItem("Poisson");
  mesher_cb_->setCurrentIndex(0);
  grid_layout_->addWidget(mesher_cb_, grid_layout_->rowCount() - 1, 1);

  AddSpacer();

  // --- DA3 (Depth Anything V3) options ---
  QLabel* sparse_mode_label = new QLabel(tr("Sparse mode"), this);
  sparse_mode_label->setFont(font());
  sparse_mode_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(sparse_mode_label, grid_layout_->rowCount(), 0);

  sparse_mode_cb_ = new QComboBox(this);
  sparse_mode_cb_->addItem("COLMAP (native SfM)");
  sparse_mode_cb_->addItem("DA3 (depth+pose)");
  sparse_mode_cb_->setCurrentIndex(0);
  grid_layout_->addWidget(sparse_mode_cb_, grid_layout_->rowCount() - 1, 1);

  QLabel* stereo_mode_label = new QLabel(tr("Stereo mode"), this);
  stereo_mode_label->setFont(font());
  stereo_mode_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(stereo_mode_label, grid_layout_->rowCount(), 0);

  stereo_mode_cb_ = new QComboBox(this);
  DA3ReconstructionUiBindings::InitStereoComboBox(stereo_mode_cb_);
  grid_layout_->addWidget(stereo_mode_cb_, grid_layout_->rowCount() - 1, 1);

  QLabel* da3_model_label = new QLabel(tr("DA3 model"), this);
  da3_model_label->setFont(font());
  da3_model_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(da3_model_label, grid_layout_->rowCount(), 0);

  da3_model_cb_ = new QComboBox(this);
  da3_model_cb_->addItem("Base (ViT-S, fastest)");
  da3_model_cb_->addItem("Large (ViT-L)");
  da3_model_cb_->addItem("Giant (ViT-G, best quality)");
  da3_model_cb_->addItem("Nested Metric");
  da3_model_cb_->addItem("Nested AnyView");
  da3_model_cb_->setCurrentIndex(0);
  grid_layout_->addWidget(da3_model_cb_, grid_layout_->rowCount() - 1, 1);

  QLabel* da3_quant_label = new QLabel(tr("DA3 quantization"), this);
  da3_quant_label->setFont(font());
  da3_quant_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  grid_layout_->addWidget(da3_quant_label, grid_layout_->rowCount(), 0);

  da3_quant_cb_ = new QComboBox(this);
  grid_layout_->addWidget(da3_quant_cb_, grid_layout_->rowCount() - 1, 1);

  da3_ui_controls_.sparse_mode_cb = sparse_mode_cb_;
  da3_ui_controls_.stereo_mode_cb = stereo_mode_cb_;
  da3_ui_controls_.da3_model_cb = da3_model_cb_;
  da3_ui_controls_.da3_quant_cb = da3_quant_cb_;
  da3_ui_controls_.dense_cb = dense_cb_;
  DA3ReconstructionUiBindings::Install(da3_ui_controls_, this);
  da3_model_cb_->setCurrentIndex(0);

  AddOptionFilePath(&options_.da3_model_path, "DA3 model path<br>(optional, auto-download)");

  AddSpacer();

  AddOptionInt(&options_.num_threads, "num_threads", -1);
  AddOptionBool(&options_.use_gpu, "GPU");
  AddOptionText(&options_.gpu_index, "gpu_index");

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
  DA3ReconstructionUiBindings::Sync(da3_ui_controls_);
}

void AutomaticReconstructionWidget::Run() {
  WriteOptions();

  if (stereo_mode_cb_->currentIndex() ==
          DA3ReconstructionUiBindings::kStereoDa3 &&
      !DA3ModelSupportsStereo(DA3ReconstructionUiBindings::ModelTypeFromIndex(
          da3_model_cb_->currentIndex()))) {
    QMessageBox::warning(
        this, tr("DA3 stereo"),
        tr("DA3 depth inference requires a Nested model (Nested AnyView or "
           "Nested Metric). Select a nested model or use COLMAP PatchMatch."));
    return;
  }

  if (!ExistsDir(options_.workspace_path)) {
    QMessageBox::critical(this, "", tr("Invalid workspace folder"));
    return;
  }

  if (!ExistsDir(options_.image_path)) {
    QMessageBox::critical(this, "", tr("Invalid image folder"));
    return;
  }

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

  // DA3 model type
  switch (da3_model_cb_->currentIndex()) {
    case 0: options_.da3_model_type = DA3ModelType::BASE; break;
    case 1: options_.da3_model_type = DA3ModelType::LARGE; break;
    case 2: options_.da3_model_type = DA3ModelType::GIANT; break;
    case 3: options_.da3_model_type = DA3ModelType::NESTED_METRIC; break;
    case 4: options_.da3_model_type = DA3ModelType::NESTED_ANYVIEW; break;
    default: options_.da3_model_type = DA3ModelType::BASE; break;
  }

  // DA3 quantization type — resolved by text since combobox is dynamic
  {
    QString qt = da3_quant_cb_->currentText();
    if (qt.startsWith("Q4_K"))
      options_.da3_quant_type = DA3QuantType::Q4_K;
    else if (qt.startsWith("F16"))
      options_.da3_quant_type = DA3QuantType::F16;
    else if (qt.startsWith("F32"))
      options_.da3_quant_type = DA3QuantType::F32;
    else
      options_.da3_quant_type = DA3QuantType::Q8_0;
  }

  if (options_.stereo_mode == StereoPipelineMode::DA3_DEPTH_INFERENCE &&
      options_.sparse_mode != SparseModelMode::DA3_DEPTH_POSE) {
    QMessageBox::warning(
        this, tr("DA3 stereo"),
        tr("DA3 depth inference requires DA3 (depth+pose) sparse mode for "
           "consistent camera poses. Select DA3 sparse mode or use COLMAP "
           "PatchMatch for stereo."));
    return;
  }

  // --- DA3 model pre-check: ensure model(s) are cached before starting ---
  const bool uses_da3 =
      (options_.sparse_mode == SparseModelMode::DA3_DEPTH_POSE ||
       options_.stereo_mode == StereoPipelineMode::DA3_DEPTH_INFERENCE);
  if (uses_da3 && options_.da3_model_path.empty()) {
    const std::string cache_dir = DA3ModelCacheDir();
    std::filesystem::create_directories(cache_dir);

    struct ModelToCheck {
      std::string filename;
      std::string url;
      std::string* dest_path;
    };
    std::vector<ModelToCheck> needed;

    // Main model
    const std::string main_filename = DA3ModelFilename(
        options_.da3_model_type, options_.da3_quant_type);
    const auto main_cached = std::filesystem::path(cache_dir) / main_filename;
    if (!std::filesystem::exists(main_cached)) {
      needed.push_back({main_filename,
          DA3ModelDownloadURL(options_.da3_model_type, options_.da3_quant_type),
          &options_.da3_model_path});
    } else {
      options_.da3_model_path = main_cached.string();
    }

    // Metric model (only for nested types)
    const bool is_nested =
        options_.da3_model_type == DA3ModelType::NESTED_METRIC ||
        options_.da3_model_type == DA3ModelType::NESTED_ANYVIEW;
    if (is_nested && options_.da3_metric_model_path.empty()) {
      const std::string metric_filename = DA3ModelFilename(
          DA3ModelType::NESTED_METRIC, DA3QuantType::F32);
      const auto metric_cached =
          std::filesystem::path(cache_dir) / metric_filename;
      if (!std::filesystem::exists(metric_cached)) {
        needed.push_back({metric_filename,
            DA3ModelDownloadURL(DA3ModelType::NESTED_METRIC, DA3QuantType::F32),
            &options_.da3_metric_model_path});
      } else {
        options_.da3_metric_model_path = metric_cached.string();
      }
    }

    if (options_.da3_model_type == DA3ModelType::NESTED_METRIC) {
      DA3QuantType anyview_quant = DA3QuantType::Q8_0;
      if (DA3ModelExists(DA3ModelType::NESTED_ANYVIEW, options_.da3_quant_type)) {
        anyview_quant = options_.da3_quant_type;
      }
      const std::string anyview_filename =
          DA3ModelFilename(DA3ModelType::NESTED_ANYVIEW, anyview_quant);
      const auto anyview_cached =
          std::filesystem::path(cache_dir) / anyview_filename;
      if (!std::filesystem::exists(anyview_cached)) {
        needed.push_back(
            {anyview_filename,
             DA3ModelDownloadURL(DA3ModelType::NESTED_ANYVIEW, anyview_quant),
             nullptr});
      }
    }

    if (!needed.empty()) {
#ifdef COLMAP_DOWNLOAD_ENABLED
      QStringList names;
      for (const auto& m : needed)
        names << QString::fromStdString(m.filename);
      auto answer = QMessageBox::question(
          this, tr("Download DA3 Model(s)"),
          tr("The following DA3 model(s) are not cached locally:\n\n"
             "  %1\n\nDownload them now?").arg(names.join("\n  ")),
          QMessageBox::Yes | QMessageBox::No);
      if (answer != QMessageBox::Yes) return;

      for (auto& m : needed) {
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
    QMessageBox::information(
        this, "",
        tr("To visualize the reconstructed dense point cloud, navigate to the "
           "<i>dense</i> sub-folder in your workspace with <i>File > Import "
           "model from...</i>. To visualize the meshed model, you must use an "
           "external viewer such as Meshlab."));
  }
}

}  // namespace colmap

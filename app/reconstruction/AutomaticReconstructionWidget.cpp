// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AutomaticReconstructionWidget.h"

#include <ecvPointCloud.h>

#include <QApplication>
#include <QMessageBox>
#include <QProgressDialog>
#include <QShowEvent>

#include "MainWindow.h"
#include "ReconstructionWidget.h"
#include "ThreadControlWidget.h"
#include "retrieval/resources.h"
#include "util/download.h"

namespace cloudViewer {

using namespace colmap;
AutomaticReconstructionWidget::AutomaticReconstructionWidget(
        ReconstructionWidget* main_window)
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

    AddOptionFilePath(&options_.vocab_tree_path,
                      "Vocabulary tree<br>(optional)");

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
    AddOptionBool(&options_.dense, "Dense model");
    AddOptionBool(&options_.texturing, "Mesh texturing");
    AddOptionBool(&options_.autoVisualization, "Auto visualization");

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

    switch (data_type_cb_->currentIndex()) {
        case 0:
            options_.data_type =
                    AutomaticReconstructionController::DataType::INDIVIDUAL;
            break;
        case 1:
            options_.data_type =
                    AutomaticReconstructionController::DataType::VIDEO;
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
            options_.quality =
                    AutomaticReconstructionController::Quality::MEDIUM;
            break;
        case 2:
            options_.quality = AutomaticReconstructionController::Quality::HIGH;
            break;
        case 3:
            options_.quality =
                    AutomaticReconstructionController::Quality::EXTREME;
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

    // Check if vocab_tree_path is a URI and needs to be downloaded
    std::string vocab_tree_path = options_.vocab_tree_path;
    if (vocab_tree_path.empty()) {
        vocab_tree_path = retrieval::kDefaultVocabTreeUri;
    }

    // If it's a URI, check if it's already cached or needs to be downloaded
    if (!vocab_tree_path.empty() && colmap::IsURI(vocab_tree_path)) {
#ifdef COLMAP_DOWNLOAD_ENABLED
        // First, check if the file is already cached
        std::filesystem::path cached_path =
                colmap::GetCachedFilePath(vocab_tree_path);
        if (!cached_path.empty() && std::filesystem::exists(cached_path)) {
            // File already exists in cache, use it directly
            LOG(INFO) << "Using cached vocabulary tree file: " << cached_path;
            options_.vocab_tree_path = cached_path.string();
        } else {
            // File doesn't exist, show download progress dialog
            QProgressDialog progress_dialog(
                    tr("Downloading vocabulary tree..."), tr("Cancel"), 0, 100,
                    this);
            progress_dialog.setWindowModality(Qt::ApplicationModal);
            progress_dialog.setWindowTitle(tr("Downloading"));
            progress_dialog.setAutoClose(false);
            progress_dialog.setAutoReset(false);
            progress_dialog.setMinimumDuration(0);
            progress_dialog.show();
            QApplication::processEvents();

            bool download_canceled = false;
            colmap::DownloadProgressCallback progress_callback =
                    [&progress_dialog, &download_canceled](int64_t downloaded,
                                                           int64_t total) {
                        QApplication::processEvents();
                        if (progress_dialog.wasCanceled()) {
                            download_canceled = true;
                            return;
                        }

                        if (total > 0) {
                            int percent = static_cast<int>((downloaded * 100) /
                                                           total);
                            progress_dialog.setValue(percent);

                            // Update label with size information
                            double downloaded_mb =
                                    static_cast<double>(downloaded) /
                                    (1024.0 * 1024.0);
                            double total_mb = static_cast<double>(total) /
                                              (1024.0 * 1024.0);
                            progress_dialog.setLabelText(
                                    tr("Downloading vocabulary tree...\n%1 MB "
                                       "/ %2 MB (%3%)")
                                            .arg(downloaded_mb, 0, 'f', 2)
                                            .arg(total_mb, 0, 'f', 2)
                                            .arg(percent));
                        } else {
                            progress_dialog.setValue(0);
                            double downloaded_mb =
                                    static_cast<double>(downloaded) /
                                    (1024.0 * 1024.0);
                            progress_dialog.setLabelText(
                                    tr("Downloading vocabulary tree...\n%1 MB")
                                            .arg(downloaded_mb, 0, 'f', 2));
                        }
                        QApplication::processEvents();
                    };

            try {
                std::string downloaded_path = colmap::DownloadAndCacheFile(
                        vocab_tree_path, progress_callback);
                progress_dialog.close();

                if (download_canceled || downloaded_path.empty()) {
                    QMessageBox::warning(
                            this, tr("Download Canceled"),
                            tr("Vocabulary tree download was canceled. Please "
                               "provide a local path."));
                    return;
                }

                // Update options with the downloaded path
                options_.vocab_tree_path = downloaded_path;
            } catch (const std::exception& e) {
                progress_dialog.close();
                QMessageBox::critical(
                        this, tr("Download Failed"),
                        tr("Failed to download vocabulary tree: %1")
                                .arg(e.what()));
                return;
            }
        }
#else
        QMessageBox::warning(this, tr("Download Disabled"),
                             tr("Download support is disabled. Please provide "
                                "a local vocabulary tree path."));
        return;
#endif
    }

    main_window_->reconstruction_manager_.Clear();
    main_window_->reconstruction_manager_widget_->Update();
    main_window_->RenderClear();
    main_window_->RenderNow();

    AutomaticReconstructionController* controller =
            new AutomaticReconstructionController(
                    options_, &main_window_->reconstruction_manager_);

    controller->AddCallback(Thread::FINISHED_CALLBACK, [this, controller]() {
        fused_points_ = controller->fused_points_;
        meshing_paths_ = controller->meshing_paths_;
        textured_paths_ = controller->textured_paths_;
        texturing_success_ = controller->texturing_success_;
        controller->fused_points_.clear();
        controller->meshing_paths_.clear();
        controller->textured_paths_.clear();
        render_result_->trigger();
    });

    thread_control_widget_->StartThread("Reconstructing...", true, controller);
}

void AutomaticReconstructionWidget::showEvent(QShowEvent* event) {
    // Ensure vocab_tree_path has default value before reading options
    // This ensures that even if WriteOptions() previously saved an empty value,
    // we restore the default value when the window is shown
    if (options_.vocab_tree_path.empty()) {
        options_.vocab_tree_path = retrieval::kDefaultVocabTreeUri;
    }

    // Call base class showEvent to read all options (including the default
    // value)
    OptionsWidget::showEvent(event);

    // Double-check: if UI is still empty after ReadOptions, set it explicitly
    // This handles the case where options_.vocab_tree_path was empty before
    for (auto& option : options_path_) {
        if (option.second == &options_.vocab_tree_path) {
            if (option.first->text().isEmpty() &&
                !options_.vocab_tree_path.empty()) {
                option.first->setText(
                        QString::fromStdString(options_.vocab_tree_path));
            }
            break;
        }
    }
}

void AutomaticReconstructionWidget::RenderResult() {
    if (main_window_->reconstruction_manager_.Size() > 0) {
        main_window_->reconstruction_manager_widget_->Update();
        main_window_->RenderClear();
        main_window_->RenderNow();
    }

    if (options_.sparse) {
        QMessageBox::information(this, "",
                                 tr("Imported the reconstructed sparse models "
                                    "for visualization. The "
                                    "models were also exported to the "
                                    "<i>sparse</i> sub-folder in the "
                                    "workspace."));
    }

    if (options_.dense) {
        if (options_.autoVisualization) {
            // add dense point cloud
            if (!fused_points_.empty()) {
                // we create a new group to store all fused dense point cloud
                ccHObject* fusedCloudGroup = new ccHObject("fusedCloudGroup");
                fusedCloudGroup->setVisible(true);

                // for each cluster
                for (std::size_t i = 0; i < fused_points_.size(); ++i) {
                    ccPointCloud* cloud =
                            new ccPointCloud(QString("%1-denseCloud").arg(i));
                    if (cloud) {
                        unsigned nPoints =
                                static_cast<unsigned>(fused_points_.size());
                        if (nPoints > 0 &&
                            cloud->reserveThePointsTable(nPoints)) {
                            if (cloud->reserveTheRGBTable()) {
                                for (const auto& point : fused_points_[i]) {
                                    cloud->addPoint(CCVector3(point.x, point.y,
                                                              point.z));
                                    cloud->addRGBColor(ecvColor::Rgb(
                                            point.r, point.g, point.b));
                                }
                                cloud->showColors(true);
                                fusedCloudGroup->addChild(cloud);
                            } else {
                                CVLog::Error(
                                        "[AutomaticReconstructionWidget::"
                                        "RenderResult] Not enough memory!");
                            }
                        } else {
                            CVLog::Warning(
                                    "[RenderResult] Ignore empty fused points "
                                    "for index %i!",
                                    i);
                        }
                    } else {
                        CVLog::Error(
                                "[AutomaticReconstructionWidget::RenderResult] "
                                "Not enough memory!");
                    }
                }

                if (fusedCloudGroup->getChildrenNumber() == 0) {
                    delete fusedCloudGroup;
                    fusedCloudGroup = nullptr;
                    CVLog::Warning(
                            "[AutomaticReconstructionWidget::RenderResult] "
                            "some unknown error!");
                } else {
                    if (main_window_->app_) {
                        main_window_->app_->addToDB(fusedCloudGroup);
                    }
                }
            }

            // Add meshed model or textured model
            if (main_window_->app_) {
                QStringList filenames;

                // If texturing was enabled and successful, add only textured
                // meshes
                if (options_.texturing && texturing_success_ &&
                    !textured_paths_.empty()) {
                    for (const std::string& path : textured_paths_) {
                        if (!ExistsFile(path)) {
                            CVLog::Warning(
                                    "[RenderResult] Ignore invalid textured "
                                    "mesh for file [%s]",
                                    path.c_str());
                            continue;
                        }
                        filenames.push_back(path.c_str());
                    }
                    if (!filenames.isEmpty()) {
                        CVLog::Print("Adding %d textured mesh(es) to scene",
                                     filenames.size());
                        main_window_->app_->addToDBAuto(filenames, false);
                    }
                }
                // Otherwise, add non-textured meshes
                else if (!meshing_paths_.empty()) {
                    for (const std::string& path : meshing_paths_) {
                        if (!ExistsFile(path)) {
                            CVLog::Warning(
                                    "[RenderResult] Ignore invalid meshed "
                                    "model for file [%s]",
                                    path.c_str());
                            continue;
                        }
                        filenames.push_back(path.c_str());
                    }
                    if (!filenames.isEmpty()) {
                        CVLog::Print("Adding %d mesh(es) to scene",
                                     filenames.size());
                        main_window_->app_->addToDBAuto(filenames, false);
                    }
                }
            }
        } else {
            QMessageBox::information(
                    this, "",
                    tr("To visualize the reconstructed dense point cloud, "
                       "navigate to the "
                       "<i>dense</i> sub-folder in your workspace with <i>File "
                       "> Import "
                       "model from...</i>. To visualize the meshed model, "
                       "you can only drop meshed file into the main window."));
        }
    }
}

}  // namespace cloudViewer

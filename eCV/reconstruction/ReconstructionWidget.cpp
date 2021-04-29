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
// this SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF this SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "ReconstructionWidget.h"

#include "MainWindow.h"
#include "util/version.h"
#include "RenderOptions.h"
#include "RenderOptionsWidget.h"
#include "ReconstructionOptionsWidget.h"

#include <ecvDisplayTools.h>

namespace cloudViewer {

using namespace colmap;

ReconstructionWidget::ReconstructionWidget(MainWindow* app)
    : QWidget(ecvDisplayTools::GetMainScreen()),
      app_(app),
      thread_control_widget_(new ThreadControlWidget(this)),
      statusbar_timer_label_(nullptr) {

  CreateWidgets();
  CreateActions();
  CreateMenus();
  CreateToolbar();
  CreateStatusbar();
  CreateControllers();
  options_.AddAllOptions();

  hideLog();
  iniEnvironment();
}

void ReconstructionWidget::iniEnvironment()
{
    QSettings settings;
    settings.beginGroup("Reconstruction");

    // import path
    std::string import_path = CVTools::FromQString(settings.value("import_path", "").toString());

    // project path
    std::string project_path =
            CVTools::FromQString(settings.value("project_path", "").toString());
    if (!import_path.empty() && project_path.empty())
    {
      project_path = JoinPaths(import_path, "project.ini");
    }

    if (project_path != "") {
      if (options_.ReRead(project_path)) {
        *options_.project_path = project_path;
        project_widget_->SetDatabasePath(*options_.database_path);
        project_widget_->SetImagePath(*options_.image_path);
      } else {
          std::string database_path =
                  CVTools::FromQString(settings.value("database_path", "").toString());
          std::string image_path =
                  CVTools::FromQString(settings.value("image_path", "").toString());
        if (!database_path.empty() && !image_path.empty())
        {
            *options_.project_path = project_path;
            *options_.database_path = database_path;
            *options_.image_path = image_path;
            project_widget_->SetDatabasePath(database_path);
            project_widget_->SetImagePath(image_path);
        } else {
            ShowInvalidProjectError();
        }
      }
    }

    settings.endGroup();
}

void ReconstructionWidget::ImportReconstruction(const std::string& path) {
  const size_t idx = reconstruction_manager_.Read(path);
  reconstruction_manager_widget_->Update();
  reconstruction_manager_widget_->SelectReconstruction(idx);
  RenderNow();
}

void ReconstructionWidget::close() {
  if (project_widget_->IsValid() && *options_.project_path == "") {
    // Project was created, but not yet saved
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(
        this, "",
        tr("You have not saved your reconstruction project. Do you want to save it?"),
        QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::Yes) {
      ProjectSave();
    }
  }

  if (mapper_controller_) {
    mapper_controller_->Stop();
    mapper_controller_->Wait();
  }

  log_widget_->close();
}

void ReconstructionWidget::release()
{
    if (model_viewer_widget_)
    {
        model_viewer_widget_->Release();
    }
    this->close();
}

void ReconstructionWidget::CreateWidgets() {
  model_viewer_widget_ = new ModelViewerWidget(this, &options_, app_);

  project_widget_ = new ProjectWidget(this, &options_);
  project_widget_->SetDatabasePath(*options_.database_path);
  project_widget_->SetImagePath(*options_.image_path);

  feature_extraction_widget_ = new FeatureExtractionWidget(this, &options_);
  feature_matching_widget_ = new FeatureMatchingWidget(this, &options_);
  database_management_widget_ = new DatabaseManagementWidget(this, &options_);
  automatic_reconstruction_widget_ = new AutomaticReconstructionWidget(this);
  reconstruction_options_widget_ =
      new ReconstructionOptionsWidget(this, &options_);
  bundle_adjustment_widget_ = new BundleAdjustmentWidget(this, &options_);
  dense_reconstruction_widget_ = new DenseReconstructionWidget(this, &options_);
  render_options_widget_ =
      new RenderOptionsWidget(this, &options_, model_viewer_widget_);
  log_widget_ = new LogWidget(this);
  undistortion_widget_ = new UndistortionWidget(this, &options_);
  reconstruction_manager_widget_ =
      new ReconstructionManagerWidget(this, &reconstruction_manager_);
  reconstruction_stats_widget_ = new ReconstructionStatsWidget(this);
  match_matrix_widget_ = new MatchMatrixWidget(this, &options_);

  dock_log_widget_ = new QDockWidget("Log", this);
  dock_log_widget_->setWidget(log_widget_);
}

void ReconstructionWidget::CreateActions() {
  //////////////////////////////////////////////////////////////////////////////
  // File actions
  //////////////////////////////////////////////////////////////////////////////

  action_project_new_ =
      new QAction(QIcon(":/media/project-new.png"), tr("New project"), this);
//  action_project_new_->setShortcuts(QKeySequence::New);
  connect(action_project_new_, &QAction::triggered, this,
          &ReconstructionWidget::ProjectNew);

  action_project_open_ =
      new QAction(QIcon(":/media/project-open.png"), tr("Open project"), this);
//  action_project_open_->setShortcuts(QKeySequence::Open);
  connect(action_project_open_, &QAction::triggered, this,
          &ReconstructionWidget::ProjectOpen);

  action_project_edit_ =
      new QAction(QIcon(":/media/project-edit.png"), tr("Edit project"), this);
  connect(action_project_edit_, &QAction::triggered, this,
          &ReconstructionWidget::ProjectEdit);

  action_project_save_ =
      new QAction(QIcon(":/media/project-save.png"), tr("Save project"), this);
//  action_project_save_->setShortcuts(QKeySequence::Save);
  connect(action_project_save_, &QAction::triggered, this,
          &ReconstructionWidget::ProjectSave);

  action_project_save_as_ = new QAction(QIcon(":/media/project-save-as.png"),
                                        tr("Save project as..."), this);
//  action_project_save_as_->setShortcuts(QKeySequence::SaveAs);
  connect(action_project_save_as_, &QAction::triggered, this,
          &ReconstructionWidget::ProjectSaveAs);

  action_import_ =
      new QAction(QIcon(":/media/import.png"), tr("Import model"), this);
  connect(action_import_, &QAction::triggered, this, &ReconstructionWidget::Import);
  blocking_actions_.push_back(action_import_);

  action_import_from_ = new QAction(QIcon(":/media/import-from.png"),
                                    tr("Import model from..."), this);
  connect(action_import_from_, &QAction::triggered, this,
          &ReconstructionWidget::ImportFrom);
  blocking_actions_.push_back(action_import_from_);

  action_export_ =
      new QAction(QIcon(":/media/export.png"), tr("Export model"), this);
  connect(action_export_, &QAction::triggered, this, &ReconstructionWidget::Export);
  blocking_actions_.push_back(action_export_);

  action_export_all_ = new QAction(QIcon(":/media/export-all.png"),
                                   tr("Export all models"), this);
  connect(action_export_all_, &QAction::triggered, this,
          &ReconstructionWidget::ExportAll);
  blocking_actions_.push_back(action_export_all_);

  action_export_as_ = new QAction(QIcon(":/media/export-as.png"),
                                  tr("Export model as..."), this);
  connect(action_export_as_, &QAction::triggered, this, &ReconstructionWidget::ExportAs);
  blocking_actions_.push_back(action_export_as_);

  action_export_as_text_ = new QAction(QIcon(":/media/export-as-text.png"),
                                       tr("Export model as text"), this);
  connect(action_export_as_text_, &QAction::triggered, this,
          &ReconstructionWidget::ExportAsText);
  blocking_actions_.push_back(action_export_as_text_);

  action_quit_ = new QAction(tr("Quit"), this);
  connect(action_quit_, &QAction::triggered, this, &ReconstructionWidget::close);

  //////////////////////////////////////////////////////////////////////////////
  // Processing action
  //////////////////////////////////////////////////////////////////////////////

  action_feature_extraction_ = new QAction(
      QIcon(":/media/feature-extraction.png"), tr("Feature extraction"), this);
  connect(action_feature_extraction_, &QAction::triggered, this,
          &ReconstructionWidget::FeatureExtraction);
  blocking_actions_.push_back(action_feature_extraction_);

  action_feature_matching_ = new QAction(QIcon(":/media/feature-matching.png"),
                                         tr("Feature matching"), this);
  connect(action_feature_matching_, &QAction::triggered, this,
          &ReconstructionWidget::FeatureMatching);
  blocking_actions_.push_back(action_feature_matching_);

  action_database_management_ =
      new QAction(QIcon(":/media/database-management.png"),
                  tr("Database management"), this);
  connect(action_database_management_, &QAction::triggered, this,
          &ReconstructionWidget::DatabaseManagement);
  blocking_actions_.push_back(action_database_management_);

  //////////////////////////////////////////////////////////////////////////////
  // Reconstruction actions
  //////////////////////////////////////////////////////////////////////////////

  action_automatic_reconstruction_ =
      new QAction(QIcon(":/media/automatic-reconstruction.png"),
                  tr("Automatic reconstruction"), this);
  connect(action_automatic_reconstruction_, &QAction::triggered, this,
          &ReconstructionWidget::AutomaticReconstruction);

  action_reconstruction_start_ =
      new QAction(QIcon(":/media/reconstruction-start.png"),
                  tr("Start reconstruction"), this);
  connect(action_reconstruction_start_, &QAction::triggered, this,
          &ReconstructionWidget::ReconstructionStart);
  blocking_actions_.push_back(action_reconstruction_start_);

  action_reconstruction_step_ =
      new QAction(QIcon(":/media/reconstruction-step.png"),
                  tr("Reconstruct next image"), this);
  connect(action_reconstruction_step_, &QAction::triggered, this,
          &ReconstructionWidget::ReconstructionStep);
  blocking_actions_.push_back(action_reconstruction_step_);

  action_reconstruction_pause_ =
      new QAction(QIcon(":/media/reconstruction-pause.png"),
                  tr("Pause reconstruction"), this);
  connect(action_reconstruction_pause_, &QAction::triggered, this,
          &ReconstructionWidget::ReconstructionPause);
  action_reconstruction_pause_->setEnabled(false);
  blocking_actions_.push_back(action_reconstruction_pause_);

  action_reconstruction_reset_ =
      new QAction(QIcon(":/media/reconstruction-reset.png"),
                  tr("Reset reconstruction"), this);
  connect(action_reconstruction_reset_, &QAction::triggered, this,
          &ReconstructionWidget::ReconstructionOverwrite);

  action_reconstruction_normalize_ =
      new QAction(QIcon(":/media/reconstruction-normalize.png"),
                  tr("Normalize reconstruction"), this);
  connect(action_reconstruction_normalize_, &QAction::triggered, this,
          &ReconstructionWidget::ReconstructionNormalize);
  blocking_actions_.push_back(action_reconstruction_normalize_);

  action_reconstruction_options_ =
      new QAction(QIcon(":/media/reconstruction-options.png"),
                  tr("Reconstruction options"), this);
  connect(action_reconstruction_options_, &QAction::triggered, this,
          &ReconstructionWidget::ReconstructionOptions);
  blocking_actions_.push_back(action_reconstruction_options_);

  action_bundle_adjustment_ = new QAction(
      QIcon(":/media/bundle-adjustment.png"), tr("Bundle adjustment"), this);
  connect(action_bundle_adjustment_, &QAction::triggered, this,
          &ReconstructionWidget::BundleAdjustment);
  action_bundle_adjustment_->setEnabled(false);
  blocking_actions_.push_back(action_bundle_adjustment_);

  action_dense_reconstruction_ =
      new QAction(QIcon(":/media/dense-reconstruction.png"),
                  tr("Dense reconstruction"), this);
  connect(action_dense_reconstruction_, &QAction::triggered, this,
          &ReconstructionWidget::DenseReconstruction);

  //////////////////////////////////////////////////////////////////////////////
  // Render actions
  //////////////////////////////////////////////////////////////////////////////

  action_render_toggle_ = new QAction(QIcon(":/media/render-enabled.png"),
                                      tr("Disable rendering"), this);
  connect(action_render_toggle_, &QAction::triggered, this,
          &ReconstructionWidget::RenderToggle);

  action_render_reset_view_ = new QAction(
      QIcon(":/media/render-reset-view.png"), tr("Reset view"), this);
  connect(action_render_reset_view_, &QAction::triggered, model_viewer_widget_,
          &ModelViewerWidget::ResetView);

  action_render_options_ = new QAction(QIcon(":/media/render-options.png"),
                                       tr("Render options"), this);
  connect(action_render_options_, &QAction::triggered, this,
          &ReconstructionWidget::RenderOptions);

  connect(
      reconstruction_manager_widget_,
      static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),
      this, &ReconstructionWidget::SelectReconstructionIdx);

  //////////////////////////////////////////////////////////////////////////////
  // Extras actions
  //////////////////////////////////////////////////////////////////////////////

  action_reconstruction_stats_ =
      new QAction(QIcon(":/media/reconstruction-stats.png"),
                  tr("Show model statistics"), this);
  connect(action_reconstruction_stats_, &QAction::triggered, this,
          &ReconstructionWidget::ReconstructionStats);

  action_match_matrix_ = new QAction(QIcon(":/media/match-matrix.png"),
                                     tr("Show match matrix"), this);
  connect(action_match_matrix_, &QAction::triggered, this,
          &ReconstructionWidget::MatchMatrix);

  action_log_show_ =
      new QAction(QIcon(":/media/log.png"), tr("Show log"), this);
  connect(action_log_show_, &QAction::triggered, this, &ReconstructionWidget::showLog);

  action_grab_image_ =
      new QAction(QIcon(":/media/grab-image.png"), tr("Grab image"), this);
  connect(action_grab_image_, &QAction::triggered, this,
          &ReconstructionWidget::GrabImage);

  action_grab_movie_ =
      new QAction(QIcon(":/media/grab-movie.png"), tr("Grab movie"), this);
  connect(action_grab_movie_, &QAction::triggered, model_viewer_widget_,
          &ModelViewerWidget::GrabMovie);

  action_undistort_ =
      new QAction(QIcon(":/media/undistort.png"), tr("Undistortion"), this);
  connect(action_undistort_, &QAction::triggered, this,
          &ReconstructionWidget::UndistortImages);
  blocking_actions_.push_back(action_undistort_);

  action_extract_colors_ = new QAction(tr("Extract colors"), this);
  connect(action_extract_colors_, &QAction::triggered, this,
          &ReconstructionWidget::ExtractColors);

  action_set_options_ = new QAction(tr("Set options for ..."), this);
  connect(action_set_options_, &QAction::triggered, this,
          &ReconstructionWidget::SetOptions);

  action_reset_options_ = new QAction(tr("Set default options"), this);
  connect(action_reset_options_, &QAction::triggered, this,
          &ReconstructionWidget::ResetOptions);

  //////////////////////////////////////////////////////////////////////////////
  // Misc actions
  //////////////////////////////////////////////////////////////////////////////

  action_render_ = new QAction(tr("Render"), this);
  connect(action_render_, &QAction::triggered, this, &ReconstructionWidget::Render,
          Qt::BlockingQueuedConnection);

  action_render_now_ = new QAction(tr("Render now"), this);
  connect(action_render_now_, &QAction::triggered, this, &ReconstructionWidget::RenderNow,
          Qt::BlockingQueuedConnection);

  action_reconstruction_finish_ =
      new QAction(tr("Finish reconstruction"), this);
  connect(action_reconstruction_finish_, &QAction::triggered, this,
          &ReconstructionWidget::ReconstructionFinish, Qt::BlockingQueuedConnection);
}

void ReconstructionWidget::CreateMenus() {
  QMenu* file_menu = new QMenu(tr("ProjectFile"), this);
  file_menu->addAction(action_project_new_);
  file_menu->addAction(action_project_open_);
  file_menu->addAction(action_project_edit_);
  file_menu->addAction(action_project_save_);
  file_menu->addAction(action_project_save_as_);
  file_menu->addSeparator();
  file_menu->addAction(action_import_);
  file_menu->addAction(action_import_from_);
  file_menu->addSeparator();
  file_menu->addAction(action_export_);
  file_menu->addAction(action_export_all_);
  file_menu->addAction(action_export_as_);
  file_menu->addAction(action_export_as_text_);
  file_menu->addSeparator();
  file_menu->addAction(action_quit_);
  menus_list_.push_back(file_menu);

  QMenu* preprocessing_menu = new QMenu(tr("Processing"), this);
  preprocessing_menu->addAction(action_feature_extraction_);
  preprocessing_menu->addAction(action_feature_matching_);
  preprocessing_menu->addAction(action_database_management_);
  menus_list_.push_back(preprocessing_menu);

  QMenu* reconstruction_menu = new QMenu(tr("Reconstruction"), this);
  reconstruction_menu->addAction(action_automatic_reconstruction_);
  reconstruction_menu->addSeparator();
  reconstruction_menu->addAction(action_reconstruction_start_);
  reconstruction_menu->addAction(action_reconstruction_pause_);
  reconstruction_menu->addAction(action_reconstruction_step_);
  reconstruction_menu->addSeparator();
  reconstruction_menu->addAction(action_reconstruction_reset_);
  reconstruction_menu->addAction(action_reconstruction_normalize_);
  reconstruction_menu->addAction(action_reconstruction_options_);
  reconstruction_menu->addSeparator();
  reconstruction_menu->addAction(action_bundle_adjustment_);
  reconstruction_menu->addAction(action_dense_reconstruction_);
  menus_list_.push_back(reconstruction_menu);

  QMenu* render_menu = new QMenu(tr("Render"), this);
  render_menu->addAction(action_render_toggle_);
  render_menu->addAction(action_render_reset_view_);
  render_menu->addAction(action_render_options_);
  menus_list_.push_back(render_menu);

  QMenu* extras_menu = new QMenu(tr("Extras"), this);
  extras_menu->addAction(action_log_show_);
  extras_menu->addAction(action_match_matrix_);
  extras_menu->addAction(action_reconstruction_stats_);
  extras_menu->addSeparator();
  extras_menu->addAction(action_grab_image_);
  extras_menu->addAction(action_grab_movie_);
  extras_menu->addSeparator();
  extras_menu->addAction(action_undistort_);
  extras_menu->addAction(action_extract_colors_);
  extras_menu->addSeparator();
  extras_menu->addAction(action_set_options_);
  extras_menu->addAction(action_reset_options_);
  menus_list_.push_back(extras_menu);
}

void ReconstructionWidget::CreateToolbar() {
  file_toolbar_ = new QToolBar(tr("ProjectFile"), this);
  file_toolbar_->setObjectName(QString::fromUtf8("ProjectFile"));
  file_toolbar_->addAction(action_project_new_);
  file_toolbar_->addAction(action_project_open_);
  file_toolbar_->addAction(action_project_edit_);
  file_toolbar_->addAction(action_project_save_);
  file_toolbar_->addAction(action_import_);
  file_toolbar_->addAction(action_export_);
  file_toolbar_->setIconSize(QSize(16, 16));
  toolbar_list_.push_back(file_toolbar_);

  preprocessing_toolbar_ = new QToolBar(tr("Processing"), this);
  preprocessing_toolbar_->setObjectName(QString::fromUtf8("Processing"));
  preprocessing_toolbar_->addAction(action_feature_extraction_);
  preprocessing_toolbar_->addAction(action_feature_matching_);
  preprocessing_toolbar_->addAction(action_database_management_);
  preprocessing_toolbar_->setIconSize(QSize(16, 16));
  toolbar_list_.push_back(preprocessing_toolbar_);

  reconstruction_toolbar_ = new QToolBar(tr("Reconstruction"), this);
  reconstruction_toolbar_->setObjectName(QString::fromUtf8("Reconstruction"));
  reconstruction_toolbar_->addAction(action_automatic_reconstruction_);
  reconstruction_toolbar_->addAction(action_reconstruction_start_);
  reconstruction_toolbar_->addAction(action_reconstruction_step_);
  reconstruction_toolbar_->addAction(action_reconstruction_pause_);
  reconstruction_toolbar_->addAction(action_reconstruction_reset_);
  reconstruction_toolbar_->addAction(action_reconstruction_normalize_);
  reconstruction_toolbar_->addAction(action_reconstruction_options_);
  reconstruction_toolbar_->addAction(action_bundle_adjustment_);
  reconstruction_toolbar_->addAction(action_dense_reconstruction_);
  reconstruction_toolbar_->setIconSize(QSize(16, 16));
  toolbar_list_.push_back(reconstruction_toolbar_);

  render_toolbar_ = new QToolBar(tr("Render"), this);
  render_toolbar_->setObjectName(QString::fromUtf8("Render"));
  render_toolbar_->addAction(action_render_toggle_);
  render_toolbar_->addAction(action_render_reset_view_);
  render_toolbar_->addAction(action_render_options_);
  render_toolbar_->addWidget(reconstruction_manager_widget_);
  render_toolbar_->setIconSize(QSize(16, 16));
  toolbar_list_.push_back(render_toolbar_);

  extras_toolbar_ = new QToolBar(tr("Extras"), this);
  extras_toolbar_->setObjectName(QString::fromUtf8("Extras"));
  extras_toolbar_->addAction(action_log_show_);
  extras_toolbar_->addAction(action_match_matrix_);
  extras_toolbar_->addAction(action_reconstruction_stats_);
  extras_toolbar_->addAction(action_grab_image_);
  extras_toolbar_->addAction(action_grab_movie_);
  extras_toolbar_->setIconSize(QSize(16, 16));
  toolbar_list_.push_back(extras_toolbar_);
}

void ReconstructionWidget::CreateStatusbar() {
  QFont font;
  font.setPointSize(11);

  statusbar_timer_label_ = new QLabel("Time 00:00:00:00", this);
  statusbar_timer_label_->setFont(font);
  statusbar_timer_label_->setAlignment(Qt::AlignCenter);
  statusbar_timer_ = new QTimer(this);
  connect(statusbar_timer_, &QTimer::timeout, this, &ReconstructionWidget::UpdateTimer);
  statusbar_timer_->start(1000);

  model_viewer_widget_->statusbar_status_label =
      new QLabel("0 Images - 0 Points", this);
  model_viewer_widget_->statusbar_status_label->setFont(font);
  model_viewer_widget_->statusbar_status_label->setAlignment(Qt::AlignCenter);
}

void ReconstructionWidget::CreateControllers() {
  if (mapper_controller_) {
    mapper_controller_->Stop();
    mapper_controller_->Wait();
  }

  mapper_controller_.reset(new IncrementalMapperController(
      options_.mapper.get(), *options_.image_path, *options_.database_path,
      &reconstruction_manager_));
  mapper_controller_->AddCallback(
      IncrementalMapperController::INITIAL_IMAGE_PAIR_REG_CALLBACK, [this]() {
        if (!mapper_controller_->IsStopped()) {
          action_render_now_->trigger();
        }
      });
  mapper_controller_->AddCallback(
      IncrementalMapperController::NEXT_IMAGE_REG_CALLBACK, [this]() {
        if (!mapper_controller_->IsStopped()) {
          action_render_->trigger();
        }
      });
  mapper_controller_->AddCallback(
      IncrementalMapperController::LAST_IMAGE_REG_CALLBACK, [this]() {
        if (!mapper_controller_->IsStopped()) {
          action_render_now_->trigger();
        }
      });
  mapper_controller_->AddCallback(
      IncrementalMapperController::FINISHED_CALLBACK, [this]() {
        if (!mapper_controller_->IsStopped()) {
          action_render_now_->trigger();
          action_reconstruction_finish_->trigger();
        }
        if (reconstruction_manager_.Size() == 0) {
          action_reconstruction_reset_->trigger();
        }
      });
}

void ReconstructionWidget::ProjectNew() {
  if (ReconstructionOverwrite()) {
    project_widget_->Reset();
    project_widget_->show();
    project_widget_->raise();
  }
}

bool ReconstructionWidget::ProjectOpen() {
  if (!ReconstructionOverwrite()) {
    return false;
  }

  QSettings settings;
  settings.beginGroup("Reconstruction");
  QString last_project_path = settings.value("project_path", "").toString();
  settings.endGroup();

  const std::string project_path =
      QFileDialog::getOpenFileName(this, tr("Select project file"),
                                   QFileInfo(last_project_path).dir().absolutePath(),
                                   tr("Project file (*.ini)"))
          .toUtf8()
          .constData();
  // If selection not canceled
  if (project_path != "") {
    if (options_.ReRead(project_path)) {
      *options_.project_path = project_path;
      project_widget_->SetDatabasePath(*options_.database_path);
      project_widget_->SetImagePath(*options_.image_path);

      project_widget_->persistSave(*options_.project_path,
                                   *options_.database_path,
                                   *options_.image_path);
      return true;
    } else {
      ShowInvalidProjectError();
    }
  }

  return false;
}

void ReconstructionWidget::ProjectEdit() {
  project_widget_->show();
  project_widget_->raise();
}

void ReconstructionWidget::ProjectSave() {
  if (!ExistsFile(*options_.project_path)) {
    QSettings settings;
    settings.beginGroup("Reconstruction");
    QString last_project_path = settings.value("project_path", "").toString();
    settings.endGroup();

    std::string project_path =
        QFileDialog::getSaveFileName(this, tr("Select project file"),
                                     QFileInfo(last_project_path).dir().absolutePath(),
                                     tr("Project file (*.ini)"))
            .toUtf8()
            .constData();
    // If selection not canceled
    if (project_path != "") {
      if (!HasFileExtension(project_path, ".ini")) {
        project_path += ".ini";
      }
      *options_.project_path = project_path;
      options_.Write(*options_.project_path);
    }
  } else {
    // Project path was chosen previously, either here or via command-line.
    options_.Write(*options_.project_path);
  }

  if (ExistsFile(*options_.project_path))
  {
    project_widget_->persistSave(*options_.project_path,
                                 *options_.database_path,
                                 *options_.image_path);
  }

}

void ReconstructionWidget::ProjectSaveAs() {
  QSettings settings;
  settings.beginGroup("Reconstruction");
  QString last_project_path = settings.value("project_path", "").toString();

  const std::string new_project_path =
      QFileDialog::getSaveFileName(this,
                                   tr("Select project file"),
                                   QFileInfo(last_project_path).dir().absolutePath(),
                                   tr("Project file (*.ini)"))
          .toUtf8()
          .constData();
  if (new_project_path != "") {
    *options_.project_path = new_project_path;
    options_.Write(*options_.project_path);

    project_widget_->persistSave(*options_.project_path,
                                 *options_.database_path,
                                 *options_.image_path);
  }

  settings.endGroup();
}

void ReconstructionWidget::Import() {

  QSettings settings;
  settings.beginGroup("Reconstruction");
  QString last_import_path = settings.value("import_path", "").toString();

  const std::string import_path =
      QFileDialog::getExistingDirectory(this,
                                        tr("Select source..."),
                                        last_import_path,
                                        QFileDialog::ShowDirsOnly)
          .toUtf8()
          .constData();

  // Selection canceled?
  if (import_path == "") {
    settings.endGroup();
    return;
  }

  const std::string project_path = JoinPaths(import_path, "project.ini");
  const std::string cameras_bin_path = JoinPaths(import_path, "cameras.bin");
  const std::string images_bin_path = JoinPaths(import_path, "images.bin");
  const std::string points3D_bin_path = JoinPaths(import_path, "points3D.bin");
  const std::string cameras_txt_path = JoinPaths(import_path, "cameras.txt");
  const std::string images_txt_path = JoinPaths(import_path, "images.txt");
  const std::string points3D_txt_path = JoinPaths(import_path, "points3D.txt");

  if ((!ExistsFile(cameras_bin_path) || !ExistsFile(images_bin_path) ||
       !ExistsFile(points3D_bin_path)) &&
      (!ExistsFile(cameras_txt_path) || !ExistsFile(images_txt_path) ||
       !ExistsFile(points3D_txt_path))) {
    QMessageBox::critical(this, "",
                          tr("cameras, images, and points3D files do not exist "
                             "in chosen directory."));
    settings.endGroup();
    return;
  }

  settings.setValue("import_path", CVTools::ToQString(import_path));
  settings.setValue("project_path", CVTools::ToQString(project_path));
  settings.setValue("cameras_path", CVTools::ToQString(cameras_bin_path));
  settings.setValue("images_path", CVTools::ToQString(images_bin_path));
  settings.setValue("points3D_path", CVTools::ToQString(points3D_bin_path));
  settings.endGroup();

  if (!ReconstructionOverwrite()) {
    return;
  }

  bool edit_project = false;
  if (ExistsFile(project_path)) {
    options_.ReRead(project_path);
  } else {
    QMessageBox::StandardButton reply = QMessageBox::question(
        this, "",
        tr("Directory does not contain a <i>project.ini</i>. To "
           "resume the reconstruction, you need to specify a valid "
           "database and image path. Do you want to select the paths "
           "now (or press <i>No</i> to only visualize the reconstruction)?"),
        QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::Yes) {
      edit_project = true;
    }
  }

  thread_control_widget_->StartFunction(
      "Importing...", [this, import_path, edit_project]() {
        const size_t idx = reconstruction_manager_.Read(import_path);
        reconstruction_manager_widget_->Update();
        reconstruction_manager_widget_->SelectReconstruction(idx);
        action_bundle_adjustment_->setEnabled(true);
        action_render_now_->trigger();
        if (edit_project) {
          action_project_edit_->trigger();
        }
      });
}

void ReconstructionWidget::ImportFrom() {

  QSettings settings;
  settings.beginGroup("Reconstruction");
  QString last_model_import_path = settings.value("model_import_path", "").toString();

  const std::string import_path =
      QFileDialog::getOpenFileName(this, tr("Select source..."),
                                   last_model_import_path).toUtf8().constData();

  // Selection canceled?
  if (import_path == "") {
    settings.endGroup();
    return;
  }

  if (!ExistsFile(import_path)) {
    QMessageBox::critical(this, "", tr("Invalid file"));
    settings.endGroup();
    return;
  }

  if (!HasFileExtension(import_path, ".ply")) {
    QMessageBox::critical(this, "",
                          tr("Invalid file format (supported formats: PLY)"));
    settings.endGroup();
    return;
  }

  settings.setValue("model_import_path", CVTools::ToQString(import_path));
  settings.endGroup();

  thread_control_widget_->StartFunction("Importing...", [this, import_path]() {
    const size_t reconstruction_idx = reconstruction_manager_.Add();
    reconstruction_manager_.Get(reconstruction_idx).ImportPLY(import_path);
    options_.render->min_track_len = 0;
    reconstruction_manager_widget_->Update();
    reconstruction_manager_widget_->SelectReconstruction(reconstruction_idx);
    action_render_now_->trigger();
  });
}

void ReconstructionWidget::Export() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }

  QSettings settings;
  settings.beginGroup("Reconstruction");
  QString last_export_path = settings.value("export_path", "").toString();

  const std::string export_path =
      QFileDialog::getExistingDirectory(this,
                                        tr("Select destination..."),
                                        last_export_path,
                                        QFileDialog::ShowDirsOnly)
          .toUtf8()
          .constData();

  // Selection canceled?
  if (export_path == "") {
    settings.endGroup();
    return;
  }

  const std::string cameras_name = "cameras.bin";
  const std::string images_name = "images.bin";
  const std::string points3D_name = "points3D.bin";

  const std::string project_path = JoinPaths(export_path, "project.ini");
  const std::string cameras_path = JoinPaths(export_path, cameras_name);
  const std::string images_path = JoinPaths(export_path, images_name);
  const std::string points3D_path = JoinPaths(export_path, points3D_name);

  settings.setValue("project_path", CVTools::ToQString(project_path));
  settings.setValue("export_path", CVTools::ToQString(export_path));
  settings.setValue("cameras_path", CVTools::ToQString(cameras_path));
  settings.setValue("images_path", CVTools::ToQString(images_path));
  settings.setValue("points3D_path", CVTools::ToQString(points3D_path));
  settings.endGroup();

  if (ExistsFile(cameras_path) || ExistsFile(images_path) ||
      ExistsFile(points3D_path)) {
    QMessageBox::StandardButton reply = QMessageBox::question(
        this, "",
        StringPrintf(
            "The files <i>%s</i>, <i>%s</i>, or <i>%s</i> already "
            "exist in the selected destination. Do you want to overwrite them?",
            cameras_name.c_str(), images_name.c_str(), points3D_name.c_str())
            .c_str(),
        QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::No) {
      return;
    }
  }

  thread_control_widget_->StartFunction(
      "Exporting...", [this, export_path, project_path]() {
        const auto& reconstruction =
            reconstruction_manager_.Get(SelectedReconstructionIdx());
        reconstruction.WriteBinary(export_path);
        options_.Write(project_path);
      });
}

void ReconstructionWidget::ExportAll() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }

  QSettings settings;
  settings.beginGroup("Reconstruction");
  QString last_export_path = settings.value("export_path", "").toString();

  const std::string export_path =
      QFileDialog::getExistingDirectory(this,
                                        tr("Select destination..."),
                                        last_export_path,
                                        QFileDialog::ShowDirsOnly)
                                      .toUtf8()
                                      .constData();

  // Selection canceled?
  if (export_path == "") {
    settings.endGroup();
    return;
  } else {
    settings.setValue("export_path", CVTools::ToQString(export_path));
    settings.endGroup();

    thread_control_widget_->StartFunction("Exporting...", [this, export_path]() {
    reconstruction_manager_.Write(export_path, &options_);
    });
  }

}

void ReconstructionWidget::ExportAs() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }

  QSettings settings;
  settings.beginGroup("Reconstruction");
  QString last_export_path = settings.value("model_export_path", "").toString();

  QString filter("NVM (*.nvm)");
  const std::string export_path =
      QFileDialog::getSaveFileName(
          this, tr("Select destination..."), last_export_path,
          "NVM (*.nvm);;Bundler (*.out);;PLY (*.ply);;VRML (*.wrl)", &filter)
          .toUtf8()
          .constData();

  // Selection canceled?
  if (export_path == "") {
    settings.endGroup();
    return;
  }

  settings.setValue("model_export_path", CVTools::ToQString(export_path));
  settings.endGroup();

  thread_control_widget_->StartFunction(
      "Exporting...", [this, export_path, filter]() {
        const Reconstruction& reconstruction =
            reconstruction_manager_.Get(SelectedReconstructionIdx());
        if (filter == "NVM (*.nvm)") {
          reconstruction.ExportNVM(export_path);
        } else if (filter == "Bundler (*.out)") {
          reconstruction.ExportBundler(export_path, export_path + ".list.txt");
        } else if (filter == "PLY (*.ply)") {
          reconstruction.ExportPLY(export_path);
        } else if (filter == "VRML (*.wrl)") {
          const auto base_path =
              export_path.substr(0, export_path.find_last_of("."));
          reconstruction.ExportVRML(base_path + ".images.wrl",
                                    base_path + ".points3D.wrl", 1,
                                    Eigen::Vector3d(1, 0, 0));
        }
      });
}

void ReconstructionWidget::ExportAsText() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }

  QSettings settings;
  settings.beginGroup("Reconstruction");
  QString last_export_path = settings.value("export_path", "").toString();

  const std::string export_path =
      QFileDialog::getExistingDirectory(this,
                                        tr("Select destination..."),
                                        last_export_path,
                                        QFileDialog::ShowDirsOnly)
          .toUtf8()
          .constData();

  // Selection canceled?
  if (export_path == "") {
    settings.endGroup();
    return;
  }

  const std::string cameras_name = "cameras.txt";
  const std::string images_name = "images.txt";
  const std::string points3D_name = "points3D.txt";

  const std::string project_path = JoinPaths(export_path, "project.ini");
  const std::string cameras_path = JoinPaths(export_path, cameras_name);
  const std::string images_path = JoinPaths(export_path, images_name);
  const std::string points3D_path = JoinPaths(export_path, points3D_name);

  settings.setValue("export_path", CVTools::ToQString(export_path));
  settings.setValue("project_path", CVTools::ToQString(project_path));
  settings.setValue("cameras_path", CVTools::ToQString(cameras_path));
  settings.setValue("images_path", CVTools::ToQString(images_path));
  settings.setValue("points3D_path", CVTools::ToQString(points3D_path));
  settings.endGroup();

  if (ExistsFile(cameras_path) || ExistsFile(images_path) ||
      ExistsFile(points3D_path)) {
    QMessageBox::StandardButton reply = QMessageBox::question(
        this, "",
        StringPrintf(
            "The files <i>%s</i>, <i>%s</i>, or <i>%s</i> already "
            "exist in the selected destination. Do you want to overwrite them?",
            cameras_name.c_str(), images_name.c_str(), points3D_name.c_str())
            .c_str(),
        QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::No) {
      return;
    }
  }

  thread_control_widget_->StartFunction(
      "Exporting...", [this, export_path, project_path]() {
        const auto& reconstruction =
            reconstruction_manager_.Get(SelectedReconstructionIdx());
        reconstruction.WriteText(export_path);
        options_.Write(project_path);
      });
}

void ReconstructionWidget::FeatureExtraction() {
  if (options_.Check()) {
    feature_extraction_widget_->show();
    feature_extraction_widget_->raise();
  } else {
    ShowInvalidProjectError();
  }
}

void ReconstructionWidget::FeatureMatching() {
  if (options_.Check()) {
    feature_matching_widget_->show();
    feature_matching_widget_->raise();
  } else {
    ShowInvalidProjectError();
  }
}

void ReconstructionWidget::DatabaseManagement() {
  if (options_.Check()) {
    database_management_widget_->show();
    database_management_widget_->raise();
  } else {
    ShowInvalidProjectError();
  }
}

void ReconstructionWidget::AutomaticReconstruction() {
  automatic_reconstruction_widget_->show();
  automatic_reconstruction_widget_->raise();
}

void ReconstructionWidget::ReconstructionStart() {
  if (!mapper_controller_->IsStarted() && !options_.Check()) {
    ShowInvalidProjectError();
    return;
  }

  if (mapper_controller_->IsFinished() && HasSelectedReconstruction()) {
    QMessageBox::critical(this, "",
                          tr("Reset reconstruction before starting."));
    return;
  }

  if (mapper_controller_->IsStarted()) {
    // Resume existing reconstruction.
    timer_.Resume();
    mapper_controller_->Resume();
  } else {
    // Start new reconstruction.
    CreateControllers();
    timer_.Restart();
    mapper_controller_->Start();
    action_reconstruction_start_->setText(tr("Resume reconstruction"));
  }

  DisableBlockingActions();
  action_reconstruction_pause_->setEnabled(true);
}

void ReconstructionWidget::ReconstructionStep() {
  if (mapper_controller_->IsFinished() && HasSelectedReconstruction()) {
    QMessageBox::critical(this, "",
                          tr("Reset reconstruction before starting."));
    return;
  }

  action_reconstruction_step_->setEnabled(false);
  ReconstructionStart();
  ReconstructionPause();
  action_reconstruction_step_->setEnabled(true);
}

void ReconstructionWidget::ReconstructionPause() {
  timer_.Pause();
  mapper_controller_->Pause();
  EnableBlockingActions();
  action_reconstruction_pause_->setEnabled(false);
}

void ReconstructionWidget::ReconstructionOptions() {
  reconstruction_options_widget_->show();
  reconstruction_options_widget_->raise();
}

void ReconstructionWidget::ReconstructionFinish() {
  timer_.Pause();
  mapper_controller_->Stop();
  EnableBlockingActions();
  action_reconstruction_start_->setEnabled(false);
  action_reconstruction_step_->setEnabled(false);
  action_reconstruction_pause_->setEnabled(false);
}

void ReconstructionWidget::ReconstructionReset() {
  CreateControllers();

  reconstruction_manager_.Clear();
  reconstruction_manager_widget_->Update();

  timer_.Reset();
  UpdateTimer();

  EnableBlockingActions();
  action_reconstruction_start_->setText(tr("Start reconstruction"));
  action_reconstruction_pause_->setEnabled(false);

  RenderClear();
}

void ReconstructionWidget::ReconstructionNormalize() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }
  action_reconstruction_step_->setEnabled(false);
  reconstruction_manager_.Get(SelectedReconstructionIdx()).Normalize();
  action_reconstruction_step_->setEnabled(true);
}

bool ReconstructionWidget::ReconstructionOverwrite() {
  if (reconstruction_manager_.Size() == 0) {
    ReconstructionReset();
    return true;
  }

  QMessageBox::StandardButton reply = QMessageBox::question(
      this, "",
      tr("Do you really want to overwrite the existing reconstruction?"),
      QMessageBox::Yes | QMessageBox::No);
  if (reply == QMessageBox::No) {
    return false;
  } else {
    ReconstructionReset();
    return true;
  }
}

void ReconstructionWidget::BundleAdjustment() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }

  bundle_adjustment_widget_->Show(
      &reconstruction_manager_.Get(SelectedReconstructionIdx()));
}

void ReconstructionWidget::DenseReconstruction() {
  if (HasSelectedReconstruction()) {
    dense_reconstruction_widget_->Show(
        &reconstruction_manager_.Get(SelectedReconstructionIdx()));
  } else {
    dense_reconstruction_widget_->Show(nullptr);
  }
}

void ReconstructionWidget::Render() {
  if (reconstruction_manager_.Size() == 0) {
    return;
  }

  const Reconstruction& reconstruction =
      reconstruction_manager_.Get(SelectedReconstructionIdx());

  int refresh_rate;
  if (options_.render->adapt_refresh_rate) {
    refresh_rate = static_cast<int>(reconstruction.NumRegImages() / 50 + 1);
  } else {
    refresh_rate = options_.render->refresh_rate;
  }

  if (!render_options_widget_->automatic_update ||
      render_options_widget_->counter % refresh_rate != 0) {
    render_options_widget_->counter += 1;
    return;
  }

  render_options_widget_->counter += 1;

  RenderNow();
}

void ReconstructionWidget::RenderNow() {
  reconstruction_manager_widget_->Update();
  RenderSelectedReconstruction();
}

void ReconstructionWidget::RenderSelectedReconstruction() {
  if (reconstruction_manager_.Size() == 0) {
    RenderClear();
    return;
  }

  const size_t reconstruction_idx = SelectedReconstructionIdx();
  model_viewer_widget_->reconstruction =
      &reconstruction_manager_.Get(reconstruction_idx);
  model_viewer_widget_->ReloadReconstruction();
}

void ReconstructionWidget::RenderClear() {
  reconstruction_manager_widget_->SelectReconstruction(
      ReconstructionManagerWidget::kNewestReconstructionIdx);
  model_viewer_widget_->ClearReconstruction();
}

void ReconstructionWidget::RenderOptions()
{
    render_options_widget_->show();
    render_options_widget_->raise();
}

void ReconstructionWidget::SelectReconstructionIdx(const size_t) {
  RenderSelectedReconstruction();
}

size_t ReconstructionWidget::SelectedReconstructionIdx() {
  size_t reconstruction_idx =
      reconstruction_manager_widget_->SelectedReconstructionIdx();
  if (reconstruction_idx ==
      ReconstructionManagerWidget::kNewestReconstructionIdx) {
    if (reconstruction_manager_.Size() > 0) {
      reconstruction_idx = reconstruction_manager_.Size() - 1;
    }
  }
  return reconstruction_idx;
}

bool ReconstructionWidget::HasSelectedReconstruction() {
  const size_t reconstruction_idx =
      reconstruction_manager_widget_->SelectedReconstructionIdx();
  if (reconstruction_idx ==
      ReconstructionManagerWidget::kNewestReconstructionIdx) {
    if (reconstruction_manager_.Size() == 0) {
      return false;
    }
  }
  return true;
}

bool ReconstructionWidget::IsSelectedReconstructionValid() {
  if (!HasSelectedReconstruction()) {
    QMessageBox::critical(this, "", tr("No reconstruction selected"));
    return false;
  }
  return true;
}

void ReconstructionWidget::GrabImage() {
  QString file_name = QFileDialog::getSaveFileName(this, tr("Save image"), "",
                                                   tr("Images (*.png *.jpg)"));
  if (file_name != "") {
    if (!HasFileExtension(file_name.toUtf8().constData(), ".png") &&
        !HasFileExtension(file_name.toUtf8().constData(), ".jpg")) {
      file_name += ".png";
    }
    QImage image = model_viewer_widget_->GrabImage();
    image.save(file_name);
  }
}

void ReconstructionWidget::UndistortImages() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }
  undistortion_widget_->Show(
      reconstruction_manager_.Get(SelectedReconstructionIdx()));
}

void ReconstructionWidget::ReconstructionStats() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }
  reconstruction_stats_widget_->show();
  reconstruction_stats_widget_->raise();
  reconstruction_stats_widget_->Show(
      reconstruction_manager_.Get(SelectedReconstructionIdx()));
}

void ReconstructionWidget::MatchMatrix() { match_matrix_widget_->Show(); }

void ReconstructionWidget::showLog() {
  log_widget_->show();
  log_widget_->raise();
  dock_log_widget_->show();
  dock_log_widget_->raise();
}

void ReconstructionWidget::hideLog() {
  log_widget_->hide();
  dock_log_widget_->hide();
}

void ReconstructionWidget::ExtractColors() {
  if (!IsSelectedReconstructionValid()) {
    return;
  }

  thread_control_widget_->StartFunction("Extracting colors...", [this]() {
    auto& reconstruction =
        reconstruction_manager_.Get(SelectedReconstructionIdx());
    reconstruction.ExtractColorsForAllImages(*options_.image_path);
  });
}

void ReconstructionWidget::SetOptions() {
  QStringList data_items;
  data_items << "Individual images"
             << "Video frames"
             << "Internet images";
  bool data_ok;
  const QString data_item =
      QInputDialog::getItem(this, "", "Data:", data_items, 0, false, &data_ok);
  if (!data_ok) {
    return;
  }

  QStringList quality_items;
  quality_items << "Low"
                << "Medium"
                << "High"
                << "Extreme";
  bool quality_ok;
  const QString quality_item = QInputDialog::getItem(
      this, "", "Quality:", quality_items, 2, false, &quality_ok);
  if (!quality_ok) {
    return;
  }

  const bool kResetPaths = false;
  options_.ResetOptions(kResetPaths);

  if (data_item == "Individual images") {
    options_.ModifyForIndividualData();
  } else if (data_item == "Video frames") {
    options_.ModifyForVideoData();
  } else if (data_item == "Internet images") {
    options_.ModifyForInternetData();
  } else {
    LOG(FATAL) << "Data type does not exist";
  }

  if (quality_item == "Low") {
    options_.ModifyForLowQuality();
  } else if (quality_item == "Medium") {
    options_.ModifyForMediumQuality();
  } else if (quality_item == "High") {
    options_.ModifyForHighQuality();
  } else if (quality_item == "Extreme") {
    options_.ModifyForExtremeQuality();
  } else {
    LOG(FATAL) << "Quality level does not exist";
  }
}

void ReconstructionWidget::ResetOptions() {
  const bool kResetPaths = false;
  options_.ResetOptions(kResetPaths);
}

void ReconstructionWidget::RenderToggle() {
    if (render_options_widget_->automatic_update) {
      render_options_widget_->automatic_update = false;
      render_options_widget_->counter = 0;
      action_render_toggle_->setIcon(QIcon(":/media/render-disabled.png"));
      action_render_toggle_->setText(tr("Enable rendering"));
    } else {
      render_options_widget_->automatic_update = true;
      render_options_widget_->counter = 0;
      Render();
      action_render_toggle_->setIcon(QIcon(":/media/render-enabled.png"));
      action_render_toggle_->setText(tr("Disable rendering"));
    }
}

void ReconstructionWidget::UpdateTimer() {
  const int elapsed_time = static_cast<int>(timer_.ElapsedSeconds());
  const int seconds = elapsed_time % 60;
  const int minutes = (elapsed_time / 60) % 60;
  const int hours = (elapsed_time / 3600) % 24;
  const int days = elapsed_time / 86400;
  if (statusbar_timer_label_)
  {
      statusbar_timer_label_->setText(QString().asprintf(
          "Time %02d:%02d:%02d:%02d", days, hours, minutes, seconds));
  }
}

void ReconstructionWidget::ShowInvalidProjectError() {
  QMessageBox::critical(this, "",
                        tr("You must create a valid project using: <i>File > "
                           "New project</i> or <i>File > Edit project</i>"));
}

void ReconstructionWidget::EnableBlockingActions() {
  for (auto& action : blocking_actions_) {
    action->setEnabled(true);
  }
}

void ReconstructionWidget::DisableBlockingActions() {
  for (auto& action : blocking_actions_) {
    action->setDisabled(true);
  }
}

}  // namespace cloudViewer

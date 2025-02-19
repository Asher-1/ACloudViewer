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

#pragma once

#include <QtCore>
#include <QtGui>
#include <QtWidgets>

#include "util/bitmap.h"
#include "base/reconstruction.h"
#include "controllers/IncrementalMapperController.h"
#include "AutomaticReconstructionWidget.h"
#include "BundleAdjustmentWidget.h"
#include "DatabaseManagementWidget.h"
#include "DenseReconstructionWidget.h"
#include "FeatureExtractionWidget.h"
#include "FeatureMatchingWidget.h"
#include "MatchMatrixWidget.h"
#include "ModelViewerWidget.h"
#include "ProjectWidget.h"
#include "LogWidget.h"
#include "ReconstructionManagerWidget.h"
#include "ReconstructionStatsWidget.h"
#include "UndistortionWidget.h"
#include "OptionManager.h"
#include "ReconstructionManager.h"

class MainWindow;

namespace cloudViewer {

class RenderOptionsWidget;
class ReconstructionOptionsWidget;
class ReconstructionWidget : public QWidget {
    Q_OBJECT
 public:
  explicit ReconstructionWidget(MainWindow* app);

  void ImportReconstruction(const std::string& path);

  void close();
  void release();

  void iniEnvironment();

  void showLog();
  void hideLog();

  QDockWidget* getLogWidget() {return dock_log_widget_; }
  std::vector<QMenu*>& getReconstructionMenus() { return menus_list_; }
  std::vector<QToolBar*>& getReconstructionToolbars() { return toolbar_list_; }

  QLabel* getTimerStatusBar() { return statusbar_timer_label_; }
  QLabel* getImageStatusBar() { return model_viewer_widget_->statusbar_status_label; }

 private:
  friend class AutomaticReconstructionWidget;
  friend class BundleAdjustmentWidget;
  friend class DenseReconstructionWidget;

  void CreateWidgets();
  void CreateActions();
  void CreateMenus();
  void CreateToolbar();
  void CreateStatusbar();
  void CreateControllers();

  void ProjectNew();
  bool ProjectOpen();
  void ProjectEdit();
  void ProjectSave();
  void ProjectSaveAs();
  void Import();
  void ImportFrom();
  void Export();
  void ExportAll();
  void ExportAs();
  void ExportAsText();

  void FeatureExtraction();
  void FeatureMatching();
  void DatabaseManagement();

  void AutomaticReconstruction();

  void ReconstructionStart();
  void ReconstructionStep();
  void ReconstructionPause();
  void ReconstructionReset();
  void ReconstructionOptions();
  void ReconstructionFinish();
  void ReconstructionNormalize();
  bool ReconstructionOverwrite();

  void BundleAdjustment();
  void DenseReconstruction();

  void Render();
  void RenderNow();
  void RenderToggle();
  void RenderOptions();
  void RenderSelectedReconstruction();
  void RenderClear();

  void SelectReconstructionIdx(const std::size_t);
  std::size_t SelectedReconstructionIdx();
  bool HasSelectedReconstruction();
  bool IsSelectedReconstructionValid();

  void GrabImage();
  void UndistortImages();

  void ReconstructionStats();
  void MatchMatrix();
  void ExtractColors();

  void SetOptions();
  void ResetOptions();

  void ShowInvalidProjectError();
  void UpdateTimer();

  void EnableBlockingActions();
  void DisableBlockingActions();

  MainWindow* app_;

  OptionManager options_;

  ReconstructionManager reconstruction_manager_;
  std::unique_ptr<IncrementalMapperController> mapper_controller_;

  colmap::Timer timer_;

  ModelViewerWidget* model_viewer_widget_;
  ProjectWidget* project_widget_;
  FeatureExtractionWidget* feature_extraction_widget_;
  FeatureMatchingWidget* feature_matching_widget_;
  DatabaseManagementWidget* database_management_widget_;
  AutomaticReconstructionWidget* automatic_reconstruction_widget_;
  ReconstructionOptionsWidget* reconstruction_options_widget_;
  BundleAdjustmentWidget* bundle_adjustment_widget_;
  DenseReconstructionWidget* dense_reconstruction_widget_;
  RenderOptionsWidget* render_options_widget_;
  LogWidget* log_widget_;
  UndistortionWidget* undistortion_widget_;
  ReconstructionManagerWidget* reconstruction_manager_widget_;
  ReconstructionStatsWidget* reconstruction_stats_widget_;
  MatchMatrixWidget* match_matrix_widget_;
  ThreadControlWidget* thread_control_widget_;

  QToolBar* file_toolbar_;
  QToolBar* preprocessing_toolbar_;
  QToolBar* reconstruction_toolbar_;
  QToolBar* render_toolbar_;
  QToolBar* extras_toolbar_;

  QDockWidget* dock_log_widget_;

  QTimer* statusbar_timer_;
  QLabel* statusbar_timer_label_;

  QAction* action_project_new_;
  QAction* action_project_open_;
  QAction* action_project_edit_;
  QAction* action_project_save_;
  QAction* action_project_save_as_;
  QAction* action_import_;
  QAction* action_import_from_;
  QAction* action_export_;
  QAction* action_export_all_;
  QAction* action_export_as_;
  QAction* action_export_as_text_;
  QAction* action_quit_;

  QAction* action_feature_extraction_;
  QAction* action_feature_matching_;
  QAction* action_database_management_;

  QAction* action_automatic_reconstruction_;

  QAction* action_reconstruction_start_;
  QAction* action_reconstruction_step_;
  QAction* action_reconstruction_pause_;
  QAction* action_reconstruction_reset_;
  QAction* action_reconstruction_finish_;
  QAction* action_reconstruction_normalize_;
  QAction* action_reconstruction_options_;

  QAction* action_bundle_adjustment_;
  QAction* action_dense_reconstruction_;

  QAction* action_render_;
  QAction* action_render_now_;
  QAction* action_render_toggle_;
  QAction* action_render_reset_view_;
  QAction* action_render_options_;

  QAction* action_reconstruction_stats_;
  QAction* action_match_matrix_;
  QAction* action_log_show_;
  QAction* action_grab_image_;
  QAction* action_grab_movie_;
  QAction* action_undistort_;
  QAction* action_extract_colors_;
  QAction* action_set_options_;
  QAction* action_reset_options_;

  std::vector<QAction*> blocking_actions_;
  std::vector<QMenu*> menus_list_;
  std::vector<QToolBar*> toolbar_list_;
};

}  // namespace cloudViewer

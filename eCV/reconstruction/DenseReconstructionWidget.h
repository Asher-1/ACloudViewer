// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore>
#include <QtWidgets>

#include "ImageViewerWidget.h"
#include "OptionsWidget.h"
#include "util/ply.h"

namespace colmap {
class Reconstruction;
}

#include "util/option_manager.h"

namespace cloudViewer {

using OptionManager = colmap::OptionManager;
class ReconstructionWidget;
class DenseReconstructionOptionsWidget : public QWidget {
public:
    DenseReconstructionOptionsWidget(QWidget* parent, OptionManager* options);
};

class ThreadControlWidget;
class DenseReconstructionWidget : public QWidget {
public:
    DenseReconstructionWidget(ReconstructionWidget* main_window,
                              OptionManager* options);

    void Show(colmap::Reconstruction* reconstruction);

private:
    void showEvent(QShowEvent* event);

    void Undistort();
    void Stereo();
    void Fusion();
    void PoissonMeshing();
    void DelaunayMeshing();
    void Texturing();

    void SelectWorkspacePath();
    std::string GetWorkspacePath();
    void RefreshWorkspace();

    void WriteFusedPoints();
    void ShowMeshingInfo();

    QWidget* GenerateTableButtonWidget(const std::string& image_name,
                                       const std::string& type);

    ReconstructionWidget* main_window_;
    OptionManager* options_;
    colmap::Reconstruction* reconstruction_;
    ThreadControlWidget* thread_control_widget_;
    DenseReconstructionOptionsWidget* options_widget_;
    ImageViewerWidget* image_viewer_widget_;
    QLineEdit* workspace_path_text_;
    QTableWidget* table_widget_;
    QPushButton* undistortion_button_;
    QPushButton* stereo_button_;
    QPushButton* fusion_button_;
    QPushButton* poisson_meshing_button_;
    QPushButton* delaunay_meshing_button_;
    QPushButton* texturing_button_;
    QAction* refresh_workspace_action_;
    QAction* write_fused_points_action_;
    QAction* show_meshing_info_action_;

    bool photometric_done_;
    bool geometric_done_;

    std::string images_path_;
    std::string depth_maps_path_;
    std::string normal_maps_path_;

    std::string out_mesh_path_;

    std::vector<colmap::PlyPoint> fused_points_;
    std::vector<std::vector<int>> fused_points_visibility_;
};

}  // namespace cloudViewer

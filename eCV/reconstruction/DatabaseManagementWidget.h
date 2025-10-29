// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore>
#include <QtWidgets>
#include <unordered_map>

#include "ImageViewerWidget.h"
#include "base/database.h"

namespace colmap {
class Reconstruction;
}

namespace cloudViewer {

class OptionManager;

////////////////////////////////////////////////////////////////////////////////
// Matches
////////////////////////////////////////////////////////////////////////////////

class TwoViewInfoTab : public QWidget {
public:
    TwoViewInfoTab() {}
    TwoViewInfoTab(QWidget* parent,
                   OptionManager* options,
                   colmap::Database* database);

    void Clear();

protected:
    void InitializeTable(const QStringList& table_header);
    void ShowMatches();
    void FillTable();

    OptionManager* options_;
    colmap::Database* database_;

    const colmap::Image* image_;
    std::vector<std::pair<const colmap::Image*, colmap::FeatureMatches>>
            matches_;
    std::vector<int> configs_;
    std::vector<size_t> sorted_matches_idxs_;

    QTableWidget* table_widget_;
    QLabel* info_label_;
    FeatureImageViewerWidget* matches_viewer_widget_;
};

class MatchesTab : public TwoViewInfoTab {
public:
    MatchesTab(QWidget* parent,
               OptionManager* options,
               colmap::Database* database);

    void Reload(const std::vector<colmap::Image>& images,
                const colmap::image_t image_id);
};

class TwoViewGeometriesTab : public TwoViewInfoTab {
public:
    TwoViewGeometriesTab(QWidget* parent,
                         OptionManager* options,
                         colmap::Database* database);

    void Reload(const std::vector<colmap::Image>& images,
                const colmap::image_t image_id);
};

class OverlappingImagesWidget : public QWidget {
public:
    OverlappingImagesWidget(QWidget* parent,
                            OptionManager* options,
                            colmap::Database* database);

    void ShowMatches(const std::vector<colmap::Image>& images,
                     const colmap::image_t image_id);

private:
    void closeEvent(QCloseEvent* event);

    QWidget* parent_;

    OptionManager* options_;

    QTabWidget* tab_widget_;
    MatchesTab* matches_tab_;
    TwoViewGeometriesTab* two_view_geometries_tab_;
};

////////////////////////////////////////////////////////////////////////////////
// Images, Cameras
////////////////////////////////////////////////////////////////////////////////

class CameraTab : public QWidget {
public:
    CameraTab(QWidget* parent, colmap::Database* database);

    void Reload();
    void Clear();

private:
    void itemChanged(QTableWidgetItem* item);
    void Add();
    void SetModel();

    colmap::Database* database_;

    std::vector<colmap::Camera> cameras_;

    QTableWidget* table_widget_;
    QLabel* info_label_;
};

class ImageTab : public QWidget {
public:
    ImageTab(QWidget* parent,
             CameraTab* camera_tab,
             OptionManager* options,
             colmap::Database* database);

    void Reload();
    void Clear();

private:
    void itemChanged(QTableWidgetItem* item);

    void ShowImage();
    void ShowMatches();
    void SetCamera();
    void SplitCamera();

    CameraTab* camera_tab_;

    OptionManager* options_;
    colmap::Database* database_;

    std::vector<colmap::Image> images_;

    QTableWidget* table_widget_;
    QLabel* info_label_;

    OverlappingImagesWidget* overlapping_images_widget_;

    FeatureImageViewerWidget* image_viewer_widget_;
};

class DatabaseManagementWidget : public QWidget {
public:
    DatabaseManagementWidget(QWidget* parent, OptionManager* options);

private:
    void showEvent(QShowEvent* event);
    void hideEvent(QHideEvent* event);

    void ClearMatches();
    void ClearTwoViewGeometries();

    QWidget* parent_;

    OptionManager* options_;
    colmap::Database database_;

    QTabWidget* tab_widget_;
    ImageTab* image_tab_;
    CameraTab* camera_tab_;
};

}  // namespace cloudViewer

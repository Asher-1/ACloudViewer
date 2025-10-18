// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore>
#include <QtWidgets>

#include "feature/types.h"

namespace colmap {
class Bitmap;
}

namespace cloudViewer {

class OptionManager;
class ModelViewerWidget;

class ImageViewerGraphicsScene : public QGraphicsScene {
public:
    ImageViewerGraphicsScene();

    QGraphicsPixmapItem* ImagePixmapItem() const;

private:
    QGraphicsPixmapItem* image_pixmap_item_ = nullptr;
};

class ImageViewerWidget : public QWidget {
public:
    explicit ImageViewerWidget(QWidget* parent);

    void ShowBitmap(const colmap::Bitmap& bitmap);
    void ShowPixmap(const QPixmap& pixmap);
    void ReadAndShow(const std::string& path);

private:
    static const double kZoomFactor;

    ImageViewerGraphicsScene graphics_scene_;
    QGraphicsView* graphics_view_;

protected:
    void resizeEvent(QResizeEvent* event);
    void closeEvent(QCloseEvent* event);
    void ZoomIn();
    void ZoomOut();
    void Save();

    QGridLayout* grid_layout_;
    QHBoxLayout* button_layout_;
};

class FeatureImageViewerWidget : public ImageViewerWidget {
public:
    FeatureImageViewerWidget(QWidget* parent, const std::string& switch_text);

    void ReadAndShowWithKeypoints(const std::string& path,
                                  const colmap::FeatureKeypoints& keypoints,
                                  const std::vector<char>& tri_mask);

    void ReadAndShowWithMatches(const std::string& path1,
                                const std::string& path2,
                                const colmap::FeatureKeypoints& keypoints1,
                                const colmap::FeatureKeypoints& keypoints2,
                                const colmap::FeatureMatches& matches);

protected:
    void ShowOrHide();

    QPixmap image1_;
    QPixmap image2_;
    bool switch_state_;
    QPushButton* switch_button_;
    const std::string switch_text_;
};

class DatabaseImageViewerWidget : public FeatureImageViewerWidget {
public:
    DatabaseImageViewerWidget(QWidget* parent,
                              ModelViewerWidget* ModelViewerWidget,
                              OptionManager* options);

    void ShowImageWithId(const colmap::image_t image_id);

private:
    void ResizeTable();
    void DeleteImage();

    ModelViewerWidget* model_viewer_widget_;

    OptionManager* options_;

    QPushButton* delete_button_;

    colmap::image_t image_id_;

    QTableWidget* table_widget_;
    QTableWidgetItem* image_id_item_;
    QTableWidgetItem* camera_id_item_;
    QTableWidgetItem* camera_model_item_;
    QTableWidgetItem* camera_params_item_;
    QTableWidgetItem* qvec_item_;
    QTableWidgetItem* tvec_item_;
    QTableWidgetItem* dimensions_item_;
    QTableWidgetItem* num_points2D_item_;
    QTableWidgetItem* num_points3D_item_;
    QTableWidgetItem* num_obs_item_;
    QTableWidgetItem* name_item_;
};

}  // namespace cloudViewer

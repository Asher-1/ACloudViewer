// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore>
#include <QtGui>
#include <QtWidgets>
#include <unordered_map>

#include "base/reconstruction.h"

namespace colmap {

class ModelViewerWidget;

class MovieGrabberWidget : public QWidget {
public:
    MovieGrabberWidget(QWidget* parent, ModelViewerWidget* model_viewer_widget);

    // List of views, used to visualize the movie grabber camera path.
    std::vector<Image> views;

    struct ViewData {
        CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW
        QMatrix4x4 model_view_matrix;
        float point_size = -1.0f;
        float image_size = -1.0f;
    };

private:
    // Add, delete, clear viewpoints.
    void Add();
    void Delete();
    void Clear();

    // Assemble movie from current viewpoints.
    void Assemble();

    // Event slot for time modification.
    void TimeChanged(QTableWidgetItem* item);

    // Event slot for changed selection.
    void SelectionChanged(const QItemSelection& selected,
                          const QItemSelection& deselected);

    // Update state when viewpoints reordered.
    void UpdateViews();

    ModelViewerWidget* model_viewer_widget_;

    QPushButton* assemble_button_;
    QPushButton* add_button_;
    QPushButton* delete_button_;
    QPushButton* clear_button_;
    QTableWidget* table_;

    QSpinBox* frame_rate_sb_;
    QCheckBox* smooth_cb_;
    QDoubleSpinBox* smoothness_sb_;

    EIGEN_STL_UMAP(const QTableWidgetItem*, ViewData) view_data_;
};

}  // namespace colmap

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(
        colmap::MovieGrabberWidget::ViewData)

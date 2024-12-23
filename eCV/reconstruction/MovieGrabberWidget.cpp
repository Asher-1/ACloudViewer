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

#include "MovieGrabberWidget.h"

#include <ecv2DViewportObject.h>
#include <ecvDisplayTools.h>
#include <ecvHObjectCaster.h>

#include "ModelViewerWidget.h"
#include "QtUtils.h"
#include "RenderOptions.h"
#include "base/pose.h"
#include "base/projection.h"
#include "controllers/ViewInterpolate.h"

namespace cloudViewer {

using namespace colmap;

MovieGrabberWidget::MovieGrabberWidget(QWidget* parent,
                                       ModelViewerWidget* model_viewer_widget)
    : QWidget(parent), model_viewer_widget_(model_viewer_widget) {
    setWindowFlags(Qt::Widget | Qt::WindowStaysOnTopHint | Qt::Tool);
    setWindowTitle("Grab movie");

    QGridLayout* grid = new QGridLayout(this);
    grid->setContentsMargins(0, 5, 0, 5);

    add_button_ = new QPushButton(tr("Add"), this);
    connect(add_button_, &QPushButton::released, this,
            &MovieGrabberWidget::Add);
    grid->addWidget(add_button_, 0, 0);

    add_from_selected_button_ = new QPushButton(tr("Add Selected"), this);
    connect(add_from_selected_button_, &QPushButton::released, this,
            &MovieGrabberWidget::AddFromSelected);
    grid->addWidget(add_from_selected_button_, 0, 1);

    delete_button_ = new QPushButton(tr("Delete"), this);
    connect(delete_button_, &QPushButton::released, this,
            &MovieGrabberWidget::Delete);
    grid->addWidget(delete_button_, 0, 2);

    clear_button_ = new QPushButton(tr("Clear"), this);
    connect(clear_button_, &QPushButton::released, this,
            &MovieGrabberWidget::Clear);
    grid->addWidget(clear_button_, 0, 3);

    table_ = new QTableWidget(this);
    table_->setColumnCount(1);
    QStringList table_header;
    table_header << "Time [seconds]";
    table_->setHorizontalHeaderLabels(table_header);
    table_->resizeColumnsToContents();
    table_->setShowGrid(true);
    table_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    table_->verticalHeader()->setVisible(true);
    table_->verticalHeader()->setDefaultSectionSize(18);
    table_->setSelectionMode(QAbstractItemView::SingleSelection);
    table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    connect(table_, &QTableWidget::itemChanged, this,
            &MovieGrabberWidget::TimeChanged);
    connect(table_->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, &MovieGrabberWidget::SelectionChanged);
    grid->addWidget(table_, 1, 0, 1, 3);

    grid->addWidget(new QLabel(tr("Frame rate"), this), 2, 1);
    frame_rate_sb_ = new QSpinBox(this);
    frame_rate_sb_->setMinimum(1);
    frame_rate_sb_->setMaximum(1000);
    frame_rate_sb_->setSingleStep(1);
    frame_rate_sb_->setValue(100);
    grid->addWidget(frame_rate_sb_, 2, 2);

    grid->addWidget(new QLabel(tr("Smooth transition"), this), 3, 1);
    smooth_cb_ = new QCheckBox(this);
    smooth_cb_->setChecked(true);
    grid->addWidget(smooth_cb_, 3, 2);

    grid->addWidget(new QLabel(tr("Smoothness"), this), 4, 1);
    smoothness_sb_ = new QDoubleSpinBox(this);
    smoothness_sb_->setMinimum(0);
    smoothness_sb_->setMaximum(1);
    smoothness_sb_->setSingleStep(0.01);
    smoothness_sb_->setValue(0.5);
    grid->addWidget(smoothness_sb_, 4, 2);

    assemble_button_ = new QPushButton(tr("Assemble movie"), this);
    connect(assemble_button_, &QPushButton::released, this,
            &MovieGrabberWidget::Assemble);
    grid->addWidget(assemble_button_, 5, 1, 1, 2);
}

void MovieGrabberWidget::AddFromSelected() {
    if (!view_data_.empty()) {
        if (QMessageBox::question(
                    this, tr("Add viewports from selected"),
                    tr("Are you sure you want to remove all history views?"),
                    QMessageBox::Yes, QMessageBox::No) != QMessageBox::Yes) {
            return;
        }

        Clear();
    }

    ccHObject::Container objects;
    ecvDisplayTools::FilterByEntityType(objects, CV_TYPES::VIEWPORT_2D_OBJECT);

    std::vector<cc2DViewportObject*> selected_viewports;
    for (const auto& obj : objects) {
        if (obj->isA(CV_TYPES::VIEWPORT_2D_OBJECT) && obj->isSelected()) {
            selected_viewports.push_back(
                    ccHObjectCaster::To2DViewportObject(obj));
        }
    }

    CVLog::Print(QString("Found selected viewports: %1")
                         .arg(selected_viewports.size()));
    for (auto& viewport : selected_viewports) {
        if (!viewport) {
            continue;
        }

        double time = 0;
        if (table_->rowCount() > 0) {
            time = table_->item(table_->rowCount() - 1, 0)->text().toDouble() +
                   1;
        }

        QTableWidgetItem* item = new QTableWidgetItem();
        item->setData(Qt::DisplayRole, time);
        item->setFlags(Qt::NoItemFlags | Qt::ItemIsEnabled |
                       Qt::ItemIsSelectable | Qt::ItemIsEditable);
        item->setTextAlignment(Qt::AlignRight);

        // Save size state of current viewpoint.
        MovieGrabberWidget::ViewData view_data;
        view_data.viewportParams = viewport->getParameters();
        view_data.model_view_matrix =
                viewport->getParameters().computeViewMatrix();
        view_data.point_size = model_viewer_widget_->PointSize();
        view_data.image_size = model_viewer_widget_->ImageSize();
        view_data_.emplace(item, view_data);

        table_->insertRow(table_->rowCount());
        table_->setItem(table_->rowCount() - 1, 0, item);
        table_->selectRow(table_->rowCount() - 1);
    }
}

void MovieGrabberWidget::Add() {
    const ccGLMatrixd matrix = model_viewer_widget_->ModelViewMatrix();

    double time = 0;
    if (table_->rowCount() > 0) {
        time = table_->item(table_->rowCount() - 1, 0)->text().toDouble() + 1;
    }

    QTableWidgetItem* item = new QTableWidgetItem();
    item->setData(Qt::DisplayRole, time);
    item->setFlags(Qt::NoItemFlags | Qt::ItemIsEnabled | Qt::ItemIsSelectable |
                   Qt::ItemIsEditable);
    item->setTextAlignment(Qt::AlignRight);

    // Save size state of current viewpoint.
    MovieGrabberWidget::ViewData view_data;
    view_data.viewportParams = ecvDisplayTools::GetViewportParameters();
    view_data.model_view_matrix = matrix;
    view_data.point_size = model_viewer_widget_->PointSize();
    view_data.image_size = model_viewer_widget_->ImageSize();
    view_data_.emplace(item, view_data);

    table_->insertRow(table_->rowCount());
    table_->setItem(table_->rowCount() - 1, 0, item);
    table_->selectRow(table_->rowCount() - 1);

    // Zoom out a little, so that we can see the newly added camera
    //  model_viewer_widget_->ChangeFocusDistance(-50.0f);
}

void MovieGrabberWidget::Delete() {
    QModelIndexList selection = table_->selectionModel()->selectedIndexes();
    std::vector<colmap::camera_t> toBeDeleted;
    for (const auto& index : selection) {
        table_->removeRow(index.row());
        Image& image = views[static_cast<std::size_t>(index.row())];
        toBeDeleted.push_back(image.CameraId());
    }

    std::vector<colmap::camera_t>::iterator it;
    for (it = camera_ids_.begin(); it != camera_ids_.end();) {
        if (std::find(toBeDeleted.begin(), toBeDeleted.end(), *it) !=
            toBeDeleted.end()) {
            it = camera_ids_.erase(it);
        } else {
            ++it;
        }
    }

    UpdateViews();
    model_viewer_widget_->UpdateMovieGrabber();
}

void MovieGrabberWidget::Clear() {
    view_data_.clear();
    while (table_->rowCount() > 0) {
        table_->removeRow(0);
    }
    views.clear();
    camera_ids_.clear();
    model_viewer_widget_->UpdateMovieGrabber();
}

void MovieGrabberWidget::Assemble() {
    if (table_->rowCount() < 2) {
        QMessageBox::critical(this, tr("Error"),
                              tr("You must add at least two control views."));
        return;
    }

    if (model_viewer_widget_->GetProjectionType() !=
        RenderOptions::ProjectionType::PERSPECTIVE) {
        QMessageBox::critical(this, tr("Error"),
                              tr("You must use perspective projection."));
        return;
    }

    const QString path = QFileDialog::getExistingDirectory(
            this, tr("Choose destination..."), "", QFileDialog::ShowDirsOnly);

    // File dialog cancelled?
    if (path == "") {
        return;
    }

    const QDir dir = QDir(path);

    const ccGLMatrixd model_view_matrix_cached =
            model_viewer_widget_->ModelViewMatrix();
    const float point_size_cached = model_viewer_widget_->PointSize();
    const float image_size_cached = model_viewer_widget_->ImageSize();
    const std::vector<Image> views_cached = views;

    // Make sure we do not render movie grabber path.
    views.clear();
    model_viewer_widget_->UpdateMovieGrabber();
    bool coords_changed = false;
    bool lengend_changed = false;
    bool coords_shown = ecvDisplayTools::OrientationMarkerShown();
    bool lengend_shown = ecvDisplayTools::OverlayEntitiesAreDisplayed();
    if (lengend_shown) {
        ecvDisplayTools::DisplayOverlayEntities(false);
        lengend_changed = true;
    }
    if (coords_shown) {
        ecvDisplayTools::ToggleOrientationMarker(false);
        coords_changed = true;
    }

    const float frame_rate = frame_rate_sb_->value();
    const float frame_time = 1.0f / frame_rate;
    size_t frame_number = 0;

    // precompute frame count
    float totalTime = 0;
    for (int row = 1; row < table_->rowCount(); ++row) {
        const auto logical_idx = table_->verticalHeader()->logicalIndex(row);
        QTableWidgetItem* prev_table_item = table_->item(logical_idx - 1, 0);
        QTableWidgetItem* table_item = table_->item(logical_idx, 0);
        // Time difference between previous and current view.
        const float dt = std::abs(table_item->text().toFloat() -
                                  prev_table_item->text().toFloat());
        totalTime += dt;
    }

    int frameCount = static_cast<int>(frame_rate * totalTime);

    // show progress dialog
    QProgressDialog progressDialog(QString("Frames: %1").arg(frameCount),
                                   "Cancel", 0, frameCount, this);
    progressDialog.setWindowTitle("Generate movie frames");
    progressDialog.setModal(true);
    // progressDialog.setAutoClose(false);
    progressDialog.show();
    QApplication::processEvents();

    // Data of first view.
    ecvViewportParameters firstViewport =
            view_data_[table_->item(0, 0)].viewportParams;
    for (int row = 1; row < table_->rowCount(); ++row) {
        const auto logical_idx = table_->verticalHeader()->logicalIndex(row);
        QTableWidgetItem* prev_table_item = table_->item(logical_idx - 1, 0);
        QTableWidgetItem* table_item = table_->item(logical_idx, 0);

        const MovieGrabberWidget::ViewData& prev_view_data =
                view_data_.at(prev_table_item);
        const MovieGrabberWidget::ViewData& view_data =
                view_data_.at(table_item);

        // Time difference between previous and current view.
        const float dt = std::abs(table_item->text().toFloat() -
                                  prev_table_item->text().toFloat());

        // Point size differences between previous and current view.
        const float dpoint_size =
                view_data.point_size - prev_view_data.point_size;
        const float dimage_size =
                view_data.image_size - prev_view_data.image_size;

        const auto num_frames = dt * frame_rate;

        for (size_t i = 0; i < num_frames; ++i) {
            const float t = i * frame_time;
            float tt = t / dt;

            if (smooth_cb_->isChecked()) {
                tt = ScaleSigmoid(tt,
                                  static_cast<float>(smoothness_sb_->value()));
            }

            ViewInterpolate interpolator(prev_view_data.viewportParams,
                                         view_data.viewportParams);
            ecvViewportParameters currentViewport;
            interpolator.interpolate(currentViewport, static_cast<double>(tt));

            // Set point and image sizes.
            model_viewer_widget_->StartRender();
            model_viewer_widget_->SetPointSize(
                    prev_view_data.point_size + dpoint_size * tt, false);
            model_viewer_widget_->SetImageSize(
                    prev_view_data.image_size + dimage_size * tt, false);
            ecvDisplayTools::SetViewportParameters(currentViewport);
            model_viewer_widget_->EndRender();
            model_viewer_widget_->update();

            QImage image = model_viewer_widget_->GrabImage();
            image.save(dir.filePath("frame" +
                                    QString().asprintf("%06zu", frame_number) +
                                    ".png"));
            frame_number += 1;
            progressDialog.setValue(frame_number);
            QApplication::processEvents();
            if (progressDialog.wasCanceled()) {
                firstViewport = currentViewport;
                break;
            }
        }
    }

    views = views_cached;
    model_viewer_widget_->SetPointSize(point_size_cached);
    model_viewer_widget_->SetImageSize(image_size_cached);
    model_viewer_widget_->UpdateMovieGrabber();
    if (lengend_changed) {
        ecvDisplayTools::DisplayOverlayEntities(lengend_shown);
    }
    if (coords_changed) {
        ecvDisplayTools::ToggleOrientationMarker(coords_shown);
    }
    ecvDisplayTools::SetViewportParameters(firstViewport);
    ecvDisplayTools::UpdateScreen();
}

void MovieGrabberWidget::TimeChanged(QTableWidgetItem* item) {
    table_->sortItems(0, Qt::AscendingOrder);
    UpdateViews();
    model_viewer_widget_->UpdateMovieGrabber();
}

void MovieGrabberWidget::SelectionChanged(const QItemSelection& selected,
                                          const QItemSelection& deselected) {
    for (const auto& index : table_->selectionModel()->selectedIndexes()) {
        model_viewer_widget_->SelectMoviewGrabberView(index.row());
    }
}

void MovieGrabberWidget::UpdateViews() {
    views.clear();
    colmap::camera_t base_id = 1000;
    std::size_t lastNum = camera_ids_.size();
    for (int row = 0; row < table_->rowCount(); ++row) {
        const auto logical_idx = table_->verticalHeader()->logicalIndex(row);
        QTableWidgetItem* item = table_->item(logical_idx, 0);

        const Eigen::Matrix4d model_view_matrix = ccGLMatrixd::ToEigenMatrix4(
                view_data_.at(item).model_view_matrix);

        Image image;
        colmap::camera_t camera_id;
        std::size_t curIndex = static_cast<std::size_t>(row);
        if (curIndex < lastNum) {
            camera_id = camera_ids_[curIndex];
        } else {
            if (camera_ids_.empty()) {
                camera_id = base_id + static_cast<colmap::camera_t>(row);
            } else {
                camera_id = camera_ids_[camera_ids_.size() - 1] + 1;
            }

            camera_ids_.push_back(camera_id);
        }

        image.SetImageId(camera_id);
        image.SetCameraId(camera_id);

        image.Qvec() =
                RotationMatrixToQuaternion(model_view_matrix.block<3, 3>(0, 0));
        image.Tvec() = model_view_matrix.block<3, 1>(0, 3);
        views.push_back(image);
    }
}

}  // namespace cloudViewer

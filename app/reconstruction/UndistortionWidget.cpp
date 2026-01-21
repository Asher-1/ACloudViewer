// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "UndistortionWidget.h"

#include "base/reconstruction.h"
#include "util/misc.h"
#include "util/option_manager.h"

namespace cloudViewer {

using namespace colmap;

UndistortionWidget::UndistortionWidget(QWidget* parent,
                                       const OptionManager* options)
    : OptionsWidget(parent),
      options_(options),
      reconstruction_(nullptr),
      thread_control_widget_(new ThreadControlWidget(this)) {
    setWindowFlags(Qt::Dialog);
    setWindowModality(Qt::ApplicationModal);
    setWindowTitle("Undistortion");

    output_format_ = new QComboBox(this);
    output_format_->addItem("COLMAP");
    output_format_->addItem("PMVS");
    output_format_->addItem("CMP-MVS");
    output_format_->setFont(font());
    AddWidgetRow("format", output_format_);

    AddOptionDouble(&undistortion_options_.min_scale, "min_scale", 0);
    AddOptionDouble(&undistortion_options_.max_scale, "max_scale", 0);
    AddOptionInt(&undistortion_options_.max_image_size, "max_image_size", -1);
    AddOptionDouble(&undistortion_options_.blank_pixels, "blank_pixels", 0);
    AddOptionDouble(&undistortion_options_.roi_min_x, "roi_min_x", 0.0, 1.0);
    AddOptionDouble(&undistortion_options_.roi_min_y, "roi_min_y", 0.0, 1.0);
    AddOptionDouble(&undistortion_options_.roi_max_x, "roi_max_x", 0.0, 1.0);
    AddOptionDouble(&undistortion_options_.roi_max_y, "roi_max_y", 0.0, 1.0);
    AddOptionDirPath(&output_path_, "output_path");

    AddSpacer();

    QPushButton* undistort_button = new QPushButton(tr("Undistort"), this);
    connect(undistort_button, &QPushButton::released, this,
            &UndistortionWidget::Undistort);
    grid_layout_->addWidget(undistort_button, grid_layout_->rowCount(), 1);
}

void UndistortionWidget::Show(const Reconstruction& reconstruction) {
    reconstruction_ = &reconstruction;
    show();
    raise();
}

bool UndistortionWidget::IsValid() const { return ExistsDir(output_path_); }

void UndistortionWidget::Undistort() {
    CHECK_NOTNULL(reconstruction_);

    WriteOptions();

    if (IsValid()) {
        Thread* undistorter = nullptr;

        if (output_format_->currentIndex() == 0) {
            undistorter = new COLMAPUndistorter(
                    undistortion_options_,
                    const_cast<colmap::Reconstruction*>(reconstruction_),
                    *options_->image_path, output_path_);
        } else if (output_format_->currentIndex() == 1) {
            undistorter = new PMVSUndistorter(
                    undistortion_options_,
                    const_cast<colmap::Reconstruction*>(reconstruction_),
                    *options_->image_path, output_path_);
        } else if (output_format_->currentIndex() == 2) {
            undistorter = new CMPMVSUndistorter(
                    undistortion_options_,
                    const_cast<colmap::Reconstruction*>(reconstruction_),
                    *options_->image_path, output_path_);
        } else {
            QMessageBox::critical(this, "", tr("Invalid output format"));
            return;
        }

        thread_control_widget_->StartThread("Undistorting...", true,
                                            undistorter);
    } else {
        QMessageBox::critical(this, "", tr("Invalid output path"));
    }
}

}  // namespace cloudViewer

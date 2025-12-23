// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore>
#include <QtWidgets>

#include "ui/options_widget.h"
#include "ThreadControlWidget.h"
#include "base/undistortion.h"

namespace colmap {
class Reconstruction;
}

#include "util/option_manager.h"

namespace cloudViewer {

using OptionManager = colmap::OptionManager;

class UndistortionWidget : public colmap::OptionsWidget {
public:
    UndistortionWidget(QWidget* parent, const OptionManager* options);

    void Show(const colmap::Reconstruction& reconstruction);
    bool IsValid() const;

private:
    void Undistort();

    const OptionManager* options_;
    const colmap::Reconstruction* reconstruction_;

    ThreadControlWidget* thread_control_widget_;

    QComboBox* output_format_;
    colmap::UndistortCameraOptions undistortion_options_;
    std::string output_path_;
};

}  // namespace cloudViewer

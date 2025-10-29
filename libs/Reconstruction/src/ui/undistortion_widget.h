// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore>
#include <QtWidgets>

#include "base/reconstruction.h"
#include "base/undistortion.h"
#include "ui/options_widget.h"
#include "ui/thread_control_widget.h"
#include "util/misc.h"
#include "util/option_manager.h"

namespace colmap {

class UndistortionWidget : public OptionsWidget {
public:
    UndistortionWidget(QWidget* parent, const OptionManager* options);

    void Show(const Reconstruction& reconstruction);
    bool IsValid() const;

private:
    void Undistort();

    const OptionManager* options_;
    const Reconstruction* reconstruction_;

    ThreadControlWidget* thread_control_widget_;

    QComboBox* output_format_;
    UndistortCameraOptions undistortion_options_;
    std::string output_path_;
};

}  // namespace colmap

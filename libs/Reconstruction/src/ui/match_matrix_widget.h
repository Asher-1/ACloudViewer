// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ui/image_viewer_widget.h"
#include "util/option_manager.h"

namespace colmap {

// Widget to visualize match matrix.
class MatchMatrixWidget : public ImageViewerWidget {
public:
    MatchMatrixWidget(QWidget* parent, OptionManager* options);

    void Show();

private:
    OptionManager* options_;
};

}  // namespace colmap

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ImageViewerWidget.h"

namespace cloudViewer {

class OptionManager;

// Widget to visualize match matrix.
class MatchMatrixWidget : public ImageViewerWidget {
public:
    MatchMatrixWidget(QWidget* parent, OptionManager* options);

    void Show();

private:
    OptionManager* options_;
};

}  // namespace cloudViewer

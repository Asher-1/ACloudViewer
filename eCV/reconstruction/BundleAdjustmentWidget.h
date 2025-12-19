// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore>
#include <QtWidgets>

#include "OptionsWidget.h"
#include "base/reconstruction.h"
#include "util/option_manager.h"

namespace cloudViewer {

using OptionManager = colmap::OptionManager;
class ReconstructionWidget;
class ThreadControlWidget;

class BundleAdjustmentWidget : public OptionsWidget {
public:
    BundleAdjustmentWidget(ReconstructionWidget* main_window,
                           OptionManager* options);

    void Show(colmap::Reconstruction* reconstruction);

private:
    void Run();
    void Render();

    ReconstructionWidget* main_window_;
    OptionManager* options_;
    colmap::Reconstruction* reconstruction_;
    ThreadControlWidget* thread_control_widget_;
    QAction* render_action_;
};

}  // namespace cloudViewer

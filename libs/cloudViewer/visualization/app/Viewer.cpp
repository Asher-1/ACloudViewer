// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/visualization/app/Viewer.h"

#include <iostream>
#include <string>

#include "cloudViewer/utility/Logging.h"
#include "cloudViewer/visualization/gui/Application.h"
#include "cloudViewer/visualization/gui/Native.h"
#include "cloudViewer/visualization/visualizer/GuiVisualizer.h"

namespace cloudViewer {
namespace visualization {
namespace app {

static const std::string usage = "usage: ./CloudViewer [meshfile|pointcloud]";
static const int width = 1280;
static const int height = 960;

void RunViewer(int argc, const char *argv[]) {
    std::function<void(const std::string &)> print_fcn =
            utility::Logger::GetInstance().GetPrintFunction();
    utility::Logger::GetInstance().ResetPrintFunction();

    const char *path = nullptr;
    if (argc == 2) {
        path = argv[1];
    } else if (argc > 2) {
        utility::LogWarning(usage.c_str());
    }

    auto &app = gui::Application::GetInstance();
    app.Initialize(argc, argv);

    auto vis = std::make_shared<cloudViewer::visualization::GuiVisualizer>(
            "CloudViewer", width, height);
    if (path && path[0] != '\0') {
        vis->LoadGeometry(path);
    }
    gui::Application::GetInstance().AddWindow(vis);
    // When Run() ends, Filament will be stopped, so we can't be holding on
    // to any GUI objects.
    vis.reset();

    app.Run();

    utility::Logger::GetInstance().SetPrintFunction(print_fcn);
}

}  // namespace app
}  // namespace visualization
}  // namespace cloudViewer

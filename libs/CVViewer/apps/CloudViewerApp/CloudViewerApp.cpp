// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CloudViewerApp.h"

#include <string>

#include <CloudViewer.h>
#include "visualization/gui/Native.h"
#include "visualization/visualizer/GuiVisualizer.h"

using namespace cloudViewer;
using namespace cloudViewer::geometry;
using namespace cloudViewer::visualization;

namespace {
static const std::string gUsage = "Usage: CloudViewer [meshfile|pointcloud]";
}  // namespace

int Run(int argc, const char *argv[]) {
    const char *path = nullptr;
    if (argc > 1) {
        path = argv[1];
        if (argc > 2) {
            utility::LogWarning(gUsage.c_str());
        }
    }

    auto &app = gui::Application::GetInstance();
    app.Initialize(argc, argv);

    auto vis = cloudViewer::make_shared<GuiVisualizer>("CloudViewer", WIDTH, HEIGHT);
    bool is_path_valid = (path && path[0] != '\0');
    if (is_path_valid) {
        vis->LoadGeometry(path);
    }
    gui::Application::GetInstance().AddWindow(vis);
    // when Run() ends, Filament will be stopped, so we can't be holding on
    // to any GUI objects.
    vis.reset();

    app.Run();

    return 0;
}

#if __APPLE__
// CloudViewerApp_mac.mm
#else
int main(int argc, const char *argv[]) { return Run(argc, argv); }
#endif  // __APPLE__

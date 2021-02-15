// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
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
            cloudViewer::utility::LogWarning(gUsage.c_str());
        }
    }

    auto &app = gui::Application::GetInstance();
    app.Initialize(argc, argv);

    auto vis = std::make_shared<GuiVisualizer>("CloudViewer", WIDTH, HEIGHT);
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
// CloudViewer_mac.mm
#else
int main(int argc, const char *argv[]) { return Run(argc, argv); }
#endif  // __APPLE__

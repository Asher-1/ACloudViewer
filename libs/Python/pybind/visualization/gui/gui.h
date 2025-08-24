// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace geometry {
class Image;
}

namespace visualization {
namespace rendering {
class CloudViewerScene;
}

namespace gui {

void InitializeForPython(std::string resource_path = "");
std::shared_ptr<geometry::Image> RenderToImageWithoutWindow(
        rendering::CloudViewerScene *scene, int width, int height);
std::shared_ptr<geometry::Image> RenderToDepthImageWithoutWindow(
        rendering::CloudViewerScene *scene, int width, int height);

void pybind_gui(py::module &m);

void pybind_gui_events(py::module &m);
void pybind_gui_classes(py::module &m);

}  // namespace gui
}  // namespace visualization
}  // namespace cloudViewer

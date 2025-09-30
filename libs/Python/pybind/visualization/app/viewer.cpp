// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/visualization/app/viewer.h"

#include "cloudViewer/visualization/app/Viewer.h"
#include "pybind/docstring.h"

namespace cloudViewer {
namespace visualization {
namespace app {

void pybind_app(py::module &m) {
    py::module m_app = m.def_submodule(
            "app", "Functionality for running the cloudViewer viewer.");
    m_app.def(
            "run_viewer",
            [](const std::vector<std::string> &args) {
                const char **argv = new const char *[args.size()];
                for (size_t it = 0; it < args.size(); it++) {
                    argv[it] = args[it].c_str();
                }
                RunViewer(args.size(), argv);
                delete[] argv;
            },
            "args"_a);
    docstring::FunctionDocInject(
            m_app, "run_viewer",
            {{"args",
              "List of arguments containing the path of the calling program "
              "(which should be in the same directory as the gui resources "
              "folder) and the optional path of the geometry to visualize."}});
}

}  // namespace app
}  // namespace visualization
}  // namespace cloudViewer

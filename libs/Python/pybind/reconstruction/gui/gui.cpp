// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pipelines/application_gui.h"

#include "pybind/docstring.h"
#include "pybind/reconstruction/gui/gui.h"
#include "pybind/reconstruction/reconstruction_options.h"

namespace cloudViewer {
namespace reconstruction {
namespace gui {

// Reconstruction project functions have similar arguments, sharing arg
// docstrings
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"database_path",
                 "Path to database in which to store the extracted data"},
                {"image_path",
                 "Root path to folder which contains the images."},
                {"import_path",
                 "The project import path containing *.ini project file."},
                {"output_path", "The options saving output path."},
                {"quality",
                 "The supported project processing quality types are {low, "
                 "medium, high, extreme}."}};

void pybind_gui_methods(py::module &m) {
    m.def("run_graphical_gui", &GraphicalUserInterface,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the project gui application", "database_path"_a = "",
          "image_path"_a = "", "import_path"_a = "");
    docstring::FunctionDocInject(m, "run_graphical_gui",
                                 map_shared_argument_docstrings);

    m.def("generate_project", &GenerateProject,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the generation of project", "output_path"_a,
          "quality"_a = "high");
    docstring::FunctionDocInject(m, "generate_project",
                                 map_shared_argument_docstrings);
}

void pybind_gui(py::module &m) {
    py::module m_submodule = m.def_submodule("gui", "Reconstruction GUI.");
    pybind_gui_methods(m_submodule);
}

}  // namespace gui
}  // namespace reconstruction
}  // namespace cloudViewer

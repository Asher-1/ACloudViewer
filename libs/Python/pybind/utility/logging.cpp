// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Logging.h>

#include "pybind/cloudViewer_pybind.h"
#include "pybind/docstring.h"

namespace cloudViewer {
namespace utility {

void pybind_logging(py::module& m) {
    py::native_enum<VerbosityLevel>(m, "VerbosityLevel", "enum.IntEnum",
                                    "Enum class for VerbosityLevel.")
            .value("Error", VerbosityLevel::Error)
            .value("Warning", VerbosityLevel::Warning)
            .value("Info", VerbosityLevel::Info)
            .value("Debug", VerbosityLevel::Debug)
            .export_values()
            .finalize();

    m.def("set_verbosity_level", &SetVerbosityLevel,
          "Set global verbosity level of CloudViewer",
          py::arg("verbosity_level"));
    docstring::FunctionDocInject(
            m, "set_verbosity_level",
            {{"verbosity_level",
              "Messages with equal or less than ``verbosity_level`` verbosity "
              "will be printed."}});

    m.def("get_verbosity_level", &GetVerbosityLevel,
          "Get global verbosity level of CloudViewer");
    docstring::FunctionDocInject(m, "get_verbosity_level");

    m.def("reset_print_function", []() {
        utility::LogInfo("Resetting default logger to print to terminal.");
        utility::Logger::GetInstance().ResetPrintFunction();
    }, "Reset the print function to the default (print to terminal)");
    docstring::FunctionDocInject(m, "reset_print_function");

    py::class_<VerbosityContextManager>(m, "VerbosityContextManager",
                                        "A context manager to "
                                        "temporally change the "
                                        "verbosity level of CloudViewer")
            .def(py::init<VerbosityLevel>(),
                 "Create a VerbosityContextManager with a given VerbosityLevel",
                 "level"_a)
            .def(
                    "__enter__",
                    [&](VerbosityContextManager& cm) { cm.Enter(); },
                    "Enter the context manager")
            .def(
                    "__exit__",
                    [&](VerbosityContextManager& cm, pybind11::object exc_type,
                        pybind11::object exc_value,
                        pybind11::object traceback) { cm.Exit(); },
                    "Exit the context manager");
}

}  // namespace utility
}  // namespace cloudViewer

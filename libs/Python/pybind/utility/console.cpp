// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                          -
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

// CV_CORE_LIB
#include <Console.h>

#include "pybind/docstring.h"
#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace utility {
	
void pybind_console(py::module &m) {
	py::enum_<CVLib::utility::VerbosityLevel> vl(m, "VerbosityLevel", py::arithmetic(),
		"VerbosityLevel");
	vl.value("Error", CVLib::utility::VerbosityLevel::Error)
		.value("Warning", CVLib::utility::VerbosityLevel::Warning)
		.value("Info", CVLib::utility::VerbosityLevel::Info)
		.value("Debug", CVLib::utility::VerbosityLevel::Debug)
		.export_values();
	// Trick to write docs without listing the members in the enum class again.
	vl.attr("__doc__") = docstring::static_property(
		py::cpp_function([](py::handle arg) -> std::string {
		return "Enum class for VerbosityLevel.";
	}),
		py::none(), py::none(), "");

	m.def("set_verbosity_level", &CVLib::utility::SetVerbosityLevel,
        "Set global verbosity level of CloudViewer", py::arg("verbosity_level"));
	docstring::FunctionDocInject(
		m, "set_verbosity_level",
		{ {"verbosity_level",
		  "Messages with equal or less than ``verbosity_level`` verbosity "
		  "will be printed."} });

	m.def("get_verbosity_level", &CVLib::utility::GetVerbosityLevel,
        "Get global verbosity level of CloudViewer");
	docstring::FunctionDocInject(m, "get_verbosity_level");

	py::class_<CVLib::utility::VerbosityContextManager>(m, "VerbosityContextManager",
		"A context manager to "
		"temporally change the "
        "verbosity level of CloudViewer")
		.def(py::init<CVLib::utility::VerbosityLevel>(),
			"Create a VerbosityContextManager with a given VerbosityLevel",
			"level"_a)
		.def(
			"__enter__",
			[&](CVLib::utility::VerbosityContextManager& cm) { cm.enter(); },
			"Enter the context manager")
		.def(
			"__exit__",
			[&](CVLib::utility::VerbosityContextManager& cm, pybind11::object exc_type,
				pybind11::object exc_value,
				pybind11::object traceback) { cm.exit(); },
			"Exit the context manager");
}

}  // namespace utility
}  // namespace cloudViewer

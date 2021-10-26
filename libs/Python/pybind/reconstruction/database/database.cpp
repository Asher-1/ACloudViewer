// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
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

#include "pybind/reconstruction/database/database.h"

#include <memory>
#include "pipelines/database.h"
#include "pybind/docstring.h"

namespace cloudViewer {
namespace reconstruction {
namespace database {

// Reconstruction feature functions have similar arguments, sharing arg
// docstrings
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"database_path",
                 "Path to database in which to store the extracted data"},
                {"first_database_path", "The first imported database directory."},
                {"second_database_path", "The other imported database directory"},
                {"merged_database_path", "The merged database directory"},
                {"type", "supported type {all, images, features, matches}"}};

void pybind_database_methods(py::module &m) {
    m.def("clean_database", &CleanDatabase,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the clearance of database", "database_path"_a,
          "type"_a);
    docstring::FunctionDocInject(m, "clean_database",
                                 map_shared_argument_docstrings);

    m.def("create_database", &CreateDatabase,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the creation of database", "database_path"_a);
    docstring::FunctionDocInject(m, "create_database",
                                 map_shared_argument_docstrings);

    m.def("merge_database", &MergeDatabase,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the merge between two databases", "first_database_path"_a,
          "second_database_path"_a, "merged_database_path"_a);
    docstring::FunctionDocInject(m, "merge_database",
                                 map_shared_argument_docstrings);
}

void pybind_database(py::module &m) {
    py::module m_submodule =
            m.def_submodule("database", "Reconstruction Database.");
    pybind_database_methods(m_submodule);
}

}  // namespace database
}  // namespace reconstruction
}  // namespace cloudViewer

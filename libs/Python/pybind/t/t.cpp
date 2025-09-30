// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/t/t.h"

#include "pybind/cloudViewer_pybind.h"
#include "pybind/t/geometry/geometry.h"
#include "pybind/t/io/io.h"
#include "pybind/t/pipelines/pipelines.h"

namespace cloudViewer {
namespace t {

void pybind_t(py::module& m) {
    py::module m_submodule = m.def_submodule("t");
    pipelines::pybind_pipelines(m_submodule);
    geometry::pybind_geometry(m_submodule);
    io::pybind_io(m_submodule);
}

}  // namespace t
}  // namespace cloudViewer

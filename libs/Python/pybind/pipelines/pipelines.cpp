// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/pipelines/pipelines.h"

#include "pybind/cloudViewer_pybind.h"
#include "pybind/pipelines/color_map/color_map.h"
#include "pybind/pipelines/integration/integration.h"
#include "pybind/pipelines/odometry/odometry.h"
#include "pybind/pipelines/registration/registration.h"

namespace cloudViewer {
namespace pipelines {

void pybind_pipelines(py::module& m) {
    py::module m_pipelines = m.def_submodule("pipelines");
    color_map::pybind_color_map(m_pipelines);
    integration::pybind_integration(m_pipelines);
    registration::pybind_registration(m_pipelines);
    odometry::pybind_odometry(m_pipelines);
}

}  // namespace pipelines
}  // namespace cloudViewer

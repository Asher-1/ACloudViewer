// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/t/pipelines/pipelines.h"

#include "pybind/cloudViewer_pybind.h"
#include "pybind/t/pipelines/odometry/odometry.h"
#include "pybind/t/pipelines/registration/registration.h"
#include "pybind/t/pipelines/slac/slac.h"
#include "pybind/t/pipelines/slam/slam.h"

namespace cloudViewer {
namespace t {
namespace pipelines {

void pybind_pipelines(py::module& m) {
    py::module m_pipelines = m.def_submodule(
            "pipelines", "Tensor-based geometry processing pipelines.");
    odometry::pybind_odometry(m_pipelines);
    registration::pybind_registration(m_pipelines);
    slac::pybind_slac(m_pipelines);
    slam::pybind_slam(m_pipelines);
}

}  // namespace pipelines
}  // namespace t
}  // namespace cloudViewer

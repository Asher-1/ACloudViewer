// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace visualization {
namespace webrtc_server {

void pybind_webrtc_server(py::module &m);

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace cloudViewer

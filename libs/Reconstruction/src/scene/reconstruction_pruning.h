// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "base/reconstruction.h"

namespace colmap {

std::vector<point3D_t> FindRedundantPoints3D(
        double min_coverage_gain, const Reconstruction& reconstruction);

}  // namespace colmap

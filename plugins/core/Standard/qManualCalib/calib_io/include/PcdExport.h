// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

#include "CalibTypes.h"

namespace mcalib {

bool writePcdBinaryXYZIRT(const std::string& path,
                          const std::vector<PointXYZIRT>& points,
                          const std::string& frame_id = "lidar");

// Writes PCD with per-point RGB (packed as float, PCL convention).
// rgb is packed as (r << 16) | (g << 8) | b in the low 24 bits.
bool writePcdBinaryXYZIRGB(const std::string& path,
                           const std::vector<PointXYZIRT>& points,
                           const std::vector<uint32_t>& rgb_packed,
                           const std::string& frame_id = "lidar");

}  // namespace mcalib

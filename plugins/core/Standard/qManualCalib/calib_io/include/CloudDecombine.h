// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "CalibTypes.h"
#include "ProtoDecoder.h"

namespace mcalib {

/// Split combined lidar clouds and transform to sensing frame.
/// Priority: PointCloud2.lidar_configs (field 12), then VehicleCalibConfig
/// fallback.
bool decombinePointCloud(const ProtoDecoder::PointCloud2Data& cloud_data,
                         const VehicleCalibConfig* calib_config,
                         std::vector<PointXYZIRT>& cloud_out,
                         std::string& frame_id_out,
                         int64_t& cloud_stamp_us);

}  // namespace mcalib

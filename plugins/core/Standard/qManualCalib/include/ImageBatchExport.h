// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <map>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "BatchExportTypes.h"
#include "BevRemapBackend.h"
#include "CalibTypes.h"

namespace mcalib {

class RosBagReader;
class BirdEyeView;

struct ImageBatchExportOptions {
    int view_mode = 0;  // 0=BEV, 1=LiDAR projection
    int num_samples = 20;
    BevRemapMode remap_mode = BevRemapMode::Auto;
    int point_size = 2;
    BatchExportProgress progress;
};

struct ImageBatchExportContext {
    VehicleCalibConfig config;
    std::map<std::string, Vector6d> delta_extrinsics;
    std::vector<Vector6d> delta_lidar_extrinsics;
    std::vector<std::pair<std::string, std::string>> camera_topics;
    std::vector<std::string> cloud_topics;
    std::map<std::string, std::string> bev_slot_map;
    std::string projection_camera;
    int calib_mode = 0;
    std::string current_sensor;
};

BatchExportResult exportImagesBatch(RosBagReader& reader,
                                    const ImageBatchExportContext& ctx,
                                    const std::string& output_dir,
                                    const ImageBatchExportOptions& options);

}  // namespace mcalib

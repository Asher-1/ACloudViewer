// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "BatchExportTypes.h"
#include "CalibTypes.h"

namespace mcalib {

class RosBagReader;

struct PcdBatchExportContext {
    VehicleCalibConfig config;
    std::vector<Vector6d> delta_lidar_extrinsics;
    std::vector<bool> selected_lidars;
    std::vector<std::pair<std::string, std::string>> camera_topics;
    std::vector<std::string> cloud_topics;
    std::deque<std::pair<int64_t, Eigen::Isometry3d>> vehicle_poses;
    float ground_filter_min = -5.f;
    float ground_filter_max = 30.f;
    float dist_filter_min = 0.f;
    float dist_filter = 200.f;
};

struct PcdBatchExportOptions {
    int num_samples = 20;
    BatchExportProgress progress;
};

BatchExportResult exportPcdsBatch(RosBagReader& reader,
                                  const PcdBatchExportContext& ctx,
                                  const std::string& output_dir,
                                  const PcdBatchExportOptions& options);

}  // namespace mcalib

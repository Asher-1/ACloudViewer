// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Geometry>
#include <map>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "BatchExportTypes.h"
#include "CalibTypes.h"

namespace mcalib {
class RosBagReader;
class BirdEyeView;
}  // namespace mcalib

namespace mcalib {

struct BevBatchExportOptions {
    int num_samples = 20;
    bool use_parallel_remap = true;
    BatchExportProgress progress;
};

struct BevBatchExportContext {
    VehicleCalibConfig config;
    std::map<std::string, mcalib::Vector6d> delta_extrinsics;
    std::vector<std::pair<std::string, std::string>> camera_topics;
    std::vector<std::string> cloud_topics;
};

BatchExportResult exportBevImagesBatch(
        RosBagReader& reader,
        const BevBatchExportContext& ctx,
        const std::string& output_dir,
        const BevBatchExportOptions& options = BevBatchExportOptions());

}  // namespace mcalib

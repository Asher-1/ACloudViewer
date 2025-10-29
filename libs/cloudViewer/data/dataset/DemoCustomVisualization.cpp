// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Logging.h>

#include <string>
#include <vector>

#include "cloudViewer/data/Dataset.h"

namespace cloudViewer {
namespace data {

const static DataDescriptor data_descriptor = {
        CloudViewerDownloadsPrefix() +
                "20220301-data/DemoCustomVisualization.zip",
        "04cb716145c51d0119b59c7876249891"};

DemoCustomVisualization::DemoCustomVisualization(const std::string& data_root)
    : DownloadDataset("DemoCustomVisualization", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    point_cloud_path_ = extract_dir + "/fragment.ply";
    camera_trajectory_path_ = extract_dir + "/camera_trajectory.json";
    render_option_path_ = extract_dir + "/renderoption.json";
}

}  // namespace data
}  // namespace cloudViewer

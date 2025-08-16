// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <string>
#include <vector>

#include "cloudViewer/data/Dataset.h"
#include <Logging.h>

namespace cloudViewer {
namespace data {

const static DataDescriptor data_descriptor = {
        CloudViewerDownloadsPrefix() + "20220201-data/DemoPoseGraphOptimization.zip",
        "af085b28d79dea7f0a50aef50c96b62c"};

DemoPoseGraphOptimization::DemoPoseGraphOptimization(
        const std::string& data_root)
    : DownloadDataset("DemoPoseGraphOptimization", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    pose_graph_fragment_path_ =
            extract_dir + "/pose_graph_example_fragment.json";
    pose_graph_global_path_ = extract_dir + "/pose_graph_example_global.json";
}

}  // namespace data
}  // namespace cloudViewer

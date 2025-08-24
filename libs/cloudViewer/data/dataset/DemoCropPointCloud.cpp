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
        CloudViewerDownloadsPrefix() + "20220201-data/DemoCropPointCloud.zip",
        "12dbcdddd3f0865d8312929506135e23"};

DemoCropPointCloud::DemoCropPointCloud(const std::string& data_root)
    : DownloadDataset("DemoCropPointCloud", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    point_cloud_path_ = extract_dir + "/fragment.ply";
    cropped_json_path_ = extract_dir + "/cropped.json";
}

}  // namespace data
}  // namespace cloudViewer

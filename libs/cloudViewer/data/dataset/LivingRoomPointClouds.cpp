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
        CloudViewerDownloadsPrefix() + "redwood/livingroom1-fragments-ply.zip",
        "36e0eb23a66ccad6af52c05f8390d33e"};

LivingRoomPointClouds::LivingRoomPointClouds(const std::string& data_root)
    : DownloadDataset("LivingRoomPointClouds", data_descriptor, data_root) {
    paths_.reserve(57);
    for (int i = 0; i < 57; ++i) {
        paths_.push_back(GetExtractDir() + "/cloud_bin_" + std::to_string(i) +
                         ".ply");
    }
}

std::string LivingRoomPointClouds::GetPaths(size_t index) const {
    if (index > 56) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 56 but got {}.",
                index);
    }
    return paths_[index];
}

}  // namespace data
}  // namespace cloudViewer

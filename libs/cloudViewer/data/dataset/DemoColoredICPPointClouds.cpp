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
                "20220201-data/DemoColoredICPPointClouds.zip",
        "bf8d469e892d76f2e69e1213207c0e30"};

DemoColoredICPPointClouds::DemoColoredICPPointClouds(
        const std::string& data_root)
    : DownloadDataset("DemoColoredICPPointClouds", data_descriptor, data_root) {
    paths_.push_back(GetExtractDir() + "/frag_115.ply");
    paths_.push_back(GetExtractDir() + "/frag_116.ply");
}

std::string DemoColoredICPPointClouds::GetPaths(size_t index) const {
    if (index > 1) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 1 but got {}.",
                index);
    }
    return paths_[index];
}

}  // namespace data
}  // namespace cloudViewer

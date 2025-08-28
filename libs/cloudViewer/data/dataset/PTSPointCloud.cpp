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
        CloudViewerDownloadsPrefix() + "20220301-data/point_cloud_sample1.pts",
        "5c2c618b703d0161e6e333fcbf55a1e9"};

PTSPointCloud::PTSPointCloud(const std::string& data_root)
    : DownloadDataset("PTSPointCloud", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/point_cloud_sample1.pts";
}

}  // namespace data
}  // namespace cloudViewer

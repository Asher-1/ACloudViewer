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
        CloudViewerDownloadsPrefix() + "20220201-data/fragment.pcd",
        "f3a613fd2bdecd699aabdd858fb29606"};

PCDPointCloud::PCDPointCloud(const std::string& data_root)
    : DownloadDataset("PCDPointCloud", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/fragment.pcd";
}

}  // namespace data
}  // namespace cloudViewer

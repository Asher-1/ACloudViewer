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
        CloudViewerDownloadsPrefix() + "20220201-data/EaglePointCloud.ply",
        "e4e6c77bc548e7eb7548542a0220ad78"};

EaglePointCloud::EaglePointCloud(const std::string& data_root)
    : DownloadDataset("EaglePointCloud", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/EaglePointCloud.ply";
}

}  // namespace data
}  // namespace cloudViewer

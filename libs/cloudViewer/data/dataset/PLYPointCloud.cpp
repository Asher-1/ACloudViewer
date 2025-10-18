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
        CloudViewerDownloadsPrefix() + "20220201-data/fragment.ply",
        "831ecffd4d7cbbbe02494c5c351aa6e5"};

PLYPointCloud::PLYPointCloud(const std::string& data_root)
    : DownloadDataset("PLYPointCloud", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/fragment.ply";
}

}  // namespace data
}  // namespace cloudViewer

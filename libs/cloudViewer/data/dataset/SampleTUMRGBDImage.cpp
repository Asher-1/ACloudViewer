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
        CloudViewerDownloadsPrefix() + "20220201-data/SampleTUMRGBDImage.zip",
        "91758d42b142dbad7b0d90e857ad47a8"};

SampleTUMRGBDImage::SampleTUMRGBDImage(const std::string& data_root)
    : DownloadDataset("SampleTUMRGBDImage", data_descriptor, data_root) {
    color_path_ = GetExtractDir() + "/TUM_color.png";
    depth_path_ = GetExtractDir() + "/TUM_depth.png";
}

}  // namespace data
}  // namespace cloudViewer

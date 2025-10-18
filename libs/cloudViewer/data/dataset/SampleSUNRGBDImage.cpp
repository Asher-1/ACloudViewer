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
        CloudViewerDownloadsPrefix() + "20220201-data/SampleSUNRGBDImage.zip",
        "b1a430586547c8986bdf8b36179a8e67"};

SampleSUNRGBDImage::SampleSUNRGBDImage(const std::string& data_root)
    : DownloadDataset("SampleSUNRGBDImage", data_descriptor, data_root) {
    color_path_ = GetExtractDir() + "/SUN_color.jpg";
    depth_path_ = GetExtractDir() + "/SUN_depth.png";
}

}  // namespace data
}  // namespace cloudViewer

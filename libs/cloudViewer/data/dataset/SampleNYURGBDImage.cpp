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
        CloudViewerDownloadsPrefix() + "20220201-data/SampleNYURGBDImage.zip",
        "b0baaf892c7ff9b202eb5fb40c5f7b58"};

SampleNYURGBDImage::SampleNYURGBDImage(const std::string& data_root)
    : DownloadDataset("SampleNYURGBDImage", data_descriptor, data_root) {
    color_path_ = GetExtractDir() + "/NYU_color.ppm";
    depth_path_ = GetExtractDir() + "/NYU_depth.pgm";
}

}  // namespace data
}  // namespace cloudViewer

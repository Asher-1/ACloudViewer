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
        CloudViewerDownloadsPrefix() + "20220301-data/WoodTexture.zip",
        "28788c7ecc42d78d4d623afbab2301e9"};

WoodTexture::WoodTexture(const std::string& data_root)
    : DownloadDataset("WoodTexture", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    map_filename_to_path_ = {
            {"albedo", extract_dir + "/Wood049_Color.jpg"},
            {"normal", extract_dir + "/Wood049_NormalDX.jpg"},
            {"roughness", extract_dir + "/Wood049_Roughness.jpg"}};
}

}  // namespace data
}  // namespace cloudViewer

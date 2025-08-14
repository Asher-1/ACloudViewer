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
        CloudViewerDownloadsPrefix() + "20220301-data/TerrazzoTexture.zip",
        "8d67f191fb5d80a27d8110902cac008e"};

TerrazzoTexture::TerrazzoTexture(const std::string& data_root)
    : DownloadDataset("TerrazzoTexture", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    map_filename_to_path_ = {
            {"albedo", extract_dir + "/Terrazzo018_Color.jpg"},
            {"normal", extract_dir + "/Terrazzo018_NormalDX.jpg"},
            {"roughness", extract_dir + "/Terrazzo018_Roughness.jpg"}};
}

}  // namespace data
}  // namespace cloudViewer

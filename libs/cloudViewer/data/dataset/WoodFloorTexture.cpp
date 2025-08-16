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
        CloudViewerDownloadsPrefix() + "20220301-data/WoodFloorTexture.zip",
        "f11b3e50208095e87340049b9ac3c319"};

WoodFloorTexture::WoodFloorTexture(const std::string& data_root)
    : DownloadDataset("WoodFloorTexture", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    map_filename_to_path_ = {
            {"albedo", extract_dir + "/WoodFloor050_Color.jpg"},
            {"normal", extract_dir + "/WoodFloor050_NormalDX.jpg"},
            {"roughness", extract_dir + "/WoodFloor050_Roughness.jpg"}};
}

}  // namespace data
}  // namespace cloudViewer

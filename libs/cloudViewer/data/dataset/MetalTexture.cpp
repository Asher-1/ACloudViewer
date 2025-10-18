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
        CloudViewerDownloadsPrefix() + "20220301-data/MetalTexture.zip",
        "2b6a17e41157138868a2cd2926eedcc7"};

MetalTexture::MetalTexture(const std::string& data_root)
    : DownloadDataset("MetalTexture", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    map_filename_to_path_ = {
            {"albedo", extract_dir + "/Metal008_Color.jpg"},
            {"normal", extract_dir + "/Metal008_NormalDX.jpg"},
            {"roughness", extract_dir + "/Metal008_Roughness.jpg"},
            {"metallic", extract_dir + "/Metal008_Metalness.jpg"}};
}

}  // namespace data
}  // namespace cloudViewer

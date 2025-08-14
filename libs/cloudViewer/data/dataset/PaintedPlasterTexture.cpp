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
        CloudViewerDownloadsPrefix() + "20220301-data/PaintedPlasterTexture.zip",
        "344096b29b06f14aac58f9ad73851dc2"};

PaintedPlasterTexture::PaintedPlasterTexture(const std::string& data_root)
    : DownloadDataset("PaintedPlasterTexture", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    map_filename_to_path_ = {
            {"albedo", extract_dir + "/PaintedPlaster017_Color.jpg"},
            {"normal", extract_dir + "/PaintedPlaster017_NormalDX.jpg"},
            {"roughness", extract_dir + "/noiseTexture.png"}};
}

}  // namespace data
}  // namespace cloudViewer

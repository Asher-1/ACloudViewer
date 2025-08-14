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
        CloudViewerDownloadsPrefix() + "20220301-data/MonkeyModel.zip",
        "fc330bf4fd8e022c1e5ded76139785d4"};

MonkeyModel::MonkeyModel(const std::string& data_root)
    : DownloadDataset("MonkeyModel", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    map_filename_to_path_ = {
            {"albedo", extract_dir + "/albedo.png"},
            {"ao", extract_dir + "/ao.png"},
            {"metallic", extract_dir + "/metallic.png"},
            {"monkey_material", extract_dir + "/monkey.mtl"},
            {"monkey_model", extract_dir + "/monkey.obj"},
            {"monkey_solid_material", extract_dir + "/monkey_solid.mtl"},
            {"monkey_solid_model", extract_dir + "/monkey_solid.obj"},
            {"normal", extract_dir + "/normal.png"},
            {"roughness", extract_dir + "/roughness.png"}};
}

}  // namespace data
}  // namespace cloudViewer

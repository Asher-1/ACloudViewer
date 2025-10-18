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
        CloudViewerDownloadsPrefix() + "20220301-data/CrateModel.zip",
        "20413eada103969bb3ca5df9aebc2034"};

CrateModel::CrateModel(const std::string& data_root)
    : DownloadDataset("CrateModel", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    map_filename_to_path_ = {{"crate_material", extract_dir + "/crate.mtl"},
                             {"crate_model", extract_dir + "/crate.obj"},
                             {"texture_image", extract_dir + "/crate.jpg"}};
}

}  // namespace data
}  // namespace cloudViewer

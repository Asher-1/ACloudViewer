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
        CloudViewerDownloadsPrefix() + "20220301-data/DamagedHelmetModel.glb",
        "a3af6ad5a8329f22ba08b7f16e4a97d8"};

DamagedHelmetModel::DamagedHelmetModel(const std::string& data_root)
    : DownloadDataset("DamagedHelmetModel", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/DamagedHelmetModel.glb";
}

}  // namespace data
}  // namespace cloudViewer

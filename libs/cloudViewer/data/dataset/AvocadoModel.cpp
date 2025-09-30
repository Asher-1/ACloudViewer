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
        CloudViewerDownloadsPrefix() + "20220301-data/AvocadoModel.glb",
        "829f96a0a3a7d5556e0a263ea0699217"};

AvocadoModel::AvocadoModel(const std::string& data_root)
    : DownloadDataset("AvocadoModel", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/AvocadoModel.glb";
}

}  // namespace data
}  // namespace cloudViewer

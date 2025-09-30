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
        CloudViewerDownloadsPrefix() + "20220201-data/BunnyMesh.ply",
        "568f871d1a221ba6627569f1e6f9a3f2"};

BunnyMesh::BunnyMesh(const std::string& data_root)
    : DownloadDataset("BunnyMesh", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/BunnyMesh.ply";
}

}  // namespace data
}  // namespace cloudViewer

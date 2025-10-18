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
        CloudViewerDownloadsPrefix() + "20220201-data/KnotMesh.ply",
        "bfc9f132ecdfb7f9fdc42abf620170fc"};

KnotMesh::KnotMesh(const std::string& data_root)
    : DownloadDataset("KnotMesh", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/KnotMesh.ply";
}

}  // namespace data
}  // namespace cloudViewer

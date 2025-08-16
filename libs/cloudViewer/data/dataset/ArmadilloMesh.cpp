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
        CloudViewerDownloadsPrefix() + "20220201-data/ArmadilloMesh.ply",
        "9e68ff1b1cc914ed88cd84f6a8235021"};

ArmadilloMesh::ArmadilloMesh(const std::string& data_root)
    : DownloadDataset("ArmadilloMesh", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/" + "ArmadilloMesh.ply";
}

}  // namespace data
}  // namespace cloudViewer

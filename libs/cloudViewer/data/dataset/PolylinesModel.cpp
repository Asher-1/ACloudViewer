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

PolylinesModel::PolylinesModel(const std::string& data_root)
    : DownloadDataset("PolylinesModel",
                      {CloudViewerDownloadsPrefix() + "bin-data/polylines.bin",
                       "9fa4241bfbb11567ffba839afccfee5c"},
                      data_root) {
    path_ = GetExtractDir() + "/" + "polylines.bin";
}

}  // namespace data
}  // namespace cloudViewer

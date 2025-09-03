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

FacetsModel::FacetsModel(const std::string& data_root)
    : DownloadDataset("FacetsModel",
                      {CloudViewerDownloadsPrefix() + "bin-data/facets.bin",
                       "0771af8714b4c1d039d99934a0966001"},
                      data_root) {
    path_ = GetExtractDir() + "/" + "facets.bin";
}

}  // namespace data
}  // namespace cloudViewer

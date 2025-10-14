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

BalusterVase::BalusterVase(const std::string& data_root)
    : DownloadDataset("BalusterVase",
                      {GetCustomDownloadsPrefix() +
                               "20220301-data/F1980_baluster_vase.glb",
                       "86ada54d74685cc9bdd4e82ce25e6fd3"},
                      data_root) {
    path_ = GetExtractDir() + "/" + "F1980_baluster_vase.glb";
}

}  // namespace data
}  // namespace cloudViewer

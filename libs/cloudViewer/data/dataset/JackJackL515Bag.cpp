// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Logging.h>

#include <string>

#include "cloudViewer/data/Dataset.h"

namespace cloudViewer {
namespace data {

const static DataDescriptor data_descriptor = {
        CloudViewerDownloadsPrefix() + "20220301-data/JackJackL515Bag.bag",
        "9f670dc92569b986b739c4179a659176"};

JackJackL515Bag::JackJackL515Bag(const std::string& data_root)
    : DownloadDataset("JackJackL515Bag", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/JackJackL515Bag.bag";
}

}  // namespace data
}  // namespace cloudViewer

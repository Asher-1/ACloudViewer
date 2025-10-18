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
        CloudViewerDownloadsPrefix() + "20220301-data/SampleL515Bag.zip",
        "9770eeb194c78103037dbdbec78b9c8c"};

SampleL515Bag::SampleL515Bag(const std::string& data_root)
    : DownloadDataset("SampleL515Bag", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/L515_test_s.bag";
}

}  // namespace data
}  // namespace cloudViewer

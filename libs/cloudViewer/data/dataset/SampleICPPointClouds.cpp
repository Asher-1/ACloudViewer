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
        CloudViewerDownloadsPrefix() + "xxx/SampleICPPointClouds.zip",
        "9d1ead73e678fa2f51a70a933b0bf017"};

SampleICPPointClouds::SampleICPPointClouds(const std::string& data_root)
    : DownloadDataset("SampleICPPointClouds", data_descriptor, data_root) {}

}  // namespace data
}  // namespace cloudViewer

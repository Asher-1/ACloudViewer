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
        CloudViewerDownloadsPrefix() + "20220201-data/JuneauImage.jpg",
        "a090f6342893bdf0caefd83c6debbecd"};

JuneauImage::JuneauImage(const std::string& data_root)
    : DownloadDataset("JuneauImage", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/JuneauImage.jpg";
}

}  // namespace data
}  // namespace cloudViewer

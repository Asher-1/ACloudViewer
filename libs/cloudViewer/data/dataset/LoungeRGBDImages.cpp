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
        CloudViewerDownloadsPrefix() + "20220301-data/LoungeRGBDImages.zip",
        "cdd307caef898519a8829ce1b6ab9f75"};

LoungeRGBDImages::LoungeRGBDImages(const std::string& data_root)
    : DownloadDataset("LoungeRGBDImages", data_descriptor, data_root) {
    color_paths_.reserve(3000);
    depth_paths_.reserve(3000);
    const std::string extract_dir = GetExtractDir();
    const size_t n_zero = 6;
    for (int i = 1; i < 3000; ++i) {
        std::string idx = std::to_string(i);
        idx = std::string(n_zero - std::min(n_zero, idx.length()), '0') + idx;
        color_paths_.push_back(extract_dir + "/color/" + idx + ".png");
        depth_paths_.push_back(extract_dir + "/depth/" + idx + ".png");
    }

    trajectory_log_path_ = extract_dir + "/lounge_trajectory.log";
    reconstruction_path_ = extract_dir + "/lounge.ply";
}

}  // namespace data
}  // namespace cloudViewer

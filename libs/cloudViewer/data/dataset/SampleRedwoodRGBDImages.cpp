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
        CloudViewerDownloadsPrefix() +
                "20220301-data/SampleRedwoodRGBDImages.zip",
        "43971c5f690c9cfc52dda8c96a0140ee"};

SampleRedwoodRGBDImages::SampleRedwoodRGBDImages(const std::string& data_root)
    : DownloadDataset("SampleRedwoodRGBDImages", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();

    color_paths_ = {
            extract_dir + "/color/00000.jpg", extract_dir + "/color/00001.jpg",
            extract_dir + "/color/00002.jpg", extract_dir + "/color/00003.jpg",
            extract_dir + "/color/00004.jpg"};

    depth_paths_ = {
            extract_dir + "/depth/00000.png", extract_dir + "/depth/00001.png",
            extract_dir + "/depth/00002.png", extract_dir + "/depth/00003.png",
            extract_dir + "/depth/00004.png"};

    trajectory_log_path_ = extract_dir + "/trajectory.log";
    odometry_log_path_ = extract_dir + "/odometry.log";
    rgbd_match_path_ = extract_dir + "/rgbd.match";
    reconstruction_path_ = extract_dir + "/example_tsdf_pcd.ply";
    camera_intrinsic_path_ = extract_dir + "/camera_primesense.json";
}

}  // namespace data
}  // namespace cloudViewer

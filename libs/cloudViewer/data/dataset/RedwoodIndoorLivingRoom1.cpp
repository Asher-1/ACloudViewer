// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <string>
#include <vector>

#include "cloudViewer/data/Dataset.h"
#include <FileSystem.h>
#include <Logging.h>

namespace cloudViewer {
namespace data {

const static std::vector<DataDescriptor> data_descriptors = {
        {CloudViewerDownloadsPrefix() + "augmented-icl-nuim/livingroom.ply.zip",
         "841f32ff6294bb52d5f9574834e0925e"},
        {CloudViewerDownloadsPrefix() + "augmented-icl-nuim/livingroom1-color.zip",
         "c23481094ca3859b204d2fcbb4e435a4", "color"},
        {CloudViewerDownloadsPrefix() +
                 "augmented-icl-nuim/livingroom1-depth-clean.zip",
         "6867ddac0e7aafe828a218b385f61985", "depth"},
        {CloudViewerDownloadsPrefix() +
                 "augmented-icl-nuim/livingroom1-depth-simulated.zip",
         "2fec03a29258a29b9ffedde88fddd676", "depth_noisy"},
        {CloudViewerDownloadsPrefix() + "augmented-icl-nuim/livingroom1-traj.txt",
         "601ac4b51aba2455a90aed8aa1158c6a"},
        {CloudViewerDownloadsPrefix() + "augmented-icl-nuim/livingroom1.oni.zip",
         "fb201903f211f31ccd01886457bb004c"},
        {CloudViewerDownloadsPrefix() + "augmented-icl-nuim/dist-model.txt",
         "d8d7b6d29e754c2993a6eba4fd8d89ea"},
};

RedwoodIndoorLivingRoom1::RedwoodIndoorLivingRoom1(const std::string& data_root)
    : DownloadDataset("RedwoodIndoorLivingRoom1", data_descriptors, data_root) {
    const std::string extract_dir = GetExtractDir();
    std::vector<std::string> all_paths;

    // point_cloud_path_
    point_cloud_path_ = extract_dir + "/livingroom.ply";
    all_paths.push_back(point_cloud_path_);

    // color_paths_
    for (int i = 0; i <= 2869; i++) {
        const std::string path =
                extract_dir + "/color/" + fmt::format("{:05d}.jpg", i);
        color_paths_.push_back(path);
        all_paths.push_back(path);
    }

    // depth_paths_
    for (int i = 0; i <= 2869; i++) {
        const std::string path =
                extract_dir + "/depth/" + fmt::format("{:05d}.png", i);
        depth_paths_.push_back(path);
        all_paths.push_back(path);
    }

    // noisy_depth_paths_
    for (int i = 0; i <= 2869; i++) {
        const std::string path =
                extract_dir + "/depth_noisy/" + fmt::format("{:05d}.png", i);
        noisy_depth_paths_.push_back(path);
        all_paths.push_back(path);
    }

    // oni_path_
    oni_path_ = extract_dir + "/livingroom1.oni";
    all_paths.push_back(oni_path_);

    // trajectory_path_
    trajectory_path_ = extract_dir + "/livingroom1-traj.txt";
    all_paths.push_back(trajectory_path_);

    // noise_model_path_
    noise_model_path_ = extract_dir + "/dist-model.txt";
    all_paths.push_back(noise_model_path_);

    // Check all files exist.
    CheckPathsExist(all_paths);
}

}  // namespace data
}  // namespace cloudViewer

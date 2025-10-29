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

const static std::vector<DataDescriptor> data_descriptors = {
        {CloudViewerDownloadsPrefix() + "augmented-icl-nuim/office.ply.zip",
         "ba3640bba38f19c8f2d5e86e045eeae5"},
        {CloudViewerDownloadsPrefix() + "augmented-icl-nuim/office2-color.zip",
         "487cdfacffc8fc8247a5b21e860a3794", "color"},
        {CloudViewerDownloadsPrefix() +
                 "augmented-icl-nuim/office2-depth-clean.zip",
         "386203a7a5dae21db6d8430ae36dcc8b", "depth"},
        {CloudViewerDownloadsPrefix() +
                 "augmented-icl-nuim/office2-depth-simulated.zip",
         "13f1c5b7c4f44524fa91f7ba87e44bb5", "depth_noisy"},
        {CloudViewerDownloadsPrefix() + "augmented-icl-nuim/office2-traj.txt",
         "698f2f09da7d2ed3fcb604889e6f8479"},
        {CloudViewerDownloadsPrefix() + "augmented-icl-nuim/office2.oni.zip",
         "dcbd567442f29bd6080d74b5a384cd0d"},
        {CloudViewerDownloadsPrefix() + "augmented-icl-nuim/dist-model.txt",
         "d8d7b6d29e754c2993a6eba4fd8d89ea"},
};

RedwoodIndoorOffice2::RedwoodIndoorOffice2(const std::string& data_root)
    : DownloadDataset("RedwoodIndoorOffice2", data_descriptors, data_root) {
    const std::string extract_dir = GetExtractDir();
    std::vector<std::string> all_paths;

    // point_cloud_path_
    point_cloud_path_ = extract_dir + "/office.ply";
    all_paths.push_back(point_cloud_path_);

    // color_paths_
    for (int i = 0; i <= 2537; i++) {
        const std::string path =
                extract_dir + "/color/" + fmt::format("{:05d}.jpg", i);
        color_paths_.push_back(path);
        all_paths.push_back(path);
    }

    // depth_paths_
    for (int i = 0; i <= 2537; i++) {
        const std::string path =
                extract_dir + "/depth/" + fmt::format("{:05d}.png", i);
        depth_paths_.push_back(path);
        all_paths.push_back(path);
    }

    // noisy_depth_paths_
    for (int i = 0; i <= 2537; i++) {
        const std::string path =
                extract_dir + "/depth_noisy/" + fmt::format("{:05d}.png", i);
        noisy_depth_paths_.push_back(path);
        all_paths.push_back(path);
    }

    // oni_path_
    oni_path_ = extract_dir + "/office2.oni";
    all_paths.push_back(oni_path_);

    // trajectory_path_
    trajectory_path_ = extract_dir + "/office2-traj.txt";
    all_paths.push_back(trajectory_path_);

    // noise_model_path_
    noise_model_path_ = extract_dir + "/dist-model.txt";
    all_paths.push_back(noise_model_path_);

    // Check all files exist.
    CheckPathsExist(all_paths);
}

}  // namespace data
}  // namespace cloudViewer

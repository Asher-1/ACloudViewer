// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <CVLog.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "BagAlignment.h"
#include "RosBagWriter.h"

namespace {

std::vector<std::string> collectCameraTopics(mcalib::RosBagReader& reader) {
    std::vector<std::string> topics;
    for (const auto& topic : reader.getTopics()) {
        if (topic.find("/sensors/camera/") != std::string::npos &&
            topic.find("compressed_proto") != std::string::npos) {
            topics.push_back(topic);
        }
    }
    std::sort(topics.begin(), topics.end());
    return topics;
}

std::vector<std::string> collectPanoramicTopics(
        const std::vector<std::string>& camera_topics) {
    std::vector<std::string> topics;
    for (const auto& topic : camera_topics) {
        if (topic.find("panoramic_") != std::string::npos) {
            topics.push_back(topic);
        }
    }
    return topics;
}

std::vector<std::string> collectSvmTopics(
        const std::vector<std::string>& camera_topics) {
    std::vector<std::string> topics;
    for (const auto& topic : camera_topics) {
        if (topic.find("panoramic_") == std::string::npos) {
            topics.push_back(topic);
        }
    }
    return topics;
}

std::vector<std::string> collectCloudTopics(mcalib::RosBagReader& reader) {
    std::vector<std::string> topics;
    for (const auto& topic : reader.getTopics()) {
        if (topic.find("combined_point_cloud") != std::string::npos) {
            topics.push_back(topic);
        }
    }
    return topics;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage:\n"
                     "  mcalib_rosbag_slice <input.bag> <output.bag> "
                     "<start_sec> <end_sec>\n"
                     "  mcalib_rosbag_slice --align-3frames <input.bag> "
                     "<output.bag>\n"
                     "      Merge SVM+AVM+LiDAR aligned groups into one "
                     "remapped bag.\n";
        return 1;
    }

    bool align_mode = false;
    int argi = 1;
    if (std::string(argv[1]) == "--align-3frames") {
        align_mode = true;
        argi = 2;
    }

    if (argc < argi + 2) {
        std::cerr << "Missing input/output bag paths\n";
        return 1;
    }

    const std::string input = argv[argi];
    const std::string output = argv[argi + 1];

    if (align_mode) {
        mcalib::RosBagReader reader;
        if (!reader.open(input)) {
            CVLog::Error("[rosbag_slice] failed to open: %s", input.c_str());
            return 2;
        }

        const auto camera_topics = collectCameraTopics(reader);
        if (camera_topics.empty()) {
            CVLog::Error("[rosbag_slice] no camera topics in %s",
                         input.c_str());
            return 2;
        }

        const auto svm_topics = collectSvmTopics(camera_topics);
        const auto pan_topics = collectPanoramicTopics(camera_topics);
        if (svm_topics.size() < 7) {
            CVLog::Error("[rosbag_slice] expected 7 SVM topics, found %zu",
                         svm_topics.size());
            return 2;
        }
        if (pan_topics.size() < 4) {
            CVLog::Error(
                    "[rosbag_slice] expected 4 panoramic topics, found %zu",
                    pan_topics.size());
            return 2;
        }

        const auto cloud_topics = collectCloudTopics(reader);
        if (cloud_topics.empty()) {
            CVLog::Error("[rosbag_slice] no combined_point_cloud topic found");
            return 2;
        }

        constexpr double kOutputDurationSec = 0.6;
        mcalib::MergedBagExportOptions options;
        options.num_sync_groups = 3;
        options.output_duration_sec = kOutputDurationSec;
        options.frame_window_sec = 0.25;
        options.sync_frames_only = true;
        options.include_ancillary = false;
        if (!mcalib::exportMergedAlignedRosBag(input, output, svm_topics,
                                               pan_topics, cloud_topics,
                                               options)) {
            CVLog::Error("[rosbag_slice] merged aligned export failed");
            return 2;
        }
        std::cout << "[rosbag_slice] merged export done (svm="
                  << svm_topics.size() << " avm=" << pan_topics.size()
                  << " lidar=" << cloud_topics.size() << ")\n";
        return 0;
    }

    if (argc < argi + 4) {
        std::cerr << "Time slice requires <start_sec> <end_sec>\n";
        return 1;
    }

    const double start_sec = std::atof(argv[argi + 2]);
    const double end_sec = std::atof(argv[argi + 3]);
    if (end_sec <= start_sec) {
        std::cerr << "end_sec must be greater than start_sec\n";
        return 1;
    }

    const uint64_t start_ns = static_cast<uint64_t>(start_sec * 1e9);
    const uint64_t end_ns = static_cast<uint64_t>(end_sec * 1e9);
    if (!mcalib::filterRosBagByTime(input, output, start_ns, end_ns)) {
        CVLog::Error("[rosbag_slice] failed: %s", input.c_str());
        return 2;
    }
    return 0;
}

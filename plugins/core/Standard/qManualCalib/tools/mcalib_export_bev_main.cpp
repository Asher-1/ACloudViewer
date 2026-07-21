// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <CVLog.h>

#include <iostream>
#include <string>
#include <vector>

#include "BevBatchExport.h"
#include "CalibConfigParser.h"
#include "RosBagReader.h"

namespace {

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --config <dir> --bag <path> --export-bev <output_dir>"
              << " [--samples N]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_dir;
    std::string bag_path;
    std::string export_dir;
    int samples = 20;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_dir = argv[++i];
        } else if (arg == "--bag" && i + 1 < argc) {
            bag_path = argv[++i];
        } else if ((arg == "--export-bev" || arg == "-e") && i + 1 < argc) {
            export_dir = argv[++i];
        } else if (arg == "--samples" && i + 1 < argc) {
            samples = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (config_dir.empty() || bag_path.empty() || export_dir.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    mcalib::VehicleCalibConfig config;
    if (!mcalib::CalibConfigParser::loadCameraConfig(
                config_dir + "/cameras.cfg", config)) {
        return 2;
    }
    mcalib::CalibConfigParser::loadLidarConfig(config_dir + "/lidars.cfg",
                                               config);
    mcalib::CalibConfigParser::loadGroundConfig(config_dir + "/ground.cfg",
                                                config);

    mcalib::RosBagReader reader;
    if (!reader.open(bag_path)) return 3;

    std::vector<std::pair<std::string, std::string>> camera_topics;
    for (const auto& topic : reader.getTopics()) {
        if (topic.find("/sensors/camera/") != std::string::npos &&
            topic.find("_raw_data/compressed") != std::string::npos) {
            const auto pos = topic.rfind('/');
            if (pos == std::string::npos) continue;
            const std::string cam = topic.substr(pos + 1);
            const auto cam_pos = cam.find("_raw_data");
            if (cam_pos == std::string::npos) continue;
            camera_topics.emplace_back(topic, cam.substr(0, cam_pos));
        }
    }

    mcalib::BevBatchExportContext ctx;
    ctx.config = config;
    ctx.camera_topics = camera_topics;

    mcalib::BevBatchExportOptions options;
    options.num_samples = samples;
    return mcalib::exportBevImagesBatch(reader, ctx, export_dir, options).ok()
                   ? 0
                   : 4;
}

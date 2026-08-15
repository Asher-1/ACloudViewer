// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <CVLog.h>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>

#include "PcdExport.h"
#include "ProtoDecoder.h"
#include "RosBagReader.h"
#include "common/mcalib_tool_common.h"

namespace fs = std::filesystem;

namespace {

constexpr const char* kDefaultCloudTopic =
        "/sensors/lidar/combined_point_cloud_proto";

void exportBagPcd(const fs::path& bag_path,
                  const fs::path& output_dir,
                  const std::string& cloud_topic) {
    mcalib::RosBagReader reader;
    if (!reader.open(bag_path.string())) return;
    fs::create_directories(output_dir);

    const std::set<std::string> topics{cloud_topic};
    reader.readMessages(
            [&](const mcalib::BagMessage& msg) {
                mcalib::ProtoDecoder::PointCloud2Data cloud_data;
                if (!mcalib::ProtoDecoder::decodePointCloud2FromBag(
                            msg.data, cloud_data)) {
                    return true;
                }
                const auto points =
                        mcalib::ProtoDecoder::pointCloud2ToXYZIRT(cloud_data);
                if (points.empty()) return true;

                const uint64_t stamp_us = msg.timestamp_ns / 1000ULL;
                const uint64_t sec = stamp_us / 1000000ULL;
                const uint64_t usec = stamp_us % 1000000ULL;
                char stamp_str[32];
                std::snprintf(stamp_str, sizeof(stamp_str), "%06lu_%06lu",
                              static_cast<unsigned long>(sec),
                              static_cast<unsigned long>(usec));
                const fs::path out =
                        output_dir / (std::string(stamp_str) + ".pcd");
                mcalib::writePcdBinaryXYZIRT(out.string(), points,
                                             cloud_data.frame_id);
                return true;
            },
            topics);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: mcalib_rosbag2pcd <input_dir> [output_dir] "
                     "[cloud_topic]\n";
        return 1;
    }

    const fs::path input_dir = argv[1];
    fs::path output_dir = (argc >= 3) ? fs::path(argv[2]) : input_dir;
    const std::string cloud_topic = (argc >= 4) ? argv[3] : kDefaultCloudTopic;

    std::vector<fs::path> bags;
    mcalib::tools::collectBagFiles(input_dir, bags);
    if (bags.empty()) return 2;

    for (const auto& bag : bags) {
        const fs::path out = output_dir / (bag.stem().string() + "_pcd");
        exportBagPcd(bag, out, cloud_topic);
        CVLog::Print("[rosbag2pcd] %s -> %s", bag.string().c_str(),
                     out.string().c_str());
    }
    return 0;
}

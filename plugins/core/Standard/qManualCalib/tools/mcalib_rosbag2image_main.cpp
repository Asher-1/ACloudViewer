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
#include <map>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "ProtoDecoder.h"
#include "RosBagReader.h"
#include "common/mcalib_tool_common.h"

namespace fs = std::filesystem;

namespace {

bool isCompressedImageTopic(const std::string& topic) {
    return topic.find("/sensors/camera/") != std::string::npos &&
           (topic.find("_raw_data/compressed") != std::string::npos ||
            topic.find("/compressed_proto") != std::string::npos);
}

void exportBagImages(const fs::path& bag_path, const fs::path& output_root) {
    mcalib::RosBagReader reader;
    if (!reader.open(bag_path.string())) return;

    std::map<std::string, fs::path> topic_dirs;
    reader.readMessages([&](const mcalib::BagMessage& msg) {
        if (!isCompressedImageTopic(msg.topic)) return true;

        auto it = topic_dirs.find(msg.topic);
        if (it == topic_dirs.end()) {
            const fs::path dir = output_root / msg.topic;
            fs::create_directories(dir);
            it = topic_dirs.emplace(msg.topic, dir).first;
        }

        cv::Mat img;
        double ts_sec = 0.0;
        if (!mcalib::ProtoDecoder::decodeCompressedImageFromBag(msg.data, img,
                                                                ts_sec)) {
            return true;
        }

        const uint64_t stamp_us = msg.timestamp_ns / 1000ULL;
        const uint32_t sec = static_cast<uint32_t>(stamp_us / 1000000ULL);
        const uint32_t usec = static_cast<uint32_t>(stamp_us % 1000000ULL);
        char stamp_str[32];
        std::snprintf(stamp_str, sizeof(stamp_str), "%u_%06u", sec, usec);
        const fs::path out = it->second / (std::string(stamp_str) + ".jpg");
        cv::imwrite(out.string(), img);
        return true;
    });
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: mcalib_rosbag2image <input_dir>\n";
        return 1;
    }

    const fs::path input_dir = argv[1];
    std::vector<fs::path> bags;
    mcalib::tools::collectBagFiles(input_dir, bags);
    if (bags.empty()) {
        CVLog::Warning("[rosbag2image] no .bag in %s",
                       input_dir.string().c_str());
        return 2;
    }

    for (const auto& bag : bags) {
        const fs::path out_dir = input_dir / (bag.stem().string() + "_dir");
        fs::create_directories(out_dir);
        exportBagImages(bag, out_dir);
        CVLog::Print("[rosbag2image] %s -> %s", bag.string().c_str(),
                     out_dir.string().c_str());
    }
    return 0;
}

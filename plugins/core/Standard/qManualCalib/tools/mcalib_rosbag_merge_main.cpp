// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <CVLog.h>

#include <filesystem>
#include <iostream>
#include <vector>

#include "RosBagWriter.h"
#include "common/mcalib_tool_common.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: mcalib_rosbag_merge <input_dir> [output_bag]\n";
        return 1;
    }

    const fs::path input_dir = argv[1];
    std::vector<fs::path> bags;
    mcalib::tools::collectBagFiles(input_dir, bags);
    if (bags.size() < 2) {
        CVLog::Warning("[rosbag_merge] need at least 2 bags in %s",
                       input_dir.string().c_str());
        return 2;
    }

    const fs::path output =
            (argc >= 3) ? fs::path(argv[2]) : (input_dir / "merge.bag");

    std::vector<std::string> paths;
    paths.reserve(bags.size());
    for (const auto& bag : bags) paths.push_back(bag.string());

    if (!mcalib::mergeRosBags(paths, output.string())) return 3;

    const fs::path raw_dir = input_dir / "raw";
    fs::create_directories(raw_dir);
    for (const auto& bag : bags) {
        fs::rename(bag, raw_dir / bag.filename());
    }

    CVLog::Print("[rosbag_merge] wrote %s (%zu bags)", output.string().c_str(),
                 bags.size());
    return 0;
}

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <CVLog.h>

#include <filesystem>
#include <iostream>

#include "ExtrinsicCompare.h"
#include "common/mcalib_tool_common.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: mcalib_extrinsic_compare <model_config_dir> "
                     "<configs_dir> [flag]\n"
                  << "  flag=0 cameras.cfg (default), flag=1 lidars.cfg\n";
        return 1;
    }

    const fs::path model_dir = argv[1];
    const fs::path configs_root = argv[2];
    const int flag = (argc >= 4) ? std::stoi(argv[3]) : 0;
    const std::string cfg_name = flag ? "lidars.cfg" : "cameras.cfg";

    std::vector<fs::path> config_dirs;
    mcalib::tools::collectConfigDirs(configs_root, cfg_name, config_dirs);
    if (config_dirs.empty()) {
        CVLog::Warning("[extrinsic_compare] no %s under %s", cfg_name.c_str(),
                       configs_root.string().c_str());
        return 2;
    }

    const std::string reference = (model_dir / cfg_name).string();
    std::map<std::string, std::map<std::string, mcalib::Vector6d>> results;

    for (const auto& dir : config_dirs) {
        const std::string candidate = (dir / cfg_name).string();
        std::map<std::string, mcalib::Vector6d> delta;
        const bool ok = flag ? mcalib::ExtrinsicCompare::compareLidarConfigs(
                                       reference, candidate, delta)
                             : mcalib::ExtrinsicCompare::compareCameraConfigs(
                                       reference, candidate, delta);
        if (ok) results[candidate] = std::move(delta);
    }

    const std::string report = cfg_name + "_compare.txt";
    mcalib::ExtrinsicCompare::saveCompareReport(report, results);
    CVLog::Print("[extrinsic_compare] saved %s (%zu entries)", report.c_str(),
                 results.size());
    return 0;
}

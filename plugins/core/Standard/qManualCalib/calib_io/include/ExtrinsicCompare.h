// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <map>
#include <string>

#include "CalibTypes.h"

namespace mcalib {

class ExtrinsicCompare {
public:
    static bool compareCameraConfigs(
            const std::string& reference_cfg,
            const std::string& candidate_cfg,
            std::map<std::string, Vector6d>& delta_xyzeuler);

    static bool compareLidarConfigs(
            const std::string& reference_cfg,
            const std::string& candidate_cfg,
            std::map<std::string, Vector6d>& delta_xyzeuler);

    static bool saveCompareReport(
            const std::string& output_file,
            const std::map<std::string, std::map<std::string, Vector6d>>&
                    results);
};

}  // namespace mcalib

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ExtrinsicCompare.h"

#include <CVLog.h>

#include <fstream>
#include <iomanip>

#include "CalibConfigParser.h"

namespace mcalib {

namespace {

Vector6d extrinsicDelta(const Eigen::Isometry3d& reference,
                        const Eigen::Isometry3d& candidate) {
    const Eigen::Vector3d pos =
            (candidate.translation() - reference.translation()) * 100.0;
    const Eigen::Matrix3d rot =
            candidate.linear() * reference.linear().inverse();
    const Eigen::Vector3d euler =
            rot.eulerAngles(2, 1, 0) * (57.29577951308232);

    Vector6d delta;
    delta.segment(0, 3) = pos;
    delta.segment(3, 3) = euler;
    return delta;
}

}  // namespace

bool ExtrinsicCompare::compareCameraConfigs(
        const std::string& reference_cfg,
        const std::string& candidate_cfg,
        std::map<std::string, Vector6d>& delta_xyzeuler) {
    delta_xyzeuler.clear();

    VehicleCalibConfig ref_config;
    VehicleCalibConfig cand_config;
    if (!CalibConfigParser::loadCameraConfig(reference_cfg, ref_config) ||
        !CalibConfigParser::loadCameraConfig(candidate_cfg, cand_config)) {
        return false;
    }

    std::map<std::string, Eigen::Isometry3d> ref_ext;
    std::map<std::string, Eigen::Isometry3d> cand_ext;
    CalibConfigParser::getCameraExtrinsics(ref_config, ref_ext);
    CalibConfigParser::getCameraExtrinsics(cand_config, cand_ext);

    for (const auto& [name, ref_iso] : ref_ext) {
        auto it = cand_ext.find(name);
        if (it == cand_ext.end()) {
            CVLog::Warning("[ExtrinsicCompare] camera %s missing in candidate",
                           name.c_str());
            continue;
        }
        delta_xyzeuler[name] = extrinsicDelta(ref_iso, it->second);
    }
    return !delta_xyzeuler.empty();
}

bool ExtrinsicCompare::compareLidarConfigs(
        const std::string& reference_cfg,
        const std::string& candidate_cfg,
        std::map<std::string, Vector6d>& delta_xyzeuler) {
    delta_xyzeuler.clear();

    VehicleCalibConfig ref_config;
    VehicleCalibConfig cand_config;
    if (!CalibConfigParser::loadLidarConfig(reference_cfg, ref_config) ||
        !CalibConfigParser::loadLidarConfig(candidate_cfg, cand_config)) {
        return false;
    }

    const size_t n =
            std::min(ref_config.lidars.size(), cand_config.lidars.size());
    for (size_t i = 0; i < n; ++i) {
        const std::string name =
                "lidar_" + std::to_string(ref_config.lidars[i].lidar_idx);
        delta_xyzeuler[name] = extrinsicDelta(ref_config.lidars[i].extrinsic,
                                              cand_config.lidars[i].extrinsic);
    }
    return !delta_xyzeuler.empty();
}

bool ExtrinsicCompare::saveCompareReport(
        const std::string& output_file,
        const std::map<std::string, std::map<std::string, Vector6d>>& results) {
    std::ofstream ofs(output_file);
    if (!ofs) return false;

    ofs << "filename";
    if (!results.empty()) {
        for (const auto& [sensor, _] : results.begin()->second) {
            ofs << ',' << sensor;
        }
    }
    ofs << "\n";

    ofs << std::fixed << std::setprecision(6);
    static const char* titles[] = {"Yaw", "Pitch", "Roll", "Z", "Y", "X"};
    for (const auto& [path, sensors] : results) {
        ofs << path;
        for (int j = 5; j >= 0; --j) {
            ofs << ';' << titles[j] << ':';
            for (const auto& [sensor, delta] : sensors) {
                (void)sensor;
                ofs << delta(j) << ',';
            }
        }
        ofs << '\n';
    }
    return true;
}

}  // namespace mcalib

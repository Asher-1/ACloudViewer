// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CloudDecombine.h"

#include <CVLog.h>

#include <Eigen/Geometry>

namespace mcalib {

namespace {

void splitByRing(const std::vector<PointXYZIRT>& src,
                 int ring_start,
                 int ring_end,
                 std::vector<PointXYZIRT>& dst) {
    dst.clear();
    dst.reserve(src.size() / 4);
    for (const auto& pt : src) {
        if (pt.ring >= static_cast<uint8_t>(ring_start) &&
            pt.ring <= static_cast<uint8_t>(ring_end)) {
            dst.push_back(pt);
        }
    }
}

void transformCloud(const std::vector<PointXYZIRT>& src,
                    const Eigen::Isometry3d& transform,
                    std::vector<PointXYZIRT>& dst) {
    dst.clear();
    dst.reserve(src.size());
    const Eigen::Matrix4f T = transform.matrix().cast<float>();
    for (const auto& pt : src) {
        PointXYZIRT out = pt;
        const Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.f);
        const Eigen::Vector4f tp = T * p;
        out.x = tp.x();
        out.y = tp.y();
        out.z = tp.z();
        dst.push_back(out);
    }
}

bool decombineWithEntries(const std::vector<PointXYZIRT>& raw,
                          const std::vector<EmbeddedLidarEntry>& entries,
                          std::vector<PointXYZIRT>& cloud_out) {
    if (entries.empty()) return false;

    cloud_out.clear();
    cloud_out.reserve(raw.size());
    for (const auto& entry : entries) {
        std::vector<PointXYZIRT> split;
        splitByRing(raw, entry.ring_start, entry.ring_end, split);
        if (split.empty()) continue;

        std::vector<PointXYZIRT> transformed;
        transformCloud(split, entry.extrinsic.inverse(), transformed);
        cloud_out.insert(cloud_out.end(), transformed.begin(),
                         transformed.end());
    }
    return !cloud_out.empty();
}

bool decombineWithConfig(const std::vector<PointXYZIRT>& raw,
                         const VehicleCalibConfig& calib_config,
                         std::vector<PointXYZIRT>& cloud_out) {
    if (calib_config.lidars.empty()) {
        return false;
    }

    std::vector<EmbeddedLidarEntry> entries;
    entries.reserve(calib_config.lidars.size());
    for (const auto& lc : calib_config.lidars) {
        EmbeddedLidarEntry entry;
        entry.ring_start = lc.ring_start;
        entry.ring_end = lc.ring_end;
        entry.extrinsic = lc.extrinsic;
        entries.push_back(entry);
    }
    return decombineWithEntries(raw, entries, cloud_out);
}

}  // namespace

bool decombinePointCloud(const ProtoDecoder::PointCloud2Data& cloud_data,
                         const VehicleCalibConfig* calib_config,
                         std::vector<PointXYZIRT>& cloud_out,
                         std::string& frame_id_out,
                         int64_t& cloud_stamp_us) {
    cloud_out = ProtoDecoder::pointCloud2ToXYZIRT(cloud_data);
    frame_id_out = cloud_data.frame_id;
    cloud_stamp_us = static_cast<int64_t>(cloud_data.timestamp_sec * 1e6);
    if (cloud_data.header_stamp_us > 0) {
        cloud_stamp_us = static_cast<int64_t>(cloud_data.header_stamp_us);
    }

    if (cloud_out.empty()) {
        return false;
    }

    if (frame_id_out == "lidar_uncalibrated") {
        return true;
    }

    std::vector<PointXYZIRT> decombined;
    if (!cloud_data.embedded_lidars.empty()) {
        if (!decombineWithEntries(cloud_out, cloud_data.embedded_lidars,
                                  decombined)) {
            CVLog::Warning(
                    "[CloudDecombine] embedded lidar_configs decombine failed, "
                    "frame=%s",
                    frame_id_out.c_str());
            return false;
        }
        CVLog::Print(
                "[CloudDecombine] decombined via embedded configs: %zu -> %zu "
                "points",
                cloud_out.size(), decombined.size());
    } else if (calib_config) {
        if (!decombineWithConfig(cloud_out, *calib_config, decombined)) {
            CVLog::Warning(
                    "[CloudDecombine] calib fallback decombine failed, "
                    "frame=%s",
                    frame_id_out.c_str());
            return false;
        }
        CVLog::Print(
                "[CloudDecombine] decombined via calib fallback: %zu -> %zu "
                "points",
                cloud_out.size(), decombined.size());
    } else {
        CVLog::Warning(
                "[CloudDecombine] frame_id=%s needs decombine but no embedded "
                "configs "
                "or calib config",
                frame_id_out.c_str());
        return false;
    }

    cloud_out = std::move(decombined);
    frame_id_out = "lidar_uncalibrated";
    return true;
}

}  // namespace mcalib

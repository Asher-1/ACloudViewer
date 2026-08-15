// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PcdBatchExport.h"

#include <CVLog.h>

#include <cstdio>
#include <filesystem>
#include <sstream>

#include "BagAlignment.h"
#include "PcdExport.h"
#include "ProtoDecoder.h"
#include "RosBagReader.h"

namespace fs = std::filesystem;

namespace mcalib {

namespace {

bool exportCancelled(const PcdBatchExportOptions& options) {
    return options.progress.cancel_flag &&
           options.progress.cancel_flag->load(std::memory_order_relaxed);
}

bool reportExportStep(const PcdBatchExportOptions& options,
                      int completed,
                      int total,
                      const std::string& label) {
    if (exportCancelled(options)) {
        return false;
    }
    if (options.progress.report) {
        return options.progress.report(completed, total, label);
    }
    return true;
}

std::string findRefCameraTopic(
        const std::vector<std::pair<std::string, std::string>>& camera_topics) {
    for (const auto& [topic, name] : camera_topics) {
        if (name == kCamera1) {
            return topic;
        }
    }
    for (const auto& [topic, name] : camera_topics) {
        if (name.compare(0, 7, "camera_") == 0) {
            return topic;
        }
    }
    if (!camera_topics.empty()) {
        return camera_topics.front().first;
    }
    return {};
}

bool loadCloudAtPercent(RosBagReader& reader,
                        const PcdBatchExportContext& ctx,
                        double percent,
                        std::vector<PointXYZIRT>& cloud_raw,
                        int64_t& cloud_stamp_us,
                        std::string& frame_id) {
    cloud_raw.clear();
    cloud_stamp_us = 0;
    frame_id.clear();

    const std::string ref_topic = findRefCameraTopic(ctx.camera_topics);
    int64_t ref_stamp_ns = 0;
    int64_t ref_bag_stamp_ns = 0;
    if (!ref_topic.empty()) {
        const auto msg = reader.readMessageAtPercent(ref_topic, percent);
        if (!msg.data.empty()) {
            ref_bag_stamp_ns = static_cast<int64_t>(msg.timestamp_ns);
            double ts_sec = 0;
            if (ProtoDecoder::extractCompressedImageTimestampFromBag(msg.data,
                                                                     ts_sec)) {
                ref_stamp_ns = static_cast<int64_t>(ts_sec * 1e9);
            }
        }
    }
    if (ref_stamp_ns <= 0 && !ref_topic.empty()) {
        const auto msg = reader.readMessageAtPercent(ref_topic, percent);
        if (!msg.data.empty()) {
            ref_bag_stamp_ns = static_cast<int64_t>(msg.timestamp_ns);
            ref_stamp_ns = ref_bag_stamp_ns;
        }
    }

    return getAlignedCloud(reader, ctx.cloud_topics, ref_stamp_ns, false,
                           cloud_raw, cloud_stamp_us, &frame_id,
                           ref_bag_stamp_ns, &ctx.config) &&
           !cloud_raw.empty();
}

std::vector<Eigen::Isometry3d> buildLidarFinalExtrinsics(
        const VehicleCalibConfig& config,
        const std::vector<Vector6d>& delta_lidar_extrinsics,
        bool is_combined) {
    std::vector<Eigen::Isometry3d> extrinsic_gnss_lidar(config.lidars.size());
    for (size_t i = 0; i < config.lidars.size(); ++i) {
        Vector6d delta = Vector6d::Zero();
        if (i < delta_lidar_extrinsics.size()) {
            delta = delta_lidar_extrinsics[i];
        }
        Vector6d delta_rad = delta;
        delta_rad.segment(0, 3) *= (M_PI / 180.0);

        Eigen::Isometry3d iso_tune;
        Vec2Isometry(delta_rad, iso_tune);

        if (!is_combined) {
            extrinsic_gnss_lidar[i] = config.iso_sensing_vehicle.inverse() *
                                      iso_tune * config.lidars[i].extrinsic;
        } else {
            extrinsic_gnss_lidar[i] =
                    config.iso_sensing_vehicle.inverse() * iso_tune;
        }
    }
    return extrinsic_gnss_lidar;
}

std::vector<PointXYZIRT> processCloudForExport(
        const std::vector<PointXYZIRT>& cloud_raw,
        int64_t cloud_stamp_us,
        const std::string& frame_id,
        const PcdBatchExportContext& ctx,
        std::vector<uint32_t>& rgb_packed) {
    std::vector<PointXYZIRT> output;
    rgb_packed.clear();
    if (cloud_raw.empty()) return output;

    const bool is_combined = frame_id.empty() || frame_id == "lidar" ||
                             frame_id == "lidar_uncalibrated";

    auto passFilters = [&](const Eigen::Vector3f& pt) {
        const float dist = pt.norm();
        if (dist < ctx.dist_filter_min || dist > ctx.dist_filter) return false;
        if (pt.z() < ctx.ground_filter_min || pt.z() > ctx.ground_filter_max) {
            return false;
        }
        return true;
    };

    // Append a point and tag it with the lidar index's high-contrast color,
    // matching what displayPointCloudIn3D shows in the 3D view.
    auto appendPoint = [&](const Eigen::Vector3f& pt, uint32_t rgb) {
        if (!passFilters(pt)) return;
        PointXYZIRT out_pt;
        out_pt.x = pt.x();
        out_pt.y = pt.y();
        out_pt.z = pt.z();
        out_pt.intensity = 0;
        out_pt.ring = 0;
        out_pt.timestamp = 0;
        output.push_back(out_pt);
        rgb_packed.push_back(rgb);
    };

    if (!ctx.config.lidars.empty()) {
        const auto extrinsic_gnss_lidar = buildLidarFinalExtrinsics(
                ctx.config, ctx.delta_lidar_extrinsics, is_combined);
        for (size_t j = 0; j < ctx.config.lidars.size(); ++j) {
            if (j >= ctx.selected_lidars.size() || !ctx.selected_lidars[j]) {
                continue;
            }
            const auto& lc = ctx.config.lidars[j];
            if (j >= extrinsic_gnss_lidar.size()) continue;

            // Match the display's per-lidar color so exported RGB is
            // visually identical to the on-screen 3D view.
            uint8_t cr, cg, cb;
            highContrastLidarColor(static_cast<int>(j), cr, cg, cb);
            const uint32_t rgb = (static_cast<uint32_t>(cr) << 16) |
                                 (static_cast<uint32_t>(cg) << 8) |
                                 static_cast<uint32_t>(cb);

            const Eigen::Matrix4f T =
                    extrinsic_gnss_lidar[j].matrix().cast<float>();
            std::vector<PointXYZIRT> single_raw;
            for (const auto& pt : cloud_raw) {
                if (pt.ring < lc.ring_start || pt.ring > lc.ring_end) continue;
                PointXYZIRT tpt = pt;
                const Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.0f);
                const Eigen::Vector4f tp = T * p;
                tpt.x = tp.x();
                tpt.y = tp.y();
                tpt.z = tp.z();
                single_raw.push_back(tpt);
            }

            std::vector<Eigen::Vector3f> undistorted;
            bool did_undistort = false;
            if (!ctx.vehicle_poses.empty() && !single_raw.empty()) {
                did_undistort = undistortPointCloud(
                        single_raw, cloud_stamp_us, cloud_stamp_us,
                        ctx.vehicle_poses, Eigen::Isometry3d::Identity(),
                        undistorted);
            }

            if (did_undistort) {
                std::vector<Eigen::Vector3f> filtered_ring;
                filterGroundReflectionPoints(undistorted, filtered_ring,
                                             ctx.ground_filter_min,
                                             ctx.ground_filter_max);
                for (const auto& pt : filtered_ring) {
                    appendPoint(pt, rgb);
                }
            } else {
                std::vector<Eigen::Vector3f> raw_xyz;
                raw_xyz.reserve(single_raw.size());
                for (const auto& pt : single_raw) {
                    raw_xyz.emplace_back(pt.x, pt.y, pt.z);
                }
                std::vector<Eigen::Vector3f> filtered_ring;
                filterGroundReflectionPoints(raw_xyz, filtered_ring,
                                             ctx.ground_filter_min,
                                             ctx.ground_filter_max);
                for (const auto& pt : filtered_ring) {
                    appendPoint(pt, rgb);
                }
            }
        }
        if (!output.empty()) {
            return output;
        }
    }

    // Fallback: no lidar config — use white so the user can see the points.
    const uint32_t white_rgb = 0xFFFFFFu;
    output.reserve(cloud_raw.size());
    rgb_packed.reserve(cloud_raw.size());
    for (const auto& pt : cloud_raw) {
        appendPoint(Eigen::Vector3f(pt.x, pt.y, pt.z), white_rgb);
    }
    return output;
}

std::string makePcdFilename(int64_t cloud_stamp_us, double percent) {
    const uint64_t stamp_us =
            static_cast<uint64_t>(std::max<int64_t>(0, cloud_stamp_us));
    const uint64_t sec = stamp_us / 1000000ULL;
    const uint64_t usec = stamp_us % 1000000ULL;
    char filename[128];
    std::snprintf(filename, sizeof(filename), "pcd_%06lu_%06lu_p%03d.pcd",
                  static_cast<unsigned long>(sec),
                  static_cast<unsigned long>(usec),
                  static_cast<int>(percent * 100.0));
    return filename;
}

}  // namespace

BatchExportResult exportPcdsBatch(RosBagReader& reader,
                                  const PcdBatchExportContext& ctx,
                                  const std::string& output_dir,
                                  const PcdBatchExportOptions& options) {
    BatchExportResult result;
    result.total = options.num_samples;
    if (!reader.isOpen()) return result;
    fs::create_directories(output_dir);

    if (!reportExportStep(options, 0, options.num_samples,
                          "PCD export starting")) {
        result.cancelled = true;
        return result;
    }

    int exported = 0;
    for (int i = 0; i < options.num_samples; ++i) {
        if (exportCancelled(options)) {
            result.cancelled = true;
            break;
        }

        const double percent = (static_cast<double>(i) + 0.5) /
                               static_cast<double>(options.num_samples);
        std::ostringstream label;
        label << "PCD frame " << (i + 1) << "/" << options.num_samples << " ("
              << static_cast<int>(percent * 100.0) << "%)";
        if (!reportExportStep(options, i, options.num_samples, label.str())) {
            result.cancelled = true;
            break;
        }

        std::vector<PointXYZIRT> cloud_raw;
        int64_t cloud_stamp_us = 0;
        std::string frame_id;
        if (!loadCloudAtPercent(reader, ctx, percent, cloud_raw, cloud_stamp_us,
                                frame_id)) {
            continue;
        }

        std::vector<uint32_t> rgb_packed;
        const auto points = processCloudForExport(cloud_raw, cloud_stamp_us,
                                                  frame_id, ctx, rgb_packed);
        if (points.empty()) continue;

        const fs::path out_path =
                fs::path(output_dir) / makePcdFilename(cloud_stamp_us, percent);
        if (!writePcdBinaryXYZIRGB(out_path.string(), points, rgb_packed,
                                   frame_id)) {
            continue;
        }
        ++exported;

        if (!reportExportStep(options, i + 1, options.num_samples,
                              label.str())) {
            result.cancelled = true;
            break;
        }
    }

    result.exported = exported;
    CVLog::Print("[PcdBatchExport] exported %d/%d PCD files to %s%s", exported,
                 options.num_samples, output_dir.c_str(),
                 result.cancelled ? " (cancelled)" : "");
    return result;
}

}  // namespace mcalib

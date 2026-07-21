// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "BevBatchExport.h"

#include <CVLog.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <sstream>

#include "BagAlignment.h"
#include "BirdEyeView.h"
#include "CalibConfigParser.h"
#include "CalibTypes.h"
#include "RosBagReader.h"

namespace fs = std::filesystem;

namespace mcalib {

namespace {

std::map<std::string, Eigen::Isometry3d> buildFinalExtrinsics(
        const VehicleCalibConfig& config,
        const std::map<std::string, Vector6d>& delta_extrinsics) {
    std::map<std::string, Eigen::Isometry3d> result;
    for (const auto& [name, cam] : config.cameras) {
        const Eigen::Isometry3d& iso_sensing_cam = cam.extrinsic;
        auto it = delta_extrinsics.find(name);
        if (it != delta_extrinsics.end()) {
            Vector6d tune_rad = it->second;
            tune_rad.segment(0, 3) *= (M_PI / 180.0);
            Eigen::Isometry3d iso_tune;
            Vec2Isometry(tune_rad, iso_tune);
            Eigen::Isometry3d iso_result;
            iso_result.linear() = iso_tune.linear() * iso_sensing_cam.linear();
            iso_result.translation() =
                    iso_sensing_cam.translation() + iso_tune.translation();
            result[name] = iso_result;
        } else {
            result[name] = iso_sensing_cam;
        }
    }
    return result;
}

void updateConfigExtrinsics(
        VehicleCalibConfig& config,
        const std::map<std::string, Eigen::Isometry3d>& extrinsics) {
    for (const auto& [name, ext] : extrinsics) {
        CalibConfigParser::updateCameraExtrinsic(config, name, ext);
    }
}

std::vector<std::pair<std::string, std::string>> filterTopicsForMap(
        const std::vector<std::pair<std::string, std::string>>& camera_topics,
        const std::map<std::string, std::string>& camera_map) {
    std::set<std::string> needed;
    for (const auto& [_, source] : camera_map) {
        needed.insert(source);
    }

    std::vector<std::pair<std::string, std::string>> out;
    for (const auto& item : camera_topics) {
        if (needed.count(item.second) > 0) {
            out.push_back(item);
        }
    }
    return out;
}

std::map<std::string, cv::Mat> topicsToCameraImages(
        const std::vector<std::pair<std::string, std::string>>& topic_pairs,
        const std::map<std::string, cv::Mat>& images_by_topic) {
    std::map<std::string, cv::Mat> images;
    for (const auto& [topic, cam_name] : topic_pairs) {
        auto it = images_by_topic.find(topic);
        if (it != images_by_topic.end() && !it->second.empty()) {
            images[cam_name] = it->second;
        }
    }
    return images;
}

}  // namespace

namespace {

bool exportCancelled(const BevBatchExportOptions& options) {
    return options.progress.cancel_flag &&
           options.progress.cancel_flag->load(std::memory_order_relaxed);
}

bool reportExportStep(const BevBatchExportOptions& options,
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

}  // namespace

BatchExportResult exportBevImagesBatch(RosBagReader& reader,
                                       const BevBatchExportContext& ctx,
                                       const std::string& output_dir,
                                       const BevBatchExportOptions& options) {
    BatchExportResult result;
    if (!reader.isOpen()) {
        CVLog::Error("[BevExport] bag reader is not open");
        return result;
    }

    const std::string svm_dir = output_dir + "/svm";
    const std::string avm_dir = output_dir + "/avm";
    fs::create_directories(svm_dir);
    fs::create_directories(avm_dir);

    struct BevCameraSet {
        std::string name;
        std::string dir;
        std::map<std::string, std::string> camera_map;
        cv::Size bev_size;
        double focal_scale = 1.0;
    };

    std::vector<BevCameraSet> bev_sets;
    bool has_svm = false;
    bool has_avm = false;
    bool has_traffic2 = false;
    for (const auto& [name, _] : ctx.config.cameras) {
        if (name == kTraffic2) has_traffic2 = true;
        if (name.compare(0, 7, "camera_") == 0 ||
            name.compare(0, 3, "tra") == 0) {
            has_svm = true;
        }
        if (name.compare(0, 3, "pan") == 0) has_avm = true;
    }

    constexpr double kExportBevSvmFocalScale = 1.175;
    constexpr double kExportBevAvmFocalScale = 0.44;

    if (has_svm) {
        const auto& svm_map = has_traffic2 ? kSvmT2CameraMap : kSvmCameraMap;
        bev_sets.push_back({"SVM", svm_dir, svm_map, cv::Size(960, 1600),
                            kExportBevSvmFocalScale});
    }
    if (has_avm) {
        bev_sets.push_back({"AVM", avm_dir, kAvmCameraMap, cv::Size(640, 960),
                            kExportBevAvmFocalScale});
    }
    if (bev_sets.empty()) {
        CVLog::Error("[BevExport] no SVM/AVM cameras in config");
        return result;
    }

    result.total = static_cast<int>(bev_sets.size()) * options.num_samples;

    VehicleCalibConfig working_config = ctx.config;
    const auto final_extrinsics =
            buildFinalExtrinsics(ctx.config, ctx.delta_extrinsics);
    updateConfigExtrinsics(working_config, final_extrinsics);

    int step = 0;
    int exported_total = 0;

    if (!reportExportStep(options, 0, result.total,
                          "BEV batch export starting")) {
        result.cancelled = true;
        return result;
    }

    for (auto& bev_set : bev_sets) {
        const auto align_topics =
                filterTopicsForMap(ctx.camera_topics, bev_set.camera_map);
        if (align_topics.empty()) {
            CVLog::Warning("[BevExport] %s: no matching camera topics",
                           bev_set.name.c_str());
            continue;
        }

        BirdEyeView::Config bev_cfg;
        bev_cfg.bev_size = bev_set.bev_size;
        bev_cfg.focal_scale = bev_set.focal_scale;
        bev_cfg.use_parallel_remap = options.use_parallel_remap;

        BirdEyeView viewer;
        if (!viewer.init(working_config, bev_cfg)) {
            CVLog::Warning("[BevExport] %s: BEV init failed",
                           bev_set.name.c_str());
            continue;
        }
        if (!viewer.setCameraSlotMap(bev_set.camera_map)) {
            CVLog::Warning("[BevExport] %s: slot map rejected",
                           bev_set.name.c_str());
            continue;
        }
        viewer.updateExtrinsics(final_extrinsics);

        int exported = 0;
        std::vector<std::string> topic_names;
        topic_names.reserve(align_topics.size());
        for (const auto& [topic, _] : align_topics) {
            topic_names.push_back(topic);
        }

        for (int i = 0; i < options.num_samples; ++i) {
            if (exportCancelled(options)) {
                result.cancelled = true;
                break;
            }
            const double percent = (static_cast<double>(i) + 0.5) /
                                   static_cast<double>(options.num_samples);
            int topic_idx_ratio = static_cast<int>(percent * 100.0);
            if (topic_idx_ratio > 99) {
                topic_idx_ratio = 99;
            }

            std::ostringstream label;
            label << bev_set.name << " frame " << (i + 1) << "/"
                  << options.num_samples << " ("
                  << static_cast<int>(percent * 100.0) << "%)";
            if (!reportExportStep(options, step, result.total, label.str())) {
                result.cancelled = true;
                break;
            }

            std::map<std::string, cv::Mat> images_by_topic;
            std::map<std::string, int64_t> stamps_ns;
            int64_t cloud_stamp_ns = 0;
            VehicleStateData vehicle_state;
            if (!getAlignedImagesForBev(reader, topic_names, percent,
                                        images_by_topic, stamps_ns,
                                        cloud_stamp_ns, &vehicle_state)) {
                CVLog::Warning(
                        "[BevExport] %s sample %d: sync failed at %.2f%%",
                        bev_set.name.c_str(), i, percent * 100.0);
                continue;
            }

            const auto images =
                    topicsToCameraImages(align_topics, images_by_topic);
            if (images.empty()) continue;

            cv::Mat bev = viewer.generate(images);
            if (bev.empty()) continue;
            cv::rotate(bev, bev, cv::ROTATE_90_CLOCKWISE);

            drawVehicleStateOverlay(bev, vehicle_state, cv::Point(20, 20), 0.5,
                                    cv::Scalar(0, 0, 255), 1);

            char filename[512];
            std::snprintf(filename, sizeof(filename),
                          "%s/bev_seg%02d_p%03d.jpg", bev_set.dir.c_str(), i,
                          topic_idx_ratio);
            if (cv::imwrite(filename, bev)) {
                ++exported;
                ++exported_total;
            }
            ++step;
            if (!reportExportStep(options, step, result.total, label.str())) {
                result.cancelled = true;
                break;
            }
        }
        CVLog::Print("[BevExport] %s: exported %d/%d images to %s",
                     bev_set.name.c_str(), exported, options.num_samples,
                     bev_set.dir.c_str());
        if (result.cancelled) {
            break;
        }
    }

    result.exported = exported_total;
    if (result.cancelled) {
        CVLog::Print("[BevExport] cancelled after %d/%d images", exported_total,
                     result.total);
    }
    return result;
}

}  // namespace mcalib

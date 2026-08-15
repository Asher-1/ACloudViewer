// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ImageBatchExport.h"

#include <CVLog.h>

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <future>
#include <map>
#include <mutex>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <thread>
#include <vector>

#include "BagAlignment.h"
#include "BirdEyeView.h"
#include "CalibConfigParser.h"
#include "LidarProjBackend.h"
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

void drawProjectedCloud(cv::Mat& img,
                        const std::vector<Eigen::Vector3f>& cloud,
                        const Eigen::Isometry3d& T_cam_sensing,
                        const CameraIntrinsic& intr,
                        BevRemapMode mode,
                        int radius) {
    if (cloud.empty() || img.empty()) return;

    LidarProjResult proj;
    const Eigen::Matrix3d rot = T_cam_sensing.linear();
    const Eigen::Vector3d trans = T_cam_sensing.translation();
    bool projected = false;
    if (intr.model_type == CameraIntrinsic::PINHOLE) {
        projected = LidarProjBackend::projectPoints(mode, cloud, rot, trans,
                                                    intr.fx, intr.fy, intr.cx,
                                                    intr.cy, proj);
    } else if (intr.model_type == CameraIntrinsic::KANNALA_BRANDT) {
        const KannalaBrandtCoeffs kb{intr.k1, intr.k2, intr.k3, intr.k4};
        projected = LidarProjBackend::projectPointsKb(mode, cloud, rot, trans,
                                                      intr.fx, intr.fy, intr.cx,
                                                      intr.cy, kb, proj);
    }
    if (!projected) {
        return;
    }

    for (size_t i = 0; i < proj.image_points.size(); ++i) {
        const cv::Scalar color = colorFromDepth(proj.depths[i], 2.0f);
        if (radius > 0) {
            cv::circle(img, proj.image_points[i], radius, color, -1);
        } else {
            const int row = static_cast<int>(proj.image_points[i].y + 0.5f);
            const int col = static_cast<int>(proj.image_points[i].x + 0.5f);
            if (row >= 0 && row < img.rows && col >= 0 && col < img.cols) {
                img.at<cv::Vec3b>(row, col) =
                        cv::Vec3b(static_cast<uchar>(color[0]),
                                  static_cast<uchar>(color[1]),
                                  static_cast<uchar>(color[2]));
            }
        }
    }
}

}  // namespace

namespace {

bool exportCancelled(const ImageBatchExportOptions& options) {
    return options.progress.cancel_flag &&
           options.progress.cancel_flag->load(std::memory_order_relaxed);
}

bool reportExportStep(const ImageBatchExportOptions& options,
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

BatchExportResult exportImagesBatch(RosBagReader& reader,
                                    const ImageBatchExportContext& ctx,
                                    const std::string& output_dir,
                                    const ImageBatchExportOptions& options) {
    BatchExportResult result;
    result.total = options.num_samples;
    if (!reader.isOpen()) return result;
    fs::create_directories(output_dir);

    const auto final_extrinsics =
            buildFinalExtrinsics(ctx.config, ctx.delta_extrinsics);
    int exported = 0;

    if (options.view_mode == 0) {
        std::vector<std::string> topic_names;
        std::vector<std::pair<std::string, std::string>> align_topics;
        for (const auto& [topic, name] : ctx.camera_topics) {
            bool include = false;
            for (const auto& [_, source] : ctx.bev_slot_map) {
                if (source == name) {
                    include = true;
                    break;
                }
            }
            if (include) {
                align_topics.emplace_back(topic, name);
                topic_names.push_back(topic);
            }
        }
        if (topic_names.empty()) return result;

        VehicleCalibConfig working_config = ctx.config;
        for (const auto& [name, ext] : final_extrinsics) {
            CalibConfigParser::updateCameraExtrinsic(working_config, name, ext);
        }

        BirdEyeView::Config bev_cfg;
        bev_cfg.remap_mode = options.remap_mode;
        bev_cfg.focal_scale = 0.3 * (options.point_size + 1);

        // Two-phase pipeline for throughput:
        //   Phase 1 (serial): decode all camera frames in percent order so the
        //   VideoDecodeCache stays warm (each topic only decodes the delta
        //   from the previous frame, ~50ms vs ~1500ms cold).
        //   Phase 2 (parallel): render BEV + imwrite with N worker threads,
        //   each with its own BirdEyeView (CUDA remap maps are not thread-safe
        //   to share, but cheap to init per-thread).
        const int num_threads = std::max(
                1, std::min<int>(static_cast<int>(
                                         std::thread::hardware_concurrency()),
                                 4));
        std::vector<std::unique_ptr<BirdEyeView>> viewers(num_threads);
        for (auto& v : viewers) {
            v = std::make_unique<BirdEyeView>();
            if (!v->init(working_config, bev_cfg)) {
                CVLog::Warning("[ImageBatchExport] BEV viewer init failed");
                return result;
            }
            v->setCameraSlotMap(ctx.bev_slot_map);
            v->updateExtrinsics(final_extrinsics);
        }

        if (!reportExportStep(options, 0, options.num_samples,
                              "BEV export starting")) {
            result.cancelled = true;
            return result;
        }

        struct DecodedFrame {
            int index;
            double percent;
            std::map<std::string, cv::Mat> images;
            VehicleStateData vehicle_state;
            bool valid = false;
        };

        std::vector<DecodedFrame> frames(options.num_samples);
        for (int i = 0; i < options.num_samples; ++i) {
            if (exportCancelled(options)) {
                result.cancelled = true;
                break;
            }
            const double percent = (static_cast<double>(i) + 0.5) /
                                   static_cast<double>(options.num_samples);
            frames[i].index = i;
            frames[i].percent = percent;

            std::map<std::string, cv::Mat> images_by_topic;
            std::map<std::string, int64_t> stamps_ns;
            int64_t cloud_stamp_ns = 0;
            if (!getAlignedImagesForBev(
                        reader, topic_names, percent, images_by_topic,
                        stamps_ns, cloud_stamp_ns, &frames[i].vehicle_state)) {
                continue;
            }
            frames[i].images =
                    topicsToCameraImages(align_topics, images_by_topic);
            frames[i].valid = !frames[i].images.empty();

            if (!reportExportStep(options, i + 1, options.num_samples,
                                  "BEV decode " + std::to_string(i + 1))) {
                result.cancelled = true;
                break;
            }
        }

        // Phase 2: render + write in parallel.
        std::atomic<int> exported_atomic{0};
        std::atomic<int> next_frame{0};
        std::atomic<bool> cancelled{result.cancelled};

        auto worker = [&](int tid) {
            BirdEyeView& viewer = *viewers[tid];
            while (true) {
                if (cancelled.load(std::memory_order_relaxed)) return;
                const int idx =
                        next_frame.fetch_add(1, std::memory_order_relaxed);
                if (idx >= options.num_samples) return;
                if (!frames[idx].valid) continue;

                cv::Mat bev = viewer.generate(frames[idx].images);
                if (bev.empty()) continue;

                drawVehicleStateOverlay(bev, frames[idx].vehicle_state,
                                        cv::Point(20, 20), 0.5,
                                        cv::Scalar(0, 0, 255), 1);
                const int hint_y = std::max(20, bev.rows - 12);
                cv::putText(bev, "BEV Mode (L:measure R:clear)",
                            cv::Point(10, hint_y), cv::FONT_HERSHEY_SIMPLEX,
                            0.5, cv::Scalar(0, 255, 0), 1);

                char filename[512];
                std::snprintf(filename, sizeof(filename), "%s/bev_p%03d.jpg",
                              output_dir.c_str(),
                              static_cast<int>(frames[idx].percent * 100.0));
                if (cv::imwrite(filename, bev)) {
                    exported_atomic.fetch_add(1, std::memory_order_relaxed);
                }

                if (options.progress.report) {
                    const int done = idx + 1;
                    std::ostringstream label;
                    label << "BEV frame " << done << "/" << options.num_samples;
                    if (!options.progress.report(done, options.num_samples,
                                                 label.str())) {
                        cancelled.store(true, std::memory_order_relaxed);
                        return;
                    }
                }
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(worker, t);
        }
        for (auto& t : threads) t.join();

        exported = exported_atomic.load();
        result.cancelled = cancelled.load();
        CVLog::Print(
                "[ImageBatchExport] BEV exported %d/%d images (%d threads)",
                exported, options.num_samples, num_threads);
        return result;
    } else if (options.view_mode == 1) {
        const std::string cam_name = ctx.projection_camera;
        auto it_cam = ctx.config.cameras.find(cam_name);
        if (it_cam == ctx.config.cameras.end()) return result;

        std::string cam_topic;
        for (const auto& [topic, name] : ctx.camera_topics) {
            if (name == cam_name) {
                cam_topic = topic;
                break;
            }
        }
        if (cam_topic.empty()) return result;

        auto it_ext = final_extrinsics.find(cam_name);
        if (it_ext == final_extrinsics.end()) return result;

        if (!reportExportStep(options, 0, options.num_samples,
                              "Projection export: " + cam_name)) {
            result.cancelled = true;
            return result;
        }

        // Two-phase: serial decode (cache-friendly) + parallel render/write.
        struct ProjFrame {
            int index;
            double percent;
            cv::Mat image;
            std::vector<Eigen::Vector3f> cloud;
            bool valid = false;
        };

        std::vector<ProjFrame> frames(options.num_samples);
        for (int i = 0; i < options.num_samples; ++i) {
            if (exportCancelled(options)) {
                result.cancelled = true;
                break;
            }
            const double percent = (static_cast<double>(i) + 0.5) /
                                   static_cast<double>(options.num_samples);
            frames[i].index = i;
            frames[i].percent = percent;

            std::map<std::string, cv::Mat> images_by_topic;
            std::map<std::string, int64_t> stamps_ns;
            std::vector<PointXYZIRT> cloud_raw;
            int64_t cloud_stamp_us = 0;
            std::string frame_id;
            if (!getAlignedImagesCloud(reader, {cam_topic}, ctx.cloud_topics,
                                       percent, false, images_by_topic,
                                       stamps_ns, cloud_raw, cloud_stamp_us,
                                       &frame_id, nullptr, &ctx.config)) {
                continue;
            }

            auto it_img = images_by_topic.find(cam_topic);
            if (it_img == images_by_topic.end() || it_img->second.empty())
                continue;

            const auto& intr = it_cam->second.intrinsic;
            cv::Mat display = it_img->second.clone();
            if (intr.width > 0 && intr.height > 0 &&
                (display.cols != intr.width || display.rows != intr.height)) {
                cv::resize(display, display, cv::Size(intr.width, intr.height));
            }
            frames[i].image = display;
            frames[i].cloud.reserve(cloud_raw.size());
            for (const auto& pt : cloud_raw) {
                frames[i].cloud.emplace_back(pt.x, pt.y, pt.z);
            }
            frames[i].valid = true;

            if (!reportExportStep(options, i + 1, options.num_samples,
                                  "Proj decode " + std::to_string(i + 1))) {
                result.cancelled = true;
                break;
            }
        }

        const int num_threads = std::max(
                1, std::min<int>(static_cast<int>(
                                         std::thread::hardware_concurrency()),
                                 4));
        std::atomic<int> exported_atomic{0};
        std::atomic<int> next_frame{0};
        std::atomic<bool> cancelled{result.cancelled};

        auto worker = [&]() {
            while (true) {
                if (cancelled.load(std::memory_order_relaxed)) return;
                const int idx =
                        next_frame.fetch_add(1, std::memory_order_relaxed);
                if (idx >= options.num_samples) return;
                if (!frames[idx].valid) continue;

                cv::Mat display = frames[idx].image.clone();
                drawProjectedCloud(display, frames[idx].cloud,
                                   it_ext->second.inverse(),
                                   it_cam->second.intrinsic, options.remap_mode,
                                   options.point_size - 1);

                char filename[512];
                std::snprintf(filename, sizeof(filename),
                              "%s/proj_%s_p%03d.jpg", output_dir.c_str(),
                              cam_name.c_str(),
                              static_cast<int>(frames[idx].percent * 100.0));
                if (cv::imwrite(filename, display)) {
                    exported_atomic.fetch_add(1, std::memory_order_relaxed);
                }

                if (options.progress.report) {
                    const int done = idx + 1;
                    std::ostringstream label;
                    label << cam_name << " frame " << done << "/"
                          << options.num_samples;
                    if (!options.progress.report(done, options.num_samples,
                                                 label.str())) {
                        cancelled.store(true, std::memory_order_relaxed);
                        return;
                    }
                }
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(worker);
        }
        for (auto& t : threads) t.join();

        exported = exported_atomic.load();
        result.cancelled = cancelled.load();
        CVLog::Print(
                "[ImageBatchExport] Projection exported %d/%d images (%d "
                "threads)",
                exported, options.num_samples, num_threads);
        return result;
    }

    result.exported = exported;
    CVLog::Print("[ImageBatchExport] exported %d/%d images to %s%s", exported,
                 options.num_samples, output_dir.c_str(),
                 result.cancelled ? " (cancelled)" : "");
    return result;
}

}  // namespace mcalib

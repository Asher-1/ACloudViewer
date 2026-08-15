// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <map>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "CalibTypes.h"

namespace mcalib {

class RosBagReader;

/// Detected camera payload encoding for a bag session (JPEG vs H.264/HEVC).
enum class BagImageEncoding {
    Unknown,
    Jpeg,
    H264,
    Hevc,
    Mixed,
};

bool bagUsesVideoCodec(BagImageEncoding encoding);

/// Sample camera topics to classify JPEG / H.264 / HEVC / mixed sessions.
BagImageEncoding probeBagImageEncoding(
        RosBagReader& reader, const std::vector<std::string>& camera_topics);

constexpr int64_t kImageSyncThresholdNs = 25000000;   // 25ms
constexpr int64_t kCloudSyncThresholdNs = 100000000;  // 100ms
constexpr int64_t kAlignSearchRangeNs = 1000000000;   // 1s

/// Split camera topics into SVM (camera_*, traffic_*) vs AVM (panoramic_*).
void partitionCameraImageTopics(const std::vector<std::string>& image_topics,
                                std::vector<std::string>& svm_out,
                                std::vector<std::string>& avm_out);

/// Multi-camera image sync within 25ms
/// (codetree/repo/calibration/.../manual_sensor_calib.cpp:get_aligned_images).
bool getAlignedImages(RosBagReader& reader,
                      const std::vector<std::string>& image_topics,
                      double percent,
                      std::map<std::string, cv::Mat>& images_by_topic,
                      std::map<std::string, int64_t>& stamps_ns_by_topic,
                      VehicleStateData* vehicle_state_out = nullptr);

/// Find nearest point cloud to ref_stamp within 100ms over full bag range.
/// When ref_bag_stamp_ns > 0, bag-record time is used for indexed lookup
/// (required for remapped merge bags where bag_ts != proto).
bool getAlignedCloud(RosBagReader& reader,
                     const std::vector<std::string>& cloud_topics,
                     int64_t ref_stamp_ns,
                     bool allow_cloud_as_ref,
                     std::vector<PointXYZIRT>& cloud_raw,
                     int64_t& cloud_stamp_us,
                     std::string* frame_id_out = nullptr,
                     int64_t ref_bag_stamp_ns = 0,
                     const VehicleCalibConfig* calib_config = nullptr);

/// Load cloud nearest to image stamp within ±1s search window (100ms
/// threshold).
bool loadCloudNearImageStamp(RosBagReader& reader,
                             const std::vector<std::string>& cloud_topics,
                             int64_t ref_stamp_ns,
                             std::vector<PointXYZIRT>& cloud_raw,
                             int64_t& cloud_stamp_us,
                             std::string* frame_id_out = nullptr,
                             int64_t ref_bag_stamp_ns = 0,
                             const VehicleCalibConfig* calib_config = nullptr);

/// Images + cloud alignment (non-BEV modes).
bool getAlignedImagesCloud(RosBagReader& reader,
                           const std::vector<std::string>& image_topics,
                           const std::vector<std::string>& cloud_topics,
                           double percent,
                           bool allow_cloud_as_ref,
                           std::map<std::string, cv::Mat>& images_by_topic,
                           std::map<std::string, int64_t>& stamps_ns_by_topic,
                           std::vector<PointXYZIRT>& cloud_raw,
                           int64_t& cloud_stamp_us,
                           std::string* frame_id_out = nullptr,
                           VehicleStateData* vehicle_state_out = nullptr,
                           const VehicleCalibConfig* calib_config = nullptr);

/// BEV mode: aligned images only, cloud stamp set to reference image stamp.
bool getAlignedImagesForBev(RosBagReader& reader,
                            const std::vector<std::string>& image_topics,
                            double percent,
                            std::map<std::string, cv::Mat>& images_by_topic,
                            std::map<std::string, int64_t>& stamps_ns_by_topic,
                            int64_t& cloud_stamp_ns,
                            VehicleStateData* vehicle_state_out = nullptr);

/// Scan [search_start, search_end] for a bag percent where all image_topics
/// align.
bool findBestAlignedPercent(RosBagReader& reader,
                            const std::vector<std::string>& image_topics,
                            double search_start_percent,
                            double search_end_percent,
                            double& out_percent,
                            int64_t& out_ref_stamp_ns,
                            double search_step = 0.005);

/// Scan for a bag percent where SVM and AVM groups each fully align.
bool findBestAlignedPercentBevGroups(
        RosBagReader& reader,
        const std::vector<std::string>& svm_image_topics,
        const std::vector<std::string>& avm_image_topics,
        double search_start_percent,
        double search_end_percent,
        double& out_percent,
        int64_t& out_ref_stamp_ns,
        double search_step = 0.005);

/// Compute bag-record time window that contains aligned SVM/AVM payloads.
bool computeBevGroupSliceWindowNs(
        RosBagReader& reader,
        const std::vector<std::string>& svm_image_topics,
        const std::vector<std::string>& avm_image_topics,
        double sync_percent,
        uint64_t pad_ns,
        uint64_t& out_start_ns,
        uint64_t& out_end_ns);

/// Export merged calibration bag: SVM/AVM/LiDAR groups aligned independently
/// in the source bag, then bag-record timestamps remapped to a continuous
/// window.
struct MergedBagExportOptions {
    int num_sync_groups = 1;
    double output_duration_sec = 0.6;
    double frame_window_sec = 0.25;
    /// When true, export only the aligned sync frame per camera (smaller bags).
    bool sync_frames_only = false;
    /// When false, skip copying non-camera/cloud messages in the selected span.
    bool include_ancillary = true;
    /// Search centers in [0,1] for each sync group; auto-spaced if empty.
    std::vector<double> source_centers;
};

bool exportMergedAlignedRosBag(
        const std::string& input_bag,
        const std::string& output_bag,
        const std::vector<std::string>& svm_image_topics,
        const std::vector<std::string>& avm_image_topics,
        const std::vector<std::string>& cloud_topics,
        const MergedBagExportOptions& options = MergedBagExportOptions());

bool exportMergedAlignedRosBag(const std::string& input_bag,
                               const std::string& output_bag,
                               const std::vector<std::string>& svm_image_topics,
                               const std::vector<std::string>& avm_image_topics,
                               const std::vector<std::string>& cloud_topics,
                               double output_duration_sec,
                               double frame_window_sec);

/// Legacy alias used by older call sites.
bool exportBevAlignedRosBag(const std::string& input_bag,
                            const std::string& output_bag,
                            const std::vector<std::string>& svm_image_topics,
                            const std::vector<std::string>& avm_image_topics,
                            double sync_percent,
                            double proto_window_sec = 0.6);

/// Detect remapped bags where SVM/AVM occupy different bag-record regions.
bool isSplitTimelineBevBag(RosBagReader& reader,
                           const std::vector<std::string>& svm_image_topics,
                           const std::vector<std::string>& avm_image_topics);

/// BEV: load SVM (camera_*) and AVM (panoramic_*) groups independently, then
/// merge.
bool getAlignedImagesBevGroups(
        RosBagReader& reader,
        const std::vector<std::string>& svm_image_topics,
        const std::vector<std::string>& avm_image_topics,
        double percent,
        std::map<std::string, cv::Mat>& images_by_topic,
        std::map<std::string, int64_t>& stamps_ns_by_topic,
        VehicleStateData* vehicle_state_out = nullptr);

}  // namespace mcalib

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <map>
#include <string>
#include <vector>

#include "CalibTypes.h"

namespace mcalib {

// Parser for camera/lidar config files (.cfg) in protobuf text format.
// Extracts camera intrinsics, extrinsics, and lidar configurations without
// requiring the full protobuf library.
class CalibConfigParser {
public:
    static bool loadCameraConfig(const std::string& config_file,
                                 VehicleCalibConfig& config);

    static bool loadLidarConfig(const std::string& config_file,
                                VehicleCalibConfig& config);

    static bool loadGroundConfig(const std::string& config_file,
                                 VehicleCalibConfig& config);

    /// Require cameras.cfg + lidars.cfg + ground.cfg in one directory.
    static bool loadConfigDirectory(const std::string& dir,
                                    VehicleCalibConfig& config);

    static bool saveCameraConfig(const std::string& config_file,
                                 const VehicleCalibConfig& config);

    /// Preserve full proto text; only patch extrinsics (aligned with
    /// save_multi_camera_extrinsic).
    static bool saveMultiCameraExtrinsic(
            const std::string& output_file,
            const std::map<std::string, Eigen::Isometry3d>&
                    extrinsics_cam_sensing,
            const VehicleCalibConfig& config);

    static bool saveLidarConfig(const std::string& config_file,
                                const VehicleCalibConfig& config);

    static bool saveGnssMultiLidarExtrinsic(
            const std::string& config_file,
            const std::vector<Eigen::Isometry3d>& extrinsics_gnss_lidar,
            const VehicleCalibConfig& config,
            const std::string& car_id = "manual_calib");

    static bool getCameraIntrinsic(const VehicleCalibConfig& config,
                                   const std::string& camera_name,
                                   cv::Mat& camera_matrix,
                                   cv::Mat& distor_coeffs,
                                   cv::Size& img_size);

    static bool getCameraExtrinsics(
            const VehicleCalibConfig& config,
            std::map<std::string, Eigen::Isometry3d>& extrinsics);

    static void updateCameraExtrinsic(VehicleCalibConfig& config,
                                      const std::string& camera_name,
                                      const Eigen::Isometry3d& extrinsic);

    // Scale SVM camera intrinsics to match camera_2 resolution (original
    // bird_eye_view).
    static void alignCameraSizes(VehicleCalibConfig& config);

private:
    struct TextProtoNode {
        std::string name;
        std::string value;
        std::vector<TextProtoNode> children;
        bool is_message = false;
    };

    static std::vector<TextProtoNode> parseTextProto(
            const std::string& content);
    static std::vector<TextProtoNode> parseBlock(const std::string& content,
                                                 size_t& pos);

    static void parseCameraConfigNode(const TextProtoNode& node,
                                      CameraSensorConfig& cam_config);
    static void parseIntrinsicNode(const TextProtoNode& node,
                                   CameraIntrinsic& intrinsic);
    static void parseExtrinsicNode(const TextProtoNode& node,
                                   Eigen::Isometry3d& extrinsic);
    static void parseTransformation3(const TextProtoNode& node,
                                     Eigen::Isometry3d& transform);

    static void parseLidarConfigNode(const TextProtoNode& node,
                                     LidarSensorConfig& lidar_config);

    static std::string serializeCameraConfig(const VehicleCalibConfig& config);

    static void skipWhitespace(const std::string& s, size_t& pos);
    static std::string readToken(const std::string& s, size_t& pos);
    static std::string readValue(const std::string& s, size_t& pos);
};

}  // namespace mcalib

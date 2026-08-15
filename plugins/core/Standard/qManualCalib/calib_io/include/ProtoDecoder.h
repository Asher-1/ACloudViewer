// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Geometry>
#include <cstdint>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

#include "CalibTypes.h"

namespace mcalib {

class RosBagReader;
class VideoDecodeCache;
struct BagMessage;

/// Single lidar entry from PointCloud2.lidar_configs (field 12).
struct EmbeddedLidarEntry {
    int ring_start = 0;
    int ring_end = 0;
    Eigen::Isometry3d extrinsic = Eigen::Isometry3d::Identity();
};

// Minimal protobuf wire format decoder for specific message types.
// Avoids requiring the full protobuf library as a dependency.
class ProtoDecoder {
public:
    enum WireType {
        VARINT = 0,
        FIXED64 = 1,
        LENGTH_DELIMITED = 2,
        FIXED32 = 5
    };

    struct ProtoField {
        uint32_t field_number;
        WireType wire_type;
        uint64_t varint_val;
        std::string bytes_val;
        uint32_t fixed32_val;
        uint64_t fixed64_val;
    };

    static std::vector<ProtoField> parseFields(const std::string& data);
    static std::vector<ProtoField> parseFields(const char* data, size_t len);

    // Extract CompressedImage.data from proto-encoded CompressedImage
    // Proto schema:
    //   field 1: Header header
    //   field 2: string format
    //   field 3: bytes data (the actual JPEG/PNG image data)
    //   field 4: uint64 exposure_duration_ns
    static bool decodeCompressedImage(const std::string& proto_data,
                                      std::string& image_data,
                                      double& timestamp_sec,
                                      std::string* format_out = nullptr);

    // Strip church header ("$$$$" + 4-byte header_len + header) if present
    static std::string stripChurchHeader(const std::string& data);

    // Decode CompressedImage from a bag message (std_msgs/String wrapper)
    static bool decodeCompressedImageFromBag(const std::string& bag_msg_data,
                                             cv::Mat& image,
                                             double& timestamp_sec);

    static bool peekCompressedImageFormat(const std::string& bag_msg_data,
                                          std::string& format);

    static bool decodeCompressedImageFromBag(RosBagReader& reader,
                                             const BagMessage& msg,
                                             cv::Mat& image,
                                             double& timestamp_sec,
                                             VideoDecodeCache& cache);

    static bool extractCompressedImageTimestampFromBag(
            const std::string& bag_msg_data, double& timestamp_sec);

    static bool extractPointCloudTimestampFromBag(
            const std::string& bag_msg_data, double& timestamp_sec);

    // Decode the image bytes (JPEG/PNG) into cv::Mat
    static cv::Mat decodeImageBuffer(const std::string& image_buffer);
    static cv::Mat decodeImageBuffer(const std::vector<uint8_t>& image_buffer);

    // Extract Header.timestamp_sec from a proto Header message
    // Proto schema:
    //   field 1: double timestamp_sec
    //   field 2: string module_name
    //   ...
    static double decodeHeaderTimestamp(const std::string& header_data);
    static void decodeHeader(const std::string& header_data,
                             double& timestamp_sec,
                             std::string& frame_id);

    static bool decodeVehicleState(const std::string& proto_data,
                                   VehicleStateData& state);
    static bool decodeVehicleStateFromBag(const std::string& bag_msg_data,
                                          VehicleStateData& state);

    // Extract PointCloud2 fields from proto-encoded PointCloud2
    // Proto schema:
    //   field 1: Header header
    //   field 2: uint32 height
    //   field 3: uint32 width
    //   field 4: repeated PointField fields
    //   field 5: bool is_bigendian
    //   field 6: uint32 point_step
    //   field 7: uint32 row_step
    //   field 8: bytes data
    //   field 9: bool is_dense
    struct PointCloud2Data {
        double timestamp_sec = 0;
        uint64_t header_stamp_us = 0;
        std::string frame_id;
        uint32_t height = 0;
        uint32_t width = 0;
        uint32_t point_step = 0;
        uint32_t row_step = 0;
        std::string data;
        bool is_dense = false;
        std::vector<EmbeddedLidarEntry> embedded_lidars;
    };

    static bool decodePointCloud2(const std::string& proto_data,
                                  PointCloud2Data& cloud_data);

    static bool decodePointCloud2FromBag(const std::string& bag_msg_data,
                                         PointCloud2Data& cloud_data);

    // Convert PointCloud2Data to vector of XYZ points
    static std::vector<Eigen::Vector3f> pointCloud2ToXYZ(
            const PointCloud2Data& cloud);

    // Convert PointCloud2Data to vector of XYZIRT points
    static std::vector<PointXYZIRT> pointCloud2ToXYZIRT(
            const PointCloud2Data& cloud);

    struct InsPose {
        int64_t measurement_time_us = 0;
        double pos_x = 0, pos_y = 0, pos_z = 0;
        double euler_x = 0, euler_y = 0, euler_z = 0;
    };

    static bool decodeInsPose(const std::string& proto_data, InsPose& pose);
    static bool decodeInsPoseFromBag(const std::string& bag_msg_data,
                                     InsPose& pose);

    static bool decodeEmbeddedLidarConfigs(
            const std::string& proto_data,
            std::vector<EmbeddedLidarEntry>& out);

private:
    static bool decodeTransformation3(const std::string& proto_data,
                                      Eigen::Isometry3d& transform);
    static bool decodeSingleLidarConfig(const std::string& proto_data,
                                        EmbeddedLidarEntry& out);

    static uint64_t readVarint(const char* data, size_t len, size_t& pos);
    static uint32_t readFixed32(const char* data, size_t len, size_t& pos);
    static uint64_t readFixed64(const char* data, size_t len, size_t& pos);
    static std::string readLengthDelimited(const char* data,
                                           size_t len,
                                           size_t& pos);
};

}  // namespace mcalib

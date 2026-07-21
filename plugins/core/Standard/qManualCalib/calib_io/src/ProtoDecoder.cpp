// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ProtoDecoder.h"

#include <CVLog.h>

#include <cstring>
#include <functional>

#include "RosBagReader.h"
#include "VideoCodecDecoder.h"

namespace mcalib {

namespace {

constexpr uint32_t kWireStartGroup = 3;
constexpr uint32_t kWireEndGroup = 4;

}  // namespace

uint64_t ProtoDecoder::readVarint(const char* data, size_t len, size_t& pos) {
    uint64_t result = 0;
    int shift = 0;
    while (pos < len) {
        uint8_t byte = static_cast<uint8_t>(data[pos++]);
        result |= static_cast<uint64_t>(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) break;
        shift += 7;
        if (shift >= 64) break;
    }
    return result;
}

uint32_t ProtoDecoder::readFixed32(const char* data, size_t len, size_t& pos) {
    if (pos + 4 > len) {
        pos = len;
        return 0;
    }
    uint32_t val;
    std::memcpy(&val, data + pos, 4);
    pos += 4;
    return val;
}

uint64_t ProtoDecoder::readFixed64(const char* data, size_t len, size_t& pos) {
    if (pos + 8 > len) {
        pos = len;
        return 0;
    }
    uint64_t val;
    std::memcpy(&val, data + pos, 8);
    pos += 8;
    return val;
}

std::string ProtoDecoder::readLengthDelimited(const char* data,
                                              size_t len,
                                              size_t& pos) {
    uint64_t str_len = readVarint(data, len, pos);
    if (pos + str_len > len) {
        pos = len;
        return {};
    }
    std::string result(data + pos, str_len);
    pos += str_len;
    return result;
}

std::vector<ProtoDecoder::ProtoField> ProtoDecoder::parseFields(
        const char* data, size_t len) {
    std::vector<ProtoField> fields;
    size_t pos = 0;

    std::function<void(bool, uint32_t)> parse_range;
    parse_range = [&](bool stop_on_end_group, uint32_t end_group_field) {
        while (pos < len) {
            const size_t tag_pos = pos;
            uint64_t tag = readVarint(data, len, pos);
            if (pos > len || pos == tag_pos) break;

            const uint32_t field_number = static_cast<uint32_t>(tag >> 3);
            const uint32_t wire_type = static_cast<uint32_t>(tag & 0x07);

            if (stop_on_end_group && field_number == end_group_field &&
                wire_type == kWireEndGroup) {
                return;
            }

            if (wire_type == kWireStartGroup) {
                parse_range(true, field_number);
                continue;
            }

            ProtoField f;
            f.field_number = field_number;
            f.wire_type = static_cast<WireType>(wire_type);
            f.varint_val = 0;
            f.fixed32_val = 0;
            f.fixed64_val = 0;

            switch (wire_type) {
                case VARINT:
                    f.varint_val = readVarint(data, len, pos);
                    break;
                case FIXED64:
                    f.fixed64_val = readFixed64(data, len, pos);
                    break;
                case LENGTH_DELIMITED:
                    f.bytes_val = readLengthDelimited(data, len, pos);
                    break;
                case FIXED32:
                    f.fixed32_val = readFixed32(data, len, pos);
                    break;
                default:
                    pos = len;
                    break;
            }

            if (pos <= len) {
                fields.push_back(f);
            }
        }
    };

    parse_range(false, 0);
    return fields;
}

std::vector<ProtoDecoder::ProtoField> ProtoDecoder::parseFields(
        const std::string& data) {
    return parseFields(data.data(), data.size());
}

double ProtoDecoder::decodeHeaderTimestamp(const std::string& header_data) {
    double ts = 0;
    std::string frame_id;
    decodeHeader(header_data, ts, frame_id);
    return ts;
}

void ProtoDecoder::decodeHeader(const std::string& header_data,
                                double& timestamp_sec,
                                std::string& frame_id) {
    timestamp_sec = 0;
    frame_id.clear();
    auto fields = parseFields(header_data);
    for (const auto& f : fields) {
        if (f.field_number == 1 && f.wire_type == FIXED64) {
            std::memcpy(&timestamp_sec, &f.fixed64_val, 8);
        } else if (f.field_number == 9 && f.wire_type == LENGTH_DELIMITED) {
            frame_id = f.bytes_val;
        }
    }
}

bool ProtoDecoder::decodeVehicleState(const std::string& proto_data,
                                      VehicleStateData& state) {
    state = VehicleStateData{};
    auto fields = parseFields(proto_data);
    for (const auto& f : fields) {
        if (f.field_number == 4 && f.wire_type == VARINT) {
            state.timestamp_us = static_cast<int64_t>(f.varint_val);
        } else if (f.field_number == 46 && f.wire_type == LENGTH_DELIMITED) {
            state.has_air_susp_report = true;
            auto sub = parseFields(f.bytes_val);
            for (const auto& sf : sub) {
                switch (sf.field_number) {
                    case 1:
                        if (sf.wire_type == VARINT) {
                            state.air_susp_lvl =
                                    static_cast<int>(sf.varint_val);
                        }
                        break;
                    case 2:
                        if (sf.wire_type == FIXED64) {
                            state.has_susp_lf = true;
                            std::memcpy(&state.susp_posn_vert_lf,
                                        &sf.fixed64_val, 8);
                        }
                        break;
                    case 3:
                        if (sf.wire_type == FIXED64) {
                            state.has_susp_lr = true;
                            std::memcpy(&state.susp_posn_vert_lr,
                                        &sf.fixed64_val, 8);
                        }
                        break;
                    case 4:
                        if (sf.wire_type == FIXED64) {
                            state.has_susp_rf = true;
                            std::memcpy(&state.susp_posn_vert_rf,
                                        &sf.fixed64_val, 8);
                        }
                        break;
                    case 5:
                        if (sf.wire_type == FIXED64) {
                            state.has_susp_rr = true;
                            std::memcpy(&state.susp_posn_vert_rr,
                                        &sf.fixed64_val, 8);
                        }
                        break;
                    default:
                        break;
                }
            }
        }
    }
    return state.has_air_susp_report || state.timestamp_us > 0;
}

bool ProtoDecoder::decodeVehicleStateFromBag(const std::string& bag_msg_data,
                                             VehicleStateData& state) {
    state = VehicleStateData{};
    if (bag_msg_data.size() < 4) return false;

    uint32_t str_len;
    std::memcpy(&str_len, bag_msg_data.data(), 4);
    if (4 + str_len > bag_msg_data.size()) return false;

    const std::string proto_data =
            stripChurchHeader(bag_msg_data.substr(4, str_len));
    return decodeVehicleState(proto_data, state);
}

bool ProtoDecoder::decodeCompressedImage(const std::string& proto_data,
                                         std::string& image_data,
                                         double& timestamp_sec,
                                         std::string* format_out) {
    auto fields = parseFields(proto_data);
    timestamp_sec = 0;
    if (format_out) format_out->clear();

    for (const auto& f : fields) {
        switch (f.field_number) {
            case 1:  // Header header (LENGTH_DELIMITED)
                if (f.wire_type == LENGTH_DELIMITED) {
                    timestamp_sec = decodeHeaderTimestamp(f.bytes_val);
                }
                break;
            case 2:  // string format
                if (f.wire_type == LENGTH_DELIMITED && format_out) {
                    *format_out = f.bytes_val;
                }
                break;
            case 3:  // bytes data (the JPEG/PNG image buffer)
                if (f.wire_type == LENGTH_DELIMITED) {
                    image_data = f.bytes_val;
                }
                break;
        }
    }

    return !image_data.empty();
}

std::string ProtoDecoder::stripChurchHeader(const std::string& data) {
    if (data.size() >= 8 && data.compare(0, 4, "$$$$") == 0) {
        uint32_t header_len;
        std::memcpy(&header_len, data.data() + 4, 4);
        size_t payload_start = 8 + header_len;
        if (payload_start <= data.size()) {
            CVLog::Print(
                    "[ProtoDecoder] stripped church header: "
                    "header_len=%u, payload=%zu bytes",
                    header_len, data.size() - payload_start);
            return data.substr(payload_start);
        }
    }
    return data;
}

bool ProtoDecoder::extractCompressedImageTimestampFromBag(
        const std::string& bag_msg_data, double& timestamp_sec) {
    timestamp_sec = 0;
    if (bag_msg_data.size() < 4) return false;

    uint32_t str_len;
    std::memcpy(&str_len, bag_msg_data.data(), 4);
    if (4 + str_len > bag_msg_data.size()) return false;

    const std::string proto_data =
            stripChurchHeader(bag_msg_data.substr(4, str_len));
    std::string image_buffer;
    return decodeCompressedImage(proto_data, image_buffer, timestamp_sec);
}

bool ProtoDecoder::extractPointCloudTimestampFromBag(
        const std::string& bag_msg_data, double& timestamp_sec) {
    timestamp_sec = 0;
    if (bag_msg_data.size() < 4) return false;

    uint32_t str_len;
    std::memcpy(&str_len, bag_msg_data.data(), 4);
    if (4 + str_len > bag_msg_data.size()) return false;

    const std::string proto_data =
            stripChurchHeader(bag_msg_data.substr(4, str_len));
    PointCloud2Data cloud_data;
    if (!decodePointCloud2(proto_data, cloud_data)) return false;
    timestamp_sec = cloud_data.timestamp_sec;
    return timestamp_sec > 0;
}

bool ProtoDecoder::peekCompressedImageFormat(const std::string& bag_msg_data,
                                             std::string& format) {
    format.clear();
    if (bag_msg_data.size() < 4) return false;

    uint32_t str_len = 0;
    std::memcpy(&str_len, bag_msg_data.data(), 4);
    if (4 + str_len > bag_msg_data.size()) return false;

    const std::string proto_data =
            stripChurchHeader(bag_msg_data.substr(4, str_len));
    std::string image_buffer;
    double timestamp_sec = 0;
    return decodeCompressedImage(proto_data, image_buffer, timestamp_sec,
                                 &format);
}

bool ProtoDecoder::decodeCompressedImageFromBag(const std::string& bag_msg_data,
                                                cv::Mat& image,
                                                double& timestamp_sec) {
    if (bag_msg_data.size() < 4) {
        CVLog::Warning("[ProtoDecoder] decodeImage: msg too small (%zu bytes)",
                       bag_msg_data.size());
        return false;
    }

    uint32_t str_len;
    std::memcpy(&str_len, bag_msg_data.data(), 4);
    if (4 + str_len > bag_msg_data.size()) {
        CVLog::Warning("[ProtoDecoder] decodeImage: str_len=%u > available=%zu",
                       str_len, bag_msg_data.size() - 4);
        return false;
    }

    std::string proto_data = stripChurchHeader(bag_msg_data.substr(4, str_len));

    std::string image_buffer;
    std::string format;
    if (!decodeCompressedImage(proto_data, image_buffer, timestamp_sec,
                               &format)) {
        CVLog::Warning(
                "[ProtoDecoder] decodeImage: proto parse failed "
                "(proto_data=%zu bytes, fields found image_data=%zu)",
                proto_data.size(), image_buffer.size());
        return false;
    }

    const bool video =
            VideoCodecDecoder::isVideoFormat(format) ||
            (format.empty() &&
             VideoCodecDecoder::looksLikeVideoBitstream(
                     reinterpret_cast<const uint8_t*>(image_buffer.data()),
                     image_buffer.size()));
    if (video) {
        CVLog::Warning(
                "[ProtoDecoder] decodeImage: format '%s' requires sequential "
                "video decode (use decodeCompressedImageFromBag with reader)",
                format.c_str());
        return false;
    }

    image = decodeImageBuffer(image_buffer);
    if (image.empty()) {
        CVLog::Warning(
                "[ProtoDecoder] decodeImage: cv::imdecode failed "
                "(buffer=%zu bytes, first bytes: %02x %02x %02x %02x)",
                image_buffer.size(),
                image_buffer.size() > 0 ? (uint8_t)image_buffer[0] : 0,
                image_buffer.size() > 1 ? (uint8_t)image_buffer[1] : 0,
                image_buffer.size() > 2 ? (uint8_t)image_buffer[2] : 0,
                image_buffer.size() > 3 ? (uint8_t)image_buffer[3] : 0);
        return false;
    }
    return true;
}

bool ProtoDecoder::decodeCompressedImageFromBag(RosBagReader& reader,
                                                const BagMessage& msg,
                                                cv::Mat& image,
                                                double& timestamp_sec,
                                                VideoDecodeCache& cache) {
    std::string format;
    peekCompressedImageFormat(msg.data, format);
    return cache.decodeMessage(reader, msg, format, image, timestamp_sec);
}

cv::Mat ProtoDecoder::decodeImageBuffer(const std::string& image_buffer) {
    cv::Mat raw_mat(1, static_cast<int>(image_buffer.size()), CV_8UC1,
                    const_cast<char*>(image_buffer.data()));
    return cv::imdecode(raw_mat, cv::IMREAD_COLOR);
}

cv::Mat ProtoDecoder::decodeImageBuffer(
        const std::vector<uint8_t>& image_buffer) {
    cv::Mat raw_mat(1, static_cast<int>(image_buffer.size()), CV_8UC1,
                    const_cast<uint8_t*>(image_buffer.data()));
    return cv::imdecode(raw_mat, cv::IMREAD_COLOR);
}

namespace {

float fixed32ToFloat(uint32_t bits) {
    float val = 0.f;
    std::memcpy(&val, &bits, sizeof(float));
    return val;
}

}  // namespace

bool ProtoDecoder::decodeTransformation3(const std::string& proto_data,
                                         Eigen::Isometry3d& transform) {
    Eigen::Quaterniond q(1, 0, 0, 0);
    Eigen::Vector3d t(0, 0, 0);
    bool has_pose = false;

    for (const auto& f : parseFields(proto_data)) {
        if (f.wire_type != LENGTH_DELIMITED) continue;
        if (f.field_number == 1) {  // Vector3 position
            for (const auto& vf : parseFields(f.bytes_val)) {
                if (vf.wire_type != FIXED32) continue;
                const float v = fixed32ToFloat(vf.fixed32_val);
                if (vf.field_number == 1)
                    t.x() = v;
                else if (vf.field_number == 2)
                    t.y() = v;
                else if (vf.field_number == 3)
                    t.z() = v;
            }
            has_pose = true;
        } else if (f.field_number == 2) {  // Quaternion_f orientation
            double qx = 0, qy = 0, qz = 0, qw = 1;
            for (const auto& qf : parseFields(f.bytes_val)) {
                if (qf.wire_type != FIXED32) continue;
                const float v = fixed32ToFloat(qf.fixed32_val);
                if (qf.field_number == 1)
                    qx = v;
                else if (qf.field_number == 2)
                    qy = v;
                else if (qf.field_number == 3)
                    qz = v;
                else if (qf.field_number == 4)
                    qw = v;
            }
            q = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
            has_pose = true;
        }
    }

    if (!has_pose) return false;
    transform = Eigen::Isometry3d::Identity();
    transform.linear() = q.toRotationMatrix();
    transform.translation() = t;
    return true;
}

bool ProtoDecoder::decodeSingleLidarConfig(const std::string& proto_data,
                                           EmbeddedLidarEntry& out) {
    out = EmbeddedLidarEntry{};
    bool has_transform = false;

    for (const auto& f : parseFields(proto_data)) {
        if (f.field_number == 26 && f.wire_type == LENGTH_DELIMITED) {
            if (!has_transform &&
                decodeTransformation3(f.bytes_val, out.extrinsic)) {
                has_transform = true;
            }
        } else if (f.field_number == 27 && f.wire_type == VARINT) {
            out.ring_start = static_cast<int>(f.varint_val);
        } else if (f.field_number == 28 && f.wire_type == VARINT) {
            out.ring_end = static_cast<int>(f.varint_val);
        }
    }

    return has_transform;
}

bool ProtoDecoder::decodeEmbeddedLidarConfigs(
        const std::string& proto_data, std::vector<EmbeddedLidarEntry>& out) {
    out.clear();
    for (const auto& f : parseFields(proto_data)) {
        if (f.field_number != 2 || f.wire_type != LENGTH_DELIMITED) continue;
        EmbeddedLidarEntry entry;
        if (decodeSingleLidarConfig(f.bytes_val, entry)) {
            out.push_back(entry);
        }
    }
    return !out.empty();
}

bool ProtoDecoder::decodePointCloud2(const std::string& proto_data,
                                     PointCloud2Data& cloud_data) {
    cloud_data.embedded_lidars.clear();
    auto fields = parseFields(proto_data);

    for (const auto& f : fields) {
        switch (f.field_number) {
            case 1:  // Header
                if (f.wire_type == LENGTH_DELIMITED) {
                    decodeHeader(f.bytes_val, cloud_data.timestamp_sec,
                                 cloud_data.frame_id);
                    cloud_data.header_stamp_us = static_cast<uint64_t>(
                            cloud_data.timestamp_sec * 1e6);
                }
                break;
            case 2:  // uint32 height
                if (f.wire_type == VARINT) {
                    cloud_data.height = static_cast<uint32_t>(f.varint_val);
                }
                break;
            case 3:  // uint32 width
                if (f.wire_type == VARINT) {
                    cloud_data.width = static_cast<uint32_t>(f.varint_val);
                }
                break;
            case 6:  // uint32 point_step
                if (f.wire_type == VARINT) {
                    cloud_data.point_step = static_cast<uint32_t>(f.varint_val);
                }
                break;
            case 7:  // uint32 row_step
                if (f.wire_type == VARINT) {
                    cloud_data.row_step = static_cast<uint32_t>(f.varint_val);
                }
                break;
            case 8:  // bytes data
                if (f.wire_type == LENGTH_DELIMITED) {
                    cloud_data.data = f.bytes_val;
                }
                break;
            case 9:  // bool is_dense
                if (f.wire_type == VARINT) {
                    cloud_data.is_dense = (f.varint_val != 0);
                }
                break;
            case 12:  // lidar_configs
                if (f.wire_type == LENGTH_DELIMITED) {
                    decodeEmbeddedLidarConfigs(f.bytes_val,
                                               cloud_data.embedded_lidars);
                }
                break;
        }
    }

    return !cloud_data.data.empty();
}

bool ProtoDecoder::decodePointCloud2FromBag(const std::string& bag_msg_data,
                                            PointCloud2Data& cloud_data) {
    if (bag_msg_data.size() < 4) {
        CVLog::Warning("[ProtoDecoder] decodeCloud: msg too small (%zu bytes)",
                       bag_msg_data.size());
        return false;
    }
    uint32_t str_len;
    std::memcpy(&str_len, bag_msg_data.data(), 4);
    if (4 + str_len > bag_msg_data.size()) {
        CVLog::Warning("[ProtoDecoder] decodeCloud: str_len=%u > available=%zu",
                       str_len, bag_msg_data.size() - 4);
        return false;
    }

    std::string proto_data = stripChurchHeader(bag_msg_data.substr(4, str_len));
    bool ok = decodePointCloud2(proto_data, cloud_data);
    if (ok) {
        CVLog::Print(
                "[ProtoDecoder] decodeCloud: %ux%u points, step=%u, data=%zu "
                "bytes, ts=%.3f",
                cloud_data.width, cloud_data.height, cloud_data.point_step,
                cloud_data.data.size(), cloud_data.timestamp_sec);
    } else {
        CVLog::Warning(
                "[ProtoDecoder] decodeCloud: proto parse failed (proto=%zu "
                "bytes)",
                proto_data.size());
    }
    return ok;
}

std::vector<Eigen::Vector3f> ProtoDecoder::pointCloud2ToXYZ(
        const PointCloud2Data& cloud) {
    std::vector<Eigen::Vector3f> points;
    if (cloud.point_step == 0 || cloud.data.empty()) return points;

    uint32_t num_points =
            static_cast<uint32_t>(cloud.data.size()) / cloud.point_step;
    points.reserve(num_points);

    for (uint32_t i = 0; i < num_points; ++i) {
        const char* ptr = cloud.data.data() + i * cloud.point_step;
        float x, y, z;
        std::memcpy(&x, ptr + 0, 4);
        std::memcpy(&y, ptr + 4, 4);
        std::memcpy(&z, ptr + 8, 4);

        if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
            points.emplace_back(x, y, z);
        }
    }
    return points;
}

std::vector<PointXYZIRT> ProtoDecoder::pointCloud2ToXYZIRT(
        const PointCloud2Data& cloud) {
    std::vector<PointXYZIRT> points;
    if (cloud.point_step == 0 || cloud.data.empty()) return points;

    uint32_t num_points =
            static_cast<uint32_t>(cloud.data.size()) / cloud.point_step;
    points.reserve(num_points);

    for (uint32_t i = 0; i < num_points; ++i) {
        const char* ptr = cloud.data.data() + i * cloud.point_step;
        PointXYZIRT pt{};
        std::memcpy(&pt.x, ptr + 0, 4);
        std::memcpy(&pt.y, ptr + 4, 4);
        std::memcpy(&pt.z, ptr + 8, 4);

        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) ||
            !std::isfinite(pt.z)) {
            continue;
        }

        // point_step >= 16 means we have intensity + ring + timestamp fields
        if (cloud.point_step >= 16) {
            std::memcpy(&pt.intensity, ptr + 12, 1);
            std::memcpy(&pt.ring, ptr + 13, 1);
            std::memcpy(&pt.timestamp, ptr + 14, 2);
        }
        points.push_back(pt);
    }
    return points;
}

static bool decodePoint3D(const std::string& data,
                          double& x,
                          double& y,
                          double& z) {
    auto fields = ProtoDecoder::parseFields(data);
    x = y = z = 0;
    for (const auto& f : fields) {
        if (f.wire_type != ProtoDecoder::FIXED64) continue;
        double val;
        std::memcpy(&val, &f.fixed64_val, 8);
        switch (f.field_number) {
            case 1:
                x = val;
                break;
            case 2:
                y = val;
                break;
            case 3:
                z = val;
                break;
        }
    }
    return true;
}

bool ProtoDecoder::decodeInsPose(const std::string& proto_data, InsPose& pose) {
    auto fields = parseFields(proto_data);
    pose = {};
    bool has_time = false;

    for (const auto& f : fields) {
        switch (f.field_number) {
            case 2:
                if (f.wire_type == FIXED64) {
                    std::memcpy(&pose.measurement_time_us, &f.fixed64_val, 8);
                    has_time = true;
                }
                break;
            case 5:
                if (f.wire_type == LENGTH_DELIMITED) {
                    decodePoint3D(f.bytes_val, pose.pos_x, pose.pos_y,
                                  pose.pos_z);
                }
                break;
            case 6:
                if (f.wire_type == LENGTH_DELIMITED) {
                    decodePoint3D(f.bytes_val, pose.euler_x, pose.euler_y,
                                  pose.euler_z);
                }
                break;
        }
    }
    return has_time;
}

bool ProtoDecoder::decodeInsPoseFromBag(const std::string& bag_msg_data,
                                        InsPose& pose) {
    if (bag_msg_data.size() < 4) return false;
    uint32_t str_len;
    std::memcpy(&str_len, bag_msg_data.data(), 4);
    if (4 + str_len > bag_msg_data.size()) return false;
    std::string proto_data = stripChurchHeader(bag_msg_data.substr(4, str_len));
    return decodeInsPose(proto_data, pose);
}

}  // namespace mcalib

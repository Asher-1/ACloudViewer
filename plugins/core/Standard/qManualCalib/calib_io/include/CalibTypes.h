// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cstdint>
#include <deque>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <utility>
#include <vector>

namespace mcalib {

typedef int64_t timestamp_t;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 8, 1> Vector8d;

#define stamp2double(x) ((x) / 1000000.0)
#define double2stamp(x) (static_cast<int64_t>((x) * 1000000))

inline constexpr const char* kCamera1 = "camera_1";
inline constexpr const char* kCamera2 = "camera_2";
inline constexpr const char* kCamera3 = "camera_3";
inline constexpr const char* kCamera4 = "camera_4";
inline constexpr const char* kCamera5 = "camera_5";
inline constexpr const char* kCamera6 = "camera_6";
inline constexpr const char* kTraffic2 = "traffic_2";

inline constexpr const char* kPanoramic1 = "panoramic_1";
inline constexpr const char* kPanoramic2 = "panoramic_2";
inline constexpr const char* kPanoramic3 = "panoramic_3";
inline constexpr const char* kPanoramic4 = "panoramic_4";

// BEV layout slot -> image/config key (aligned with bird_eye_viewer.h).
inline const std::map<std::string, std::string> kSvmCameraMap = {
        {kCamera1, kCamera1}, {kCamera2, kCamera2}, {kCamera3, kCamera3},
        {kCamera4, kCamera4}, {kCamera5, kCamera5}, {kCamera6, kCamera6}};

inline const std::map<std::string, std::string> kSvmT2CameraMap = {
        {kCamera1, kTraffic2}, {kCamera2, kCamera2}, {kCamera3, kCamera3},
        {kCamera4, kCamera4},  {kCamera5, kCamera5}, {kCamera6, kCamera6}};

inline const std::map<std::string, std::string> kAvmCameraMap = {
        {kCamera1, kPanoramic1},
        {kCamera2, kPanoramic2},
        {kCamera3, kPanoramic3},
        {kCamera4, kPanoramic4}};

inline constexpr const char* kImgEncodingMono8 = "mono8";
inline constexpr const char* kImgEncodingnv12 = "nv12";

struct PointXYZIRT {
    float x, y, z;
    uint8_t intensity;
    uint8_t ring;
    uint16_t timestamp;
};
static_assert(sizeof(PointXYZIRT) == 16, "PointXYZIRT should be 16 bytes");

struct CameraIntrinsic {
    int width = 0;
    int height = 0;
    double fx = 0, fy = 0;
    double cx = 0, cy = 0;
    double k1 = 0, k2 = 0, k3 = 0, k4 = 0, k5 = 0, k6 = 0;
    double p1 = 0, p2 = 0;
    double xi = 0;

    enum ModelType {
        PINHOLE = 0,
        KANNALA_BRANDT = 1,
        MEI = 2,
        FULLPINHOLE = 3
    };
    ModelType model_type = KANNALA_BRANDT;

    cv::Mat getCameraMatrix() const {
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0, 0) = fx;
        K.at<double>(1, 1) = fy;
        K.at<double>(0, 2) = cx;
        K.at<double>(1, 2) = cy;
        return K;
    }

    cv::Mat getDistCoeffs() const {
        cv::Mat D = cv::Mat::zeros(4, 1, CV_64F);
        D.at<double>(0) = k1;
        D.at<double>(1) = k2;
        D.at<double>(2) = k3;
        D.at<double>(3) = k4;
        return D;
    }

    cv::Size getImageSize() const { return cv::Size(width, height); }
};

struct CameraSensorConfig {
    std::string frame_id;
    CameraIntrinsic intrinsic;
    Eigen::Isometry3d extrinsic = Eigen::Isometry3d::Identity();
};

struct LidarSensorConfig {
    std::string frame_id;
    std::string model;
    int lidar_idx = 0;
    int ring_start = 0;
    int ring_end = 0;
    Eigen::Isometry3d extrinsic = Eigen::Isometry3d::Identity();
};

struct GroundPlane {
    double a = 0, b = 0, c = 1, d = 0;
};

struct VehicleCalibConfig {
    std::map<std::string, CameraSensorConfig> cameras;
    std::vector<LidarSensorConfig> lidars;
    Eigen::Isometry3d iso_sensing_vehicle = Eigen::Isometry3d::Identity();
    GroundPlane ground;

    std::string camera_cfg_path;
    std::string camera_cfg_raw;
    std::string lidar_cfg_path;
    std::string lidar_cfg_raw;
};

// Angle-axis rotation vector + translation (aligned with calibration
// math_utils).
inline void Vec2Isometry(const Vector6d& v, Eigen::Isometry3d& iso) {
    iso = Eigen::Isometry3d::Identity();
    const Eigen::Vector3d rv = v.head<3>();
    const double angle = rv.norm();
    if (angle > 1e-12) {
        iso.linear() =
                Eigen::AngleAxisd(angle, rv.normalized()).toRotationMatrix();
    }
    iso.translation() = v.tail<3>();
}

inline void Isometry2Vec(const Eigen::Isometry3d& iso, Vector6d& v) {
    const Eigen::AngleAxisd aa(iso.rotation());
    v.head<3>() = aa.axis() * aa.angle();
    v.tail<3>() = iso.translation();
}

inline void xyzeuler2isometry(const Vector6d& pose, Eigen::Isometry3d& iso) {
    Eigen::Quaterniond q =
            Eigen::AngleAxisd(pose(5), Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pose(4), Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(pose(3), Eigen::Vector3d::UnitX());
    iso = Eigen::Isometry3d::Identity();
    iso.linear() = q.toRotationMatrix();
    iso.translation() = pose.head<3>();
}

inline bool motionInterpolate(
        const std::deque<std::pair<int64_t, Eigen::Isometry3d>>& poses,
        int64_t stamp_us,
        Eigen::Isometry3d& result) {
    if (poses.empty()) return false;
    const double t = stamp_us / 1e6;
    const int pose_bound = static_cast<int>(poses.size()) - 1;
    for (int i = 0; i < pose_bound; ++i) {
        const double t1 = poses[i].first / 1e6;
        const double t2 = poses[i + 1].first / 1e6;
        if (t1 <= t && t2 >= t) {
            const Eigen::Isometry3d& iso1 = poses[i].second;
            const Eigen::Isometry3d& iso2 = poses[i + 1].second;
            Eigen::Isometry3d delta = iso1.inverse() * iso2;
            Vector6d dv;
            Isometry2Vec(delta, dv);
            const Vector6d interp = dv * ((t - t1) / (t2 - t1));
            Eigen::Isometry3d delta_interp;
            Vec2Isometry(interp, delta_interp);
            result = iso1 * delta_interp;
            return true;
        }
    }
    return false;
}

inline bool undistortPointCloud(
        const std::vector<PointXYZIRT>& cloud_raw,
        int64_t cloud_stamp_us,
        int64_t target_stamp_us,
        const std::deque<std::pair<int64_t, Eigen::Isometry3d>>& vehicle_poses,
        const Eigen::Isometry3d& iso_vehicle_lidar,
        std::vector<Eigen::Vector3f>& output) {
    output.clear();
    if (cloud_raw.empty() || vehicle_poses.empty()) return false;

    std::deque<std::pair<int64_t, Eigen::Isometry3d>> lidar_poses;
    for (const auto& vp : vehicle_poses) {
        lidar_poses.emplace_back(vp.first, vp.second * iso_vehicle_lidar);
    }

    uint16_t max_inner = 0;
    for (const auto& pt : cloud_raw) {
        if (pt.timestamp > max_inner) max_inner = pt.timestamp;
    }

    Eigen::Isometry3d start_pose, end_pose, target_pose;
    if (!motionInterpolate(lidar_poses, cloud_stamp_us, start_pose))
        return false;
    int64_t delta_us = static_cast<int64_t>(max_inner) * 2;
    if (!motionInterpolate(lidar_poses, cloud_stamp_us + delta_us, end_pose))
        return false;
    if (!motionInterpolate(lidar_poses, target_stamp_us, target_pose))
        return false;

    Eigen::Isometry3d delta = start_pose.inverse() * end_pose;
    Vector6d v;
    Isometry2Vec(delta, v);
    const double delta_sec = delta_us * 1e-6;
    if (delta_sec > 1e-12) v /= delta_sec;

    Eigen::Isometry3d iso_target_start = target_pose.inverse() * start_pose;
    output.reserve(cloud_raw.size());

    for (const auto& pt : cloud_raw) {
        const double dt = pt.timestamp * 2e-6;
        const Vector6d v2 = v * dt;
        Eigen::Isometry3d d2;
        Vec2Isometry(v2, d2);
        d2 = iso_target_start * d2;

        Eigen::Vector3d p(pt.x, pt.y, pt.z);
        Eigen::Vector3d pu = d2.linear() * p + d2.translation();
        output.emplace_back(static_cast<float>(pu.x()),
                            static_cast<float>(pu.y()),
                            static_cast<float>(pu.z()));
    }
    return true;
}

inline cv::Scalar colorFromDepth(float depth, float unit_depth = 1.0f) {
    float normalized = std::fmod(depth / unit_depth, 6.0f);
    if (normalized < 0) normalized += 6.0f;

    int section = static_cast<int>(normalized);
    float frac = normalized - section;

    int r = 0, g = 0, b = 0;
    switch (section) {
        case 0:
            r = 255;
            g = static_cast<int>(255 * frac);
            b = 0;
            break;
        case 1:
            r = static_cast<int>(255 * (1 - frac));
            g = 255;
            b = 0;
            break;
        case 2:
            r = 0;
            g = 255;
            b = static_cast<int>(255 * frac);
            break;
        case 3:
            r = 0;
            g = static_cast<int>(255 * (1 - frac));
            b = 255;
            break;
        case 4:
            r = static_cast<int>(255 * frac);
            g = 0;
            b = 255;
            break;
        case 5:
            r = 255;
            g = 0;
            b = static_cast<int>(255 * (1 - frac));
            break;
    }
    return cv::Scalar(b, g, r);
}

inline void highContrastLidarColor(int idx,
                                   uint8_t& r,
                                   uint8_t& g,
                                   uint8_t& b) {
    static const uint8_t palette[][3] = {
            {255, 0, 0},   {0, 255, 0},    {0, 0, 255},   {255, 255, 0},
            {255, 0, 255}, {0, 255, 255},  {255, 128, 0}, {128, 0, 255},
            {0, 255, 128}, {128, 128, 255}};
    int ci = idx % 10;
    r = palette[ci][0];
    g = palette[ci][1];
    b = palette[ci][2];
}

struct VehicleStateData {
    int64_t timestamp_us = 0;
    bool has_air_susp_report = false;
    int air_susp_lvl = 0;
    bool has_susp_lf = false;
    bool has_susp_lr = false;
    bool has_susp_rf = false;
    bool has_susp_rr = false;
    double susp_posn_vert_lf = 0;
    double susp_posn_vert_lr = 0;
    double susp_posn_vert_rf = 0;
    double susp_posn_vert_rr = 0;
};

inline const char* airSuspLvlName(int level) {
    switch (level) {
        case 0:
            return "NORMAL";
        case 1:
            return "LOW_LEVEL1";
        case 2:
            return "LOW_LEVEL2";
        case 3:
            return "HIGH_LEVEL1";
        case 4:
            return "HIGH_LEVEL2";
        case 5:
            return "HIGH_LEVEL3";
        default:
            return "UNKNOWN";
    }
}

inline void filterGroundReflectionPoints(
        const std::vector<Eigen::Vector3f>& cloud_in,
        std::vector<Eigen::Vector3f>& cloud_out,
        double ground_floor,
        double ground_ceiling) {
    cloud_out.clear();
    cloud_out.reserve(cloud_in.size());
    for (const auto& pt : cloud_in) {
        if (pt.z() >= ground_floor && pt.z() <= ground_ceiling) {
            cloud_out.push_back(pt);
        }
    }
}

inline void drawVehicleStateOverlay(cv::Mat& img,
                                    const VehicleStateData& state,
                                    cv::Point pos,
                                    double font_scale = 0.5,
                                    const cv::Scalar& color = cv::Scalar(0,
                                                                         0,
                                                                         255),
                                    int thickness = 1) {
    if (!state.has_air_susp_report) return;

    std::vector<std::string> lines;
    lines.push_back(std::string("AirSuspLvl: ") +
                    airSuspLvlName(state.air_susp_lvl));
    if (state.has_susp_lf) {
        lines.push_back("SuspLeftFront: " +
                        std::to_string(state.susp_posn_vert_lf));
    }
    if (state.has_susp_lr) {
        lines.push_back("SuspLeftRear: " +
                        std::to_string(state.susp_posn_vert_lr));
    }
    if (state.has_susp_rf) {
        lines.push_back("SuspRightFront: " +
                        std::to_string(state.susp_posn_vert_rf));
    }
    if (state.has_susp_rr) {
        lines.push_back("SuspRightRear: " +
                        std::to_string(state.susp_posn_vert_rr));
    }

    int y = pos.y;
    for (const auto& line : lines) {
        cv::putText(img, line, cv::Point(pos.x, y), cv::FONT_HERSHEY_COMPLEX,
                    font_scale, color, thickness, cv::LINE_AA);
        y += static_cast<int>(40 * font_scale);
    }
}

}  // namespace mcalib

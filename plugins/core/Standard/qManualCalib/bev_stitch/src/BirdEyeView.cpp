// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "BirdEyeView.h"

#include <CVLog.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>

#include "BevAlphaFusion.h"
#include "BevRemapBackend.h"
#include "CalibConfigParser.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <future>
#include <set>

namespace mcalib {

BirdEyeView::BirdEyeView() = default;

bool BirdEyeView::setCameraSlotMap(
        const std::map<std::string, std::string>& slot_to_source) {
    std::vector<std::string> new_order;
    for (const auto& [slot, source] : slot_to_source) {
        if (!camera_system_.getCamera(source)) continue;
        new_order.push_back(slot);
    }
    if (new_order.empty()) return false;

    const bool changed =
            (new_order != camera_order_) || (slot_to_source_ != slot_to_source);
    camera_order_ = std::move(new_order);
    slot_to_source_ = slot_to_source;
    CVLog::Print("[BEV] setCameraSlotMap: %zu slots", camera_order_.size());
    if (changed && initialized_ && !last_extrinsics_.empty()) {
        buildRemapMaps(last_extrinsics_);
        buildAlphaFusion();
    }
    return changed;
}

bool BirdEyeView::setCameraSubset(const std::vector<std::string>& cameras) {
    std::map<std::string, std::string> identity;
    for (const auto& name : cameras) {
        identity[name] = name;
    }
    return setCameraSlotMap(identity);
}

void BirdEyeView::resetCameraSubset() {
    camera_order_ = all_camera_order_;
    CVLog::Print("[BEV] resetCameraSubset: %zu cameras", camera_order_.size());
}

void BirdEyeView::setFocalScale(double scale) {
    if (scale <= 0.01) scale = 0.01;
    if (std::fabs(config_.focal_scale - scale) < 1e-6) return;
    config_.focal_scale = scale;
    double f = config_.virtual_camera_focal * config_.focal_scale;
    K_bev_.at<float>(0, 0) = static_cast<float>(f);
    K_bev_.at<float>(1, 1) = static_cast<float>(f);
    buildBevScene();
    if (initialized_) {
        if (last_extrinsics_.empty()) {
            for (const auto& [name, cam] : calib_config_.cameras) {
                last_extrinsics_[name] = cam.extrinsic;
            }
        }
        buildRemapMaps(last_extrinsics_);
        buildAlphaFusion();
    }
    CVLog::Print("[BEV] setFocalScale: %.3f (f=%.1f)", scale, f);
}

void BirdEyeView::setRemapMode(BevRemapMode mode) {
    if (config_.remap_mode == mode) return;
    config_.remap_mode = mode;
    if (!initialized_) return;

    std::map<std::string, Eigen::Isometry3d> extrinsics;
    if (!last_extrinsics_.empty()) {
        extrinsics = last_extrinsics_;
    } else {
        for (const auto& [name, cam] : calib_config_.cameras) {
            extrinsics[name] = cam.extrinsic;
        }
    }
    buildRemapMaps(extrinsics);
    buildAlphaFusion();
    CVLog::Print("[BEV] setRemapMode: %s (active %s)",
                 BevRemapper::modeName(config_.remap_mode),
                 BevRemapper::modeName(getActiveRemapMode()));
}

BevRemapMode BirdEyeView::getActiveRemapMode() const {
    if (!remap_backends_.empty()) {
        auto it = remap_backends_.begin();
        if (it->second) return it->second->activeMode();
    }
    return BevRemapper::resolveMode(config_.remap_mode);
}

BirdEyeView::~BirdEyeView() = default;

bool BirdEyeView::init(const VehicleCalibConfig& config,
                       const Config& bev_config) {
    config_ = bev_config;
    calib_config_ = config;

    if (calib_config_.ground.d > 0.1) {
        config_.sensing_height = calib_config_.ground.d;
        CVLog::Print("[BEV] sensing_height from ground.cfg: %.3f",
                     config_.sensing_height);
    }

    if (calib_config_.cameras.empty()) {
        CVLog::Warning("[BEV] No cameras in config");
        return false;
    }

    camera_system_.loadFromConfig(calib_config_);

    camera_order_.clear();
    slot_to_source_.clear();
    for (const auto& [name, _] : calib_config_.cameras) {
        camera_order_.push_back(name);
        slot_to_source_[name] = name;
    }
    all_camera_order_ = camera_order_;

    K_bev_ = cv::Mat::zeros(3, 3, CV_32FC1);
    double f = config_.virtual_camera_focal * config_.focal_scale;
    K_bev_.at<float>(0, 0) = static_cast<float>(f);
    K_bev_.at<float>(1, 1) = static_cast<float>(f);
    K_bev_.at<float>(0, 2) = config_.bev_size.width / 2.f;
    K_bev_.at<float>(1, 2) = config_.bev_size.height / 2.f;
    K_bev_.at<float>(2, 2) = 1.f;

    iso_sensing_ground_ = Eigen::Isometry3d::Identity();
    Eigen::Vector3d ground_normal(calib_config_.ground.a,
                                  calib_config_.ground.b,
                                  calib_config_.ground.c);
    double norm = ground_normal.norm();
    if (norm > 1e-6) {
        ground_normal /= norm;
    } else {
        ground_normal = Eigen::Vector3d(0, 0, 1);
    }
    iso_sensing_ground_.linear() =
            Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d(0, 0, 1),
                                               ground_normal)
                    .toRotationMatrix();
    iso_sensing_ground_.translation() =
            Eigen::Vector3d(0, 0, -config_.sensing_height);

    iso_ground_bev_ = Eigen::Isometry3d::Identity();
    Eigen::Matrix3d R_ground_bev;
    R_ground_bev << 0, -1, 0, -1, 0, 0, 0, 0, -1;
    iso_ground_bev_.linear() = R_ground_bev;
    iso_ground_bev_.translation()(2) = config_.virtual_camera_height;

    iso_sensing_bev_ = iso_sensing_ground_ * iso_ground_bev_;

    generateCameraPairs();
    buildBevScene();

    std::map<std::string, Eigen::Isometry3d> extrinsics;
    for (const auto& [name, cam] : calib_config_.cameras) {
        extrinsics[name] = cam.extrinsic;
    }
    last_extrinsics_ = extrinsics;
    buildRemapMaps(extrinsics);
    buildAlphaFusion();

    initialized_ = true;
    CVLog::Print("[BEV] Initialized: %dx%d, %zu cameras",
                 config_.bev_size.width, config_.bev_size.height,
                 camera_order_.size());
    return true;
}

static int cameraLogicalIndex(const std::string& name) {
    auto pos = name.find_last_of('_');
    if (pos != std::string::npos && pos + 1 < name.size()) {
        try {
            return std::stoi(name.substr(pos + 1));
        } catch (...) {
        }
    }
    return -1;
}

static std::string findCamByIndex(const std::vector<std::string>& order,
                                  int idx) {
    for (const auto& n : order) {
        if (cameraLogicalIndex(n) == idx) return n;
    }
    return {};
}

void BirdEyeView::generateCameraPairs() {
    camera_pairs_.clear();

    auto cam = [this](int idx) { return findCamByIndex(camera_order_, idx); };

    if (camera_order_.size() == 4) {
        auto c1 = cam(1), c2 = cam(2), c3 = cam(3), c4 = cam(4);
        if (!c1.empty() && !c2.empty()) camera_pairs_.emplace_back(c1, c2);
        if (!c2.empty() && !c3.empty()) camera_pairs_.emplace_back(c2, c3);
        if (!c3.empty() && !c4.empty()) camera_pairs_.emplace_back(c3, c4);
        if (!c4.empty() && !c1.empty()) camera_pairs_.emplace_back(c4, c1);
    } else {
        auto c1 = cam(1), c2 = cam(2), c3 = cam(3);
        auto c4 = cam(4), c5 = cam(5), c6 = cam(6);
        if (!c1.empty() && !c3.empty()) camera_pairs_.emplace_back(c1, c3);
        if (!c3.empty() && !c2.empty()) camera_pairs_.emplace_back(c3, c2);
        if (!c2.empty() && !c4.empty()) camera_pairs_.emplace_back(c2, c4);
        if (!c4.empty() && !c6.empty()) camera_pairs_.emplace_back(c4, c6);
        if (!c6.empty() && !c5.empty()) camera_pairs_.emplace_back(c6, c5);
        if (!c5.empty() && !c1.empty()) camera_pairs_.emplace_back(c5, c1);
    }
}

void BirdEyeView::buildBevScene() {
    Eigen::Matrix3f K_bev_eigen;
    cv::cv2eigen(K_bev_, K_bev_eigen);
    Eigen::Matrix3f K_bev_inv = K_bev_eigen.inverse();

    bev_scene_ = cv::Mat::zeros(config_.bev_size, CV_32FC3);

    const Eigen::Vector3d ground_z(0, 0, -1);
    const Eigen::Vector3d bev_origin = iso_ground_bev_.translation();
    const Eigen::Matrix3d R_g = iso_ground_bev_.linear();
    const double cam_h = config_.virtual_camera_height;

    const int W = config_.bev_size.width;
    const int H = config_.bev_size.height;

    for (int v = 0; v < H; v++) {
        float* row = bev_scene_.ptr<float>(v);
        for (int u = 0; u < W; u++) {
            Eigen::Vector3f p(static_cast<float>(u), static_cast<float>(v),
                              1.f);
            Eigen::Vector3f dir = K_bev_inv * p;
            dir.normalize();

            Eigen::Vector3d dir_g = R_g * dir.cast<double>();
            double cos_angle = dir_g.dot(ground_z);

            if (cos_angle > 0) {
                double scale = cam_h / cos_angle;
                Eigen::Vector3d pt = bev_origin + scale * dir_g;
                row[u * 3 + 0] = static_cast<float>(pt(0));
                row[u * 3 + 1] = static_cast<float>(pt(1));
                row[u * 3 + 2] = static_cast<float>(pt(2));
            }
        }
    }
}

void BirdEyeView::buildRemapMaps(
        const std::map<std::string, Eigen::Isometry3d>& extrinsics) {
    const std::set<std::string> active_slots(camera_order_.begin(),
                                             camera_order_.end());
    for (auto it = remap_map1_.begin(); it != remap_map1_.end();) {
        if (active_slots.count(it->first) == 0) {
            remap_map2_.erase(it->first);
            remap_backends_.erase(it->first);
            it = remap_map1_.erase(it);
        } else {
            ++it;
        }
    }

    const int W = config_.bev_size.width;
    const int H = config_.bev_size.height;
    const double angle_thresh = M_PI * 0.49;
    const BevRemapMode active_mode = getActiveRemapMode();

    for (const auto& slot : camera_order_) {
        const auto map_it = slot_to_source_.find(slot);
        const std::string source =
                map_it != slot_to_source_.end() ? map_it->second : slot;

        auto it_ext = extrinsics.find(source);
        if (it_ext == extrinsics.end()) continue;

        auto cam_model = camera_system_.getCamera(source);
        if (!cam_model) continue;

        // cfg stores sensor_to_cam; BEV remap expects cam_to_sensing (see
        // original get_camera_extrinsics which stores inverse(sensor_to_cam)).
        Eigen::Isometry3d iso_cam_ground =
                it_ext->second.inverse() * iso_sensing_ground_;

        const Eigen::Matrix3d R = iso_cam_ground.linear();
        const Eigen::Vector3d t = iso_cam_ground.translation();

        cv::Mat& map_x = remap_map1_[slot];
        cv::Mat& map_y = remap_map2_[slot];
        if (map_x.size() != config_.bev_size || map_x.type() != CV_32FC1) {
            map_x.create(config_.bev_size, CV_32FC1);
        } else {
            map_x.setTo(-1.f);
        }
        if (map_y.size() != config_.bev_size || map_y.type() != CV_32FC1) {
            map_y.create(config_.bev_size, CV_32FC1);
        } else {
            map_y.setTo(-1.f);
        }

        for (int v = 0; v < H; v++) {
            const float* scene_row = bev_scene_.ptr<float>(v);
            float* mx_row = map_x.ptr<float>(v);
            float* my_row = map_y.ptr<float>(v);

            for (int u = 0; u < W; u++) {
                double sx = scene_row[u * 3];
                double sy = scene_row[u * 3 + 1];
                double sz = scene_row[u * 3 + 2];

                if (sx == 0 && sy == 0 && sz == 0) continue;

                double pcx = R(0, 0) * sx + R(0, 1) * sy + R(0, 2) * sz + t(0);
                double pcy = R(1, 0) * sx + R(1, 1) * sy + R(1, 2) * sz + t(1);
                double pcz = R(2, 0) * sx + R(2, 1) * sy + R(2, 2) * sz + t(2);

                if (pcz <= 0) continue;
                double angle =
                        std::atan2(std::sqrt(pcx * pcx + pcy * pcy), pcz);
                if (angle >= angle_thresh) continue;

                Eigen::Vector3d pc(pcx, pcy, pcz);
                Eigen::Vector2d p;
                cam_model->spaceToPlane(pc, p);
                mx_row[u] = static_cast<float>(p(0));
                my_row[u] = static_cast<float>(p(1));
            }
        }

        const cv::Size src_size(cam_model->getParameters().width,
                                cam_model->getParameters().height);
        auto backend_it = remap_backends_.find(slot);
        if (backend_it != remap_backends_.end() && backend_it->second &&
            backend_it->second->activeMode() == active_mode) {
            backend_it->second->updateMaps(map_x, map_y);
        } else {
            auto backend = std::make_unique<BevRemapper>(src_size, map_x, map_y,
                                                         config_.remap_mode);
            if (backend->valid()) {
                remap_backends_[slot] = std::move(backend);
            } else {
                remap_backends_.erase(slot);
            }
        }
    }
    if (!remap_backends_.empty()) {
        CVLog::Print("[BEV] remap backend: %s (resolved %s)",
                     BevRemapper::modeName(config_.remap_mode),
                     BevRemapper::modeName(
                             remap_backends_.begin()->second->activeMode()));
    }
    CVLog::Print("[BEV] buildRemapMaps: %zu maps created", remap_map1_.size());
}

void BirdEyeView::getAngleRange4(const std::string& name,
                                 double& start,
                                 double& end) {
    const int delta = -20;
    int idx = cameraLogicalIndex(name);
    switch (idx) {
        case 2:
            start = -45 + delta;
            end = 45 - delta;
            break;
        case 3:
            start = 45 - delta;
            end = 135 + delta;
            break;
        case 4:
            start = 135 + delta;
            end = 225 - delta;
            break;
        case 1:
            start = 225 - delta;
            end = 315 + delta;
            break;
        default:
            start = 0;
            end = 360;
            break;
    }
    start *= (M_PI / 180);
    end *= (M_PI / 180);
}

void BirdEyeView::getAngleRange6(const std::string& name,
                                 double& start,
                                 double& end) {
    int idx = cameraLogicalIndex(name);
    switch (idx) {
        case 1:
            start = 240;
            end = 300;
            break;
        case 3:
            start = 300;
            end = 360;
            break;
        case 2:
            start = 0;
            end = 60;
            break;
        case 4:
            start = 60;
            end = 120;
            break;
        case 6:
            start = 120;
            end = 180;
            break;
        case 5:
            start = 180;
            end = 240;
            break;
        default:
            start = 0;
            end = 360;
            break;
    }
    start *= (M_PI / 180);
    end *= (M_PI / 180);
}

void BirdEyeView::generatePriorMasks(double front_to_sensing,
                                     double back_to_sensing) {
    const int num_cams = static_cast<int>(camera_order_.size());

    struct Pt3 {
        double x, y, z;
    };
    double bh = config_.virtual_camera_height;
    double hw = config_.car_width / 2;
    std::vector<Pt3> pts3d = {{hw, -front_to_sensing, bh},
                              {hw, -back_to_sensing, bh},
                              {-hw, -back_to_sensing, bh},
                              {-hw, -front_to_sensing, bh},
                              {hw, 0, bh},
                              {-hw, 0, bh}};

    float fx = K_bev_.at<float>(0, 0), fy = K_bev_.at<float>(1, 1);
    float cx = K_bev_.at<float>(0, 2), cy = K_bev_.at<float>(1, 2);

    std::vector<cv::Point2f> pts2d;
    for (const auto& p : pts3d) {
        if (std::fabs(p.z) < 1e-6) {
            pts2d.emplace_back(-1, -1);
            continue;
        }
        pts2d.emplace_back(fx * static_cast<float>(p.x / p.z) + cx,
                           fy * static_cast<float>(p.y / p.z) + cy);
    }

    cv::Mat car_mask = cv::Mat::zeros(config_.bev_size, CV_8UC3);
    cv::Point center(config_.bev_size.width / 2, config_.bev_size.height / 2);
    double radius = std::sqrt(
            double(config_.bev_size.width) * config_.bev_size.width +
            double(config_.bev_size.height) * config_.bev_size.height);

    for (const auto& cam_name : camera_order_) {
        int logical_idx = cameraLogicalIndex(cam_name);
        int start_idx = 0, end_idx = 0;
        if (num_cams <= 4) {
            switch (logical_idx) {
                case 1:
                    start_idx = 3;
                    end_idx = 0;
                    break;
                case 2:
                    start_idx = 0;
                    end_idx = 1;
                    break;
                case 3:
                    start_idx = 1;
                    end_idx = 2;
                    break;
                case 4:
                    start_idx = 2;
                    end_idx = 3;
                    break;
                default:
                    continue;
            }
        } else {
            switch (logical_idx) {
                case 1:
                    start_idx = 3;
                    end_idx = 0;
                    break;
                case 2:
                    start_idx = 4;
                    end_idx = 1;
                    break;
                case 3:
                    start_idx = 0;
                    end_idx = 4;
                    break;
                case 4:
                    start_idx = 1;
                    end_idx = 2;
                    break;
                case 5:
                    start_idx = 5;
                    end_idx = 3;
                    break;
                case 6:
                    start_idx = 2;
                    end_idx = 5;
                    break;
                default:
                    continue;
            }
        }

        cv::Point p1(static_cast<int>(pts2d[start_idx].x),
                     static_cast<int>(pts2d[start_idx].y));
        cv::Point p2(static_cast<int>(pts2d[end_idx].x),
                     static_cast<int>(pts2d[end_idx].y));

        double start_angle, end_angle;
        if (num_cams <= 4) {
            getAngleRange4(cam_name, start_angle, end_angle);
        } else {
            getAngleRange6(cam_name, start_angle, end_angle);
        }

        cv::Point p3(
                center.x + static_cast<int>(radius * std::cos(start_angle)),
                center.y + static_cast<int>(radius * std::sin(start_angle)));
        cv::Point p4(center.x + static_cast<int>(radius * std::cos(end_angle)),
                     center.y + static_cast<int>(radius * std::sin(end_angle)));

        std::vector<cv::Point> pts = {p1, p2, p4, p3};
        cv::Mat prior_mask = car_mask.clone();
        cv::fillConvexPoly(prior_mask, pts, cv::Scalar::all(255));

        start_angle -= 24.0 * (M_PI / 180.0);
        end_angle += 24.0 * (M_PI / 180.0);
        double scale_r = std::max(1.0 / std::abs(std::cos(start_angle)),
                                  1.0 / std::abs(std::sin(start_angle)));
        double bigger_radius = radius * scale_r;
        cv::Point p5(center.x + static_cast<int>(bigger_radius *
                                                 std::cos(start_angle)),
                     center.y + static_cast<int>(bigger_radius *
                                                 std::sin(start_angle)));
        cv::Point p6(center.x + static_cast<int>(bigger_radius *
                                                 std::cos(end_angle)),
                     center.y + static_cast<int>(bigger_radius *
                                                 std::sin(end_angle)));
        std::vector<cv::Point> bigger_pts = {p1, p2, p6, p5};
        cv::Mat bigger_prior_mask = car_mask.clone();
        cv::fillConvexPoly(bigger_prior_mask, bigger_pts, cv::Scalar::all(255));

        if (bigger_bev_masks_.count(cam_name)) {
            bigger_bev_masks_[cam_name] &= bigger_prior_mask;
        }
    }
}

void BirdEyeView::generateWeights() {
    bev_weights_.clear();

    for (const auto& cam_name : camera_order_) {
        cv::Mat weight = cv::Mat::zeros(config_.bev_size, CV_32FC1);
        bev_weights_[cam_name] = weight;
    }

    for (const auto& pair : camera_pairs_) {
        if (bigger_bev_masks_.find(pair.first) == bigger_bev_masks_.end() ||
            bigger_bev_masks_.find(pair.second) == bigger_bev_masks_.end()) {
            continue;
        }

        cv::Mat mask_l = bigger_bev_masks_[pair.first];
        cv::Mat mask_r = bigger_bev_masks_[pair.second];

        std::vector<cv::Mat> ch_l, ch_r;
        cv::split(mask_l, ch_l);
        cv::split(mask_r, ch_r);

        cv::Mat overlap = ch_l[0] & ch_r[0];
        cv::Mat overlap_inv = ~overlap;

        cv::Mat single_l = ch_l[0] & overlap_inv;
        cv::Mat single_r = ch_r[0] & overlap_inv;

        cv::Mat dist_l, dist_r;
        cv::distanceTransform(255 - single_l, dist_l, cv::DIST_L2, 0);
        cv::distanceTransform(255 - single_r, dist_r, cv::DIST_L2, 0);

        for (int i = 0; i < overlap.rows; i++) {
            for (int j = 0; j < overlap.cols; j++) {
                if (overlap.at<uchar>(i, j) > 0) {
                    float d1 = dist_l.at<float>(i, j);
                    float d2 = dist_r.at<float>(i, j);
                    float d = d1 + d2;
                    if (d > 1e-6f) {
                        bev_weights_[pair.first].at<float>(i, j) = d2 / d;
                        bev_weights_[pair.second].at<float>(i, j) = d1 / d;
                    }
                }
            }
        }
    }

    for (const auto& [name, weight] : bev_weights_) {
        if (bigger_bev_masks_.find(name) == bigger_bev_masks_.end()) continue;
        const cv::Mat& mask = bigger_bev_masks_[name];
        const cv::Mat& overlap = bev_overlap_masks_[name];
        for (int i = 0; i < overlap.rows; i++) {
            for (int j = 0; j < overlap.cols; j++) {
                if (overlap.at<uchar>(i, j) == 0 &&
                    mask.at<cv::Vec3b>(i, j)[0] != 0) {
                    bev_weights_[name].at<float>(i, j) = 1.0f;
                }
            }
        }
    }
}

void BirdEyeView::generateBlendAreas(cv::Rect car_zone) {
    bev_blend_areas_.clear();
    int xx[4] = {0, car_zone.tl().x, car_zone.br().x,
                 config_.bev_size.width - 1};
    int yy[4] = {0, car_zone.tl().y, car_zone.br().y,
                 config_.bev_size.height - 1};

    std::vector<cv::Rect> zones;
    for (int i = 0; i <= 2; i++) {
        for (int j = 0; j <= 2; j++) {
            zones.emplace_back(cv::Point(xx[j], yy[i]),
                               cv::Point(xx[j + 1], yy[i + 1]));
        }
    }

    std::string c1 = findCamByIndex(camera_order_, 1);
    std::string c2 = findCamByIndex(camera_order_, 2);
    std::string c3 = findCamByIndex(camera_order_, 3);
    std::string c4 = findCamByIndex(camera_order_, 4);

    if (c1.empty() || c2.empty() || c3.empty() || c4.empty()) return;

    auto set = [&](const std::string& id, int zone_idx, const std::string& a,
                   const std::string& b) {
        BlendArea area;
        area.zone = zones[zone_idx];
        area.blend_pair = std::make_pair(a, b);
        bev_blend_areas_[id] = area;
    };

    set("00", 0, c1, c4);
    set("01", 1, c1, c1);
    set("02", 2, c1, c2);
    set("10", 3, c4, c4);
    set("12", 5, c2, c2);
    set("20", 6, c3, c4);
    set("21", 7, c3, c3);
    set("22", 8, c2, c3);
}

void BirdEyeView::buildAlphaFusion() {
    generateCameraPairs();

    bigger_bev_masks_.clear();
    smaller_bev_masks_.clear();
    bev_overlap_masks_.clear();

    for (const auto& cam_name : camera_order_) {
        auto cam_model = camera_system_.getCamera(cam_name);
        if (!cam_model) continue;

        cv::Size img_size(cam_model->getParameters().width,
                          cam_model->getParameters().height);
        cv::Mat white_img(img_size, CV_8UC3, cv::Scalar(255, 255, 255));

        if (remap_map1_.find(cam_name) == remap_map1_.end()) continue;

        cv::Mat mask_warp;
        cv::remap(white_img, mask_warp, remap_map1_[cam_name],
                  remap_map2_[cam_name], cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                  cv::Scalar(0, 0, 0));

        for (int i = 0; i < mask_warp.rows; i++) {
            for (int j = 0; j < mask_warp.cols; j++) {
                cv::Vec3b& v = mask_warp.at<cv::Vec3b>(i, j);
                if (v[0] != 255 || v[1] != 255 || v[2] != 255) {
                    v = cv::Vec3b(0, 0, 0);
                }
            }
        }
        bigger_bev_masks_[cam_name] = mask_warp;
    }

    if (camera_order_.size() == 4) {
        std::string c2 = findCamByIndex(camera_order_, 2);
        std::string c4 = findCamByIndex(camera_order_, 4);
        auto it2 = bigger_bev_masks_.find(c2);
        auto it4 = bigger_bev_masks_.find(c4);
        if (it2 != bigger_bev_masks_.end() && it4 != bigger_bev_masks_.end()) {
            cv::Mat overlap_c2c4 = it2->second & it4->second;
            it2->second &= (~overlap_c2c4);
            it4->second &= (~overlap_c2c4);
        }
    }

    double front_to_sensing = config_.car_length / 2;
    double back_to_sensing = -config_.car_length / 2;

    auto resolveSource = [this](const std::string& slot) -> std::string {
        const auto it = slot_to_source_.find(slot);
        return it != slot_to_source_.end() ? it->second : slot;
    };
    auto findExtrinsic =
            [this](const std::string& source) -> const Eigen::Isometry3d* {
        if (auto it = last_extrinsics_.find(source);
            it != last_extrinsics_.end()) {
            return &it->second;
        }
        if (auto it = calib_config_.cameras.find(source);
            it != calib_config_.cameras.end()) {
            return &it->second.extrinsic;
        }
        return nullptr;
    };

    // codetree bird_eye_viewer::bev_alpha_fusion_init
    if (camera_order_.size() == 4) {
        if (const auto* ext1 = findExtrinsic(resolveSource("camera_1"))) {
            front_to_sensing = ext1->translation().x() + 0.05;
        }
        if (const auto* ext3 = findExtrinsic(resolveSource("camera_3"))) {
            back_to_sensing = ext3->translation().x() - 0.05;
        }
    } else if (camera_order_.size() == 6) {
        if (const auto* ext4 = findExtrinsic(resolveSource("camera_4"))) {
            back_to_sensing = ext4->translation().x();
            front_to_sensing = config_.car_length + back_to_sensing;
        }
    }

    generatePriorMasks(front_to_sensing, back_to_sensing);

    for (const auto& [name, mask] : bigger_bev_masks_) {
        smaller_bev_masks_[name] = mask.clone();
        bev_overlap_masks_[name] = cv::Mat::zeros(mask.size(), CV_8UC1);

        for (const auto& [other_name, other_mask] : bigger_bev_masks_) {
            if (name == other_name) continue;
            cv::Mat overlap = mask & other_mask;
            smaller_bev_masks_[name] &= (~overlap);

            std::vector<cv::Mat> overlap_channels;
            cv::split(overlap, overlap_channels);
            bev_overlap_masks_[name] |= overlap_channels[0];
        }
    }

    generateWeights();

    Eigen::Isometry3d iso_bev_ground = iso_ground_bev_.inverse();

    Eigen::Matrix3d R_bg = iso_bev_ground.linear();
    Eigen::Vector3d t_bg = iso_bev_ground.translation();

    auto projectGround = [&](double gx, double gy, double gz) -> cv::Point2f {
        Eigen::Vector3d pg(gx, gy, gz);
        Eigen::Vector3d pc = R_bg * pg + t_bg;
        if (std::fabs(pc(2)) < 1e-6) return cv::Point2f(-1, -1);
        float u = K_bev_.at<float>(0, 0) * static_cast<float>(pc(0) / pc(2)) +
                  K_bev_.at<float>(0, 2);
        float v = K_bev_.at<float>(1, 1) * static_cast<float>(pc(1) / pc(2)) +
                  K_bev_.at<float>(1, 2);
        return cv::Point2f(u, v);
    };

    double hw = config_.car_width / 2;
    cv::Point2f p0 = projectGround(front_to_sensing, hw, 0);
    cv::Point2f p2 = projectGround(back_to_sensing, -hw, 0);

    black_zone_ =
            cv::Rect(cv::Point(static_cast<int>(p0.x), static_cast<int>(p0.y)),
                     cv::Point(static_cast<int>(p2.x), static_cast<int>(p2.y)));

    car_zone_ = black_zone_;

    if (camera_order_.size() == 4) {
        generateBlendAreas(black_zone_);
    }

    CVLog::Print(
            "[BEV] buildAlphaFusion: %zu cameras, black_zone=(%d,%d,%d,%d), "
            "front=%.3f back=%.3f",
            camera_order_.size(), black_zone_.x, black_zone_.y,
            black_zone_.width, black_zone_.height, front_to_sensing,
            back_to_sensing);

    alpha_fusion_.reset();
    const BevRemapMode active = getActiveRemapMode();
    if (active == BevRemapMode::CUDA || active == BevRemapMode::OpenCL) {
        alpha_fusion_ = std::make_unique<BevAlphaFusion>(
                config_.bev_size, camera_order_, bev_weights_, active);
        if (alpha_fusion_ && alpha_fusion_->valid()) {
            CVLog::Print("[BEV] alpha fusion backend: %s",
                         BevRemapper::modeName(alpha_fusion_->activeMode()));
        } else {
            alpha_fusion_.reset();
            CVLog::Warning(
                    "[BEV] GPU alpha fusion init failed, using CPU blend");
        }
    }
}

void BirdEyeView::updateExtrinsics(
        const std::map<std::string, Eigen::Isometry3d>& extrinsics) {
    last_extrinsics_ = extrinsics;
    buildRemapMaps(extrinsics);
    buildAlphaFusion();
}

cv::Mat BirdEyeView::generate(const std::map<std::string, cv::Mat>& images) {
    if (!initialized_) {
        CVLog::Warning("[BEV] generate: not initialized!");
        return cv::Mat();
    }

    bev_image_ = cv::Mat::zeros(config_.bev_size, CV_8UC3);

    struct RemapTask {
        std::string slot;
        cv::Mat img_warp;
        bool ok = false;
    };

    std::vector<RemapTask> tasks;
    tasks.reserve(camera_order_.size());

    auto remap_one = [&](const std::string& slot) -> RemapTask {
        RemapTask task;
        task.slot = slot;
        const auto map_it = slot_to_source_.find(slot);
        const std::string source =
                map_it != slot_to_source_.end() ? map_it->second : slot;

        auto it_img = images.find(source);
        if (it_img == images.end() || it_img->second.empty()) return task;

        auto cam_model = camera_system_.getCamera(source);
        if (!cam_model) return task;

        cv::Mat img = it_img->second;
        int target_w = cam_model->getParameters().width;
        int target_h = cam_model->getParameters().height;
        if (img.cols != target_w || img.rows != target_h) {
            cv::resize(img, img, cv::Size(target_w, target_h));
        }

        auto backend_it = remap_backends_.find(slot);
        if (backend_it != remap_backends_.end() && backend_it->second) {
            task.img_warp = backend_it->second->remap(img);
        } else if (remap_map1_.count(slot) && remap_map2_.count(slot)) {
            cv::remap(img, task.img_warp, remap_map1_[slot], remap_map2_[slot],
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        }
        task.ok = !task.img_warp.empty();
        return task;
    };

    const bool cpu_parallel = config_.use_parallel_remap &&
                              (config_.remap_mode == BevRemapMode::CPU ||
                               remap_backends_.empty() ||
                               (remap_backends_.begin()->second &&
                                remap_backends_.begin()->second->activeMode() ==
                                        BevRemapMode::CPU));

    int matched = 0;
    if (cpu_parallel && camera_order_.size() > 1) {
        std::vector<std::future<RemapTask>> futures;
        futures.reserve(camera_order_.size());
        for (const auto& slot : camera_order_) {
            futures.push_back(std::async(std::launch::async, remap_one, slot));
        }
        for (auto& fut : futures) {
            tasks.push_back(fut.get());
        }
    } else {
        for (const auto& slot : camera_order_) {
            tasks.push_back(remap_one(slot));
        }
    }

    std::map<std::string, cv::Mat> bigger_bev_images;
    std::map<std::string, cv::Mat> smaller_bev_images;

    for (auto& task : tasks) {
        if (!task.ok) continue;
        ++matched;
        const std::string& slot = task.slot;
        if (bigger_bev_masks_.count(slot)) {
            bigger_bev_images[slot] = task.img_warp & bigger_bev_masks_[slot];
        }
        if (smaller_bev_masks_.count(slot)) {
            smaller_bev_images[slot] = task.img_warp & smaller_bev_masks_[slot];
        }
    }

    CVLog::Print("[BEV] generate: %d cameras matched, order=%zu, input=%zu",
                 matched, camera_order_.size(), images.size());

    if (matched == 0) {
        // No source image matched the current slot map — usually a stale
        // m_images after a mode/sensor switch. Skip fusion entirely to avoid
        // the misleading "GPU alpha fusion failed" warning and let the
        // caller (e.g. updateBevView) refresh the frame.
        CVLog::Warning(
                "[BEV] generate: 0 cameras matched (slot map and input "
                "images out of sync), skipping fusion");
        return bev_image_;
    }

    bool use_gpu_fusion = alpha_fusion_ && alpha_fusion_->valid() &&
                          (alpha_fusion_->activeMode() == BevRemapMode::CUDA ||
                           alpha_fusion_->activeMode() == BevRemapMode::OpenCL);

    if (use_gpu_fusion) {
        std::map<std::string, cv::Mat> full_warped;
        for (auto& task : tasks) {
            if (!task.ok) continue;
            full_warped[task.slot] = task.img_warp;
        }
        // full_warped cannot be empty here because matched > 0.
        if (!alpha_fusion_->fuse(full_warped, bev_image_)) {
            CVLog::Warning(
                    "[BEV] GPU alpha fusion failed (%s), fallback CPU blend",
                    BevRemapper::modeName(alpha_fusion_->activeMode()));
            use_gpu_fusion = false;
            bev_image_.setTo(0);
        }
    }

    if (!use_gpu_fusion) {
        if (camera_order_.size() == 4) {
            for (const auto& [id, area] : bev_blend_areas_) {
                const auto& pair = area.blend_pair;
                cv::Rect zone = area.zone;
                if (zone.width <= 0 || zone.height <= 0) continue;

                zone &= cv::Rect(0, 0, config_.bev_size.width,
                                 config_.bev_size.height);
                if (zone.width <= 0 || zone.height <= 0) continue;

                if (pair.first == pair.second) {
                    if (bigger_bev_images.count(pair.first) == 0) continue;
                    cv::Mat roi_src = bigger_bev_images[pair.first](zone);
                    cv::Mat roi_dst = bev_image_(zone);
                    roi_src.copyTo(roi_dst);
                } else {
                    if (bigger_bev_images.count(pair.first) == 0 ||
                        bigger_bev_images.count(pair.second) == 0)
                        continue;
                    cv::Mat roi_a = bigger_bev_images[pair.first](zone);
                    cv::Mat roi_b = bigger_bev_images[pair.second](zone);
                    cv::Mat w_a = bev_weights_[pair.first](zone);
                    cv::Mat w_b = bev_weights_[pair.second](zone);
                    cv::Mat roi_dst = bev_image_(zone);
                    cv::blendLinear(roi_a, roi_b, w_a, w_b, roi_dst);
                }
            }
        } else {
            for (const auto& [name, img] : smaller_bev_images) {
                bev_image_ += img;
            }
            for (const auto& [name, weight] : bev_weights_) {
                if (bigger_bev_images.count(name) == 0) continue;
                const cv::Mat& overlap = bev_overlap_masks_[name];

                for (int i = 0; i < overlap.rows; i++) {
                    for (int j = 0; j < overlap.cols; j++) {
                        if (overlap.at<uchar>(i, j)) {
                            float w = weight.at<float>(i, j);
                            cv::Vec3b src =
                                    bigger_bev_images[name].at<cv::Vec3b>(i, j);
                            cv::Vec3b& dst = bev_image_.at<cv::Vec3b>(i, j);
                            dst[0] = cv::saturate_cast<uchar>(dst[0] +
                                                              src[0] * w);
                            dst[1] = cv::saturate_cast<uchar>(dst[1] +
                                                              src[1] * w);
                            dst[2] = cv::saturate_cast<uchar>(dst[2] +
                                                              src[2] * w);
                        }
                    }
                }
            }
        }
    }

    if (config_.show_car_outline && black_zone_.width > 0 &&
        black_zone_.height > 0) {
        cv::Rect safe = black_zone_ & cv::Rect(0, 0, config_.bev_size.width,
                                               config_.bev_size.height);
        if (safe.width > 0 && safe.height > 0) {
            bev_image_(safe).setTo(cv::Scalar(40, 40, 40));
            cv::rectangle(bev_image_, safe, cv::Scalar(0, 200, 255), 2);
            cv::putText(bev_image_, "CAR",
                        cv::Point(safe.x + safe.width / 2 - 15,
                                  safe.y + safe.height / 2 + 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 255),
                        1);
        }
    }

    if (config_.rotate_cw) {
        last_pre_rotate_size_ = bev_image_.size();
        cv::Mat rotated;
        cv::rotate(bev_image_, rotated, cv::ROTATE_90_CLOCKWISE);
        bev_image_ = rotated;
    } else {
        last_pre_rotate_size_ = bev_image_.size();
    }

    return bev_image_;
}

cv::Mat BirdEyeView::generateSingleIPM(const std::string& camera_name,
                                       const cv::Mat& image) {
    if (!initialized_) return cv::Mat();
    if (remap_map1_.find(camera_name) == remap_map1_.end()) return cv::Mat();

    auto cam_model = camera_system_.getCamera(camera_name);
    if (!cam_model) return cv::Mat();

    cv::Mat img = image;
    int target_w = cam_model->getParameters().width;
    int target_h = cam_model->getParameters().height;
    if (img.cols != target_w || img.rows != target_h) {
        cv::resize(img, img, cv::Size(target_w, target_h));
    }

    cv::Mat warped;
    cv::remap(img, warped, remap_map1_[camera_name], remap_map2_[camera_name],
              cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    if (bigger_bev_masks_.find(camera_name) != bigger_bev_masks_.end()) {
        warped &= bigger_bev_masks_[camera_name];
    }

    return warped;
}

cv::Point2f BirdEyeView::groundToBevPixel(
        const Eigen::Vector3d& ground_pt) const {
    Eigen::Isometry3d iso_bev_ground = iso_ground_bev_.inverse();
    Eigen::Vector3d pt_bev =
            iso_bev_ground.linear() * ground_pt + iso_bev_ground.translation();
    if (std::fabs(pt_bev(2)) < 1e-6) return cv::Point2f(-1, -1);

    float u =
            K_bev_.at<float>(0, 0) * static_cast<float>(pt_bev(0) / pt_bev(2)) +
            K_bev_.at<float>(0, 2);
    float v =
            K_bev_.at<float>(1, 1) * static_cast<float>(pt_bev(1) / pt_bev(2)) +
            K_bev_.at<float>(1, 2);
    return cv::Point2f(u, v);
}

Eigen::Vector3d BirdEyeView::bevPixelToGround(const cv::Point2f& px) const {
    int iu = static_cast<int>(px.x);
    int iv = static_cast<int>(px.y);
    if (iu < 0 || iu >= config_.bev_size.width || iv < 0 ||
        iv >= config_.bev_size.height) {
        return Eigen::Vector3d::Zero();
    }
    cv::Vec3f v = bev_scene_.at<cv::Vec3f>(iv, iu);
    return Eigen::Vector3d(v[0], v[1], v[2]);
}

bool BirdEyeView::unprojectBevPixelToGround(const cv::Point& bev_px,
                                            Eigen::Vector3d& ground_pt) const {
    if (bev_px.x < 0 || bev_px.y < 0 || bev_px.x >= config_.bev_size.width ||
        bev_px.y >= config_.bev_size.height) {
        return false;
    }

    const double fx = K_bev_.at<float>(0, 0);
    const double fy = K_bev_.at<float>(1, 1);
    const double cx = K_bev_.at<float>(0, 2);
    const double cy = K_bev_.at<float>(1, 2);
    const double x = (static_cast<double>(bev_px.x) - cx) / fx;
    const double y = (static_cast<double>(bev_px.y) - cy) / fy;
    Eigen::Vector3d dir_cam(x, y, 1.0);
    dir_cam.normalize();

    Eigen::Isometry3d iso_bev_ground = iso_ground_bev_.inverse();
    const Eigen::Vector3d t = iso_bev_ground.translation();
    const Eigen::Vector3d dir_g = iso_bev_ground.linear() * dir_cam;
    const double denom = dir_g(2);
    if (std::abs(denom) < 1e-9) return false;

    const double s = -t(2) / denom;
    ground_pt = t + s * dir_g;
    return true;
}

bool BirdEyeView::projectGroundToBevPixel(const Eigen::Vector3d& ground_pt,
                                          cv::Point& bev_px) const {
    cv::Point2f px = groundToBevPixel(ground_pt);
    if (px.x < 0 || px.y < 0 || px.x >= config_.bev_size.width ||
        px.y >= config_.bev_size.height) {
        return false;
    }
    bev_px = cv::Point(static_cast<int>(std::round(px.x)),
                       static_cast<int>(std::round(px.y)));
    return true;
}

}  // namespace mcalib

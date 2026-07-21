// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "BevAlphaFusion.h"
#include "BevRemapBackend.h"
#include "CalibTypes.h"
#include "CameraModel.h"

namespace mcalib {

struct BlendArea {
    cv::Rect zone;
    std::pair<std::string, std::string> blend_pair;
};

class BirdEyeView {
public:
    struct Config {
        cv::Size bev_size;
        double virtual_camera_height;
        double virtual_camera_focal;
        double focal_scale;
        double sensing_height;
        double car_width;
        double car_length;
        bool show_car_outline;
        bool rotate_cw;
        BevRemapMode remap_mode;
        bool use_parallel_remap;

        Config()
            : bev_size(960, 1600),
              virtual_camera_height(3.44),
              virtual_camera_focal(162.6),
              focal_scale(1.0),
              sensing_height(1.72),
              car_width(1.93),
              car_length(4.77),
              show_car_outline(true),
              rotate_cw(false),
              remap_mode(BevRemapMode::Auto),
              use_parallel_remap(true) {}
    };

    BirdEyeView();
    ~BirdEyeView();

    bool init(const VehicleCalibConfig& config,
              const Config& bev_config = Config());

    void updateExtrinsics(
            const std::map<std::string, Eigen::Isometry3d>& extrinsics);

    cv::Mat generate(const std::map<std::string, cv::Mat>& images);

    cv::Mat generateSingleIPM(const std::string& camera_name,
                              const cv::Mat& image);

    const cv::Mat& getLastBEV() const { return bev_image_; }
    const cv::Mat& getBevIntrinsic() const { return K_bev_; }
    cv::Size getBevSize() const { return config_.bev_size; }
    double getSensingHeight() const { return config_.sensing_height; }

    cv::Point2f groundToBevPixel(const Eigen::Vector3d& ground_pt) const;
    Eigen::Vector3d bevPixelToGround(const cv::Point2f& px) const;
    bool unprojectBevPixelToGround(const cv::Point& bev_px,
                                   Eigen::Vector3d& ground_pt) const;
    bool projectGroundToBevPixel(const Eigen::Vector3d& ground_pt,
                                 cv::Point& bev_px) const;

    const CameraSystem& getCameraSystem() const { return camera_system_; }
    cv::Size getLastPreRotateSize() const { return last_pre_rotate_size_; }
    const Eigen::Isometry3d& getIsoGroundBev() const { return iso_ground_bev_; }

    bool setCameraSlotMap(
            const std::map<std::string, std::string>& slot_to_source);
    bool setCameraSubset(const std::vector<std::string>& cameras);
    void resetCameraSubset();
    void setFocalScale(double scale);
    void setRemapMode(BevRemapMode mode);
    BevRemapMode getRemapMode() const { return config_.remap_mode; }
    BevRemapMode getActiveRemapMode() const;

private:
    void buildBevScene();
    void buildRemapMaps(
            const std::map<std::string, Eigen::Isometry3d>& extrinsics);
    void buildAlphaFusion();
    void generateCameraPairs();
    void generatePriorMasks(double front_to_sensing, double back_to_sensing);
    void generateWeights();
    void generateBlendAreas(cv::Rect car_zone);

    static void getAngleRange4(const std::string& name,
                               double& start,
                               double& end);
    static void getAngleRange6(const std::string& name,
                               double& start,
                               double& end);

    Config config_;
    VehicleCalibConfig calib_config_;
    CameraSystem camera_system_;

    cv::Mat K_bev_;
    Eigen::Isometry3d iso_sensing_ground_;
    Eigen::Isometry3d iso_ground_bev_;
    Eigen::Isometry3d iso_sensing_bev_;

    cv::Mat bev_scene_;

    std::map<std::string, cv::Mat> remap_map1_;
    std::map<std::string, cv::Mat> remap_map2_;
    std::map<std::string, std::unique_ptr<BevRemapper>> remap_backends_;
    std::unique_ptr<BevAlphaFusion> alpha_fusion_;

    std::map<std::string, cv::Mat> bigger_bev_masks_;
    std::map<std::string, cv::Mat> smaller_bev_masks_;
    std::map<std::string, cv::Mat> bev_overlap_masks_;
    std::map<std::string, cv::Mat> bev_weights_;
    std::map<std::string, BlendArea> bev_blend_areas_;

    std::vector<std::pair<std::string, std::string>> camera_pairs_;

    cv::Mat bev_image_;
    cv::Rect car_zone_;
    cv::Rect black_zone_;

    bool initialized_ = false;
    std::vector<std::string> camera_order_;
    std::vector<std::string> all_camera_order_;
    std::map<std::string, std::string> slot_to_source_;
    std::map<std::string, Eigen::Isometry3d> last_extrinsics_;
    cv::Size last_pre_rotate_size_;
};

}  // namespace mcalib

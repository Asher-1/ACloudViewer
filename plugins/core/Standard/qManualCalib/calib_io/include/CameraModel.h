// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>
#include <map>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <string>

#include "CalibTypes.h"

namespace mcalib {

// Camera model hierarchy ported from calibration/modules/camera_model.
// Supports PINHOLE, KANNALA_BRANDT (equidistant), MEI (cata), FULLPINHOLE.
class CameraModel {
public:
    virtual ~CameraModel() = default;

    virtual CameraIntrinsic::ModelType modelType() const = 0;
    virtual const CameraIntrinsic& getParameters() const = 0;

    virtual void spaceToPlane(const Eigen::Vector3d& P,
                              Eigen::Vector2d& p) const = 0;
    virtual void liftProjective(const Eigen::Vector2d& p,
                                Eigen::Vector3d& P) const = 0;

    virtual cv::Mat initUndistortRectifyMap(
            cv::Mat& map1,
            cv::Mat& map2,
            float fx = -1.0f,
            float fy = -1.0f,
            cv::Size imageSize = cv::Size(0, 0),
            float cx = -1.0f,
            float cy = -1.0f,
            cv::Mat rmat = cv::Mat::eye(3, 3, CV_32F)) const;

    virtual void initUndistortMap(cv::Mat& map1,
                                  cv::Mat& map2,
                                  double fScale = 1.0) const;
};

using CameraModelPtr = std::shared_ptr<CameraModel>;

class PinholeCameraModel : public CameraModel {
public:
    explicit PinholeCameraModel(const CameraIntrinsic& intrinsic);

    CameraIntrinsic::ModelType modelType() const override;
    const CameraIntrinsic& getParameters() const override;
    void spaceToPlane(const Eigen::Vector3d& P,
                      Eigen::Vector2d& p) const override;
    void liftProjective(const Eigen::Vector2d& p,
                        Eigen::Vector3d& P) const override;
    void initUndistortMap(cv::Mat& map1,
                          cv::Mat& map2,
                          double fScale = 1.0) const override;

private:
    void distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u) const;

    CameraIntrinsic params_;
    double m_inv_K11 = 1.0, m_inv_K13 = 0.0;
    double m_inv_K22 = 1.0, m_inv_K23 = 0.0;
    bool m_noDistortion = true;
};

class EquidistantCameraModel : public CameraModel {
public:
    explicit EquidistantCameraModel(const CameraIntrinsic& intrinsic);

    CameraIntrinsic::ModelType modelType() const override;
    const CameraIntrinsic& getParameters() const override;
    void spaceToPlane(const Eigen::Vector3d& P,
                      Eigen::Vector2d& p) const override;
    void liftProjective(const Eigen::Vector2d& p,
                        Eigen::Vector3d& P) const override;
    void initUndistortMap(cv::Mat& map1,
                          cv::Mat& map2,
                          double fScale = 1.0) const override;

private:
    static double r_func(
            double k2, double k3, double k4, double k5, double theta);
    void backprojectSymmetric(const Eigen::Vector2d& p_u,
                              double& theta,
                              double& phi) const;

    CameraIntrinsic params_;
    double m_inv_K11 = 1.0, m_inv_K13 = 0.0;
    double m_inv_K22 = 1.0, m_inv_K23 = 0.0;
};

class CataCameraModel : public CameraModel {
public:
    explicit CataCameraModel(const CameraIntrinsic& intrinsic);

    CameraIntrinsic::ModelType modelType() const override;
    const CameraIntrinsic& getParameters() const override;
    void spaceToPlane(const Eigen::Vector3d& P,
                      Eigen::Vector2d& p) const override;
    void liftProjective(const Eigen::Vector2d& p,
                        Eigen::Vector3d& P) const override;
    void initUndistortMap(cv::Mat& map1,
                          cv::Mat& map2,
                          double fScale = 1.0) const override;

private:
    void distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u) const;

    CameraIntrinsic params_;
    double m_inv_K11 = 1.0, m_inv_K13 = 0.0;
    double m_inv_K22 = 1.0, m_inv_K23 = 0.0;
    bool m_noDistortion = true;
};

class FullpinholeCameraModel : public CameraModel {
public:
    explicit FullpinholeCameraModel(const CameraIntrinsic& intrinsic);

    CameraIntrinsic::ModelType modelType() const override;
    const CameraIntrinsic& getParameters() const override;
    void spaceToPlane(const Eigen::Vector3d& P,
                      Eigen::Vector2d& p) const override;
    void liftProjective(const Eigen::Vector2d& p,
                        Eigen::Vector3d& P) const override;
    void initUndistortMap(cv::Mat& map1,
                          cv::Mat& map2,
                          double fScale = 1.0) const override;

private:
    void distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u) const;

    CameraIntrinsic params_;
    double m_inv_K11 = 1.0, m_inv_K13 = 0.0;
    double m_inv_K22 = 1.0, m_inv_K23 = 0.0;
    bool m_noDistortion = true;
};

class CameraModelFactory {
public:
    static CameraModelPtr create(const CameraIntrinsic& intrinsic);
};

class CameraSystem {
public:
    CameraSystem() = default;

    bool loadFromConfig(const VehicleCalibConfig& config);

    CameraModelPtr getCamera(const std::string& name) const;
    std::map<std::string, CameraModelPtr> getCameras() const;

private:
    std::map<std::string, CameraModelPtr> cameras_;
};

}  // namespace mcalib

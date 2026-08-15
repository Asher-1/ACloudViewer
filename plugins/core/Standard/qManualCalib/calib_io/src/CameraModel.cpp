// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CameraModel.h"

#include <CVLog.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>

namespace mcalib {
namespace {

bool is_equal(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) < eps;
}

bool is_zero(double v, double eps = 1e-12) { return std::fabs(v) < eps; }

void updateInverseK(const CameraIntrinsic& params,
                    double& inv_K11,
                    double& inv_K13,
                    double& inv_K22,
                    double& inv_K23) {
    inv_K11 = (params.fx != 0.0) ? 1.0 / params.fx : 1.0;
    inv_K13 = -params.cx / ((params.fx != 0.0) ? params.fx : 1.0);
    inv_K22 = (params.fy != 0.0) ? 1.0 / params.fy : 1.0;
    inv_K23 = -params.cy / ((params.fy != 0.0) ? params.fy : 1.0);
}

void mergeMaps(const cv::Mat& mapX,
               const cv::Mat& mapY,
               cv::Mat& map1,
               cv::Mat& map2) {
    std::vector<cv::Mat> maps = {mapX, mapY};
    cv::merge(maps, map1);
    map2 = cv::Mat();
}

cv::Mat buildRectifyK(float fx,
                      float fy,
                      cv::Size imageSize,
                      float cx,
                      float cy,
                      const CameraIntrinsic& params) {
    Eigen::Matrix3f K_rect;
    if (is_equal(cx, -1.0f) && is_equal(cy, -1.0f)) {
        K_rect << fx, 0, imageSize.width / 2.0f, 0, fy, imageSize.height / 2.0f,
                0, 0, 1;
    } else {
        K_rect << fx, 0, cx, 0, fy, cy, 0, 0, 1;
    }

    if (is_equal(fx, -1.0f) || is_equal(fy, -1.0f)) {
        K_rect(0, 0) = static_cast<float>(params.fx);
        K_rect(1, 1) = static_cast<float>(params.fy);
    } else if (fx < 0 && fy < 0) {
        K_rect(0, 0) = static_cast<float>(params.fx) * std::fabs(fx);
        K_rect(1, 1) = static_cast<float>(params.fy) * std::fabs(fy);
    }

    cv::Mat K_rect_cv;
    cv::eigen2cv(K_rect, K_rect_cv);
    return K_rect_cv;
}

}  // namespace

// ---------------------------------------------------------------------------
// CameraModel base
// ---------------------------------------------------------------------------

cv::Mat CameraModel::initUndistortRectifyMap(cv::Mat& map1,
                                             cv::Mat& map2,
                                             float fx,
                                             float fy,
                                             cv::Size imageSize,
                                             float cx,
                                             float cy,
                                             cv::Mat rmat) const {
    if (imageSize == cv::Size(0, 0)) {
        imageSize = getParameters().getImageSize();
    }

    cv::Mat mapX = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);

    Eigen::Matrix3f R, R_inv;
    cv::cv2eigen(rmat, R);
    R_inv = R.inverse();

    Eigen::Matrix3f K_rect;
    if (is_equal(cx, -1.0f) && is_equal(cy, -1.0f)) {
        K_rect << fx, 0, imageSize.width / 2.0f, 0, fy, imageSize.height / 2.0f,
                0, 0, 1;
    } else {
        K_rect << fx, 0, cx, 0, fy, cy, 0, 0, 1;
    }

    const auto& params = getParameters();
    if (is_equal(fx, -1.0f) || is_equal(fy, -1.0f)) {
        K_rect(0, 0) = static_cast<float>(params.fx);
        K_rect(1, 1) = static_cast<float>(params.fy);
    } else if (fx < 0 && fy < 0) {
        K_rect(0, 0) = static_cast<float>(params.fx) * std::fabs(fx);
        K_rect(1, 1) = static_cast<float>(params.fy) * std::fabs(fy);
    }

    Eigen::Matrix3f K_rect_inv = K_rect.inverse();

    for (int v = 0; v < imageSize.height; ++v) {
        for (int u = 0; u < imageSize.width; ++u) {
            Eigen::Vector3f xo;
            xo << static_cast<float>(u), static_cast<float>(v), 1.0f;

            Eigen::Vector3f uo = R_inv * K_rect_inv * xo;

            Eigen::Vector2d p;
            spaceToPlane(uo.cast<double>(), p);

            mapX.at<float>(v, u) = static_cast<float>(p(0));
            mapY.at<float>(v, u) = static_cast<float>(p(1));
        }
    }

    mergeMaps(mapX, mapY, map1, map2);
    return buildRectifyK(fx, fy, imageSize, cx, cy, params);
}

void CameraModel::initUndistortMap(cv::Mat& map1,
                                   cv::Mat& map2,
                                   double fScale) const {
    const auto& params = getParameters();
    cv::Size imageSize(params.width, params.height);

    cv::Mat mapX = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize, CV_32F);

    double inv_K11, inv_K13, inv_K22, inv_K23;
    updateInverseK(params, inv_K11, inv_K13, inv_K22, inv_K23);

    for (int v = 0; v < imageSize.height; ++v) {
        for (int u = 0; u < imageSize.width; ++u) {
            double mx_u = inv_K11 / fScale * u + inv_K13 / fScale;
            double my_u = inv_K22 / fScale * v + inv_K23 / fScale;

            Eigen::Vector3d ray;
            liftProjective(Eigen::Vector2d(mx_u * params.fx, my_u * params.fy),
                           ray);

            Eigen::Vector2d p;
            spaceToPlane(ray, p);

            mapX.at<float>(v, u) = static_cast<float>(p(0));
            mapY.at<float>(v, u) = static_cast<float>(p(1));
        }
    }

    mergeMaps(mapX, mapY, map1, map2);
}

// ---------------------------------------------------------------------------
// Pinhole
// ---------------------------------------------------------------------------

PinholeCameraModel::PinholeCameraModel(const CameraIntrinsic& intrinsic) {
    params_ = intrinsic;
    updateInverseK(params_, m_inv_K11, m_inv_K13, m_inv_K22, m_inv_K23);
    m_noDistortion = is_zero(params_.k1) && is_zero(params_.k2) &&
                     is_zero(params_.p1) && is_zero(params_.p2);
}

CameraIntrinsic::ModelType PinholeCameraModel::modelType() const {
    return CameraIntrinsic::PINHOLE;
}

const CameraIntrinsic& PinholeCameraModel::getParameters() const {
    return params_;
}

void PinholeCameraModel::distortion(const Eigen::Vector2d& p_u,
                                    Eigen::Vector2d& d_u) const {
    const double mx2_u = p_u(0) * p_u(0);
    const double my2_u = p_u(1) * p_u(1);
    const double mxy_u = p_u(0) * p_u(1);
    const double rho2_u = mx2_u + my2_u;
    const double rad_dist_u =
            params_.k1 * rho2_u + params_.k2 * rho2_u * rho2_u;
    d_u << p_u(0) * rad_dist_u + 2.0 * params_.p1 * mxy_u +
                    params_.p2 * (rho2_u + 2.0 * mx2_u),
            p_u(1) * rad_dist_u + 2.0 * params_.p2 * mxy_u +
                    params_.p1 * (rho2_u + 2.0 * my2_u);
}

void PinholeCameraModel::spaceToPlane(const Eigen::Vector3d& P,
                                      Eigen::Vector2d& p) const {
    if (std::fabs(P(2)) < 1e-10) {
        p = Eigen::Vector2d(-1, -1);
        return;
    }

    Eigen::Vector2d p_u(P(0) / P(2), P(1) / P(2));
    Eigen::Vector2d p_d;
    if (m_noDistortion) {
        p_d = p_u;
    } else {
        Eigen::Vector2d d_u;
        distortion(p_u, d_u);
        p_d = p_u + d_u;
    }

    p << params_.fx * p_d(0) + params_.cx, params_.fy * p_d(1) + params_.cy;
}

void PinholeCameraModel::liftProjective(const Eigen::Vector2d& p,
                                        Eigen::Vector3d& P) const {
    const double mx_d = m_inv_K11 * p(0) + m_inv_K13;
    const double my_d = m_inv_K22 * p(1) + m_inv_K23;

    double mx_u = mx_d;
    double my_u = my_d;
    if (!m_noDistortion) {
        constexpr int n = 8;
        Eigen::Vector2d d_u;
        distortion(Eigen::Vector2d(mx_d, my_d), d_u);
        mx_u = mx_d - d_u(0);
        my_u = my_d - d_u(1);
        for (int i = 1; i < n; ++i) {
            distortion(Eigen::Vector2d(mx_u, my_u), d_u);
            mx_u = mx_d - d_u(0);
            my_u = my_d - d_u(1);
        }
    }

    P << mx_u, my_u, 1.0;
}

void PinholeCameraModel::initUndistortMap(cv::Mat& map1,
                                          cv::Mat& map2,
                                          double fScale) const {
    cv::Size imageSize(params_.width, params_.height);
    cv::Mat mapX = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize, CV_32F);

    for (int v = 0; v < imageSize.height; ++v) {
        for (int u = 0; u < imageSize.width; ++u) {
            const double mx_u = m_inv_K11 / fScale * u + m_inv_K13 / fScale;
            const double my_u = m_inv_K22 / fScale * v + m_inv_K23 / fScale;

            Eigen::Vector3d ray;
            ray << mx_u, my_u, 1.0;

            Eigen::Vector2d p;
            spaceToPlane(ray, p);

            mapX.at<float>(v, u) = static_cast<float>(p(0));
            mapY.at<float>(v, u) = static_cast<float>(p(1));
        }
    }

    mergeMaps(mapX, mapY, map1, map2);
}

// ---------------------------------------------------------------------------
// Kannala-Brandt / Equidistant
// ---------------------------------------------------------------------------

EquidistantCameraModel::EquidistantCameraModel(
        const CameraIntrinsic& intrinsic) {
    params_ = intrinsic;
    updateInverseK(params_, m_inv_K11, m_inv_K13, m_inv_K22, m_inv_K23);
}

CameraIntrinsic::ModelType EquidistantCameraModel::modelType() const {
    return CameraIntrinsic::KANNALA_BRANDT;
}

const CameraIntrinsic& EquidistantCameraModel::getParameters() const {
    return params_;
}

double EquidistantCameraModel::r_func(
        double k2, double k3, double k4, double k5, double theta) {
    const double theta2 = theta * theta;
    const double theta3 = theta2 * theta;
    const double theta5 = theta2 * theta3;
    const double theta7 = theta2 * theta5;
    const double theta9 = theta2 * theta7;
    return theta + k2 * theta3 + k3 * theta5 + k4 * theta7 + k5 * theta9;
}

void EquidistantCameraModel::backprojectSymmetric(const Eigen::Vector2d& p_u,
                                                  double& theta,
                                                  double& phi) const {
    constexpr double tol = 1e-10;
    const double p_u_norm = p_u.norm();

    phi = (p_u_norm < 1e-10) ? 0.0 : std::atan2(p_u(1), p_u(0));

    if (p_u_norm < 1e-10) {
        theta = 0.0;
        return;
    }

    theta = p_u_norm;
    for (int i = 0; i < 100; ++i) {
        const double theta2 = theta * theta;
        const double r_theta =
                r_func(params_.k1, params_.k2, params_.k3, params_.k4, theta);
        const double dr_theta =
                1.0 + 3.0 * params_.k1 * theta2 +
                5.0 * params_.k2 * theta2 * theta2 +
                7.0 * params_.k3 * theta2 * theta2 * theta2 +
                9.0 * params_.k4 * theta2 * theta2 * theta2 * theta2;

        const double d = r_theta - p_u_norm;
        theta -= d / dr_theta;
        if (std::fabs(d) < tol) break;
    }
}

void EquidistantCameraModel::spaceToPlane(const Eigen::Vector3d& P,
                                          Eigen::Vector2d& p) const {
    const double len = P.norm();
    if (len < 1e-12) {
        p = Eigen::Vector2d(-1, -1);
        return;
    }

    const double theta = std::acos(P(2) / len);
    const double phi = std::atan2(P(1), P(0));

    const Eigen::Vector2d p_u =
            r_func(params_.k1, params_.k2, params_.k3, params_.k4, theta) *
            Eigen::Vector2d(std::cos(phi), std::sin(phi));

    p << params_.fx * p_u(0) + params_.cx, params_.fy * p_u(1) + params_.cy;
}

void EquidistantCameraModel::liftProjective(const Eigen::Vector2d& p,
                                            Eigen::Vector3d& P) const {
    const Eigen::Vector2d p_u(m_inv_K11 * p(0) + m_inv_K13,
                              m_inv_K22 * p(1) + m_inv_K23);

    double theta = 0.0;
    double phi = 0.0;
    backprojectSymmetric(p_u, theta, phi);

    P << std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi),
            std::cos(theta);
}

void EquidistantCameraModel::initUndistortMap(cv::Mat& map1,
                                              cv::Mat& map2,
                                              double fScale) const {
    cv::Size imageSize(params_.width, params_.height);
    cv::Mat mapX = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize, CV_32F);

    for (int v = 0; v < imageSize.height; ++v) {
        for (int u = 0; u < imageSize.width; ++u) {
            const double mx_u = m_inv_K11 / fScale * u + m_inv_K13 / fScale;
            const double my_u = m_inv_K22 / fScale * v + m_inv_K23 / fScale;

            double theta = 0.0;
            double phi = 0.0;
            backprojectSymmetric(Eigen::Vector2d(mx_u, my_u), theta, phi);

            Eigen::Vector3d ray;
            ray << std::sin(theta) * std::cos(phi),
                    std::sin(theta) * std::sin(phi), std::cos(theta);

            Eigen::Vector2d p;
            spaceToPlane(ray, p);

            mapX.at<float>(v, u) = static_cast<float>(p(0));
            mapY.at<float>(v, u) = static_cast<float>(p(1));
        }
    }

    mergeMaps(mapX, mapY, map1, map2);
}

// ---------------------------------------------------------------------------
// MEI / Cata
// ---------------------------------------------------------------------------

CataCameraModel::CataCameraModel(const CameraIntrinsic& intrinsic) {
    params_ = intrinsic;
    updateInverseK(params_, m_inv_K11, m_inv_K13, m_inv_K22, m_inv_K23);
    m_noDistortion = is_zero(params_.k1) && is_zero(params_.k2) &&
                     is_zero(params_.p1) && is_zero(params_.p2);
}

CameraIntrinsic::ModelType CataCameraModel::modelType() const {
    return CameraIntrinsic::MEI;
}

const CameraIntrinsic& CataCameraModel::getParameters() const {
    return params_;
}

void CataCameraModel::distortion(const Eigen::Vector2d& p_u,
                                 Eigen::Vector2d& d_u) const {
    const double mx2_u = p_u(0) * p_u(0);
    const double my2_u = p_u(1) * p_u(1);
    const double mxy_u = p_u(0) * p_u(1);
    const double rho2_u = mx2_u + my2_u;
    const double rad_dist_u =
            params_.k1 * rho2_u + params_.k2 * rho2_u * rho2_u;
    d_u << p_u(0) * rad_dist_u + 2.0 * params_.p1 * mxy_u +
                    params_.p2 * (rho2_u + 2.0 * mx2_u),
            p_u(1) * rad_dist_u + 2.0 * params_.p2 * mxy_u +
                    params_.p1 * (rho2_u + 2.0 * my2_u);
}

void CataCameraModel::spaceToPlane(const Eigen::Vector3d& P,
                                   Eigen::Vector2d& p) const {
    const double norm = P.norm();
    if (norm < 1e-12) {
        p = Eigen::Vector2d(-1, -1);
        return;
    }

    const double z = P(2) + params_.xi * norm;
    if (std::fabs(z) < 1e-10) {
        p = Eigen::Vector2d(-1, -1);
        return;
    }

    Eigen::Vector2d p_u(P(0) / z, P(1) / z);
    Eigen::Vector2d p_d;
    if (m_noDistortion) {
        p_d = p_u;
    } else {
        Eigen::Vector2d d_u;
        distortion(p_u, d_u);
        p_d = p_u + d_u;
    }

    p << params_.fx * p_d(0) + params_.cx, params_.fy * p_d(1) + params_.cy;
}

void CataCameraModel::liftProjective(const Eigen::Vector2d& p,
                                     Eigen::Vector3d& P) const {
    const double mx_d = m_inv_K11 * p(0) + m_inv_K13;
    const double my_d = m_inv_K22 * p(1) + m_inv_K23;

    double mx_u = mx_d;
    double my_u = my_d;
    if (!m_noDistortion) {
        constexpr int n = 8;
        Eigen::Vector2d d_u;
        distortion(Eigen::Vector2d(mx_d, my_d), d_u);
        mx_u = mx_d - d_u(0);
        my_u = my_d - d_u(1);
        for (int i = 1; i < n; ++i) {
            distortion(Eigen::Vector2d(mx_u, my_u), d_u);
            mx_u = mx_d - d_u(0);
            my_u = my_d - d_u(1);
        }
    }

    const double xi = params_.xi;
    if (is_equal(static_cast<float>(xi), 1.0f)) {
        P << mx_u, my_u, (1.0 - mx_u * mx_u - my_u * my_u) / 2.0;
    } else {
        const double rho2_d = mx_u * mx_u + my_u * my_u;
        P << mx_u, my_u,
                1.0 - xi * (rho2_d + 1.0) /
                                (xi +
                                 std::sqrt(1.0 + (1.0 - xi * xi) * rho2_d));
    }
}

void CataCameraModel::initUndistortMap(cv::Mat& map1,
                                       cv::Mat& map2,
                                       double fScale) const {
    cv::Size imageSize(params_.width, params_.height);
    cv::Mat mapX = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize, CV_32F);

    for (int v = 0; v < imageSize.height; ++v) {
        for (int u = 0; u < imageSize.width; ++u) {
            const double mx_u = m_inv_K11 / fScale * u + m_inv_K13 / fScale;
            const double my_u = m_inv_K22 / fScale * v + m_inv_K23 / fScale;
            const double d2 = mx_u * mx_u + my_u * my_u;

            Eigen::Vector3d ray;
            ray << mx_u, my_u,
                    1.0 - params_.xi * (d2 + 1.0) /
                                    (params_.xi +
                                     std::sqrt(1.0 +
                                               (1.0 - params_.xi * params_.xi) *
                                                       d2));

            Eigen::Vector2d p;
            spaceToPlane(ray, p);

            mapX.at<float>(v, u) = static_cast<float>(p(0));
            mapY.at<float>(v, u) = static_cast<float>(p(1));
        }
    }

    mergeMaps(mapX, mapY, map1, map2);
}

// ---------------------------------------------------------------------------
// Full pinhole (rational model)
// ---------------------------------------------------------------------------

FullpinholeCameraModel::FullpinholeCameraModel(
        const CameraIntrinsic& intrinsic) {
    params_ = intrinsic;
    updateInverseK(params_, m_inv_K11, m_inv_K13, m_inv_K22, m_inv_K23);
    m_noDistortion = is_zero(params_.k1) && is_zero(params_.k2) &&
                     is_zero(params_.k3) && is_zero(params_.k4) &&
                     is_zero(params_.k5) && is_zero(params_.k6) &&
                     is_zero(params_.p1) && is_zero(params_.p2);
}

CameraIntrinsic::ModelType FullpinholeCameraModel::modelType() const {
    return CameraIntrinsic::FULLPINHOLE;
}

const CameraIntrinsic& FullpinholeCameraModel::getParameters() const {
    return params_;
}

void FullpinholeCameraModel::distortion(const Eigen::Vector2d& p_u,
                                        Eigen::Vector2d& d_u) const {
    const double x = p_u(0);
    const double y = p_u(1);
    const double x2 = x * x;
    const double y2 = y * y;
    const double r2 = x2 + y2;
    const double _2xy = 2.0 * x * y;
    const double kr =
            (1.0 + ((params_.k3 * r2 + params_.k2) * r2 + params_.k1) * r2) /
            (1.0 + ((params_.k6 * r2 + params_.k5) * r2 + params_.k4) * r2);
    d_u << x * kr + params_.p1 * _2xy + params_.p2 * (r2 + 2.0 * x2) - x,
            y * kr + params_.p1 * (r2 + 2.0 * y2) + params_.p2 * _2xy - y;
}

void FullpinholeCameraModel::spaceToPlane(const Eigen::Vector3d& P,
                                          Eigen::Vector2d& p) const {
    if (std::fabs(P(2)) < 1e-10) {
        p = Eigen::Vector2d(-1, -1);
        return;
    }

    Eigen::Vector2d p_u(P(0) / P(2), P(1) / P(2));
    Eigen::Vector2d p_d;
    if (m_noDistortion) {
        p_d = p_u;
    } else {
        Eigen::Vector2d d_u;
        distortion(p_u, d_u);
        p_d = p_u + d_u;
    }

    p << params_.fx * p_d(0) + params_.cx, params_.fy * p_d(1) + params_.cy;
}

void FullpinholeCameraModel::liftProjective(const Eigen::Vector2d& p,
                                            Eigen::Vector3d& P) const {
    const double mx_d = m_inv_K11 * p(0) + m_inv_K13;
    const double my_d = m_inv_K22 * p(1) + m_inv_K23;

    double mx_u = mx_d;
    double my_u = my_d;
    if (!m_noDistortion) {
        constexpr int n = 8;
        Eigen::Vector2d d_u;
        distortion(Eigen::Vector2d(mx_d, my_d), d_u);
        mx_u = mx_d - d_u(0);
        my_u = my_d - d_u(1);
        for (int i = 1; i < n; ++i) {
            distortion(Eigen::Vector2d(mx_u, my_u), d_u);
            mx_u = mx_d - d_u(0);
            my_u = my_d - d_u(1);
        }
    }

    P << mx_u, my_u, 1.0;
}

void FullpinholeCameraModel::initUndistortMap(cv::Mat& map1,
                                              cv::Mat& map2,
                                              double fScale) const {
    cv::Size imageSize(params_.width, params_.height);
    cv::Mat mapX = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize, CV_32F);

    for (int v = 0; v < imageSize.height; ++v) {
        for (int u = 0; u < imageSize.width; ++u) {
            const double mx_u = m_inv_K11 / fScale * u + m_inv_K13 / fScale;
            const double my_u = m_inv_K22 / fScale * v + m_inv_K23 / fScale;

            Eigen::Vector3d ray;
            ray << mx_u, my_u, 1.0;

            Eigen::Vector2d p;
            spaceToPlane(ray, p);

            mapX.at<float>(v, u) = static_cast<float>(p(0));
            mapY.at<float>(v, u) = static_cast<float>(p(1));
        }
    }

    mergeMaps(mapX, mapY, map1, map2);
}

// ---------------------------------------------------------------------------
// Factory & system
// ---------------------------------------------------------------------------

CameraModelPtr CameraModelFactory::create(const CameraIntrinsic& intrinsic) {
    switch (intrinsic.model_type) {
        case CameraIntrinsic::PINHOLE:
            return std::make_shared<PinholeCameraModel>(intrinsic);
        case CameraIntrinsic::MEI:
            return std::make_shared<CataCameraModel>(intrinsic);
        case CameraIntrinsic::FULLPINHOLE:
            return std::make_shared<FullpinholeCameraModel>(intrinsic);
        case CameraIntrinsic::KANNALA_BRANDT:
        default:
            return std::make_shared<EquidistantCameraModel>(intrinsic);
    }
}

bool CameraSystem::loadFromConfig(const VehicleCalibConfig& config) {
    cameras_.clear();
    for (const auto& [name, cam_config] : config.cameras) {
        auto model = CameraModelFactory::create(cam_config.intrinsic);
        cameras_[name] = std::move(model);
        CVLog::Print("[CameraSystem] loaded camera '%s' model=%d", name.c_str(),
                     static_cast<int>(cam_config.intrinsic.model_type));
    }
    CVLog::Print("[CameraSystem] loaded %zu cameras", cameras_.size());
    return !cameras_.empty();
}

CameraModelPtr CameraSystem::getCamera(const std::string& name) const {
    auto it = cameras_.find(name);
    if (it != cameras_.end()) return it->second;
    return nullptr;
}

std::map<std::string, CameraModelPtr> CameraSystem::getCameras() const {
    return cameras_;
}

}  // namespace mcalib

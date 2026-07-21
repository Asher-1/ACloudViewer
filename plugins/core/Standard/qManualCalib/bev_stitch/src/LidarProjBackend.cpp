// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LidarProjBackend.h"

#include <CVLog.h>

#ifdef MCALIB_BEV_OPENCL_ENABLED
#include "BevRemapOcl.h"
#endif

#ifdef MCALIB_BEV_CUDA_ENABLED
#include "BevRemapCuda.cuh"
#endif

#include <algorithm>
#include <cmath>

namespace mcalib {

namespace {

bool projectPointsCpu(const std::vector<Eigen::Vector3f>& points_sensing,
                      const Eigen::Matrix3d& rotation,
                      const Eigen::Vector3d& translation,
                      double fx,
                      double fy,
                      double cx,
                      double cy,
                      LidarProjResult& out) {
    out.image_points.clear();
    out.depths.clear();
    out.image_points.reserve(points_sensing.size());
    out.depths.reserve(points_sensing.size());

    for (const auto& pt : points_sensing) {
        const Eigen::Vector3d p(pt.x(), pt.y(), pt.z());
        const Eigen::Vector3d pc = rotation * p + translation;
        if (pc.z() <= 0) continue;
        out.image_points.emplace_back(
                static_cast<float>(fx * pc.x() / pc.z() + cx),
                static_cast<float>(fy * pc.y() / pc.z() + cy));
        out.depths.push_back(static_cast<float>(pc.z()));
    }
    return !out.image_points.empty();
}

float kbRadius(double theta, const KannalaBrandtCoeffs& kb) {
    const double t2 = theta * theta;
    const double t3 = t2 * theta;
    const double t5 = t2 * t3;
    const double t7 = t2 * t5;
    const double t9 = t2 * t7;
    return static_cast<float>(theta + kb.k1 * t3 + kb.k2 * t5 + kb.k3 * t7 +
                              kb.k4 * t9);
}

bool projectPointsKbCpu(const std::vector<Eigen::Vector3f>& points_sensing,
                        const Eigen::Matrix3d& rotation,
                        const Eigen::Vector3d& translation,
                        double fx,
                        double fy,
                        double cx,
                        double cy,
                        const KannalaBrandtCoeffs& kb,
                        LidarProjResult& out) {
    out.image_points.clear();
    out.depths.clear();
    out.image_points.reserve(points_sensing.size());
    out.depths.reserve(points_sensing.size());

    for (const auto& pt : points_sensing) {
        const Eigen::Vector3d p(pt.x(), pt.y(), pt.z());
        const Eigen::Vector3d pc = rotation * p + translation;
        const double len = pc.norm();
        if (len < 1e-12 || pc.z() <= 0) continue;

        const double theta = std::acos(pc.z() / len);
        const double phi = std::atan2(pc.y(), pc.x());
        const float r = kbRadius(theta, kb);
        const float pu_x = r * static_cast<float>(std::cos(phi));
        const float pu_y = r * static_cast<float>(std::sin(phi));
        out.image_points.emplace_back(static_cast<float>(fx * pu_x + cx),
                                      static_cast<float>(fy * pu_y + cy));
        out.depths.push_back(static_cast<float>(pc.z()));
    }
    return !out.image_points.empty();
}

std::vector<float> packPoints(
        const std::vector<Eigen::Vector3f>& points_sensing) {
    std::vector<float> packed(points_sensing.size() * 3);
    for (size_t i = 0; i < points_sensing.size(); ++i) {
        packed[i * 3 + 0] = points_sensing[i].x();
        packed[i * 3 + 1] = points_sensing[i].y();
        packed[i * 3 + 2] = points_sensing[i].z();
    }
    return packed;
}

void packRotationTranslation(const Eigen::Matrix3d& rotation,
                             const Eigen::Vector3d& translation,
                             float rot[9],
                             float trans[3]) {
    rot[0] = static_cast<float>(rotation(0, 0));
    rot[1] = static_cast<float>(rotation(0, 1));
    rot[2] = static_cast<float>(rotation(0, 2));
    rot[3] = static_cast<float>(rotation(1, 0));
    rot[4] = static_cast<float>(rotation(1, 1));
    rot[5] = static_cast<float>(rotation(1, 2));
    rot[6] = static_cast<float>(rotation(2, 0));
    rot[7] = static_cast<float>(rotation(2, 1));
    rot[8] = static_cast<float>(rotation(2, 2));
    trans[0] = static_cast<float>(translation.x());
    trans[1] = static_cast<float>(translation.y());
    trans[2] = static_cast<float>(translation.z());
}

}  // namespace

bool LidarProjBackend::projectPoints(
        BevRemapMode mode,
        const std::vector<Eigen::Vector3f>& points_sensing,
        const Eigen::Matrix3d& rotation,
        const Eigen::Vector3d& translation,
        double fx,
        double fy,
        double cx,
        double cy,
        LidarProjResult& out) {
    if (points_sensing.empty()) return false;

    const BevRemapMode resolved = BevRemapper::resolveMode(mode);
    const auto packed = packPoints(points_sensing);
    float rot[9];
    float trans[3];
    packRotationTranslation(rotation, translation, rot, trans);

#ifdef MCALIB_BEV_CUDA_ENABLED
    if (resolved == BevRemapMode::CUDA) {
        if (bev_cuda::projectPoints(
                    packed.data(), static_cast<int>(points_sensing.size()), rot,
                    trans, static_cast<float>(fx), static_cast<float>(fy),
                    static_cast<float>(cx), static_cast<float>(cy),
                    out.image_points, out.depths)) {
            return !out.image_points.empty();
        }
    }
#endif

#ifdef MCALIB_BEV_OPENCL_ENABLED
    if (resolved == BevRemapMode::OpenCL) {
        if (bev_ocl::projectPoints(
                    packed.data(), static_cast<int>(points_sensing.size()), rot,
                    trans, static_cast<float>(fx), static_cast<float>(fy),
                    static_cast<float>(cx), static_cast<float>(cy),
                    out.image_points, out.depths)) {
            return !out.image_points.empty();
        }
    }
#endif

    return projectPointsCpu(points_sensing, rotation, translation, fx, fy, cx,
                            cy, out);
}

bool LidarProjBackend::projectPointsKb(
        BevRemapMode mode,
        const std::vector<Eigen::Vector3f>& points_sensing,
        const Eigen::Matrix3d& rotation,
        const Eigen::Vector3d& translation,
        double fx,
        double fy,
        double cx,
        double cy,
        const KannalaBrandtCoeffs& kb,
        LidarProjResult& out) {
    if (points_sensing.empty()) return false;

    const BevRemapMode resolved = BevRemapper::resolveMode(mode);
    const auto packed = packPoints(points_sensing);
    float rot[9];
    float trans[3];
    packRotationTranslation(rotation, translation, rot, trans);
    const float kb_arr[4] = {
            static_cast<float>(kb.k1), static_cast<float>(kb.k2),
            static_cast<float>(kb.k3), static_cast<float>(kb.k4)};

#ifdef MCALIB_BEV_CUDA_ENABLED
    if (resolved == BevRemapMode::CUDA) {
        if (bev_cuda::projectPointsKb(
                    packed.data(), static_cast<int>(points_sensing.size()), rot,
                    trans, static_cast<float>(fx), static_cast<float>(fy),
                    static_cast<float>(cx), static_cast<float>(cy), kb_arr,
                    out.image_points, out.depths)) {
            return !out.image_points.empty();
        }
    }
#endif

#ifdef MCALIB_BEV_OPENCL_ENABLED
    if (resolved == BevRemapMode::OpenCL) {
        if (bev_ocl::projectPointsKb(
                    packed.data(), static_cast<int>(points_sensing.size()), rot,
                    trans, static_cast<float>(fx), static_cast<float>(fy),
                    static_cast<float>(cx), static_cast<float>(cy), kb_arr,
                    out.image_points, out.depths)) {
            return !out.image_points.empty();
        }
    }
#endif

    return projectPointsKbCpu(points_sensing, rotation, translation, fx, fy, cx,
                              cy, kb, out);
}

}  // namespace mcalib

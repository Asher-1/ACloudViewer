// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "sensor/models/base.h"

namespace colmap {

// Simple Pinhole camera model.
//
// No Distortion is assumed. Only focal length and principal point is modeled.
//
// Parameter list is expected in the following order:
//
//   f, cx, cy
//
// See https://en.wikipedia.org/wiki/Pinhole_camera_model
struct SimplePinholeCameraModel
    : public BasePerspectivePinholeCameraModel<SimplePinholeCameraModel> {
    PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(CameraModelId::kSimplePinhole,
                                         "SIMPLE_PINHOLE",
                                         "f, cx, cy",
                                         1,
                                         2,
                                         0,
                                         true)

    static inline std::vector<double> InitializeParams(
            const double focal_length,
            const size_t width,
            const size_t height) {
        return {focal_length, width / 2.0, height / 2.0};
    }

    template <typename T>
    static inline bool ImgFromCam(const T* params,
                                  const T& u,
                                  const T& v,
                                  const T& w,
                                  T* x,
                                  T* y,
                                  const bool check_cheirality = true) {
        if (!HasProjectableDepth(w, check_cheirality)) {
            return false;
        }

        const T f = params[0];
        const T c1 = params[1];
        const T c2 = params[2];

        // No Distortion

        // Transform to image coordinates
        *x = f * u / w + c1;
        *y = f * v / w + c2;

        return true;
    }

    static inline bool CamFromImg(
            const double* params, double x, double y, double* u, double* v) {
        const double f = params[0];
        const double c1 = params[1];
        const double c2 = params[2];

        *u = (x - c1) / f;
        *v = (y - c2) / f;

        return true;
    }
};

// Pinhole camera model.
//
// No Distortion is assumed. Only focal length and principal point is modeled.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy
//
// See https://en.wikipedia.org/wiki/Pinhole_camera_model
struct PinholeCameraModel
    : public BasePerspectivePinholeCameraModel<PinholeCameraModel> {
    PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(
            CameraModelId::kPinhole, "PINHOLE", "fx, fy, cx, cy", 2, 2, 0, true)

    static inline std::vector<double> InitializeParams(
            const double focal_length,
            const size_t width,
            const size_t height) {
        return {focal_length, focal_length, width / 2.0, height / 2.0};
    }

    template <typename T>
    static inline bool ImgFromCam(const T* params,
                                  const T& u,
                                  const T& v,
                                  const T& w,
                                  T* x,
                                  T* y,
                                  const bool check_cheirality = true) {
        if (!HasProjectableDepth(w, check_cheirality)) {
            return false;
        }

        const T f1 = params[0];
        const T f2 = params[1];
        const T c1 = params[2];
        const T c2 = params[3];

        // No Distortion

        // Transform to image coordinates
        *x = f1 * u / w + c1;
        *y = f2 * v / w + c2;

        return true;
    }

    static inline bool CamFromImg(
            const double* params, double x, double y, double* u, double* v) {
        const double f1 = params[0];
        const double f2 = params[1];
        const double c1 = params[2];
        const double c2 = params[3];

        *u = (x - c1) / f1;
        *v = (y - c2) / f2;

        return true;
    }
};

// Perspective pinhole camera with non-orthogonal pixel axes. The skew is in
// pixels and is refined as an extra parameter.
struct SkewedPinholeCameraModel
    : public BasePerspectivePinholeCameraModel<SkewedPinholeCameraModel> {
    PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(CameraModelId::kSkewedPinhole,
                                         "SKEWED_PINHOLE",
                                         "fx, fy, cx, cy, s",
                                         2,
                                         2,
                                         1,
                                         true)

    static inline std::vector<double> InitializeParams(
            const double focal_length,
            const size_t width,
            const size_t height) {
        return {focal_length, focal_length, width / 2.0, height / 2.0, 0.0};
    }

    template <typename T>
    static inline bool HasBogusExtraParams(const std::vector<T>& params,
                                           const T max_extra_param) {
        const T mean_focal_length = (params[0] + params[1]) / T(2);
        return std::abs(params[4]) >
               max_extra_param * std::abs(mean_focal_length);
    }

    static inline void Rescale(const double scale_x,
                               const double scale_y,
                               std::vector<double>* params) {
        BasePerspectiveCameraModel<SkewedPinholeCameraModel>::Rescale(
                scale_x, scale_y, params);
        (*params)[4] *= scale_x;
    }

    template <typename T>
    static inline bool ImgFromCam(const T* params,
                                  const T& u,
                                  const T& v,
                                  const T& w,
                                  T* x,
                                  T* y,
                                  const bool check_cheirality = true) {
        if (!HasProjectableDepth(w, check_cheirality)) return false;
        const T uu = u / w;
        const T vv = v / w;
        *x = params[0] * uu + params[4] * vv + params[2];
        *y = params[1] * vv + params[3];
        return true;
    }

    template <typename T>
    static inline bool CamFromImg(
            const T* params, const T& x, const T& y, T* u, T* v) {
        *v = (y - params[3]) / params[1];
        *u = (x - params[2] - params[4] * *v) / params[0];
        return true;
    }
};

}  // namespace colmap

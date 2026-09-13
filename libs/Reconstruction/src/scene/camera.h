// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <optional>
#include <vector>

#include "geometry/pose.h"
#include "sensor/models.h"
#include "util/types.h"

namespace colmap {

// Camera class that holds the intrinsic parameters. Cameras may be shared
// between multiple images, e.g., if the same "physical" camera took multiple
// pictures with the exact same lens and intrinsics (focal length, etc.).
// This class has a specific distortion model defined by a camera model class.
class Camera {
public:
    Camera();

    // Access the unique identifier of the camera.
    inline camera_t CameraId() const;
    inline void SetCameraId(const camera_t camera_id);

    // Access the camera model.
    inline CameraModelId ModelId() const;
    std::string ModelName() const;
    void SetModelId(CameraModelId model_id);
    void SetModelIdFromName(const std::string& model_name);

    // Access dimensions of the camera sensor.
    inline size_t Width() const;
    inline size_t Height() const;
    inline void SetWidth(const size_t width);
    inline void SetHeight(const size_t height);

    // Access focal length parameters.
    // ---- Upstream-parity API (COLMAP 4.x scene/camera.h) ----
    // Whether the model is perspective with a finite pinhole image plane.
    // Upstream-parity factories (COLMAP 4.x scene/camera.h): initialize
    // parameters for the given model with the principal point at the image
    // center.
    static Camera CreateFromModelId(camera_t camera_id,
                                    CameraModelId model_id,
                                    double focal_length,
                                    size_t width,
                                    size_t height);
    static Camera CreateFromModelName(camera_t camera_id,
                                      const std::string& model_name,
                                      double focal_length,
                                      size_t width,
                                      size_t height);

    // Upstream-parity: the camera as a sensor in a rig.
    inline sensor_t SensorId() const {
        return sensor_t(SensorType::CAMERA, camera_id_);
    }
    inline bool IsPerspectivePinhole() const {
        return CameraModelIsPerspectivePinhole(model_id_);
    }
    // Whether the model is a spherical (equirectangular) panorama model.
    inline bool IsSpherical() const {
        return CameraModelIsSpherical(model_id_);
    }
    inline bool IsPerspectiveFisheye() const {
        return CameraModelIsPerspectiveFisheye(model_id_);
    }
    // Principal point as a 2D point (pixels).
    inline Eigen::Vector2d PrincipalPoint() const {
        return Eigen::Vector2d(PrincipalPointX(), PrincipalPointY());
    }

    double MeanFocalLength() const;
    double FocalLength() const;
    double FocalLengthX() const;
    double FocalLengthY() const;
    void SetFocalLength(const double focal_length);
    void SetFocalLengthX(const double focal_length_x);
    void SetFocalLengthY(const double focal_length_y);

    // Check if camera has prior focal length.
    inline bool HasPriorFocalLength() const;
    inline void SetPriorFocalLength(const bool prior);

    // Access principal point parameters. Only works if there are two
    // principal point parameters.
    double PrincipalPointX() const;
    double PrincipalPointY() const;
    void SetPrincipalPointX(const double ppx);
    void SetPrincipalPointY(const double ppy);

    // Get the indices of the parameter groups in the parameter vector.
    const std::vector<size_t>& FocalLengthIdxs() const;
    const std::vector<size_t>& PrincipalPointIdxs() const;
    const std::vector<size_t>& ExtraParamsIdxs() const;

    // Get intrinsic calibration matrix composed from focal length and principal
    // point parameters, excluding distortion parameters.
    Eigen::Matrix3d CalibrationMatrix() const;

    // Get human-readable information about the parameter vector ordering.
    std::string ParamsInfo() const;

    // Access the raw parameter vector.
    inline size_t NumParams() const;
    inline const std::vector<double>& Params() const;
    inline std::vector<double>& Params();
    inline double Params(const size_t idx) const;
    inline double& Params(const size_t idx);
    inline const double* ParamsData() const;
    inline double* ParamsData();
    inline void SetParams(const std::vector<double>& params);

    // Concatenate parameters as comma-separated list.
    std::string ParamsToString() const;

    // Set camera parameters from comma-separated list.
    bool SetParamsFromString(const std::string& string);

    // Check whether parameters are valid, i.e. the parameter vector has
    // the correct dimensions that match the specified camera model.
    bool VerifyParams() const;

    // Check whether camera is already undistorted
    bool IsUndistorted() const;

    // Check whether camera has bogus parameters.
    bool HasBogusParams(const double min_focal_length_ratio,
                        const double max_focal_length_ratio,
                        const double max_extra_param) const;

    // Initialize parameters for given camera model and focal length, and set
    // the principal point to be the image center.
    void InitializeWithId(CameraModelId model_id,
                          const double focal_length,
                          const size_t width,
                          const size_t height);
    void InitializeWithName(const std::string& model_name,
                            const double focal_length,
                            const size_t width,
                            const size_t height);

    // Project point in image plane to camera ray (not unit normalized).
    // Upstream parity (d3ccaf35 scene/camera.h): the legacy fork
    // ImageToWorld/WorldToImage pair is replaced by the optional-returning
    // CamFromImg/ImgFromCam API; the unprojection may fail on models with a
    // finite valid domain (e.g. division/EUCM discriminants).
    std::optional<Eigen::Vector2d> CamFromImg(
            const Eigen::Vector2d& image_point) const;

    // Unproject a pixel to a unit calibrated bearing. This is the lightweight
    // path for geometry consumers that do not need pixel derivatives.
    std::optional<Eigen::Vector3d> CamRayFromImg(
            const Eigen::Vector2d& image_point) const;

    // Convert pixel threshold in image plane to camera space.
    double CamFromImgThreshold(const double threshold) const;

    // Project point from camera frame to image plane. Without cheirality
    // check, points behind the camera are projected as well and only points
    // on the camera plane fail.
    std::optional<Eigen::Vector2d> ImgFromCam(
            const Eigen::Vector3d& camera_point,
            bool check_cheirality = true) const;

    // Project point from camera frame to image plane, additionally computing
    // the analytic Jacobian d(x, y) / d(u, v, w) (upstream parity). Pass
    // nullptr to skip the Jacobian.
    std::optional<Eigen::Vector2d> ImgFromCamWithJac(
            const Eigen::Vector3d& camera_point,
            Eigen::Matrix2x3d* J_uvw,
            bool check_cheirality = true) const;

    // Unproject a pixel to a unit bearing together with the Jacobian
    // d(u, v, w) / d(x, y) of that bearing with respect to the pixel
    // (upstream analytic tangent-Sampson path).
    std::optional<CamRayWithJac> CamRayFromImgWithJac(
            const Eigen::Vector2d& image_point) const;

    // Rescale camera dimensions and accordingly the focal length and
    // and the principal point.
    void Rescale(const double scale);
    void Rescale(const size_t width, const size_t height);

    inline bool operator==(const Camera& other) const;
    inline bool operator!=(const Camera& other) const;

private:
    // The unique identifier of the camera. If the identifier is not specified
    // it is set to `kInvalidCameraId`.
    camera_t camera_id_;

    // The identifier of the camera model. If the camera model is not specified
    // the identifier is `CameraModelId::kInvalid`.
    CameraModelId model_id_;

    // The dimensions of the image, 0 if not initialized.
    size_t width_;
    size_t height_;

    // The focal length, principal point, and extra parameters. If the camera
    // model is not specified, this vector is empty.
    std::vector<double> params_;

    // Whether there is a safe prior for the focal length,
    // e.g. manually provided or extracted from EXIF
    bool prior_focal_length_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

camera_t Camera::CameraId() const { return camera_id_; }

void Camera::SetCameraId(const camera_t camera_id) { camera_id_ = camera_id; }

CameraModelId Camera::ModelId() const { return model_id_; }

size_t Camera::Width() const { return width_; }

size_t Camera::Height() const { return height_; }

void Camera::SetWidth(const size_t width) { width_ = width; }

void Camera::SetHeight(const size_t height) { height_ = height; }

bool Camera::HasPriorFocalLength() const { return prior_focal_length_; }

void Camera::SetPriorFocalLength(const bool prior) {
    prior_focal_length_ = prior;
}

size_t Camera::NumParams() const { return params_.size(); }

const std::vector<double>& Camera::Params() const { return params_; }

std::vector<double>& Camera::Params() { return params_; }

double Camera::Params(const size_t idx) const { return params_[idx]; }

double& Camera::Params(const size_t idx) { return params_[idx]; }

const double* Camera::ParamsData() const { return params_.data(); }

double* Camera::ParamsData() { return params_.data(); }

void Camera::SetParams(const std::vector<double>& params) { params_ = params; }

bool Camera::operator==(const Camera& other) const {
    return CameraId() == other.CameraId() && ModelId() == other.ModelId() &&
           Width() == other.Width() && Height() == other.Height() &&
           Params() == other.Params() &&
           HasPriorFocalLength() == other.HasPriorFocalLength();
}

bool Camera::operator!=(const Camera& other) const { return !(*this == other); }

}  // namespace colmap

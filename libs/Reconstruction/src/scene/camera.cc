// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "scene/camera.h"

#include <cmath>
#include <iomanip>
#include <limits>

#include "sensor/models.h"
#include "util/logging.h"
#include "util/misc.h"

namespace colmap {

Camera::Camera()
    : camera_id_(kInvalidCameraId),
      model_id_(CameraModelId::kInvalid),
      width_(0),
      height_(0),
      prior_focal_length_(false) {}

std::string Camera::ModelName() const { return CameraModelIdToName(model_id_); }

void Camera::SetModelId(const CameraModelId model_id) {
  CHECK(ExistsCameraModelWithId(model_id));
  model_id_ = model_id;
  params_.resize(CameraModelNumParams(model_id_), 0);
}

void Camera::SetModelIdFromName(const std::string& model_name) {
  CHECK(ExistsCameraModelWithName(model_name));
  model_id_ = CameraModelNameToId(model_name);
  params_.resize(CameraModelNumParams(model_id_), 0);
}

const std::vector<size_t>& Camera::FocalLengthIdxs() const {
  return CameraModelFocalLengthIdxs(model_id_);
}

const std::vector<size_t>& Camera::PrincipalPointIdxs() const {
  return CameraModelPrincipalPointIdxs(model_id_);
}

const std::vector<size_t>& Camera::ExtraParamsIdxs() const {
  return CameraModelExtraParamsIdxs(model_id_);
}

Eigen::Matrix3d Camera::CalibrationMatrix() const {
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();

  const std::vector<size_t>& idxs = FocalLengthIdxs();
  if (idxs.size() == 1) {
    K(0, 0) = params_[idxs[0]];
    K(1, 1) = params_[idxs[0]];
  } else if (idxs.size() == 2) {
    K(0, 0) = params_[idxs[0]];
    K(1, 1) = params_[idxs[1]];
  } else {
    LOG(FATAL)
        << "Camera model must either have 1 or 2 focal length parameters.";
  }

  K(0, 2) = PrincipalPointX();
  K(1, 2) = PrincipalPointY();

  return K;
}

std::string Camera::ParamsInfo() const {
  return CameraModelParamsInfo(model_id_);
}

double Camera::MeanFocalLength() const {
  const auto& focal_length_idxs = FocalLengthIdxs();
  if (focal_length_idxs.empty()) {
    return 0.0;
  }
  double focal_length = 0;
  for (const auto idx : focal_length_idxs) {
    focal_length += params_[idx];
  }
  return focal_length / focal_length_idxs.size();
}

double Camera::FocalLength() const {
  const std::vector<size_t>& idxs = FocalLengthIdxs();
  CHECK_EQ(idxs.size(), 1);
  return params_[idxs[0]];
}

double Camera::FocalLengthX() const {
    // Upstream parity (dbb41680 scene/camera.h): single-focal-length models
    // are legal here (x == y); only the setters require two focal params.
    const std::vector<size_t>& idxs = FocalLengthIdxs();
    return params_[idxs[0]];
}

double Camera::FocalLengthY() const {
    const std::vector<size_t>& idxs = FocalLengthIdxs();
    return params_[idxs[(idxs.size() == 1) ? 0 : 1]];
}

void Camera::SetFocalLength(const double focal_length) {
  const std::vector<size_t>& idxs = FocalLengthIdxs();
  for (const auto idx : idxs) {
    params_[idx] = focal_length;
  }
}

void Camera::SetFocalLengthX(const double focal_length_x) {
  const std::vector<size_t>& idxs = FocalLengthIdxs();
  CHECK_EQ(idxs.size(), 2);
  params_[idxs[0]] = focal_length_x;
}

void Camera::SetFocalLengthY(const double focal_length_y) {
  const std::vector<size_t>& idxs = FocalLengthIdxs();
  CHECK_EQ(idxs.size(), 2);
  params_[idxs[1]] = focal_length_y;
}

double Camera::PrincipalPointX() const {
  const std::vector<size_t>& idxs = PrincipalPointIdxs();
  CHECK_EQ(idxs.size(), 2);
  return params_[idxs[0]];
}

double Camera::PrincipalPointY() const {
  const std::vector<size_t>& idxs = PrincipalPointIdxs();
  CHECK_EQ(idxs.size(), 2);
  return params_[idxs[1]];
}

void Camera::SetPrincipalPointX(const double ppx) {
  const std::vector<size_t>& idxs = PrincipalPointIdxs();
  CHECK_EQ(idxs.size(), 2);
  params_[idxs[0]] = ppx;
}

void Camera::SetPrincipalPointY(const double ppy) {
  const std::vector<size_t>& idxs = PrincipalPointIdxs();
  CHECK_EQ(idxs.size(), 2);
  params_[idxs[1]] = ppy;
}

std::string Camera::ParamsToString() const { return VectorToCSV(params_); }

bool Camera::SetParamsFromString(const std::string& string) {
  const std::vector<double> new_camera_params = CSVToVector<double>(string);
  if (!CameraModelVerifyParams(model_id_, new_camera_params)) {
    return false;
  }

  params_ = new_camera_params;
  return true;
}

bool Camera::VerifyParams() const {
  return CameraModelVerifyParams(model_id_, params_);
}

bool Camera::HasBogusParams(const double min_focal_length_ratio,
                            const double max_focal_length_ratio,
                            const double max_extra_param) const {
  return CameraModelHasBogusParams(model_id_, params_, width_, height_,
                                   min_focal_length_ratio,
                                   max_focal_length_ratio, max_extra_param);
}

bool Camera::IsUndistorted() const {
  for (const size_t idx : ExtraParamsIdxs()) {
    if (std::abs(params_[idx]) > 1e-8) {
      return false;
    }
  }
  return true;
}

void Camera::InitializeWithId(const CameraModelId model_id,
                              const double focal_length,
                              const size_t width, const size_t height) {
  CHECK(ExistsCameraModelWithId(model_id));
  model_id_ = model_id;
  width_ = width;
  height_ = height;
  params_ = CameraModelInitializeParams(model_id, focal_length, width, height);
}

void Camera::InitializeWithName(const std::string& model_name,
                                const double focal_length, const size_t width,
                                const size_t height) {
  InitializeWithId(CameraModelNameToId(model_name), focal_length, width,
                   height);
}

// Upstream parity (d3ccaf35 scene/camera.cc): the projection/unprojection
// methods delegate to the runtime dispatch over the model API; the fork's
// hand-rolled equirectangular branches (and the legacy void-returning
// ImageToWorld/WorldToImage pair) are gone.
std::optional<Eigen::Vector2d> Camera::CamFromImg(
        const Eigen::Vector2d& image_point) const {
  return CameraModelCamFromImg(model_id_, params_, image_point);
}

std::optional<Eigen::Vector3d> Camera::CamRayFromImg(
        const Eigen::Vector2d& image_point) const {
  return CameraModelCamRayFromImg(model_id_, params_, image_point);
}

std::optional<CamRayWithJac> Camera::CamRayFromImgWithJac(
        const Eigen::Vector2d& image_point) const {
  const std::optional<Eigen::Vector3d> cam_ray = CamRayFromImg(image_point);
  if (!cam_ray.has_value()) {
    return std::nullopt;
  }
  Eigen::Matrix2x3d J_uvw;
  if (!ImgFromCamWithJac(*cam_ray, &J_uvw).has_value()) {
    return std::nullopt;
  }
  const std::optional<Eigen::Matrix3x2d> J_ray =
          CamRayFromImgJac(*cam_ray, J_uvw);
  if (!J_ray.has_value()) {
    return std::nullopt;
  }
  return CamRayWithJac{*cam_ray, *J_ray};
}

double Camera::CamFromImgThreshold(const double threshold) const {
  return CameraModelCamFromImgThreshold(model_id_, params_, threshold);
}

std::optional<Eigen::Vector2d> Camera::ImgFromCam(
        const Eigen::Vector3d& camera_point, const bool check_cheirality) const {
  return CameraModelImgFromCam(
          model_id_, params_, camera_point, check_cheirality);
}

std::optional<Eigen::Vector2d> Camera::ImgFromCamWithJac(
        const Eigen::Vector3d& camera_point,
        Eigen::Matrix2x3d* J_uvw,
        const bool check_cheirality) const {
  return CameraModelImgFromCamWithJac(
          model_id_, params_, camera_point, J_uvw, check_cheirality);
}

void Camera::Rescale(const double scale) {
  CHECK_GT(scale, 0.0);
  const double scale_x =
      std::round(scale * width_) / static_cast<double>(width_);
  const double scale_y =
      std::round(scale * height_) / static_cast<double>(height_);
  width_ = static_cast<size_t>(std::round(scale * width_));
  height_ = static_cast<size_t>(std::round(scale * height_));
  CameraModelRescale(model_id_, scale_x, scale_y, params_);
}

void Camera::Rescale(const size_t width, const size_t height) {
  const double scale_x =
      static_cast<double>(width) / static_cast<double>(width_);
  const double scale_y =
      static_cast<double>(height) / static_cast<double>(height_);
  width_ = width;
  height_ = height;
  CameraModelRescale(model_id_, scale_x, scale_y, params_);
}


Camera Camera::CreateFromModelId(camera_t camera_id,
                                 CameraModelId model_id,
                                 double focal_length,
                                 size_t width,
                                 size_t height) {
    THROW_CHECK(ExistsCameraModelWithId(model_id));
    Camera camera;
    camera.SetModelId(model_id);
    camera.SetCameraId(camera_id);
    camera.SetWidth(width);
    camera.SetHeight(height);
    camera.Params() = CameraModelInitializeParams(model_id, focal_length,
                                                 width, height);
    camera.SetPriorFocalLength(true);
    return camera;
}

Camera Camera::CreateFromModelName(camera_t camera_id,
                                   const std::string& model_name,
                                   double focal_length,
                                   size_t width,
                                   size_t height) {
    return CreateFromModelId(
            camera_id, CameraModelNameToId(model_name), focal_length, width,
            height);
}
}  // namespace colmap

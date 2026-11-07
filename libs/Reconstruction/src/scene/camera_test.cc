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

#define TEST_NAME "base/camera"
#include "util/testing.h"

#include <cmath>

#include "scene/camera.h"
#include "sensor/models.h"

using namespace colmap;

TEST(base_camera, TestEmpty) {
  Camera camera;
  EXPECT_EQ(camera.CameraId(), kInvalidCameraId);
  EXPECT_EQ(camera.ModelId(), CameraModelId::kInvalid);
  EXPECT_EQ(camera.ModelName(), "");
  EXPECT_EQ(camera.Width(), 0);
  EXPECT_EQ(camera.Height(), 0);
  EXPECT_EQ(camera.HasPriorFocalLength(), false);
  EXPECT_THROW(camera.FocalLengthIdxs(), std::domain_error);
  EXPECT_THROW(camera.ParamsInfo(), std::domain_error);
  EXPECT_EQ(camera.ParamsToString(), "");
  EXPECT_EQ(camera.NumParams(), 0);
  EXPECT_EQ(camera.Params().size(), 0);
  EXPECT_EQ(camera.ParamsData(), camera.Params().data());
}

TEST(base_camera, TestCameraId) {
  Camera camera;
  EXPECT_EQ(camera.CameraId(), kInvalidCameraId);
  camera.SetCameraId(1);
  EXPECT_EQ(camera.CameraId(), 1);
}

TEST(base_camera, TestModelId) {
  Camera camera;
  EXPECT_EQ(camera.ModelId(), CameraModelId::kInvalid);
  EXPECT_EQ(camera.ModelName(), "");
  camera.SetModelId(SimplePinholeCameraModel::model_id);
  EXPECT_EQ(camera.ModelId(), SimplePinholeCameraModel::model_id);
  EXPECT_EQ(camera.ModelName(), "SIMPLE_PINHOLE");
  EXPECT_EQ(camera.NumParams(), SimplePinholeCameraModel::num_params);
  camera.SetModelIdFromName("SIMPLE_RADIAL");
  EXPECT_EQ(camera.ModelId(), SimpleRadialCameraModel::model_id);
  EXPECT_EQ(camera.ModelName(), "SIMPLE_RADIAL");
  EXPECT_EQ(camera.NumParams(), SimpleRadialCameraModel::num_params);
}

TEST(base_camera, TestEquirectangularCamRayWithJac) {
  Camera camera;
  camera.InitializeWithId(EquirectangularCameraModel::model_id, 0.0, 1000,
                          500);
  EXPECT_EQ(camera.ModelName(), "EQUIRECTANGULAR");
  EXPECT_EQ(camera.Params().size(), 2);
  EXPECT_EQ(camera.MeanFocalLength(), 0.0);
  EXPECT_FALSE(camera.HasBogusParams(0.1, 10.0, 1.0));

  const Eigen::Vector2d pixel(250.0, 200.0);
  const auto ray_with_jac = camera.CamRayFromImgWithJac(pixel);
  ASSERT_TRUE(ray_with_jac.has_value());
  ASSERT_LE(std::abs((ray_with_jac->ray -
                     Eigen::Vector3d(-std::cos(EIGEN_PI / 10.0),
                                     -std::sin(EIGEN_PI / 10.0),
                                     0.0))
                            .norm()), 1e-12);

  constexpr double kStep = 1e-4;
  for (int axis = 0; axis < 2; ++axis) {
    Eigen::Vector2d backward = pixel;
    Eigen::Vector2d forward = pixel;
    backward[axis] -= kStep;
    forward[axis] += kStep;
    const auto ray_backward = camera.CamRayFromImgWithJac(backward);
    const auto ray_forward = camera.CamRayFromImgWithJac(forward);
    ASSERT_TRUE(ray_backward.has_value());
    ASSERT_TRUE(ray_forward.has_value());
    ASSERT_LE(std::abs((ray_with_jac->jacobian.col(axis) -
                       (ray_forward->ray - ray_backward->ray) / (2.0 * kStep))
                              .norm()), 1e-9);
  }

  camera.Rescale(2000, 1000);
  EXPECT_EQ(camera.Params()[0], 2000.0);
  EXPECT_EQ(camera.Params()[1], 1000.0);

  const auto front = camera.ImgFromCam(Eigen::Vector3d(0, 0, 1));
  const auto rear = camera.ImgFromCam(Eigen::Vector3d(0, 0, -1));
  ASSERT_TRUE(front.has_value());
  ASSERT_TRUE(rear.has_value());
  ASSERT_LE(std::abs((*front - Eigen::Vector2d(1000, 500)).norm()), 1e-12);
  ASSERT_LE(std::abs((*rear - Eigen::Vector2d(2000, 500)).norm()), 1e-12);
  EXPECT_FALSE(camera.ImgFromCam(Eigen::Vector3d::Zero()).has_value());
}

TEST(base_camera, TestWidthHeight) {
  Camera camera;
  EXPECT_EQ(camera.Width(), 0);
  EXPECT_EQ(camera.Height(), 0);
  camera.SetWidth(1);
  EXPECT_EQ(camera.Width(), 1);
  EXPECT_EQ(camera.Height(), 0);
  camera.SetHeight(1);
  EXPECT_EQ(camera.Width(), 1);
  EXPECT_EQ(camera.Height(), 1);
}

TEST(base_camera, TestFocalLength) {
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.FocalLength(), 1.0);
  camera.SetFocalLength(2.0);
  EXPECT_EQ(camera.FocalLength(), 2.0);
  camera.InitializeWithId(PinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.FocalLengthX(), 1.0);
  EXPECT_EQ(camera.FocalLengthY(), 1.0);
  camera.SetFocalLengthX(2.0);
  EXPECT_EQ(camera.FocalLengthX(), 2.0);
  EXPECT_EQ(camera.FocalLengthY(), 1.0);
  camera.SetFocalLengthY(2.0);
  EXPECT_EQ(camera.FocalLengthX(), 2.0);
  EXPECT_EQ(camera.FocalLengthY(), 2.0);
}

TEST(base_camera, TestPriorFocalLength) {
  Camera camera;
  EXPECT_EQ(camera.HasPriorFocalLength(), false);
  camera.SetPriorFocalLength(true);
  EXPECT_EQ(camera.HasPriorFocalLength(), true);
  camera.SetPriorFocalLength(false);
  EXPECT_EQ(camera.HasPriorFocalLength(), false);
}

TEST(base_camera, TestPrincipalPoint) {
  Camera camera;
  camera.InitializeWithId(PinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.PrincipalPointX(), 0.5);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);
  camera.SetPrincipalPointX(2.0);
  EXPECT_EQ(camera.PrincipalPointX(), 2.0);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);
  camera.SetPrincipalPointY(2.0);
  EXPECT_EQ(camera.PrincipalPointX(), 2.0);
  EXPECT_EQ(camera.PrincipalPointY(), 2.0);
}

TEST(base_camera, TestSphericalPrincipalPointUnavailable) {
  Camera camera;
  camera.InitializeWithId(EquirectangularCameraModel::model_id, 0.0, 2048,
                          1024);
  ASSERT_TRUE(camera.IsSpherical());
  EXPECT_THROW(camera.PrincipalPointX(), std::invalid_argument);
  EXPECT_THROW(camera.PrincipalPointY(), std::invalid_argument);
  EXPECT_THROW(camera.SetPrincipalPointX(1.0), std::invalid_argument);
  EXPECT_THROW(camera.SetPrincipalPointY(1.0), std::invalid_argument);
}

TEST(base_camera, TestParamIdxs) {
  Camera camera;
  EXPECT_THROW(camera.FocalLengthIdxs(), std::domain_error);
  EXPECT_THROW(camera.PrincipalPointIdxs(), std::domain_error);
  EXPECT_THROW(camera.ExtraParamsIdxs(), std::domain_error);
  camera.SetModelId(FullOpenCVCameraModel::model_id);
  EXPECT_EQ(camera.FocalLengthIdxs().size(), 2);
  EXPECT_EQ(camera.PrincipalPointIdxs().size(), 2);
  EXPECT_EQ(camera.ExtraParamsIdxs().size(), 8);
}

TEST(base_camera, TestCalibrationMatrix) {
  Camera camera;
  camera.InitializeWithId(PinholeCameraModel::model_id, 1.0, 1, 1);
  const Eigen::Matrix3d K = camera.CalibrationMatrix();
  Eigen::Matrix3d K_ref;
  K_ref << 1, 0, 0.5, 0, 1, 0.5, 0, 0, 1;
  EXPECT_EQ(K, K_ref);
}

TEST(base_camera, TestParamsInfo) {
  Camera camera;
  EXPECT_THROW(camera.ParamsInfo(), std::domain_error);
  camera.SetModelId(SimpleRadialCameraModel::model_id);
  EXPECT_EQ(camera.ParamsInfo(), "f, cx, cy, k");
}

TEST(base_camera, TestParams) {
  Camera camera;
  EXPECT_EQ(camera.NumParams(), 0);
  EXPECT_EQ(camera.Params().size(), camera.NumParams());
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.NumParams(), 3);
  EXPECT_EQ(camera.Params().size(), camera.NumParams());
  EXPECT_EQ(camera.ParamsData(), camera.Params().data());
  EXPECT_EQ(camera.Params(0), 1.0);
  EXPECT_EQ(camera.Params(1), 0.5);
  EXPECT_EQ(camera.Params(2), 0.5);
  EXPECT_EQ(camera.Params()[0], 1.0);
  EXPECT_EQ(camera.Params()[1], 0.5);
  EXPECT_EQ(camera.Params()[2], 0.5);
  camera.SetParams({2.0, 1.0, 1.0});
  EXPECT_EQ(camera.Params(0), 2.0);
  EXPECT_EQ(camera.Params(1), 1.0);
  EXPECT_EQ(camera.Params(2), 1.0);
}

TEST(base_camera, TestParamsToString) {
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.ParamsToString(), "1, 0.5, 0.5");
}

TEST(base_camera, TestParamsFromString) {
  Camera camera;
  camera.SetModelId(SimplePinholeCameraModel::model_id);
  EXPECT_TRUE(camera.SetParamsFromString("1, 0.5, 0.5"));
  const std::vector<double> params{1.0, 0.5, 0.5};
  ASSERT_TRUE(std::equal(camera.Params().begin(), camera.Params().end(), params.begin(), params.end()));
  EXPECT_FALSE(camera.SetParamsFromString("1.000000, 0.500000"));
  ASSERT_TRUE(std::equal(camera.Params().begin(), camera.Params().end(), params.begin(), params.end()));
}

TEST(base_camera, TestVerifyParams) {
  Camera camera;
  EXPECT_THROW(camera.VerifyParams(), std::domain_error);
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.VerifyParams(), true);
  camera.Params().resize(2);
  EXPECT_EQ(camera.VerifyParams(), false);
}

TEST(base_camera, TestIsUndistorted) {
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.IsUndistorted());
  camera.InitializeWithId(SimpleRadialCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.IsUndistorted());
  camera.SetParams({1.0, 0.5, 0.5, 0.005});
  EXPECT_FALSE(camera.IsUndistorted());
  camera.InitializeWithId(RadialCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.IsUndistorted());
  camera.SetParams({1.0, 0.5, 0.5, 0.0, 0.005});
  EXPECT_FALSE(camera.IsUndistorted());
  camera.InitializeWithId(OpenCVCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.IsUndistorted());
  camera.SetParams({1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.001});
  EXPECT_FALSE(camera.IsUndistorted());
  camera.InitializeWithId(FullOpenCVCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.IsUndistorted());
  camera.SetParams({1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001});
  EXPECT_FALSE(camera.IsUndistorted());
}

TEST(base_camera, TestHasBogusParams) {
  Camera camera;
  EXPECT_THROW(camera.HasBogusParams(0.0, 0.0, 0.0), std::domain_error);
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.HasBogusParams(0.1, 1.1, 1.0), false);
  EXPECT_EQ(camera.HasBogusParams(0.1, 1.1, 0.0), false);
  EXPECT_EQ(camera.HasBogusParams(0.1, 0.99, 1.0), true);
  EXPECT_EQ(camera.HasBogusParams(1.01, 1.1, 1.0), true);
  camera.InitializeWithId(SimpleRadialCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.HasBogusParams(0.1, 1.1, 1.0), false);
  camera.Params(3) = 1.01;
  EXPECT_EQ(camera.HasBogusParams(0.1, 1.1, 1.0), true);
  camera.Params(3) = -0.5;
  EXPECT_EQ(camera.HasBogusParams(0.1, 1.1, 1.0), false);
  camera.Params(3) = -1.01;
  EXPECT_EQ(camera.HasBogusParams(0.1, 1.1, 1.0), true);
}

TEST(base_camera, TestInitializeWithId) {
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.CameraId(), kInvalidCameraId);
  EXPECT_EQ(camera.ModelId(), SimplePinholeCameraModel::model_id);
  EXPECT_EQ(camera.ModelName(), "SIMPLE_PINHOLE");
  EXPECT_EQ(camera.Width(), 1);
  EXPECT_EQ(camera.Height(), 1);
  EXPECT_EQ(camera.HasPriorFocalLength(), false);
  EXPECT_EQ(camera.FocalLengthIdxs().size(), 1);
  EXPECT_EQ(camera.PrincipalPointIdxs().size(), 2);
  EXPECT_EQ(camera.ExtraParamsIdxs().size(), 0);
  EXPECT_EQ(camera.ParamsInfo(), "f, cx, cy");
  EXPECT_EQ(camera.ParamsToString(), "1, 0.5, 0.5");
  EXPECT_EQ(camera.FocalLength(), 1.0);
  EXPECT_EQ(camera.PrincipalPointX(), 0.5);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);
  EXPECT_EQ(camera.VerifyParams(), true);
  EXPECT_EQ(camera.HasBogusParams(0.1, 2.0, 1.0), false);
  EXPECT_EQ(camera.HasBogusParams(0.1, 0.5, 1.0), true);
  EXPECT_EQ(camera.NumParams(),
                    static_cast<int>(SimplePinholeCameraModel::num_params));
  EXPECT_EQ(camera.Params().size(),
                    static_cast<int>(SimplePinholeCameraModel::num_params));
}

TEST(base_camera, TestInitializeWithName) {
  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  EXPECT_EQ(camera.CameraId(), kInvalidCameraId);
  EXPECT_EQ(camera.ModelId(), SimplePinholeCameraModel::model_id);
  EXPECT_EQ(camera.ModelName(), "SIMPLE_PINHOLE");
  EXPECT_EQ(camera.Width(), 1);
  EXPECT_EQ(camera.Height(), 1);
  EXPECT_EQ(camera.HasPriorFocalLength(), false);
  EXPECT_EQ(camera.FocalLengthIdxs().size(), 1);
  EXPECT_EQ(camera.PrincipalPointIdxs().size(), 2);
  EXPECT_EQ(camera.ExtraParamsIdxs().size(), 0);
  EXPECT_EQ(camera.ParamsInfo(), "f, cx, cy");
  EXPECT_EQ(camera.ParamsToString(), "1, 0.5, 0.5");
  EXPECT_EQ(camera.FocalLength(), 1.0);
  EXPECT_EQ(camera.PrincipalPointX(), 0.5);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);
  EXPECT_EQ(camera.VerifyParams(), true);
  EXPECT_EQ(camera.HasBogusParams(0.1, 2.0, 1.0), false);
  EXPECT_EQ(camera.HasBogusParams(0.1, 0.5, 1.0), true);
  EXPECT_EQ(camera.NumParams(),
                    static_cast<int>(SimplePinholeCameraModel::num_params));
  EXPECT_EQ(camera.Params().size(),
                    static_cast<int>(SimplePinholeCameraModel::num_params));
}

TEST(base_camera, TestCamFromImg) {
  Camera camera;
  EXPECT_THROW(camera.CamFromImg(Eigen::Vector2d::Zero()),
                    std::domain_error);
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  EXPECT_EQ(camera.CamFromImg(Eigen::Vector2d(0.0, 0.0))->x(), -0.5);
  EXPECT_EQ(camera.CamFromImg(Eigen::Vector2d(0.0, 0.0))->y(), -0.5);
  EXPECT_EQ(camera.CamFromImg(Eigen::Vector2d(0.5, 0.5))->x(), 0.0);
  EXPECT_EQ(camera.CamFromImg(Eigen::Vector2d(0.5, 0.5))->y(), 0.0);
}

TEST(base_camera, TestCamFromImgThreshold) {
  Camera camera;
  EXPECT_THROW(camera.CamFromImgThreshold(0), std::domain_error);
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  EXPECT_EQ(camera.CamFromImgThreshold(0), 0);
  EXPECT_EQ(camera.CamFromImgThreshold(1), 1);
  camera.SetFocalLength(2.0);
  EXPECT_EQ(camera.CamFromImgThreshold(1), 0.5);
  camera.InitializeWithName("PINHOLE", 1.0, 1, 1);
  camera.SetFocalLengthY(3.0);
  EXPECT_EQ(camera.CamFromImgThreshold(1), 0.5);
}

TEST(base_camera, TestImgFromCam) {
  Camera camera;
  EXPECT_THROW(camera.ImgFromCam(Eigen::Vector3d::Zero()), std::domain_error);
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  EXPECT_EQ(camera.ImgFromCam(Eigen::Vector3d(0.0, 0.0, 1.0))->x(), 0.5);
  EXPECT_EQ(camera.ImgFromCam(Eigen::Vector3d(0.0, 0.0, 1.0))->y(), 0.5);
  EXPECT_EQ(camera.ImgFromCam(Eigen::Vector3d(-0.5, -0.5, 1.0))->x(), 0.0);
  EXPECT_EQ(camera.ImgFromCam(Eigen::Vector3d(-0.5, -0.5, 1.0))->y(), 0.0);
}

TEST(base_camera, TestRescale) {
  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.Rescale(2.0);
  EXPECT_EQ(camera.Width(), 2);
  EXPECT_EQ(camera.Height(), 2);
  EXPECT_EQ(camera.FocalLength(), 2);
  EXPECT_EQ(camera.PrincipalPointX(), 1);
  EXPECT_EQ(camera.PrincipalPointY(), 1);

  camera.InitializeWithName("PINHOLE", 1.0, 1, 1);
  camera.Rescale(2.0);
  EXPECT_EQ(camera.Width(), 2);
  EXPECT_EQ(camera.Height(), 2);
  EXPECT_EQ(camera.FocalLengthX(), 2);
  EXPECT_EQ(camera.FocalLengthY(), 2);
  EXPECT_EQ(camera.PrincipalPointX(), 1);
  EXPECT_EQ(camera.PrincipalPointY(), 1);

  camera.InitializeWithName("PINHOLE", 1.0, 2, 2);
  camera.Rescale(0.5);
  EXPECT_EQ(camera.Width(), 1);
  EXPECT_EQ(camera.Height(), 1);
  EXPECT_EQ(camera.FocalLengthX(), 0.5);
  EXPECT_EQ(camera.FocalLengthY(), 0.5);
  EXPECT_EQ(camera.PrincipalPointX(), 0.5);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);

  camera.InitializeWithName("PINHOLE", 1.0, 2, 2);
  camera.Rescale(1, 1);
  EXPECT_EQ(camera.Width(), 1);
  EXPECT_EQ(camera.Height(), 1);
  EXPECT_EQ(camera.FocalLengthX(), 0.5);
  EXPECT_EQ(camera.FocalLengthY(), 0.5);
  EXPECT_EQ(camera.PrincipalPointX(), 0.5);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);

  camera.InitializeWithName("PINHOLE", 1.0, 2, 2);
  camera.Rescale(4, 4);
  EXPECT_EQ(camera.Width(), 4);
  EXPECT_EQ(camera.Height(), 4);
  EXPECT_EQ(camera.FocalLengthX(), 2);
  EXPECT_EQ(camera.FocalLengthY(), 2);
  EXPECT_EQ(camera.PrincipalPointX(), 2);
  EXPECT_EQ(camera.PrincipalPointY(), 2);
}

TEST(Camera, Equals) {
  Camera camera = Camera::CreateFromModelId(
      1, SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  Camera other = camera;
  EXPECT_EQ(camera, other);
  camera.SetFocalLength(2.);
  EXPECT_NE(camera, other);
  other.SetFocalLength(2.);
  EXPECT_EQ(camera, other);
  // Upstream parity (f1ea91d0): the calibration source participates in
  // equality.
  camera.SetSource(CameraSource::USER);
  EXPECT_NE(camera, other);
  other.SetSource(CameraSource::USER);
  EXPECT_EQ(camera, other);
}

// Upstream parity (f1ea91d0): the calibration source is the single source of
// truth for the prior flag; persisted enum values are decoupled from the
// reliability ranking via CameraSourcePriority.
TEST(Camera, HasPriorFocalLength) {
  EXPECT_LT(CameraSourcePriority(CameraSource::BEST),
            CameraSourcePriority(CameraSource::UNKNOWN));
  EXPECT_LT(CameraSourcePriority(CameraSource::UNKNOWN),
            CameraSourcePriority(CameraSource::GUESS));
  EXPECT_LT(CameraSourcePriority(CameraSource::GUESS),
            CameraSourcePriority(CameraSource::EXIF));
  EXPECT_LT(CameraSourcePriority(CameraSource::EXIF),
            CameraSourcePriority(CameraSource::SINGLE_VIEW));
  EXPECT_LT(CameraSourcePriority(CameraSource::SINGLE_VIEW),
            CameraSourcePriority(CameraSource::VIEW_GRAPH));
  EXPECT_LT(CameraSourcePriority(CameraSource::VIEW_GRAPH),
            CameraSourcePriority(CameraSource::USER));

  Camera camera;
  camera.SetSource(CameraSource::UNKNOWN);
  EXPECT_FALSE(camera.HasPriorFocalLength());
  camera.SetSource(CameraSource::GUESS);
  EXPECT_FALSE(camera.HasPriorFocalLength());
  camera.SetSource(CameraSource::EXIF);
  EXPECT_TRUE(camera.HasPriorFocalLength());
  camera.SetSource(CameraSource::SINGLE_VIEW);
  EXPECT_TRUE(camera.HasPriorFocalLength());
  camera.SetSource(CameraSource::USER);
  EXPECT_TRUE(camera.HasPriorFocalLength());
  camera.SetSource(CameraSource::VIEW_GRAPH);
  EXPECT_TRUE(camera.HasPriorFocalLength());

  const Camera camera_with_source = Camera::CreateFromModelId(
      1, SimplePinholeCameraModel::model_id, 1.0, 1, 1,
      CameraSource::EXIF);
  EXPECT_EQ(camera_with_source.Source(), CameraSource::EXIF);
  EXPECT_THROW(Camera::CreateFromModelId(1,
                                         SimplePinholeCameraModel::model_id,
                                         1.0,
                                         1,
                                         1,
                                         CameraSource::BEST),
               std::invalid_argument);
}

TEST(Camera, Print) {
  Camera camera = Camera::CreateFromModelId(
      1, SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  std::ostringstream stream;
  stream << camera;
  EXPECT_EQ(stream.str(),
            "Camera(camera_id=1, model=SIMPLE_PINHOLE, width=1, height=1, "
            "params=[1, 0.5, 0.5] (f, cx, cy))");
}

TEST(Camera, SensorId) {
  Camera camera;
  camera.SetCameraId(1);
  EXPECT_EQ(camera.SensorId(), sensor_t(SensorType::CAMERA, 1));
}

TEST(Camera, CreateFromModelId) {
  Camera camera = Camera::CreateFromModelId(
      1, SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.CameraId(), 1);
  EXPECT_EQ(camera.ModelId(), SimplePinholeCameraModel::model_id);
  EXPECT_EQ(camera.ModelName(), "SIMPLE_PINHOLE");
  EXPECT_EQ(camera.Width(), 1);
  EXPECT_EQ(camera.Height(), 1);
  EXPECT_FALSE(camera.HasPriorFocalLength());
  EXPECT_EQ(camera.FocalLengthIdxs().size(), 1);
  EXPECT_EQ(camera.PrincipalPointIdxs().size(), 2);
  EXPECT_EQ(camera.ExtraParamsIdxs().size(), 0);
  EXPECT_EQ(camera.MetaDataParamsIdxs().size(), 0);
  EXPECT_EQ(camera.ParamsInfo(), "f, cx, cy");
  EXPECT_EQ(camera.ParamsToString(), "1, 0.5, 0.5");
  EXPECT_EQ(camera.FocalLength(), 1.0);
  EXPECT_EQ(camera.PrincipalPointX(), 0.5);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);
  EXPECT_TRUE(camera.VerifyParams());
  EXPECT_FALSE(camera.HasBogusParams(0.1, 2.0, 1.0));
  EXPECT_TRUE(camera.HasBogusParams(0.1, 0.5, 1.0));
  EXPECT_EQ(camera.Params().size(),
            static_cast<int>(SimplePinholeCameraModel::num_params));
  EXPECT_EQ(camera.Params().size(),
            static_cast<int>(SimplePinholeCameraModel::num_params));
}

TEST(Camera, CreateFromModelName) {
  Camera camera = Camera::CreateFromModelName(1, "SIMPLE_PINHOLE", 1.0, 1, 1);
  EXPECT_EQ(camera.CameraId(), 1);
  EXPECT_EQ(camera.ModelId(), SimplePinholeCameraModel::model_id);
  EXPECT_EQ(camera.ModelName(), "SIMPLE_PINHOLE");
  EXPECT_EQ(camera.Width(), 1);
  EXPECT_EQ(camera.Height(), 1);
  EXPECT_FALSE(camera.HasPriorFocalLength());
  EXPECT_EQ(camera.FocalLengthIdxs().size(), 1);
  EXPECT_EQ(camera.PrincipalPointIdxs().size(), 2);
  EXPECT_EQ(camera.ExtraParamsIdxs().size(), 0);
  EXPECT_EQ(camera.MetaDataParamsIdxs().size(), 0);
  EXPECT_EQ(camera.ParamsInfo(), "f, cx, cy");
  EXPECT_EQ(camera.ParamsToString(), "1, 0.5, 0.5");
  EXPECT_EQ(camera.FocalLength(), 1.0);
  EXPECT_EQ(camera.PrincipalPointX(), 0.5);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);
  EXPECT_TRUE(camera.VerifyParams());
  EXPECT_FALSE(camera.HasBogusParams(0.1, 2.0, 1.0));
  EXPECT_TRUE(camera.HasBogusParams(0.1, 0.5, 1.0));
  EXPECT_EQ(camera.Params().size(),
            static_cast<int>(SimplePinholeCameraModel::num_params));
  EXPECT_EQ(camera.Params().size(),
            static_cast<int>(SimplePinholeCameraModel::num_params));
}

TEST(Camera, ImgFromCamWithoutCheiralityCheck) {
  const Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kSimplePinhole, 1.0, 1, 1);
  const Eigen::Vector3d cam_point(-0.5, -0.5, -1);
  EXPECT_FALSE(camera.ImgFromCam(cam_point).has_value());
  EXPECT_EQ(camera.ImgFromCam(cam_point, /*check_cheirality=*/false).value(),
            Eigen::Vector2d(1.0, 1.0));
}

TEST(Camera, CamRayFromImgWithJac) {
  // Covers a perspective model, a distorted one, and a spherical one. For the
  // spherical camera the sampled pixels include the back hemisphere, where
  // CamFromImg fails outright but CamRayFromImg (and hence this) must not.
  const std::vector<Camera> cameras = {
      Camera::CreateFromModelId(
          1, CameraModelId::kSimplePinhole, 650.0, 1024, 768),
      Camera::CreateFromModelId(
          2, CameraModelId::kSimpleRadial, 650.0, 1024, 768),
      Camera::CreateFromModelId(
          3, CameraModelId::kEquirectangular, 0.0, 1000, 500),
  };

  for (const Camera& camera : cameras) {
    for (const double x : {1.0, 250.0, 512.0, 800.0, 999.0}) {
      for (const double y : {1.0, 120.0, 250.0, 400.0, 499.0}) {
        const Eigen::Vector2d image_point(x, y);
        const auto ray_and_jac = camera.CamRayFromImgWithJac(image_point);
        ASSERT_TRUE(ray_and_jac.has_value())
            << "model " << camera.ModelName() << " at "
            << image_point.transpose();
        const auto& [cam_ray, J_ray] = *ray_and_jac;

        // The bearing is a unit vector that reprojects to the source pixel.
        EXPECT_NEAR(cam_ray.norm(), 1.0, 1e-12);
        const std::optional<Eigen::Vector2d> reprojected =
            camera.ImgFromCam(cam_ray);
        ASSERT_TRUE(reprojected.has_value());
        EXPECT_LE((*reprojected - image_point).norm(), 1e-8);

        // The Jacobian inverts the projection on the tangent plane.
        Eigen::Matrix2x3d J_uvw;
        ASSERT_TRUE(camera.ImgFromCamWithJac(cam_ray, &J_uvw).has_value());
        EXPECT_LE((J_uvw * J_ray - Eigen::Matrix2d::Identity()).norm(), 1e-10);
        EXPECT_LE((J_ray.transpose() * cam_ray).norm(), 1e-10 * J_ray.norm());
      }
    }
  }
}

TEST(Camera, CamRayFromImgWithJacUnprojectable) {
  // A pixel the camera cannot unproject (iterative undistortion diverges) must
  // yield no ray + Jacobian, rather than a garbage one.
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kOpenCV, 100.0, 100, 200);
  camera.Params()[4] = -0.5;  // k1
  camera.Params()[5] = 0.5;   // k2
  camera.Params()[6] = -0.5;  // p1
  const Eigen::Vector2d unprojectable(50.0, 150.0);
  ASSERT_FALSE(camera.CamFromImg(unprojectable).has_value());
  EXPECT_FALSE(camera.CamRayFromImgWithJac(unprojectable).has_value());
}

TEST(Camera, Spherical) {
  Camera camera = Camera::CreateFromModelId(
      1, EquirectangularCameraModel::model_id, /*focal_length=*/0.0, 1000, 500);
  EXPECT_EQ(camera.Params(), std::vector<double>({1000, 500}));
  EXPECT_FALSE(camera.IsPerspective());
  EXPECT_TRUE(camera.IsSpherical());

  // No focal length / pinhole image plane. The (w, h) parameters form the
  // metadata group; there is no principal point or extra (distortion).
  EXPECT_TRUE(camera.FocalLengthIdxs().empty());
  EXPECT_TRUE(camera.PrincipalPointIdxs().empty());
  EXPECT_TRUE(camera.ExtraParamsIdxs().empty());
  EXPECT_EQ(camera.MetaDataParamsIdxs().size(), 2);
  EXPECT_EQ(camera.MeanFocalLength(), 0.0);
  // CalibrationMatrix is undefined without a focal length.
  EXPECT_ANY_THROW(camera.CalibrationMatrix());
  // Spherical images have no lens distortion to undistort.
  EXPECT_TRUE(camera.IsUndistorted());

  // Rescaling keeps the (w, h) parameters consistent with the dimensions.
  camera.Rescale(0.5);
  EXPECT_EQ(camera.Width(), 500);
  EXPECT_EQ(camera.Height(), 250);
  EXPECT_EQ(camera.Params(), std::vector<double>({500, 250}));
  EXPECT_TRUE(camera.IsSpherical());

  // A perspective camera is not spherical.
  const Camera pinhole =
      Camera::CreateFromModelId(2, PinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(pinhole.IsPerspective());
  EXPECT_FALSE(pinhole.IsSpherical());
}

TEST(Camera, IsPerspectivePinhole) {
  const Camera pinhole =
      Camera::CreateFromModelId(1, PinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(pinhole.IsPerspectivePinhole());

  const Camera radial = Camera::CreateFromModelId(
      2, SimpleRadialCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(radial.IsPerspectivePinhole());

  const Camera fisheye = Camera::CreateFromModelId(
      3, OpenCVFisheyeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(fisheye.IsPerspective());
  EXPECT_TRUE(fisheye.IsPerspectiveFisheye());
  EXPECT_FALSE(fisheye.IsPerspectivePinhole());

  const Camera spherical = Camera::CreateFromModelId(
      4, EquirectangularCameraModel::model_id, 0.0, 1000, 500);
  EXPECT_FALSE(spherical.IsPerspectiveFisheye());
  EXPECT_FALSE(spherical.IsPerspectivePinhole());
}

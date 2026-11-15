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

#define TEST_NAME "base/undistortion"
#include "util/testing.h"

#include "geometry/pose.h"
#include "image/undistortion.h"

using namespace colmap;

TEST(base_undistortion, TestUndistortCamera) {
  UndistortCameraOptions options;
  Camera distorted_camera;
  Camera undistorted_camera;

  distorted_camera.InitializeWithName("SIMPLE_PINHOLE", 1, 1, 1);
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 1);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 1);
  EXPECT_EQ(undistorted_camera.Width(), 1);
  EXPECT_EQ(undistorted_camera.Height(), 1);

  distorted_camera.InitializeWithName("SIMPLE_RADIAL", 1, 1, 1);
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 1);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 1);
  EXPECT_EQ(undistorted_camera.Width(), 1);
  EXPECT_EQ(undistorted_camera.Height(), 1);

  distorted_camera.InitializeWithName("SIMPLE_RADIAL", 100, 100, 100);
  distorted_camera.Params(3) = 0.5;
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.PrincipalPointX(), 84.0 / 2.0);
  EXPECT_EQ(undistorted_camera.PrincipalPointY(), 84.0 / 2.0);
  EXPECT_EQ(undistorted_camera.Width(), 84);
  EXPECT_EQ(undistorted_camera.Height(), 84);

  options.blank_pixels = 1;
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.Width(), 90);
  EXPECT_EQ(undistorted_camera.Height(), 90);

  options.max_scale = 0.75;
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.Width(), 75);
  EXPECT_EQ(undistorted_camera.Height(), 75);

  options.max_scale = 1.0;
  options.roi_min_x = 0.1;
  options.roi_min_y = 0.2;
  options.roi_max_x = 0.9;
  options.roi_max_y = 0.8;
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.Width(), 80);
  EXPECT_EQ(undistorted_camera.Height(), 60);
  EXPECT_EQ(undistorted_camera.PrincipalPointX(), 40);
  EXPECT_EQ(undistorted_camera.PrincipalPointY(), 30);
}

TEST(base_undistortion, TestUndistortCameraMaxCamPointNorm) {
  Camera distorted_camera;
  distorted_camera.InitializeWithName("SIMPLE_FISHEYE", 130.0, 200, 100);
  distorted_camera.SetPrincipalPointX(10.0);
  distorted_camera.SetPrincipalPointY(50.0);

  UndistortCameraOptions options;
  options.blank_pixels = 1.0;
  const Camera unbounded = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(unbounded.Width(),
            distorted_camera.Width() * options.max_scale);
  EXPECT_EQ(unbounded.Height(),
            distorted_camera.Height() * options.max_scale);

  options.max_cam_point_norm = 2.0;
  const Camera bounded = UndistortCamera(options, distorted_camera);
  EXPECT_LT(bounded.Width(), unbounded.Width());
  EXPECT_LT(bounded.Height(), unbounded.Height());

  options.max_cam_point_norm = 0.0;
  EXPECT_ANY_THROW(UndistortCamera(options, distorted_camera));
}

TEST(base_undistortion, TestUndistortCameraBlankPixels) {
  UndistortCameraOptions options;
  options.blank_pixels = 1;

  Camera distorted_camera;
  distorted_camera.InitializeWithName("SIMPLE_RADIAL", 100, 100, 100);
  distorted_camera.Params(3) = 0.5;

  Bitmap distorted_image;
  distorted_image.Allocate(100, 100, false);
  distorted_image.Fill(BitmapColor<uint8_t>(255));

  Bitmap undistorted_image;
  Camera undistorted_camera;
  UndistortImage(options, distorted_image, distorted_camera, &undistorted_image,
                 &undistorted_camera);

  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.PrincipalPointX(), 90.0 / 2.0);
  EXPECT_EQ(undistorted_camera.PrincipalPointY(), 90.0 / 2.0);
  EXPECT_EQ(undistorted_camera.Width(), 90);
  EXPECT_EQ(undistorted_camera.Height(), 90);

  // Make sure that there is no blank pixel.
  size_t num_blank_pixels = 0;
  for (int y = 0; y < undistorted_image.Height(); ++y) {
    for (int x = 0; x < undistorted_image.Width(); ++x) {
      BitmapColor<uint8_t> color;
      EXPECT_TRUE(undistorted_image.GetPixel(x, y, &color));
      if (color == BitmapColor<uint8_t>(0)) {
        num_blank_pixels += 1;
      }
    }
  }

  EXPECT_GT(num_blank_pixels, 0);
}

TEST(base_undistortion, TestUndistortCameraNoBlankPixels) {
  UndistortCameraOptions options;
  options.blank_pixels = 0;

  Camera distorted_camera;
  distorted_camera.InitializeWithName("SIMPLE_RADIAL", 100, 100, 100);
  distorted_camera.Params(3) = 0.5;

  Bitmap distorted_image;
  distorted_image.Allocate(100, 100, false);
  distorted_image.Fill(BitmapColor<uint8_t>(255));

  Bitmap undistorted_image;
  Camera undistorted_camera;
  UndistortImage(options, distorted_image, distorted_camera, &undistorted_image,
                 &undistorted_camera);

  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.PrincipalPointX(), 84.0 / 2.0);
  EXPECT_EQ(undistorted_camera.PrincipalPointY(), 84.0 / 2.0);
  EXPECT_EQ(undistorted_camera.Width(), 84);
  EXPECT_EQ(undistorted_camera.Height(), 84);

  // Make sure that there is no blank pixel.
  for (int y = 0; y < undistorted_image.Height(); ++y) {
    for (int x = 0; x < undistorted_image.Width(); ++x) {
      BitmapColor<uint8_t> color;
      EXPECT_TRUE(undistorted_image.GetPixel(x, y, &color));
      EXPECT_NE(color.r, 0);
      EXPECT_EQ(color.g, color.r);
      EXPECT_EQ(color.b, color.r);
    }
  }
}

TEST(base_undistortion, TestUndistortReconstruction) {
  const size_t kNumImages = 10;
  const size_t kNumPoints2D = 10;

  Reconstruction reconstruction;

  Camera camera;
  camera.SetCameraId(1);
  camera.InitializeWithName("OPENCV", 1, 1, 1);
  camera.Params(4) = 1.0;
  reconstruction.AddCameraWithTrivialRig(camera);

  for (image_t image_id = 1; image_id <= kNumImages; ++image_id) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(1);
    image.SetName("image" + std::to_string(image_id));
    image.SetPoints2D(
        std::vector<Eigen::Vector2d>(kNumPoints2D, Eigen::Vector2d::Ones()));
    reconstruction.AddImageWithTrivialFrame(image, Rigid3d());
  }

  UndistortCameraOptions options;
  UndistortReconstruction(options, &reconstruction);
  for (const auto& camera : reconstruction.Cameras()) {
    EXPECT_EQ(camera.second.ModelName(), "PINHOLE");
  }

  for (const auto& image : reconstruction.Images()) {
    for (const auto& point2D : image.second.Points2D()) {
      EXPECT_NE(point2D.XY(), Eigen::Vector2d::Ones());
    }
  }
}

TEST(base_undistortion, TestRectifyStereoCameras) {
  Camera camera1;
  camera1.SetCameraId(1);
  camera1.InitializeWithName("PINHOLE", 1, 1, 1);

  Camera camera2;
  camera2.SetCameraId(1);
  camera2.InitializeWithName("PINHOLE", 1, 1, 1);

  const Eigen::Vector4d qvec =
      RotationMatrixToQuaternion(EulerAnglesToRotationMatrix(0.1, 0.2, 0.3));
  const Eigen::Vector3d tvec(0.1, 0.2, 0.3);

  Camera rectified_camera1;
  Camera rectified_camera2;
  Eigen::Matrix3d H1;
  Eigen::Matrix3d H2;
  Eigen::Matrix4d Q;
  RectifyStereoCameras(camera1, camera2, qvec, tvec, &H1, &H2, &Q);

  Eigen::Matrix3d H1_ref;
  H1_ref << -0.202759, -0.815848, -0.897034, 0.416329, 0.733069, -0.199657,
      0.910839, -0.175408, 0.942638;
  EXPECT_TRUE(H1.isApprox(H1_ref.transpose(), 1e-5));

  Eigen::Matrix3d H2_ref;
  H2_ref << -0.082173, -1.01288, -0.698868, 0.301854, 0.472844, -0.465336,
      0.963533, 0.292411, 1.12528;
  EXPECT_TRUE(H2.isApprox(H2_ref.transpose(), 1e-5));

  Eigen::Matrix4d Q_ref;
  Q_ref << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -2.67261, -0.5, -0.5, 1, 0;
  EXPECT_TRUE(Q.isApprox(Q_ref, 1e-5));
}

// Upstream parity (937064dd undistortion_test.cc).
TEST(base_undistortion, MaxCamPointNorm) {
  // Fisheye camera with off-center principal point: pixels far from (cx, cy)
  // have theta close to pi/2, so CamFromImg returns very large values
  // (|cam_point| = tan(theta)) that blow up the output dimensions.
  Camera distorted_camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimpleFisheye, 130, 200, 100);
  distorted_camera.SetPrincipalPointX(10);
  distorted_camera.SetPrincipalPointY(50);

  UndistortCameraOptions options;
  // Exercise the path driven by extreme samples.
  options.blank_pixels = 1.0;

  // Default (max_cam_point_norm = -1): the extreme cam_points push the scale
  // factor against max_scale, so the output is the maximum allowed size.
  const Camera undistorted_camera_unbounded =
      UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera_unbounded.Width(),
            distorted_camera.Width() * options.max_scale);
  EXPECT_EQ(undistorted_camera_unbounded.Height(),
            distorted_camera.Height() * options.max_scale);

  // With a finite threshold, extreme border samples are skipped, yielding
  // a smaller output that is no longer hitting the max_scale clamp.
  options.max_cam_point_norm = 2.0;
  const Camera undistorted_camera_bounded =
      UndistortCamera(options, distorted_camera);
  EXPECT_LT(undistorted_camera_bounded.Width(),
            undistorted_camera_unbounded.Width());
  EXPECT_LT(undistorted_camera_bounded.Height(),
            undistorted_camera_unbounded.Height());

  // max_cam_point_norm = 0 is invalid (would skip every sample).
  options.max_cam_point_norm = 0;
  EXPECT_ANY_THROW(UndistortCamera(options, distorted_camera));
}

TEST(base_undistortion, TestUndistortReconstructionRescalesAlreadyUndistorted) {
  const size_t kNumImages = 3;

  Reconstruction reconstruction;

  // Already-undistorted (PINHOLE) camera at a high resolution. focal=100,
  // width=height=100 => principal point at (50, 50).
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kPinhole, 100, 100, 100);
  ASSERT_TRUE(camera.IsUndistorted());
  reconstruction.AddCameraWithTrivialRig(camera);

  const Eigen::Vector2d point_xy(60, 40);
  for (image_t image_id = 1; image_id <= kNumImages; ++image_id) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(1);
    image.SetName("image" + std::to_string(image_id));
    image.SetPoints2D(std::vector<Eigen::Vector2d>(1, point_xy));
    reconstruction.AddImageWithTrivialFrame(image, Rigid3d());
  }

  UndistortCameraOptions options;
  options.max_image_size = 50;
  UndistortReconstruction(options, &reconstruction);

  // The camera resolution and intrinsics are rescaled to match
  // max_image_size (through the perspective undistortion path).
  const Camera& undistorted_camera = reconstruction.Camera(1);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.Width(), 50);
  EXPECT_EQ(undistorted_camera.Height(), 50);
  EXPECT_NEAR(undistorted_camera.FocalLengthX(), 50, 1e-6);
  EXPECT_NEAR(undistorted_camera.FocalLengthY(), 50, 1e-6);
  EXPECT_NEAR(undistorted_camera.PrincipalPointX(), 25, 1e-6);
  EXPECT_NEAR(undistorted_camera.PrincipalPointY(), 25, 1e-6);

  // The observations are rescaled consistently with the camera: (60, 40)
  // maps to normalized (0.1, -0.1) and back through the halved intrinsics
  // to (0.1 * 50 + 25, -0.1 * 50 + 25) = (30, 20).
  for (const auto& image : reconstruction.Images()) {
    ASSERT_EQ(image.second.NumPoints2D(), 1);
    EXPECT_NEAR(image.second.Point2D(0).XY().x(), 30, 1e-6);
    EXPECT_NEAR(image.second.Point2D(0).XY().y(), 20, 1e-6);
  }
}

TEST(base_undistortion, TestUndistortReconstructionRescalesSphericalCameras) {
  const size_t kNumImages = 3;

  Reconstruction reconstruction;

  // Spherical (EQUIRECTANGULAR) camera at a high resolution. It has no
  // pinhole plane to undistort to, but can still be resized to a smaller
  // image of the same model.
  Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0.0, 1000, 500);
  ASSERT_TRUE(camera.IsSpherical());
  reconstruction.AddCameraWithTrivialRig(camera);

  // A front-hemisphere pixel (azimuth 36 deg) and a back-hemisphere pixel
  // (azimuth -144 deg, behind the camera). The back-hemisphere observation
  // has no forward normalized (CamFromImg) representation, so it must be
  // handled by plain linear scaling rather than a bearing round-trip.
  const std::vector<Eigen::Vector2d> points_xy = {Eigen::Vector2d(600, 200),
                                                  Eigen::Vector2d(100, 400)};
  for (image_t image_id = 1; image_id <= kNumImages; ++image_id) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(1);
    image.SetName("image" + std::to_string(image_id));
    image.SetPoints2D(points_xy);
    reconstruction.AddImageWithTrivialFrame(image, Rigid3d());
  }

  UndistortCameraOptions options;
  options.max_image_size = 250;
  UndistortReconstruction(options, &reconstruction);

  // The camera keeps its model but is resized so its larger dimension
  // (width) matches max_image_size: scale = 250 / 1000 = 0.25.
  const Camera& undistorted_camera = reconstruction.Camera(1);
  EXPECT_EQ(undistorted_camera.ModelName(), "EQUIRECTANGULAR");
  EXPECT_EQ(undistorted_camera.Width(), 250);
  EXPECT_EQ(undistorted_camera.Height(), 125);

  // The observations are rescaled consistently with the camera: the
  // fractional position within the image is preserved (a plain 0.25 scaling
  // here), so both the front- and back-hemisphere points map through
  // without loss.
  for (const auto& image : reconstruction.Images()) {
    ASSERT_EQ(image.second.NumPoints2D(), 2);
    EXPECT_NEAR(image.second.Point2D(0).XY().x(), 150, 1e-6);
    EXPECT_NEAR(image.second.Point2D(0).XY().y(), 50, 1e-6);
    EXPECT_NEAR(image.second.Point2D(1).XY().x(), 25, 1e-6);
    EXPECT_NEAR(image.second.Point2D(1).XY().y(), 100, 1e-6);
  }
}

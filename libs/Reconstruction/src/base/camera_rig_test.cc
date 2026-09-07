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

#define TEST_NAME "base/camera_rig"
#include "util/testing.h"

#include <sstream>

#include "base/camera_rig.h"

using namespace colmap;

TEST(base_camera_rig, TestEmpty) {
  CameraRig camera_rig;
  EXPECT_EQ(camera_rig.NumCameras(), 0);
  EXPECT_EQ(camera_rig.NumSnapshots(), 0);
  EXPECT_EQ(camera_rig.GetCameraIds().size(), 0);
  EXPECT_EQ(camera_rig.HasCamera(0), false);
}

TEST(base_camera_rig, TestAddCamera) {
  CameraRig camera_rig;
  EXPECT_EQ(camera_rig.NumCameras(), 0);
  EXPECT_EQ(camera_rig.NumSnapshots(), 0);
  EXPECT_EQ(camera_rig.GetCameraIds().size(), 0);
  EXPECT_EQ(camera_rig.HasCamera(0), false);

  camera_rig.AddCamera(0, ComposeIdentityQuaternion(),
                       Eigen::Vector3d(0, 1, 2));
  EXPECT_EQ(camera_rig.NumCameras(), 1);
  EXPECT_EQ(camera_rig.NumSnapshots(), 0);
  EXPECT_EQ(camera_rig.GetCameraIds().size(), 1);
  EXPECT_EQ(camera_rig.GetCameraIds()[0], 0);
  EXPECT_EQ(camera_rig.HasCamera(0), true);
  EXPECT_EQ(camera_rig.RelativeQvec(0), ComposeIdentityQuaternion());
  EXPECT_EQ(camera_rig.RelativeTvec(0), Eigen::Vector3d(0, 1, 2));

  camera_rig.AddCamera(1, ComposeIdentityQuaternion(),
                       Eigen::Vector3d(3, 4, 5));
  EXPECT_EQ(camera_rig.NumCameras(), 2);
  EXPECT_EQ(camera_rig.NumSnapshots(), 0);
  EXPECT_EQ(camera_rig.GetCameraIds().size(), 2);
  EXPECT_EQ(camera_rig.HasCamera(0), true);
  EXPECT_EQ(camera_rig.HasCamera(1), true);
  EXPECT_EQ(camera_rig.RelativeQvec(1), ComposeIdentityQuaternion());
  EXPECT_EQ(camera_rig.RelativeTvec(1), Eigen::Vector3d(3, 4, 5));
}

TEST(base_camera_rig, TestAddSnapshot) {
  CameraRig camera_rig;
  EXPECT_EQ(camera_rig.NumCameras(), 0);
  EXPECT_EQ(camera_rig.NumSnapshots(), 0);
  EXPECT_EQ(camera_rig.GetCameraIds().size(), 0);
  EXPECT_EQ(camera_rig.Snapshots().size(), 0);

  camera_rig.AddCamera(0, ComposeIdentityQuaternion(),
                       Eigen::Vector3d(0, 1, 2));
  camera_rig.AddCamera(1, ComposeIdentityQuaternion(),
                       Eigen::Vector3d(3, 4, 5));
  EXPECT_EQ(camera_rig.NumCameras(), 2);
  EXPECT_EQ(camera_rig.NumSnapshots(), 0);
  EXPECT_EQ(camera_rig.Snapshots().size(), 0);

  const std::vector<image_t> image_ids1 = {0, 1};
  camera_rig.AddSnapshot(image_ids1);
  EXPECT_EQ(camera_rig.NumCameras(), 2);
  EXPECT_EQ(camera_rig.NumSnapshots(), 1);
  EXPECT_EQ(camera_rig.Snapshots().size(), 1);
  EXPECT_EQ(camera_rig.Snapshots()[0].size(), 2);
  EXPECT_EQ(camera_rig.Snapshots()[0][0], 0);
  EXPECT_EQ(camera_rig.Snapshots()[0][1], 1);

  const std::vector<image_t> image_ids2 = {2, 3};
  camera_rig.AddSnapshot(image_ids2);
  EXPECT_EQ(camera_rig.NumCameras(), 2);
  EXPECT_EQ(camera_rig.NumSnapshots(), 2);
  EXPECT_EQ(camera_rig.Snapshots().size(), 2);
  EXPECT_EQ(camera_rig.Snapshots()[0].size(), 2);
  EXPECT_EQ(camera_rig.Snapshots()[0][0], 0);
  EXPECT_EQ(camera_rig.Snapshots()[0][1], 1);
  EXPECT_EQ(camera_rig.Snapshots()[1].size(), 2);
  EXPECT_EQ(camera_rig.Snapshots()[1][0], 2);
  EXPECT_EQ(camera_rig.Snapshots()[1][1], 3);
}

TEST(base_camera_rig, TestSerializationPreservesRelativePoseAndSnapshots) {
  CameraRig source;
  source.AddCamera(7, ComposeIdentityQuaternion(), Eigen::Vector3d::Zero());
  source.AddCamera(11, Eigen::Vector4d(0.9238795325, 0, 0.3826834324, 0),
                   Eigen::Vector3d(0.3, -0.2, 1.1));
  source.SetRefCameraId(7);
  source.AddSnapshot({100, 101});
  source.AddSnapshot({102});

  std::stringstream text_stream;
  source.WriteText(&text_stream);
  CameraRig text_round_trip;
  ASSERT_TRUE(text_round_trip.ReadText(&text_stream));
  EXPECT_EQ(text_round_trip.RefCameraId(), source.RefCameraId());
  EXPECT_EQ(text_round_trip.GetCameraIds().size(), 2);
  ASSERT_LE(std::abs((text_round_trip.RelativeQvec(11) -
                     source.RelativeQvec(11))
                            .norm()), 1e-15);
  ASSERT_LE(std::abs((text_round_trip.RelativeTvec(11) -
                     source.RelativeTvec(11))
                            .norm()), 1e-15);
  ASSERT_TRUE(std::equal(text_round_trip.Snapshots()[0].begin(), text_round_trip.Snapshots()[0].end(), source.Snapshots()[0].begin(), source.Snapshots()[0].end()));

  std::stringstream binary_stream;
  source.WriteBinary(&binary_stream);
  CameraRig binary_round_trip;
  ASSERT_TRUE(binary_round_trip.ReadBinary(&binary_stream));
  EXPECT_EQ(binary_round_trip.RefCameraId(), source.RefCameraId());
  ASSERT_LE(std::abs((binary_round_trip.RelativeQvec(11) -
                     source.RelativeQvec(11))
                            .norm()), 1e-15);
  ASSERT_LE(std::abs((binary_round_trip.RelativeTvec(11) -
                     source.RelativeTvec(11))
                            .norm()), 1e-15);
  ASSERT_TRUE(std::equal(binary_round_trip.Snapshots()[1].begin(), binary_round_trip.Snapshots()[1].end(), source.Snapshots()[1].begin(), source.Snapshots()[1].end()));
}

TEST(base_camera_rig, TestCheck) {
  CameraRig camera_rig;
  camera_rig.AddCamera(0, ComposeIdentityQuaternion(),
                       Eigen::Vector3d(0, 1, 2));
  camera_rig.AddCamera(1, ComposeIdentityQuaternion(),
                       Eigen::Vector3d(3, 4, 5));
  const std::vector<image_t> image_ids1 = {0, 1};
  camera_rig.AddSnapshot(image_ids1);
  const std::vector<image_t> image_ids2 = {2, 3};
  camera_rig.AddSnapshot(image_ids2);

  Reconstruction reconstruction;

  Camera camera1;
  camera1.SetCameraId(0);
  camera1.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction.AddCameraWithTrivialRig(camera1);

  Camera camera2;
  camera2.SetCameraId(1);
  camera2.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction.AddCameraWithTrivialRig(camera2);

  Image image1;
  image1.SetImageId(0);
  image1.SetCameraId(camera1.CameraId());
  reconstruction.AddImageWithTrivialFrame(image1);

  Image image2;
  image2.SetImageId(1);
  image2.SetCameraId(camera2.CameraId());
  reconstruction.AddImageWithTrivialFrame(image2);

  Image image3;
  image3.SetImageId(2);
  image3.SetCameraId(camera1.CameraId());
  reconstruction.AddImageWithTrivialFrame(image3);

  Image image4;
  image4.SetImageId(3);
  image4.SetCameraId(camera2.CameraId());
  reconstruction.AddImageWithTrivialFrame(image4);

  camera_rig.SetRefCameraId(0);
  camera_rig.Check(reconstruction);
}

TEST(base_camera_rig, TestComputeScale) {
  CameraRig camera_rig;
  camera_rig.AddCamera(0, ComposeIdentityQuaternion(),
                       Eigen::Vector3d(0, 0, 0));
  camera_rig.AddCamera(1, ComposeIdentityQuaternion(),
                       Eigen::Vector3d(2, 4, 6));
  const std::vector<image_t> image_ids1 = {0, 1};
  camera_rig.AddSnapshot(image_ids1);

  Reconstruction reconstruction;

  Camera camera1;
  camera1.SetCameraId(0);
  camera1.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction.AddCameraWithTrivialRig(camera1);

  Camera camera2;
  camera2.SetCameraId(1);
  camera2.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction.AddCameraWithTrivialRig(camera2);

  Image image1;
  image1.SetImageId(0);
  image1.SetCameraId(camera1.CameraId());
  image1.SetQvec(ComposeIdentityQuaternion());
  image1.SetTvec(Eigen::Vector3d(0, 0, 0));
  reconstruction.AddImageWithTrivialFrame(image1);

  Image image2;
  image2.SetImageId(1);
  image2.SetCameraId(camera2.CameraId());
  image2.SetQvec(ComposeIdentityQuaternion());
  image2.SetTvec(Eigen::Vector3d(1, 2, 3));
  reconstruction.AddImageWithTrivialFrame(image2);

  camera_rig.SetRefCameraId(0);
  camera_rig.Check(reconstruction);

  EXPECT_EQ(camera_rig.ComputeScale(reconstruction), 2.0);

  reconstruction.Image(1).SetTvec(Eigen::Vector3d(0, 0, 0));
  EXPECT_TRUE(IsNaN(camera_rig.ComputeScale(reconstruction)));
}

TEST(base_camera_rig, TestComputeRelativePoses) {
  CameraRig camera_rig;
  camera_rig.AddCamera(0, ComposeIdentityQuaternion(),
                       Eigen::Vector3d(0, 0, 0));
  camera_rig.AddCamera(1, ComposeIdentityQuaternion(),
                       Eigen::Vector3d(0, 0, 0));
  const std::vector<image_t> image_ids1 = {0, 1};
  camera_rig.AddSnapshot(image_ids1);

  Reconstruction reconstruction;

  Camera camera1;
  camera1.SetCameraId(0);
  camera1.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction.AddCameraWithTrivialRig(camera1);

  Camera camera2;
  camera2.SetCameraId(1);
  camera2.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction.AddCameraWithTrivialRig(camera2);

  Image image1;
  image1.SetImageId(0);
  image1.SetCameraId(camera1.CameraId());
  image1.SetQvec(ComposeIdentityQuaternion());
  image1.SetTvec(Eigen::Vector3d(0, 0, 0));
  reconstruction.AddImageWithTrivialFrame(image1);

  Image image2;
  image2.SetImageId(1);
  image2.SetCameraId(camera2.CameraId());
  image2.SetQvec(ComposeIdentityQuaternion());
  image2.SetTvec(Eigen::Vector3d(1, 2, 3));
  reconstruction.AddImageWithTrivialFrame(image2);

  camera_rig.SetRefCameraId(0);
  camera_rig.Check(reconstruction);
  camera_rig.ComputeRelativePoses(reconstruction);
  EXPECT_EQ(camera_rig.RelativeQvec(0), ComposeIdentityQuaternion());
  EXPECT_EQ(camera_rig.RelativeTvec(0), Eigen::Vector3d(0, 0, 0));
  EXPECT_EQ(camera_rig.RelativeQvec(1), ComposeIdentityQuaternion());
  EXPECT_EQ(camera_rig.RelativeTvec(1), Eigen::Vector3d(1, 2, 3));

  const std::vector<image_t> image_ids2 = {2, 3};
  camera_rig.AddSnapshot(image_ids2);

  Image image3;
  image3.SetImageId(2);
  image3.SetCameraId(camera1.CameraId());
  image3.SetQvec(ComposeIdentityQuaternion());
  image3.SetTvec(Eigen::Vector3d(0, 0, 0));
  reconstruction.AddImageWithTrivialFrame(image3);

  Image image4;
  image4.SetImageId(3);
  image4.SetCameraId(camera2.CameraId());
  image4.SetQvec(ComposeIdentityQuaternion());
  image4.SetTvec(Eigen::Vector3d(2, 4, 6));
  reconstruction.AddImageWithTrivialFrame(image4);

  camera_rig.Check(reconstruction);
  camera_rig.ComputeRelativePoses(reconstruction);
  EXPECT_EQ(camera_rig.RelativeQvec(0), ComposeIdentityQuaternion());
  EXPECT_EQ(camera_rig.RelativeTvec(0), Eigen::Vector3d(0, 0, 0));
  EXPECT_EQ(camera_rig.RelativeQvec(1), ComposeIdentityQuaternion());
  EXPECT_EQ(camera_rig.RelativeTvec(1), Eigen::Vector3d(1.5, 3, 4.5));

  const std::vector<image_t> image_ids3 = {4};
  camera_rig.AddSnapshot(image_ids3);

  Image image5;
  image5.SetImageId(4);
  image5.SetCameraId(camera1.CameraId());
  image5.SetQvec(ComposeIdentityQuaternion());
  image5.SetTvec(Eigen::Vector3d(0, 0, 0));
  reconstruction.AddImageWithTrivialFrame(image5);

  camera_rig.Check(reconstruction);
  camera_rig.ComputeRelativePoses(reconstruction);
  EXPECT_EQ(camera_rig.RelativeQvec(0), ComposeIdentityQuaternion());
  EXPECT_EQ(camera_rig.RelativeTvec(0), Eigen::Vector3d(0, 0, 0));
  EXPECT_EQ(camera_rig.RelativeQvec(1), ComposeIdentityQuaternion());
  EXPECT_EQ(camera_rig.RelativeTvec(1), Eigen::Vector3d(1.5, 3, 4.5));
}

TEST(base_camera_rig, TestComputeRelativePosesWithRotationAndTranslation) {
  CameraRig camera_rig;
  camera_rig.AddCamera(3, ComposeIdentityQuaternion(), Eigen::Vector3d::Zero());
  camera_rig.AddCamera(5, ComposeIdentityQuaternion(), Eigen::Vector3d::Zero());
  camera_rig.SetRefCameraId(3);
  camera_rig.AddSnapshot({30, 50});

  Reconstruction reconstruction;
  Camera ref_camera;
  ref_camera.SetCameraId(3);
  ref_camera.InitializeWithName("PINHOLE", 1.0, 1, 1);
  reconstruction.AddCameraWithTrivialRig(ref_camera);
  Camera other_camera;
  other_camera.SetCameraId(5);
  other_camera.InitializeWithName("PINHOLE", 1.0, 1, 1);
  reconstruction.AddCameraWithTrivialRig(other_camera);

  const Eigen::Vector4d ref_qvec(0.9238795325112867, 0.0, 0.0,
                                  0.3826834323650898);
  const Eigen::Vector3d ref_tvec(0.4, -0.2, 1.1);
  const Eigen::Vector4d expected_relative_qvec(0.9659258262890683, 0.0,
                                                0.2588190451025207, 0.0);
  const Eigen::Vector3d expected_relative_tvec(-0.3, 0.7, 0.2);

  Image ref_image;
  ref_image.SetImageId(30);
  ref_image.SetCameraId(3);
  ref_image.SetQvec(ref_qvec);
  ref_image.SetTvec(ref_tvec);
  reconstruction.AddImageWithTrivialFrame(ref_image);

  Image other_image;
  other_image.SetImageId(50);
  other_image.SetCameraId(5);
  other_image.SetQvec(
      ConcatenateQuaternions(ref_qvec, expected_relative_qvec));
  other_image.SetTvec(expected_relative_tvec +
                       QuaternionRotatePoint(expected_relative_qvec, ref_tvec));
  reconstruction.AddImageWithTrivialFrame(other_image);

  camera_rig.Check(reconstruction);
  ASSERT_TRUE(camera_rig.ComputeRelativePoses(reconstruction));
  ASSERT_LE(std::abs((QuaternionToRotationMatrix(camera_rig.RelativeQvec(3)) -
                     Eigen::Matrix3d::Identity())
                        .norm()), 1e-12);
  ASSERT_LE(std::abs((camera_rig.RelativeTvec(3)).norm()), 1e-12);
  ASSERT_LE(std::abs((QuaternionToRotationMatrix(camera_rig.RelativeQvec(5)) -
       QuaternionToRotationMatrix(expected_relative_qvec))
          .norm()), 1e-12);
  ASSERT_LE(std::abs((camera_rig.RelativeTvec(5) - expected_relative_tvec).norm()), 1e-12);
}

TEST(base_camera_rig, TestComputeAbsolutePose) {
  CameraRig camera_rig;
  camera_rig.AddCamera(0, ComposeIdentityQuaternion(),
                       Eigen::Vector3d(0, 1, 2));
  camera_rig.AddCamera(1, ComposeIdentityQuaternion(),
                       Eigen::Vector3d(3, 4, 5));
  const std::vector<image_t> image_ids1 = {0, 1};
  camera_rig.AddSnapshot(image_ids1);

  Reconstruction reconstruction;

  Camera camera1;
  camera1.SetCameraId(0);
  camera1.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction.AddCameraWithTrivialRig(camera1);

  Camera camera2;
  camera2.SetCameraId(1);
  camera2.InitializeWithName("PINHOLE", 1, 1, 1);
  reconstruction.AddCameraWithTrivialRig(camera2);

  Image image1;
  image1.SetImageId(0);
  image1.SetCameraId(camera1.CameraId());
  image1.SetQvec(ComposeIdentityQuaternion());
  image1.SetTvec(Eigen::Vector3d(0, 0, 0));
  reconstruction.AddImageWithTrivialFrame(image1);

  Image image2;
  image2.SetImageId(1);
  image2.SetCameraId(camera2.CameraId());
  image2.SetQvec(ComposeIdentityQuaternion());
  image2.SetTvec(Eigen::Vector3d(3, 3, 3));
  reconstruction.AddImageWithTrivialFrame(image2);

  camera_rig.SetRefCameraId(0);
  camera_rig.Check(reconstruction);

  Eigen::Vector4d abs_qvec;
  Eigen::Vector3d abs_tvec;
  camera_rig.ComputeAbsolutePose(0, reconstruction, &abs_qvec, &abs_tvec);
  EXPECT_EQ(abs_qvec, ComposeIdentityQuaternion());
  EXPECT_EQ(abs_tvec, Eigen::Vector3d(0, -1, -2));
}

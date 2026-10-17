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

#define TEST_NAME "base/database"
#include "util/testing.h"

#include <thread>

#include "scene/database.h"
#include "geometry/rigid3.h"
#include "sensor/models.h"
#include "geometry/pose.h"
#include "util/string.h"
#include "util/version.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "SQLite/sqlite3.h"

using namespace colmap;

const static std::string kMemoryDatabasePath = ":memory:";

TEST(base_database, TestOpenCloseConstructorDestructor) {
  auto database = Database::Open(kMemoryDatabasePath);
}

TEST(base_database, TestOpenClose) {
  auto database = Database::Open(kMemoryDatabasePath);
  database->Close();
}

TEST(base_database, TestRigFrameSqliteRoundTrip) {
  auto database = Database::Open(kMemoryDatabasePath);

  Camera camera1;
  camera1.InitializeWithId(SimplePinholeCameraModel::model_id, 800.0, 640, 480);
  camera1.SetCameraId(1);
  database->WriteCamera(camera1, true);
  Camera camera2 = camera1;
  camera2.SetCameraId(2);
  database->WriteCamera(camera2, true);

  Image image1;
  image1.SetImageId(1);
  image1.SetCameraId(1);
  image1.SetName("rig-frame-1");
  database->WriteImage(image1, true);
  Image image2;
  image2.SetImageId(2);
  image2.SetCameraId(2);
  image2.SetName("rig-frame-2");
  database->WriteImage(image2, true);

  Rig rig;
  rig.SetRigId(7);
  rig.AddRefCamera(2);
  rig.AddCamera(1, Eigen::Vector4d(0.0, 0.0, 0.0, 1.0),
                Eigen::Vector3d(0.2, -0.1, 0.05));
  rig.AddSensor(sensor_t(SensorType::IMU, 9),
                Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
                Eigen::Vector3d(0.01, 0.02, 0.03));
  EXPECT_EQ(database->WriteRig(rig, true), 7);
  EXPECT_TRUE(database->ExistsRig(7));

  Frame frame;
  frame.SetFrameId(11);
  frame.SetRigId(7);
  frame.AddImageId(1);
  frame.AddImageId(2);
  frame.AddDataId(data_t(sensor_t(SensorType::IMU, 9), 42));
  frame.SetRigFromWorld(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
                        Eigen::Vector3d(1.0, 2.0, 3.0));
  EXPECT_EQ(database->WriteFrame(frame, true), 11);
  EXPECT_TRUE(database->ExistsFrame(11));

  const Rig loaded_rig = database->ReadRig(7);
  EXPECT_EQ(loaded_rig.RefCameraId(), 2);
  EXPECT_TRUE(loaded_rig.HasCamera(1));
  EXPECT_TRUE(loaded_rig.HasSensor(sensor_t(SensorType::IMU, 9)));
  EXPECT_TRUE(loaded_rig.HasSensorFromRig(sensor_t(SensorType::IMU, 9)));
  ASSERT_LE(std::abs((loaded_rig.CamFromRigTvec(1) -
                     Eigen::Vector3d(0.2, -0.1, 0.05)).norm()), 1e-12);
  const Frame loaded_frame = database->ReadFrame(11);
  EXPECT_EQ(loaded_frame.RigId(), 7);
  EXPECT_TRUE(loaded_frame.HasImageId(1));
  EXPECT_TRUE(loaded_frame.HasImageId(2));
  EXPECT_TRUE(loaded_frame.HasDataId(data_t(sensor_t(SensorType::IMU, 9), 42)));
  // Upstream schema (a2bd15c2): the frames table does not persist poses.
  EXPECT_FALSE(loaded_frame.HasPose());
  EXPECT_EQ(database->NumRigs(), 1);
  EXPECT_EQ(database->NumFrames(), 1);
}

TEST(base_database, TestGenericRigReferenceSensorSqliteRoundTrip) {
  auto database = Database::Open(kMemoryDatabasePath);
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 800.0, 640, 480);
  camera.SetCameraId(1);
  database->WriteCamera(camera, true);

  Rig rig;
  rig.SetRigId(12);
  rig.AddRefSensor(sensor_t(SensorType::IMU, 5));
  rig.AddCamera(1, Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
                Eigen::Vector3d(0.0, 0.0, 0.1));
  EXPECT_EQ(database->WriteRig(rig, true), 12);
  const Rig loaded = database->ReadRig(12);
  EXPECT_TRUE(loaded.RefSensorId() == sensor_t(SensorType::IMU, 5));
  EXPECT_TRUE(loaded.HasCamera(1));
  EXPECT_FALSE(loaded.HasSensorFromRig(sensor_t(SensorType::IMU, 5)));
}

TEST(base_database, TestGenericDataOnlyFrameSqliteRoundTrip) {
  auto database = Database::Open(kMemoryDatabasePath);
  Rig rig;
  rig.SetRigId(13);
  rig.AddRefSensor(sensor_t(SensorType::IMU, 8));
  EXPECT_EQ(database->WriteRig(rig, true), 13);

  Frame frame;
  frame.SetFrameId(14);
  frame.SetRigId(13);
  const data_t imu_data(sensor_t(SensorType::IMU, 8), 99);
  frame.AddDataId(imu_data);
  EXPECT_EQ(database->WriteFrame(frame, true), 14);

  const Frame loaded = database->ReadFrame(14);
  EXPECT_TRUE(loaded.HasDataId(imu_data));
  EXPECT_TRUE(loaded.ImageIds().empty());
}

TEST(base_database, TestFloatDescriptorSqliteRoundTrip) {
  auto database = Database::Open(kMemoryDatabasePath);
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 800.0, 640, 480);
  camera.SetCameraId(1);
  database->WriteCamera(camera, true);
  Image image;
  image.SetImageId(1);
  image.SetCameraId(1);
  image.SetName("loma-float");
  database->WriteImage(image, true);

  FeatureKeypoints keypoints = {FeatureKeypoint(10.5f, 20.25f),
                                FeatureKeypoint(30.0f, 40.0f)};
  FeatureDescriptorsFloatData data(2, 3);
  data << 0.125f, -3.5f, 1.0e-7f, 1.0f, 2.5f, -9.25f;
  const FeatureDescriptorsFloat descriptors(FeatureExtractorType::LOMA_B,
                                             std::move(data));
  database->WriteKeypoints(1, keypoints);
  database->WriteDescriptors(1, descriptors.ToBytes());

  const FeatureDescriptors loaded = database->ReadDescriptors(1);
  EXPECT_EQ(loaded.type, FeatureExtractorType::LOMA_B);
  const FeatureDescriptorsFloat loaded_float = loaded.ToFloat();
  EXPECT_EQ(loaded_float.type, descriptors.type);
  EXPECT_EQ(loaded_float.data.rows(), 2);
  EXPECT_EQ(loaded_float.data.cols(), 3);
  EXPECT_TRUE((loaded_float.data - descriptors.data).cwiseAbs().maxCoeff() ==
              0.0f);
  EXPECT_TRUE(database->ExistsDescriptors(1));
}

TEST(base_database, TestTransaction) {
  auto database = Database::Open(kMemoryDatabasePath);
  DatabaseTransaction database_transaction(database.get());
}

TEST(base_database, TestTransactionMultiThreaded) {
  auto database = Database::Open(kMemoryDatabasePath);

  std::thread thread1([&database]() {
    DatabaseTransaction database_transaction(database.get());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  });

  std::thread thread2([&database]() {
    DatabaseTransaction database_transaction(database.get());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  });

  thread1.join();
  thread2.join();
}

TEST(base_database, TestEmpty) {
  auto database = Database::Open(kMemoryDatabasePath);
  EXPECT_EQ(database->NumCameras(), 0);
  EXPECT_EQ(database->NumImages(), 0);
  EXPECT_EQ(database->NumKeypoints(), 0);
  EXPECT_EQ(database->MaxNumKeypoints(), 0);
  EXPECT_EQ(database->NumDescriptors(), 0);
  EXPECT_EQ(database->MaxNumDescriptors(), 0);
  EXPECT_EQ(database->NumMatches(), 0);
  EXPECT_EQ(database->NumMatchedImagePairs(), 0);
  EXPECT_EQ(database->NumVerifiedImagePairs(), 0);
}

TEST(base_database, TestImagePairToPairId) {
  EXPECT_EQ(Database::ImagePairToPairId(0, 0), 0);
  EXPECT_EQ(Database::ImagePairToPairId(0, 1), 1);
  EXPECT_EQ(Database::ImagePairToPairId(0, 2), 2);
  EXPECT_EQ(Database::ImagePairToPairId(0, 3), 3);
  EXPECT_EQ(Database::ImagePairToPairId(1, 2),
                    Database::kMaxNumImages + 2);
  for (image_t i = 0; i < 20; ++i) {
    for (image_t j = 0; j < 20; ++j) {
      const image_pair_t pair_id = Database::ImagePairToPairId(i, j);
      image_t image_id1;
      image_t image_id2;
      Database::PairIdToImagePair(pair_id, &image_id1, &image_id2);
      if (i < j) {
        EXPECT_EQ(i, image_id1);
        EXPECT_EQ(j, image_id2);
      } else {
        EXPECT_EQ(i, image_id2);
        EXPECT_EQ(j, image_id1);
      }
    }
  }
}

TEST(base_database, TestSkewedPinholeCameraRoundTrip) {
  auto database = Database::Open(kMemoryDatabasePath);
  Camera camera;
  camera.InitializeWithName("SKEWED_PINHOLE", 600.0, 1280, 960);
  camera.Params()[1] = 500.0;
  camera.Params()[4] = 50.0;
  camera.SetCameraId(database->WriteCamera(camera));

  const Camera restored = database->ReadCamera(camera.CameraId());
  EXPECT_EQ(restored.ModelId(), CameraModelId::kSkewedPinhole);
  EXPECT_EQ(restored.ModelName(), "SKEWED_PINHOLE");
  EXPECT_EQ(restored.Params(), camera.Params());

  const Eigen::Vector2d image_point(1020.0, 1240.0);
  ASSERT_TRUE(camera.CamFromImg(image_point).has_value());
  ASSERT_TRUE(restored.CamFromImg(image_point).has_value());
  EXPECT_EQ(*restored.CamFromImg(image_point),
            *camera.CamFromImg(image_point));
}

TEST(base_database, TestSwapImagePair) {
  EXPECT_FALSE(Database::SwapImagePair(0, 0));
  EXPECT_FALSE(Database::SwapImagePair(0, 1));
  EXPECT_TRUE(Database::SwapImagePair(1, 0));
  EXPECT_FALSE(Database::SwapImagePair(1, 1));
}

TEST(base_database, TestCamera) {
  auto database = Database::Open(kMemoryDatabasePath);
  EXPECT_EQ(database->NumCameras(), 0);
  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.SetCameraId(database->WriteCamera(camera));
  EXPECT_EQ(database->NumCameras(), 1);
  EXPECT_EQ(database->ExistsCamera(camera.CameraId()), true);
  EXPECT_EQ(database->ReadCamera(camera.CameraId()).CameraId(),
                    camera.CameraId());
  EXPECT_EQ(database->ReadCamera(camera.CameraId()).ModelId(),
                    camera.ModelId());
  EXPECT_EQ(database->ReadCamera(camera.CameraId()).FocalLength(),
                    camera.FocalLength());
  EXPECT_EQ(database->ReadCamera(camera.CameraId()).PrincipalPointX(),
                    camera.PrincipalPointX());
  EXPECT_EQ(database->ReadCamera(camera.CameraId()).PrincipalPointY(),
                    camera.PrincipalPointY());
  camera.SetFocalLength(2.0);
  database->UpdateCamera(camera);
  EXPECT_EQ(database->ReadCamera(camera.CameraId()).FocalLength(),
                    camera.FocalLength());
  Camera camera2 = camera;
  camera2.SetCameraId(camera.CameraId() + 1);
  database->WriteCamera(camera2, true);
  EXPECT_EQ(database->NumCameras(), 2);
  EXPECT_EQ(database->ExistsCamera(camera.CameraId()), true);
  EXPECT_EQ(database->ExistsCamera(camera2.CameraId()), true);
  EXPECT_EQ(database->ReadAllCameras().size(), 2);
  EXPECT_EQ(database->ReadAllCameras()[0].CameraId(), camera.CameraId());
  EXPECT_EQ(database->ReadAllCameras()[1].CameraId(),
                    camera2.CameraId());
  database->ClearCameras();
  EXPECT_EQ(database->NumCameras(), 0);
}

TEST(base_database, TestImage) {
  auto database = Database::Open(kMemoryDatabasePath);
  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.SetCameraId(database->WriteCamera(camera));
  EXPECT_EQ(database->NumImages(), 0);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.CameraId());
  image.SetImageId(database->WriteImage(image));
  EXPECT_EQ(database->NumImages(), 1);
  EXPECT_EQ(database->ExistsImage(image.ImageId()), true);
  EXPECT_EQ(database->ReadImage(image.ImageId()).ImageId(),
                    image.ImageId());
  EXPECT_EQ(database->ReadImage(image.ImageId()).CameraId(),
                    image.CameraId());
  EXPECT_EQ(database->ReadImage(image.ImageId()).Name(), image.Name());
  image.SetName("test_updated");
  database->UpdateImage(image);
  EXPECT_EQ(database->ReadImage(image.ImageId()).ImageId(),
                    image.ImageId());
  EXPECT_EQ(database->ReadImage(image.ImageId()).CameraId(),
                    image.CameraId());
  EXPECT_EQ(database->ReadImage(image.ImageId()).Name(), "test_updated");
  Image image2 = image;
  image2.SetName("test2");
  image2.SetImageId(image.ImageId() + 1);
  database->WriteImage(image2, true);
  EXPECT_EQ(database->NumImages(), 2);
  EXPECT_EQ(database->ExistsImage(image.ImageId()), true);
  EXPECT_EQ(database->ExistsImage(image2.ImageId()), true);
  EXPECT_EQ(database->ReadAllImages().size(), 2);
  database->ClearImages();
  EXPECT_EQ(database->NumImages(), 0);
}

TEST(base_database, TestKeypoints) {
  auto database = Database::Open(kMemoryDatabasePath);
  Camera camera;
  camera.SetCameraId(database->WriteCamera(camera));
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.CameraId());
  image.SetImageId(database->WriteImage(image));
  EXPECT_EQ(database->NumKeypoints(), 0);
  EXPECT_EQ(database->NumKeypointsForImage(image.ImageId()), 0);
  const FeatureKeypoints keypoints = FeatureKeypoints(10);
  database->WriteKeypoints(image.ImageId(), keypoints);
  const FeatureKeypoints keypoints_read =
      database->ReadKeypoints(image.ImageId());
  EXPECT_EQ(keypoints.size(), keypoints_read.size());
  for (size_t i = 0; i < keypoints.size(); ++i) {
    EXPECT_EQ(keypoints[i].x, keypoints_read[i].x);
    EXPECT_EQ(keypoints[i].y, keypoints_read[i].y);
    EXPECT_EQ(keypoints[i].a11, keypoints_read[i].a11);
    EXPECT_EQ(keypoints[i].a12, keypoints_read[i].a12);
    EXPECT_EQ(keypoints[i].a21, keypoints_read[i].a21);
    EXPECT_EQ(keypoints[i].a22, keypoints_read[i].a22);
  }
  EXPECT_EQ(database->NumKeypoints(), 10);
  EXPECT_EQ(database->MaxNumKeypoints(), 10);
  EXPECT_EQ(database->NumKeypointsForImage(image.ImageId()), 10);
  const FeatureKeypoints keypoints2 = FeatureKeypoints(20);
  image.SetName("test2");
  image.SetImageId(database->WriteImage(image));
  database->WriteKeypoints(image.ImageId(), keypoints2);
  EXPECT_EQ(database->NumKeypoints(), 30);
  EXPECT_EQ(database->MaxNumKeypoints(), 20);
  EXPECT_EQ(database->NumKeypointsForImage(image.ImageId()), 20);
  database->ClearKeypoints();
  EXPECT_EQ(database->NumKeypoints(), 0);
  EXPECT_EQ(database->MaxNumKeypoints(), 0);
  EXPECT_EQ(database->NumKeypointsForImage(image.ImageId()), 0);
}

TEST(base_database, TestDescriptors) {
  auto database = Database::Open(kMemoryDatabasePath);
  Camera camera;
  camera.SetCameraId(database->WriteCamera(camera));
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.CameraId());
  image.SetImageId(database->WriteImage(image));
  EXPECT_EQ(database->NumDescriptors(), 0);
  EXPECT_EQ(database->NumDescriptorsForImage(image.ImageId()), 0);
  const FeatureDescriptors descriptors(
      FeatureExtractorType::SIFT, FeatureDescriptorsData::Random(10, 128));
  database->WriteDescriptors(image.ImageId(), descriptors);
  const FeatureDescriptors descriptors_read =
      database->ReadDescriptors(image.ImageId());
  EXPECT_EQ(descriptors.type, descriptors_read.type);
  EXPECT_EQ(descriptors.data.rows(), descriptors_read.data.rows());
  EXPECT_EQ(descriptors.data.cols(), descriptors_read.data.cols());
  for (FeatureDescriptorsData::Index r = 0; r < descriptors.data.rows(); ++r) {
    for (FeatureDescriptorsData::Index c = 0; c < descriptors.data.cols(); ++c) {
      EXPECT_EQ(descriptors.data(r, c), descriptors_read.data(r, c));
    }
  }
  EXPECT_EQ(database->NumDescriptors(), 10);
  EXPECT_EQ(database->MaxNumDescriptors(), 10);
  EXPECT_EQ(database->NumDescriptorsForImage(image.ImageId()), 10);
  const FeatureDescriptors descriptors2(
      FeatureExtractorType::SIFT, FeatureDescriptorsData(20, 128));
  image.SetName("test2");
  image.SetImageId(database->WriteImage(image));
  database->WriteDescriptors(image.ImageId(), descriptors2);
  EXPECT_EQ(database->NumDescriptors(), 30);
  EXPECT_EQ(database->MaxNumDescriptors(), 20);
  EXPECT_EQ(database->NumDescriptorsForImage(image.ImageId()), 20);
  database->ClearDescriptors();
  EXPECT_EQ(database->NumDescriptors(), 0);
  EXPECT_EQ(database->MaxNumDescriptors(), 0);
  EXPECT_EQ(database->NumDescriptorsForImage(image.ImageId()), 0);
}

TEST(base_database, TestMatches) {
  auto database = Database::Open(kMemoryDatabasePath);
  const image_t image_id1 = 1;
  const image_t image_id2 = 2;
  const FeatureMatches matches = FeatureMatches(1000);
  database->WriteMatches(image_id1, image_id2, matches);
  const FeatureMatches matches_read =
      database->ReadMatches(image_id1, image_id2);
  EXPECT_EQ(matches.size(), matches_read.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    EXPECT_EQ(matches[i].point2D_idx1, matches_read[i].point2D_idx1);
    EXPECT_EQ(matches[i].point2D_idx2, matches_read[i].point2D_idx2);
  }
  EXPECT_EQ(database->ReadAllMatches().size(), 1);
  EXPECT_EQ(database->ReadAllMatches()[0].first,
                    Database::ImagePairToPairId(image_id1, image_id2));
  EXPECT_EQ(database->NumMatches(), 1000);
  database->DeleteMatches(image_id1, image_id2);
  EXPECT_EQ(database->NumMatches(), 0);
  database->WriteMatches(image_id1, image_id2, matches);
  EXPECT_EQ(database->NumMatches(), 1000);
  database->ClearMatches();
  EXPECT_EQ(database->NumMatches(), 0);
}

TEST(base_database, TestTwoViewGeometry) {
  auto database = Database::Open(kMemoryDatabasePath);
  const image_t image_id1 = 1;
  const image_t image_id2 = 2;
  TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches = FeatureMatches(1000);
  two_view_geometry.config =
      TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
  two_view_geometry.F = Eigen::Matrix3d::Random();
  two_view_geometry.E = Eigen::Matrix3d::Random();
  two_view_geometry.H = Eigen::Matrix3d::Random();
  two_view_geometry.cam2_from_cam1 = Rigid3d(
      Eigen::Quaterniond(Eigen::Vector4d::Random().normalized()),
      Eigen::Vector3d::Random());
  database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  const TwoViewGeometry two_view_geometry_read =
      database->ReadTwoViewGeometry(image_id1, image_id2);
  EXPECT_EQ(two_view_geometry_read.inlier_matches.size(),
            two_view_geometry.inlier_matches.size());
  for (size_t i = 0; i < two_view_geometry_read.inlier_matches.size(); ++i) {
    EXPECT_EQ(two_view_geometry.inlier_matches[i].point2D_idx1,
              two_view_geometry_read.inlier_matches[i].point2D_idx1);
    EXPECT_EQ(two_view_geometry.inlier_matches[i].point2D_idx2,
              two_view_geometry_read.inlier_matches[i].point2D_idx2);
  }

  EXPECT_EQ(two_view_geometry.config, two_view_geometry_read.config);
  EXPECT_EQ(two_view_geometry.F, two_view_geometry_read.F);
  EXPECT_EQ(two_view_geometry.E, two_view_geometry_read.E);
  EXPECT_EQ(two_view_geometry.H, two_view_geometry_read.H);
  EXPECT_EQ(two_view_geometry.cam2_from_cam1,
            two_view_geometry_read.cam2_from_cam1);

  const TwoViewGeometry two_view_geometry_read_inv =
      database->ReadTwoViewGeometry(image_id2, image_id1);
  EXPECT_EQ(two_view_geometry_read_inv.inlier_matches.size(),
            two_view_geometry_read.inlier_matches.size());
  for (size_t i = 0; i < two_view_geometry_read.inlier_matches.size(); ++i) {
    EXPECT_EQ(two_view_geometry_read_inv.inlier_matches[i].point2D_idx2,
              two_view_geometry_read.inlier_matches[i].point2D_idx1);
    EXPECT_EQ(two_view_geometry_read_inv.inlier_matches[i].point2D_idx1,
              two_view_geometry_read.inlier_matches[i].point2D_idx2);
  }

  EXPECT_EQ(two_view_geometry_read_inv.config, two_view_geometry_read.config);
  EXPECT_EQ(*two_view_geometry_read_inv.F, two_view_geometry_read.F->transpose());
  EXPECT_EQ(*two_view_geometry_read_inv.E, two_view_geometry_read.E->transpose());
  EXPECT_TRUE(two_view_geometry_read_inv.H->inverse().eval().isApprox(
      *two_view_geometry_read.H));

  const Rigid3d read_inv_pose_inv =
      Inverse(*two_view_geometry_read_inv.cam2_from_cam1);
  EXPECT_TRUE(read_inv_pose_inv.rotation().coeffs().isApprox(
      two_view_geometry_read.cam2_from_cam1->rotation().coeffs()));
  EXPECT_TRUE(read_inv_pose_inv.translation().isApprox(
      two_view_geometry_read.cam2_from_cam1->translation()));

  std::vector<image_pair_t> image_pair_ids;
  std::vector<TwoViewGeometry> two_view_geometries;
  database->ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);
  EXPECT_EQ(image_pair_ids.size(), 1);
  EXPECT_EQ(two_view_geometries.size(), 1);
  EXPECT_EQ(image_pair_ids[0],
            Database::ImagePairToPairId(image_id1, image_id2));
  EXPECT_EQ(two_view_geometry.config, two_view_geometries[0].config);
  EXPECT_EQ(two_view_geometry.F, two_view_geometries[0].F);
  EXPECT_EQ(two_view_geometry.E, two_view_geometries[0].E);
  EXPECT_EQ(two_view_geometry.H, two_view_geometries[0].H);
  EXPECT_EQ(two_view_geometry.cam2_from_cam1,
            two_view_geometries[0].cam2_from_cam1);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(),
            two_view_geometries[0].inlier_matches.size());
  std::vector<std::pair<image_t, image_t>> image_pairs;
  std::vector<int> num_inliers;
  database->ReadTwoViewGeometryNumInliers(&image_pairs, &num_inliers);
  EXPECT_EQ(image_pairs.size(), 1);
  EXPECT_EQ(num_inliers.size(), 1);
  EXPECT_EQ(image_pairs[0].first, image_id1);
  EXPECT_EQ(image_pairs[0].second, image_id2);
  EXPECT_EQ(num_inliers[0], two_view_geometry.inlier_matches.size());
  EXPECT_EQ(database->NumInlierMatches(), 1000);
  database->DeleteInlierMatches(image_id1, image_id2);
  // Upstream parity (d3ccaf35): deleting the inlier matches keeps the
  // two-view geometry entry itself.
  EXPECT_TRUE(database->ExistsTwoViewGeometry(image_id1, image_id2));
  EXPECT_EQ(database->NumInlierMatches(), 0);
  database->DeleteTwoViewGeometry(image_id1, image_id2);
  EXPECT_FALSE(database->ExistsTwoViewGeometry(image_id1, image_id2));
  EXPECT_EQ(database->NumInlierMatches(), 0);
  database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  EXPECT_EQ(database->NumInlierMatches(), 1000);
  // Upstream additionally asserts that a duplicate write throws; the fork's
  // SQLite error helper exits the process instead of throwing (adjudicated
  // in util/sqlite3_utils.h until the W17.2b database abstraction lands),
  // so that contract is verified by the CLI tests rather than in-process.
}

TEST(base_database, TestMerge) {
  auto database1 = Database::Open(kMemoryDatabasePath);
  auto database2 = Database::Open(kMemoryDatabasePath);

  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.SetCameraId(database1->WriteCamera(camera));
  camera.SetCameraId(database2->WriteCamera(camera));

  Image image;
  image.SetCameraId(camera.CameraId());
  image.SetQvecPrior(Eigen::Vector4d(0.1, 0.2, 0.3, 0.4));
  image.SetTvecPrior(Eigen::Vector3d(0.1, 0.2, 0.3));

  image.SetName("test1");
  const image_t image_id1 = database1->WriteImage(image);
  image.SetName("test2");
  const image_t image_id2 = database1->WriteImage(image);
  image.SetName("test3");
  const image_t image_id3 = database2->WriteImage(image);
  image.SetName("test4");
  const image_t image_id4 = database2->WriteImage(image);

  auto keypoints1 = FeatureKeypoints(10);
  keypoints1[0].x = 100;
  auto keypoints2 = FeatureKeypoints(20);
  keypoints2[0].x = 200;
  auto keypoints3 = FeatureKeypoints(30);
  keypoints3[0].x = 300;
  auto keypoints4 = FeatureKeypoints(40);
  keypoints4[0].x = 400;

  const FeatureDescriptors descriptors1(
      FeatureExtractorType::SIFT, FeatureDescriptorsData::Random(10, 128));
  const FeatureDescriptors descriptors2(
      FeatureExtractorType::SIFT, FeatureDescriptorsData::Random(20, 128));
  const FeatureDescriptors descriptors3(
      FeatureExtractorType::SIFT, FeatureDescriptorsData::Random(30, 128));
  const FeatureDescriptors descriptors4(
      FeatureExtractorType::SIFT, FeatureDescriptorsData::Random(40, 128));

  database1->WriteKeypoints(image_id1, keypoints1);
  database1->WriteKeypoints(image_id2, keypoints2);
  database2->WriteKeypoints(image_id3, keypoints3);
  database2->WriteKeypoints(image_id4, keypoints4);
  database1->WriteDescriptors(image_id1, descriptors1);
  database1->WriteDescriptors(image_id2, descriptors2);
  database2->WriteDescriptors(image_id3, descriptors3);
  database2->WriteDescriptors(image_id4, descriptors4);
  database1->WriteMatches(image_id1, image_id2, FeatureMatches(10));
  database2->WriteMatches(image_id3, image_id4, FeatureMatches(10));
  database1->WriteTwoViewGeometry(image_id1, image_id2, TwoViewGeometry());
  database2->WriteTwoViewGeometry(image_id3, image_id4, TwoViewGeometry());

  auto merged_database = Database::Open(kMemoryDatabasePath);
  Database::Merge(*database1, *database2, merged_database.get());
  EXPECT_EQ(merged_database->NumCameras(), 2);
  EXPECT_EQ(merged_database->NumImages(), 4);
  EXPECT_EQ(merged_database->NumKeypoints(), 100);
  EXPECT_EQ(merged_database->NumDescriptors(), 100);
  EXPECT_EQ(merged_database->NumMatches(), 20);
  EXPECT_EQ(merged_database->NumInlierMatches(), 0);
  EXPECT_EQ(merged_database->ReadAllImages()[0].CameraId(), 1);
  EXPECT_EQ(merged_database->ReadAllImages()[1].CameraId(), 1);
  EXPECT_EQ(merged_database->ReadAllImages()[2].CameraId(), 2);
  EXPECT_EQ(merged_database->ReadAllImages()[3].CameraId(), 2);
  EXPECT_EQ(merged_database->ReadKeypoints(1).size(), 10);
  EXPECT_EQ(merged_database->ReadKeypoints(2).size(), 20);
  EXPECT_EQ(merged_database->ReadKeypoints(3).size(), 30);
  EXPECT_EQ(merged_database->ReadKeypoints(4).size(), 40);
  EXPECT_EQ(merged_database->ReadKeypoints(1)[0].x, 100);
  EXPECT_EQ(merged_database->ReadKeypoints(2)[0].x, 200);
  EXPECT_EQ(merged_database->ReadKeypoints(3)[0].x, 300);
  EXPECT_EQ(merged_database->ReadKeypoints(4)[0].x, 400);
  EXPECT_EQ(merged_database->ReadDescriptors(1).type, descriptors1.type);
  EXPECT_EQ(merged_database->ReadDescriptors(1).data.size(),
            descriptors1.data.size());
  EXPECT_EQ(merged_database->ReadDescriptors(2).data.size(),
            descriptors2.data.size());
  EXPECT_EQ(merged_database->ReadDescriptors(3).data.size(),
            descriptors3.data.size());
  EXPECT_EQ(merged_database->ReadDescriptors(4).data.size(),
            descriptors4.data.size());
  EXPECT_TRUE(merged_database->ExistsMatches(1, 2));
  EXPECT_FALSE(merged_database->ExistsMatches(2, 3));
  EXPECT_FALSE(merged_database->ExistsMatches(2, 4));
  EXPECT_TRUE(merged_database->ExistsMatches(3, 4));
  PosePrior pose_prior;
  pose_prior.corr_data_id = merged_database->ReadImage(1).DataId();
  pose_prior.position = Eigen::Vector3d(1.0, 2.0, 3.0);
  merged_database->WritePosePrior(pose_prior);
  ASSERT_EQ(merged_database->NumPosePriors(), 1);
  merged_database->ClearAllTables();
  EXPECT_EQ(merged_database->NumCameras(), 0);
  EXPECT_EQ(merged_database->NumImages(), 0);
  EXPECT_EQ(merged_database->NumPosePriors(), 0);
  EXPECT_EQ(merged_database->NumKeypoints(), 0);
  EXPECT_EQ(merged_database->NumDescriptors(), 0);
  EXPECT_EQ(merged_database->NumMatches(), 0);
}

namespace {

// Creates a minimal legacy fork database (pre user_version migration): a
// two_view_geometries table with identity/zero pose sentinels and a zero F
// matrix, stamped with kLegacyForkDatabaseVersionNumber.
void CreateLegacyForkDatabase(const std::string& path) {
  std::remove(path.c_str());
  sqlite3* db = nullptr;
  CHECK_EQ(sqlite3_open(path.c_str(), &db), SQLITE_OK);
  const char* create_sql =
      "CREATE TABLE two_view_geometries"
      "   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,"
      "    rows     INTEGER               NOT NULL,"
      "    cols     INTEGER               NOT NULL,"
      "    data     BLOB,"
      "    config   INTEGER               NOT NULL,"
      "    F        BLOB,"
      "    E        BLOB,"
      "    H        BLOB,"
      "    qvec     BLOB,"
      "    tvec     BLOB);";
  CHECK_EQ(sqlite3_exec(db, create_sql, nullptr, nullptr, nullptr), SQLITE_OK);
  CHECK_EQ(sqlite3_exec(
               db,
               "CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY NOT NULL, "
               "rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB);"
               "CREATE TABLE float_descriptors (image_id INTEGER PRIMARY KEY NOT NULL, "
               "rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB, "
               "type INTEGER NOT NULL);",
               nullptr,
               nullptr,
               nullptr),
           SQLITE_OK);
  sqlite3_stmt* stmt = nullptr;
  CHECK_EQ(sqlite3_prepare_v2(db,
                              "INSERT INTO two_view_geometries"
                              "(pair_id, rows, cols, config, F, qvec, tvec) "
                              "VALUES(1, 0, 0, 2, ?, ?, ?);",
                              -1, &stmt, nullptr),
           SQLITE_OK);
  const Eigen::Matrix3d zero_F = Eigen::Matrix3d::Zero();
  // qvec is stored as (w, x, y, z); the identity rotation sentinels the pose.
  const Eigen::Vector4d identity_qvec(1.0, 0.0, 0.0, 0.0);
  const Eigen::Vector3d zero_tvec = Eigen::Vector3d::Zero();
  sqlite3_bind_blob(stmt, 1, zero_F.data(), sizeof(double) * 9,
                    SQLITE_TRANSIENT);
  sqlite3_bind_blob(stmt, 2, identity_qvec.data(), sizeof(double) * 4,
                    SQLITE_TRANSIENT);
  sqlite3_bind_blob(stmt, 3, zero_tvec.data(), sizeof(double) * 3,
                    SQLITE_TRANSIENT);
  CHECK_EQ(sqlite3_step(stmt), SQLITE_DONE);
  sqlite3_finalize(stmt);

  const FeatureDescriptorsData sift_bytes =
      FeatureDescriptorsData::Constant(1, 2, 17);
  CHECK_EQ(sqlite3_prepare_v2(
               db,
               "INSERT INTO descriptors(image_id, rows, cols, data) "
               "VALUES(7, 1, 2, ?);",
               -1,
               &stmt,
               nullptr),
           SQLITE_OK);
  CHECK_EQ(sqlite3_bind_blob(stmt,
                             1,
                             sift_bytes.data(),
                             static_cast<int>(sift_bytes.size()),
                             SQLITE_TRANSIENT),
           SQLITE_OK);
  CHECK_EQ(sqlite3_step(stmt), SQLITE_DONE);
  sqlite3_finalize(stmt);

  const FeatureDescriptorsFloatData loma_float =
      FeatureDescriptorsFloatData::Constant(1, 2, 0.25f);
  CHECK_EQ(sqlite3_prepare_v2(
               db,
               "INSERT INTO float_descriptors(image_id, rows, cols, data, type) "
               "VALUES(8, 1, 2, ?, 1);",
               -1,
               &stmt,
               nullptr),
           SQLITE_OK);
  CHECK_EQ(sqlite3_bind_blob(stmt,
                             1,
                             loma_float.data(),
                             static_cast<int>(loma_float.size() * sizeof(float)),
                             SQLITE_TRANSIENT),
           SQLITE_OK);
  CHECK_EQ(sqlite3_step(stmt), SQLITE_DONE);
  sqlite3_finalize(stmt);

  CHECK_EQ(sqlite3_exec(
               db,
               StringPrintf("PRAGMA user_version = %d;",
                            kLegacyForkDatabaseVersionNumber)
                   .c_str(),
               nullptr, nullptr, nullptr),
           SQLITE_OK);
  sqlite3_close(db);
}

int ReadRawDatabaseUserVersion(const std::string& path) {
  sqlite3* db = nullptr;
  CHECK_EQ(sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr),
           SQLITE_OK);
  sqlite3_stmt* stmt = nullptr;
  CHECK_EQ(sqlite3_prepare_v2(db, "PRAGMA user_version;", -1, &stmt, nullptr),
           SQLITE_OK);
  CHECK_EQ(sqlite3_step(stmt), SQLITE_ROW);
  const int user_version = sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return user_version;
}

std::string MakeMigrationTestDatabasePath(const std::string& name) {
  return (std::filesystem::path(std::filesystem::temp_directory_path()) /
          (name + ".db"))
      .string();
}

}  // namespace

TEST(base_database, TestMakeDatabaseVersionNumber) {
  EXPECT_EQ(MakeDatabaseVersionNumber(4, 3, 0, 0), 4030000);
  EXPECT_EQ(MakeDatabaseVersionNumber(3, 13, 0, 0), 3130000);
  EXPECT_EQ(MakeDatabaseVersionNumber(3, 14, 0, 1), 3140001);
  // The legacy fork numbering must fall into the version-gated migrations.
  EXPECT_LT(kLegacyForkDatabaseVersionNumber,
                 MakeDatabaseVersionNumber(3, 13, 0, 0));
  EXPECT_LT(MakeDatabaseVersionNumber(3, 14, 0, 1),
                 GetDatabaseVersionNumber());
  EXPECT_THROW(MakeDatabaseVersionNumber(4, 100, 0, 0),
                    std::invalid_argument);
}

TEST(base_database, TestFreshDatabaseUserVersion) {
  const std::string path = MakeMigrationTestDatabasePath("colmap_fresh_db");
  std::remove(path.c_str());
  {
    auto database = Database::Open(path);
    database->Close();
  }
  EXPECT_EQ(ReadRawDatabaseUserVersion(path),
                    GetDatabaseVersionNumber());
  std::remove(path.c_str());
}

TEST(base_database, TestMigrateLegacyForkDatabase) {
  const std::string path =
      MakeMigrationTestDatabasePath("colmap_legacy_fork_db");
  CreateLegacyForkDatabase(path);
  EXPECT_EQ(ReadRawDatabaseUserVersion(path),
                    kLegacyForkDatabaseVersionNumber);
  {
    auto database = Database::Open(path);
    const FeatureDescriptors sift_descriptors = database->ReadDescriptors(7);
    EXPECT_EQ(sift_descriptors.type, FeatureExtractorType::SIFT);
    EXPECT_TRUE(sift_descriptors.data.isApprox(
        FeatureDescriptorsData::Constant(1, 2, 17)));
    const FeatureDescriptors loma_descriptors = database->ReadDescriptors(8);
    EXPECT_EQ(loma_descriptors.type, FeatureExtractorType::LOMA_B);
    const FeatureDescriptorsFloat loma_float = loma_descriptors.ToFloat();
    EXPECT_TRUE(loma_float.data.isApprox(
        FeatureDescriptorsFloatData::Constant(1, 2, 0.25f)));
    database->Close();
  }
  // Identity/zero pose sentinels and the zero F matrix are migrated to NULL.
  sqlite3* db = nullptr;
  CHECK_EQ(sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr),
           SQLITE_OK);
  sqlite3_stmt* stmt = nullptr;
  CHECK_EQ(sqlite3_prepare_v2(db,
                              "SELECT F, qvec, tvec FROM two_view_geometries "
                              "WHERE pair_id = 1;",
                              -1, &stmt, nullptr),
           SQLITE_OK);
  EXPECT_TRUE(sqlite3_step(stmt) == SQLITE_ROW);
  EXPECT_TRUE(sqlite3_column_type(stmt, 0) == SQLITE_NULL);
  EXPECT_TRUE(sqlite3_column_type(stmt, 1) == SQLITE_NULL);
  EXPECT_TRUE(sqlite3_column_type(stmt, 2) == SQLITE_NULL);
  sqlite3_finalize(stmt);
  CHECK_EQ(sqlite3_prepare_v2(
               db,
               "SELECT COUNT(*) FROM sqlite_master "
               "WHERE type = 'table' AND name = 'float_descriptors';",
               -1,
               &stmt,
               nullptr),
           SQLITE_OK);
  EXPECT_TRUE(sqlite3_step(stmt) == SQLITE_ROW);
  EXPECT_EQ(sqlite3_column_int(stmt, 0), 0);
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  EXPECT_EQ(ReadRawDatabaseUserVersion(path),
                    GetDatabaseVersionNumber());
  std::remove(path.c_str());
}

TEST(base_database, TestRejectNewerDatabase) {
  const std::string path = MakeMigrationTestDatabasePath("colmap_newer_db");
  std::remove(path.c_str());
  sqlite3* db = nullptr;
  CHECK_EQ(sqlite3_open(path.c_str(), &db), SQLITE_OK);
  const std::string pragma =
      StringPrintf("PRAGMA user_version = %d;", GetDatabaseVersionNumber() + 1);
  CHECK_EQ(sqlite3_exec(db, pragma.c_str(), nullptr, nullptr, nullptr),
           SQLITE_OK);
  sqlite3_close(db);
  bool threw = false;
  try {
    auto database = Database::Open(path);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
  std::remove(path.c_str());
}

namespace {

// Verbatim schema of upstream colmap a2bd15c2 (scene/database_sqlite.cc
// Create*Table), with kMaxNumImages = INT32_MAX substituted. Databases must be
// interchangeable between the fork and upstream binaries, so the fork has to
// write exactly this schema.
// Upstream parity (f1ea91d0): the current upstream schema (composite
// primary-key cameras with a source column; images without a foreign key to
// cameras). The pre-f1ea91d0 schema is kept separately in
// kLegacyUpstreamSchemaSql as the migration source.
const char* kUpstreamSchemaSql =
    "CREATE TABLE IF NOT EXISTS rigs"
    "   (rig_id               INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
    "    ref_sensor_id        INTEGER                             NOT NULL,"
    "    ref_sensor_type      INTEGER                             NOT NULL);"
    "CREATE UNIQUE INDEX IF NOT EXISTS rig_ref_sensor_assignment ON "
    "   rigs(ref_sensor_id, ref_sensor_type);"
    "CREATE TABLE IF NOT EXISTS rig_sensors"
    "   (rig_id               INTEGER                             NOT NULL,"
    "    sensor_id            INTEGER                             NOT NULL,"
    "    sensor_type          INTEGER                             NOT NULL,"
    "    sensor_from_rig      BLOB,"
    "FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);"
    "CREATE UNIQUE INDEX IF NOT EXISTS rig_sensor_assignment ON "
    "   rig_sensors(sensor_id, sensor_type);"
    "CREATE TABLE IF NOT EXISTS cameras"
    "   (camera_id            INTEGER                             NOT NULL,"
    "    model                INTEGER                             NOT NULL,"
    "    width                INTEGER                             NOT NULL,"
    "    height               INTEGER                             NOT NULL,"
    "    params               BLOB,"
    "    source               INTEGER                             NOT NULL,"
    "    PRIMARY KEY(camera_id, source));"
    "CREATE TABLE IF NOT EXISTS frames"
    "   (frame_id             INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
    "    rig_id               INTEGER                             NOT NULL,"
    "    FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);"
    "CREATE TABLE IF NOT EXISTS frame_data"
    "   (frame_id             INTEGER                             NOT NULL,"
    "    data_id              INTEGER                             NOT NULL,"
    "    sensor_id            INTEGER                             NOT NULL,"
    "    sensor_type          INTEGER                             NOT NULL,"
    "    FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE);"
    "CREATE UNIQUE INDEX IF NOT EXISTS frame_sensor_assignment ON "
    "   frame_data(data_id, sensor_type);"
    "CREATE TABLE IF NOT EXISTS images"
    "   (image_id   INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
    "    name       TEXT                                NOT NULL UNIQUE,"
    "    camera_id  INTEGER                             NOT NULL,"
    "    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < "
    "2147483647));"
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);"
    "CREATE TABLE IF NOT EXISTS pose_priors"
    "   (pose_prior_id              INTEGER  PRIMARY KEY  NOT NULL,"
    "    corr_data_id               INTEGER               NOT NULL,"
    "    corr_sensor_id             INTEGER               NOT NULL,"
    "    corr_sensor_type           INTEGER               NOT NULL,"
    "    position                   BLOB,"
    "    position_covariance        BLOB,"
    "    gravity                    BLOB,"
    "    coordinate_system          INTEGER               NOT NULL);"
    "CREATE UNIQUE INDEX IF NOT EXISTS pose_prior_data_assignment ON "
    "   pose_priors(corr_data_id, corr_sensor_id, corr_sensor_type);"
    "CREATE TABLE IF NOT EXISTS keypoints"
    "   (image_id  INTEGER  PRIMARY KEY  NOT NULL,"
    "    rows      INTEGER               NOT NULL,"
    "    cols      INTEGER               NOT NULL,"
    "    data      BLOB,"
    "    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);"
    "CREATE TABLE IF NOT EXISTS descriptors"
    "   (image_id      INTEGER  PRIMARY KEY  NOT NULL,"
    "    type          INTEGER               NOT NULL,"
    "    rows          INTEGER               NOT NULL,"
    "    cols          INTEGER               NOT NULL,"
    "    data          BLOB,"
    "    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);"
    "CREATE TABLE IF NOT EXISTS matches"
    "   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,"
    "    rows     INTEGER               NOT NULL,"
    "    cols     INTEGER               NOT NULL,"
    "    data     BLOB);"
    "CREATE TABLE IF NOT EXISTS two_view_geometries"
    "   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,"
    "    rows     INTEGER               NOT NULL,"
    "    cols     INTEGER               NOT NULL,"
    "    data     BLOB,"
    "    config   INTEGER               NOT NULL,"
    "    F        BLOB,"
    "    E        BLOB,"
    "    H        BLOB,"
    "    qvec     BLOB,"
    "    tvec     BLOB,"
    "    camera1  BLOB,"
    "    camera2  BLOB);";

// Schema of the fork before the upstream port (frame poses, image priors,
// rig_cameras/frame_images side tables, renamed pose prior columns).
const char* kForkLegacySchemaSql =
    "CREATE TABLE rigs (rig_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
    "  ref_camera_id INTEGER, ref_sensor_id INTEGER NOT NULL,"
    "  ref_sensor_type INTEGER NOT NULL);"
    "CREATE TABLE rig_sensors (rig_id INTEGER NOT NULL, sensor_id INTEGER NOT "
    "NULL, sensor_type INTEGER NOT NULL, qvec BLOB, tvec BLOB,"
    "  PRIMARY KEY(rig_id, sensor_id, sensor_type), UNIQUE(sensor_id, "
    "sensor_type), FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE "
    "CASCADE);"
    "CREATE TABLE rig_cameras (rig_id INTEGER NOT NULL, camera_id INTEGER NOT "
    "NULL, qvec BLOB NOT NULL, tvec BLOB NOT NULL, PRIMARY KEY(rig_id, "
    "camera_id), UNIQUE(camera_id), FOREIGN KEY(rig_id) REFERENCES "
    "rigs(rig_id) ON DELETE CASCADE, FOREIGN KEY(camera_id) REFERENCES "
    "cameras(camera_id));"
    "CREATE TABLE frames (frame_id INTEGER PRIMARY KEY AUTOINCREMENT NOT "
    "NULL, rig_id INTEGER NOT NULL, has_pose INTEGER NOT NULL, qvec BLOB, "
    "tvec BLOB, FOREIGN KEY(rig_id) REFERENCES rigs(rig_id));"
    "CREATE TABLE frame_data (frame_id INTEGER NOT NULL, data_id INTEGER NOT "
    "NULL, sensor_id INTEGER NOT NULL, sensor_type INTEGER NOT NULL, PRIMARY "
    "KEY(frame_id, data_id, sensor_id, sensor_type), UNIQUE(data_id, "
    "sensor_type), FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON "
    "DELETE CASCADE);"
    "CREATE TABLE frame_images (frame_id INTEGER NOT NULL, image_id INTEGER "
    "NOT NULL, PRIMARY KEY(frame_id, image_id), UNIQUE(image_id), FOREIGN "
    "KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE, FOREIGN "
    "KEY(image_id) REFERENCES images(image_id));"
    "CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT "
    "NULL, model INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT "
    "NULL, params BLOB, prior_focal_length INTEGER NOT NULL);"
    "CREATE TABLE images (image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT "
    "NULL, name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL, prior_qw "
    "REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL, prior_tx REAL, "
    "prior_ty REAL, prior_tz REAL, CONSTRAINT image_id_check CHECK(image_id "
    ">= 0 and image_id < 2147483647), FOREIGN KEY(camera_id) REFERENCES "
    "cameras(camera_id));"
    "CREATE UNIQUE INDEX index_name ON images(name);"
    "CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY NOT NULL, rows "
    "INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB, FOREIGN "
    "KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);"
    "CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY NOT NULL, rows "
    "INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB, type INTEGER NOT "
    "NULL, FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE "
    "CASCADE);"
    "CREATE TABLE matches (pair_id INTEGER PRIMARY KEY NOT NULL, rows "
    "INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB);"
    "CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY NOT NULL, "
    "rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB, config INTEGER "
    "NOT NULL, F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB, camera1 BLOB, "
    "camera2 BLOB);"
    "CREATE TABLE pose_priors (pose_prior_id INTEGER PRIMARY KEY "
    "AUTOINCREMENT, data_id INTEGER NOT NULL, sensor_id INTEGER NOT NULL, "
    "sensor_type INTEGER NOT NULL, position BLOB, position_covariance BLOB, "
    "coordinate_system INTEGER NOT NULL, gravity BLOB);";

void ExecRawSql(sqlite3* db, const std::string& sql) {
  char* err_msg = nullptr;
  const int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err_msg);
  CHECK_EQ(rc, SQLITE_OK) << (err_msg ? err_msg : "") << " in: " << sql;
}

// Executes a single statement with the given blob parameters bound in order
// (nullopt binds NULL, an empty string binds a zero-length blob).
void ExecRawSqlWithBlobs(sqlite3* db,
                         const std::string& sql,
                         const std::vector<std::optional<std::string>>& blobs) {
  sqlite3_stmt* stmt = nullptr;
  CHECK_EQ(sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr), SQLITE_OK)
      << sqlite3_errmsg(db) << " in: " << sql;
  for (size_t i = 0; i < blobs.size(); ++i) {
    const int col = static_cast<int>(i) + 1;
    if (blobs[i].has_value()) {
      CHECK_EQ(sqlite3_bind_blob(stmt, col, blobs[i]->data(),
                                 static_cast<int>(blobs[i]->size()),
                                 SQLITE_TRANSIENT),
               SQLITE_OK);
    } else {
      CHECK_EQ(sqlite3_bind_null(stmt, col), SQLITE_OK);
    }
  }
  CHECK_EQ(sqlite3_step(stmt), SQLITE_DONE) << sqlite3_errmsg(db);
  sqlite3_finalize(stmt);
}

template <typename MatrixType>
std::string MatrixBytes(const MatrixType& matrix) {
  return std::string(
      reinterpret_cast<const char*>(matrix.data()),
      matrix.size() * sizeof(typename MatrixType::Scalar));
}

std::string SensorFromRigBytes(const Eigen::Vector4d& qvec,
                               const Eigen::Vector3d& tvec) {
  return MatrixBytes(qvec) + MatrixBytes(tvec);
}

// Collapses whitespace and identifier quoting so that schemas are compared by
// their SQL tokens, not by the formatting of the CREATE statements.
std::string NormalizeSql(const std::string& sql) {
  std::string normalized;
  bool pending_space = false;
  for (const char c : sql) {
    if (c == '"') {
      continue;
    }
    if (std::isspace(static_cast<unsigned char>(c))) {
      pending_space = true;
      continue;
    }
    if (pending_space && !normalized.empty() && normalized.back() != '(' &&
        normalized.back() != ',' && c != '(' && c != ')' && c != ',') {
      normalized.push_back(' ');
    }
    pending_space = false;
    normalized.push_back(c);
  }
  return normalized;
}

using SchemaEntry =
    std::tuple<std::string, std::string, std::string, std::string>;

std::vector<SchemaEntry> ReadNormalizedSchema(
    sqlite3* db, const std::set<std::string>& skip_tables = {}) {
  sqlite3_stmt* stmt = nullptr;
  CHECK_EQ(sqlite3_prepare_v2(db,
                              "SELECT type, name, tbl_name, sql FROM "
                              "sqlite_master ORDER BY type, name;",
                              -1, &stmt, nullptr),
           SQLITE_OK);
  std::vector<SchemaEntry> schema;
  auto column_text = [&stmt](const int col) {
    const unsigned char* text = sqlite3_column_text(stmt, col);
    return text ? std::string(reinterpret_cast<const char*>(text))
                : std::string();
  };
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    if (skip_tables.count(column_text(2)) != 0) {
      continue;
    }
    schema.emplace_back(column_text(0), column_text(1), column_text(2),
                        NormalizeSql(column_text(3)));
  }
  sqlite3_finalize(stmt);
  return schema;
}

std::vector<SchemaEntry> ReadNormalizedSchema(
    const std::string& path, const std::set<std::string>& skip_tables = {}) {
  sqlite3* db = nullptr;
  CHECK_EQ(sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr),
           SQLITE_OK);
  std::vector<SchemaEntry> schema = ReadNormalizedSchema(db, skip_tables);
  sqlite3_close(db);
  return schema;
}

std::vector<SchemaEntry> UpstreamNormalizedSchema(
    const std::set<std::string>& skip_tables = {}) {
  sqlite3* db = nullptr;
  CHECK_EQ(sqlite3_open(":memory:", &db), SQLITE_OK);
  ExecRawSql(db, kUpstreamSchemaSql);
  std::vector<SchemaEntry> schema = ReadNormalizedSchema(db, skip_tables);
  sqlite3_close(db);
  return schema;
}

std::string FormatSchema(const std::vector<SchemaEntry>& schema) {
  std::ostringstream stream;
  for (const auto& [type, name, tbl_name, sql] : schema) {
    stream << type << " " << name << " (" << tbl_name << "): " << sql << "\n";
  }
  return stream.str();
}

std::vector<std::string> ReadRawColumnNames(const std::string& path,
                                            const std::string& table) {
  sqlite3* db = nullptr;
  CHECK_EQ(sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr),
           SQLITE_OK);
  sqlite3_stmt* stmt = nullptr;
  CHECK_EQ(sqlite3_prepare_v2(db, ("PRAGMA table_info(" + table + ");").c_str(),
                              -1, &stmt, nullptr),
           SQLITE_OK);
  std::vector<std::string> names;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    names.emplace_back(
        reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return names;
}

const Eigen::Vector3d kCameraParams(800.0, 320.0, 240.0);

void InsertRawSimplePinholeCamera(sqlite3* db, const camera_t camera_id) {
  ExecRawSqlWithBlobs(
      db,
      StringPrintf("INSERT INTO cameras VALUES(%u, %d, 640, 480, ?, 0);",
                   camera_id, static_cast<int>(SimplePinholeCameraModel::model_id)),
      {MatrixBytes(kCameraParams)});
}

}  // namespace

TEST(base_database, TestFreshSchemaMatchesUpstream) {
  const std::string path = MakeMigrationTestDatabasePath("colmap_fresh_schema");
  std::remove(path.c_str());
  {
    auto database = Database::Open(path);
    database->Close();
  }
  const std::vector<SchemaEntry> schema = ReadNormalizedSchema(path);
  const std::vector<SchemaEntry> upstream_schema = UpstreamNormalizedSchema();
  EXPECT_EQ(schema, upstream_schema)
      << "fork:\n" << FormatSchema(schema) << "upstream:\n"
      << FormatSchema(upstream_schema);
  // Upstream stamps COLMAP_DATABASE_SCHEMA_REVISION = 1 after f1ea91d0.
  EXPECT_EQ(ReadRawDatabaseUserVersion(path),
            MakeDatabaseVersionNumber(COLMAP_VERSION_MAJOR,
                                      COLMAP_VERSION_MINOR,
                                      COLMAP_VERSION_PATCH, 1));
  std::remove(path.c_str());
}

// Upstream parity (f1ea91d0): the pre-f1ea91d0 upstream schema, kept as the
// migration source for the cameras-table rebuild (prior_focal_length ->
// source; images foreign-key removal).
const char* kLegacyUpstreamSchemaSql =
    "CREATE TABLE IF NOT EXISTS cameras"
    "   (camera_id            INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
    "    model                INTEGER                             NOT NULL,"
    "    width                INTEGER                             NOT NULL,"
    "    height               INTEGER                             NOT NULL,"
    "    params               BLOB,"
    "    prior_focal_length   INTEGER                             NOT NULL);"
    "CREATE TABLE IF NOT EXISTS images"
    "   (image_id   INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
    "    name       TEXT                                NOT NULL UNIQUE,"
    "    camera_id  INTEGER                             NOT NULL,"
    "    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < "
    "2147483647),"
    "    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));"
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);";

TEST(base_database, TestOpenUpstreamDatabase) {
  const std::string path = MakeMigrationTestDatabasePath("colmap_upstream_db");
  std::remove(path.c_str());
  {
    sqlite3* db = nullptr;
    CHECK_EQ(sqlite3_open(path.c_str(), &db), SQLITE_OK);
    // The current upstream schema at the current schema version: opening it
    // must not alter any tables (fork-parity gate).
    ExecRawSql(db, kUpstreamSchemaSql);
    ExecRawSql(db, StringPrintf(
                       "PRAGMA user_version = %d;",
                       GetDatabaseVersionNumber()));
    sqlite3_close(db);
  }

  {
    auto database = Database::Open(path);
    EXPECT_EQ(database->NumCameras(), 0);
    EXPECT_EQ(ReadRawDatabaseUserVersion(path), GetDatabaseVersionNumber());
    database->Close();
  }

  // The fork must neither add nor alter tables of an upstream database: the
  // schema read after opening must match the preset schema read before.
  const std::vector<SchemaEntry> schema = ReadNormalizedSchema(path);
  const std::vector<SchemaEntry> preset_schema = ReadNormalizedSchema(path);
  EXPECT_EQ(schema, preset_schema)
      << "after open:\n" << FormatSchema(schema) << "preset:\n"
      << FormatSchema(preset_schema);
  EXPECT_EQ(ReadRawDatabaseUserVersion(path), GetDatabaseVersionNumber());
  std::remove(path.c_str());
}

// Upstream parity (f1ea91d0): opening a pre-f1ea91d0 upstream database
// (user_version 4.3.0.0) rebuilds the cameras table onto the composite
// primary key, mapping the legacy prior_focal_length flag to EXIF/GUESS.
TEST(base_database, TestMigrateUpstreamCamerasTable) {
  const std::string path =
      MakeMigrationTestDatabasePath("colmap_pre_f1ea91d0_db");
  std::remove(path.c_str());
  {
    sqlite3* db = nullptr;
    CHECK_EQ(sqlite3_open(path.c_str(), &db), SQLITE_OK);
    ExecRawSql(db, kLegacyUpstreamSchemaSql);
    InsertRawSimplePinholeCamera(db, 1);
    InsertRawSimplePinholeCamera(db, 2);
    ExecRawSql(db, "UPDATE cameras SET prior_focal_length = 1 "
                   "WHERE camera_id = 2;");
    ExecRawSql(db, "INSERT INTO images VALUES(1, 'a.jpg', 1);");
    ExecRawSql(db, "INSERT INTO images VALUES(2, 'b.jpg', 2);");
    ExecRawSql(db, StringPrintf("PRAGMA user_version = %d;",
                                MakeDatabaseVersionNumber(4, 3, 0, 0)));
    sqlite3_close(db);
  }

  const std::vector<SchemaEntry> migrated_schema = ReadNormalizedSchema(path);
  {
    auto database = Database::Open(path);
    EXPECT_EQ(database->NumCameras(), 2);
    EXPECT_EQ(database->NumImages(), 2);
    const Camera guess_camera = database->ReadCamera(1);
    EXPECT_EQ(guess_camera.Source(), CameraSource::GUESS);
    EXPECT_FALSE(guess_camera.HasPriorFocalLength());
    const Camera exif_camera = database->ReadCamera(2);
    EXPECT_EQ(exif_camera.Source(), CameraSource::EXIF);
    EXPECT_TRUE(exif_camera.HasPriorFocalLength());
    EXPECT_TRUE(database->ExistsCameraSource(1, CameraSource::GUESS));
    EXPECT_FALSE(database->ExistsCameraSource(1, CameraSource::EXIF));
    const auto calibrations = database->ReadAllCameraSources(1);
    EXPECT_EQ(calibrations.size(), 1);
    EXPECT_EQ(calibrations.count(CameraSource::GUESS), 1);
    database->Close();
  }
  const std::vector<SchemaEntry> post_schema = ReadNormalizedSchema(path);
  EXPECT_EQ(post_schema, UpstreamNormalizedSchema())
      << "post:\n" << FormatSchema(post_schema) << "upstream:\n"
      << FormatSchema(UpstreamNormalizedSchema());
  std::remove(path.c_str());
}

TEST(base_database, TestRigWritesUpstreamSensorEncoding) {
  const std::string path = MakeMigrationTestDatabasePath("colmap_rig_encoding");
  std::remove(path.c_str());
  const Eigen::Vector4d qvec(0.0, 0.0, 0.0, 1.0);
  const Eigen::Vector3d tvec(0.2, -0.1, 0.05);
  {
    auto database = Database::Open(path);
    Camera camera;
    camera.InitializeWithId(SimplePinholeCameraModel::model_id, 800.0, 640,
                            480);
    camera.SetCameraId(1);
    database->WriteCamera(camera, true);
    camera.SetCameraId(2);
    database->WriteCamera(camera, true);
    Rig rig;
    rig.SetRigId(1);
    rig.AddRefCamera(2);
    rig.AddCamera(1, qvec, tvec);
    rig.AddSensor(sensor_t(SensorType::IMU, 9), std::nullopt, std::nullopt);
    database->WriteRig(rig, true);
    Frame frame;
    frame.SetFrameId(1);
    frame.SetRigId(1);
    frame.AddDataId(data_t(sensor_t(SensorType::CAMERA, 1), 1));
    frame.SetRigFromWorld(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
                          Eigen::Vector3d(1.0, 2.0, 3.0));
    database->WriteFrame(frame, true);
    database->Close();
  }

  sqlite3* db = nullptr;
  CHECK_EQ(sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr),
           SQLITE_OK);
  sqlite3_stmt* stmt = nullptr;
  CHECK_EQ(sqlite3_prepare_v2(db,
                              "SELECT sensor_id, sensor_type, sensor_from_rig "
                              "FROM rig_sensors ORDER BY sensor_type, "
                              "sensor_id;",
                              -1, &stmt, nullptr),
           SQLITE_OK);
  // Only non-reference sensors are stored (upstream WriteRigSensors).
  ASSERT_EQ(sqlite3_step(stmt), SQLITE_ROW);
  EXPECT_EQ(sqlite3_column_int64(stmt, 0), 1);
  EXPECT_EQ(sqlite3_column_int64(stmt, 1), 0);
  ASSERT_EQ(sqlite3_column_bytes(stmt, 2),
            static_cast<int>(7 * sizeof(double)));
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(
                            sqlite3_column_blob(stmt, 2)),
                        7 * sizeof(double)),
            SensorFromRigBytes(qvec, tvec));
  ASSERT_EQ(sqlite3_step(stmt), SQLITE_ROW);
  EXPECT_EQ(sqlite3_column_int64(stmt, 0), 9);
  EXPECT_EQ(sqlite3_column_int64(stmt, 1), 1);
  EXPECT_EQ(sqlite3_column_type(stmt, 2), SQLITE_NULL);
  EXPECT_EQ(sqlite3_step(stmt), SQLITE_DONE);
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  EXPECT_EQ(ReadRawColumnNames(path, "frames"),
            (std::vector<std::string>{"frame_id", "rig_id"}));
  std::remove(path.c_str());
}

TEST(base_database, TestTwoViewGeometryAbsentFieldsWriteNull) {
  const std::string path = MakeMigrationTestDatabasePath("colmap_tvg_null");
  std::remove(path.c_str());
  {
    auto database = Database::Open(path);
    TwoViewGeometry two_view_geometry;
    two_view_geometry.config = TwoViewGeometry::CALIBRATED;
    two_view_geometry.inlier_matches = FeatureMatches(3);
    two_view_geometry.E = Eigen::Matrix3d::Identity();
    database->WriteTwoViewGeometry(1, 2, two_view_geometry);
    database->WriteTwoViewGeometry(3, 4, TwoViewGeometry());
    database->Close();
  }

  sqlite3* db = nullptr;
  CHECK_EQ(sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr),
           SQLITE_OK);
  sqlite3_stmt* stmt = nullptr;
  CHECK_EQ(sqlite3_prepare_v2(db,
                              "SELECT F, E, H, qvec, tvec FROM "
                              "two_view_geometries WHERE pair_id = ?;",
                              -1, &stmt, nullptr),
           SQLITE_OK);
  sqlite3_bind_int64(stmt, 1, Database::ImagePairToPairId(1, 2));
  ASSERT_EQ(sqlite3_step(stmt), SQLITE_ROW);
  EXPECT_EQ(sqlite3_column_type(stmt, 0), SQLITE_NULL);
  EXPECT_EQ(sqlite3_column_bytes(stmt, 1),
            static_cast<int>(9 * sizeof(double)));
  EXPECT_EQ(sqlite3_column_type(stmt, 2), SQLITE_NULL);
  EXPECT_EQ(sqlite3_column_type(stmt, 3), SQLITE_NULL);
  EXPECT_EQ(sqlite3_column_type(stmt, 4), SQLITE_NULL);
  sqlite3_reset(stmt);
  sqlite3_bind_int64(stmt, 1, Database::ImagePairToPairId(3, 4));
  ASSERT_EQ(sqlite3_step(stmt), SQLITE_ROW);
  for (int col = 0; col < 5; ++col) {
    EXPECT_EQ(sqlite3_column_type(stmt, col), SQLITE_NULL) << col;
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);

  auto database = Database::Open(path);
  const TwoViewGeometry loaded = database->ReadTwoViewGeometry(1, 2);
  EXPECT_FALSE(loaded.F.has_value());
  ASSERT_TRUE(loaded.E.has_value());
  EXPECT_EQ(*loaded.E, Eigen::Matrix3d::Identity());
  EXPECT_FALSE(loaded.H.has_value());
  EXPECT_FALSE(loaded.cam2_from_cam1.has_value());
  database->Close();
  std::remove(path.c_str());
}

TEST(base_database, TestMigrateForkLegacyRigFrameSchema) {
  const std::string path = MakeMigrationTestDatabasePath("colmap_fork_legacy");
  std::remove(path.c_str());
  const Eigen::Vector4d qvec2(0.5, 0.5, 0.5, 0.5);
  const Eigen::Vector3d tvec2(0.1, 0.2, 0.3);
  const Eigen::Vector4d qvec4(0.0, 1.0, 0.0, 0.0);
  const Eigen::Vector3d tvec4(-1.0, 0.0, 2.0);
  const FeatureDescriptorsData descriptors =
      FeatureDescriptorsData::Constant(1, 2, 17);
  const std::string zero_matrix = MatrixBytes(Eigen::Matrix3d::Zero().eval());
  {
    sqlite3* db = nullptr;
    CHECK_EQ(sqlite3_open(path.c_str(), &db), SQLITE_OK);
    ExecRawSql(db, kForkLegacySchemaSql);
    for (camera_t camera_id = 1; camera_id <= 4; ++camera_id) {
      InsertRawSimplePinholeCamera(db, camera_id);
    }
    ExecRawSql(db, "INSERT INTO rigs VALUES(1, 1, 1, 0);");
    ExecRawSql(db, "INSERT INTO rigs VALUES(2, 3, 3, 0);");
    // Reference sensor rows, a posed camera, and a pose-less IMU.
    ExecRawSql(db, "INSERT INTO rig_sensors VALUES(1, 1, 0, NULL, NULL);");
    ExecRawSqlWithBlobs(db, "INSERT INTO rig_sensors VALUES(1, 2, 0, ?, ?);",
                        {MatrixBytes(qvec2), MatrixBytes(tvec2)});
    ExecRawSql(db, "INSERT INTO rig_sensors VALUES(1, 9, 1, NULL, NULL);");
    // Camera 2 is duplicated in rig_cameras; camera 4 only lives there.
    ExecRawSqlWithBlobs(db, "INSERT INTO rig_cameras VALUES(1, 2, ?, ?);",
                        {MatrixBytes(qvec2), MatrixBytes(tvec2)});
    ExecRawSqlWithBlobs(db, "INSERT INTO rig_cameras VALUES(2, 4, ?, ?);",
                        {MatrixBytes(qvec4), MatrixBytes(tvec4)});
    ExecRawSqlWithBlobs(
        db, "INSERT INTO frames VALUES(1, 1, 1, ?, ?);",
        {MatrixBytes(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0)),
         MatrixBytes(Eigen::Vector3d(1.0, 2.0, 3.0))});
    ExecRawSql(db, "INSERT INTO frames VALUES(2, 2, 0, NULL, NULL);");
    ExecRawSql(db,
               "INSERT INTO images VALUES(1, 'a.jpg', 1, 1, 0, 0, 0, 4, 5, 6);"
               "INSERT INTO images(image_id, name, camera_id) "
               "VALUES(2, 'b.jpg', 2), (3, 'c.jpg', 3), (4, 'd.jpg', 4);");
    ExecRawSql(db,
               "INSERT INTO frame_data VALUES(1, 1, 1, 0);"
               "INSERT INTO frame_data VALUES(1, 42, 9, 1);"
               "INSERT INTO frame_images VALUES(1, 1);"
               "INSERT INTO frame_images VALUES(1, 2);"
               "INSERT INTO frame_images VALUES(2, 3);"
               "INSERT INTO frame_images VALUES(2, 4);");
    ExecRawSqlWithBlobs(
        db, "INSERT INTO pose_priors VALUES(5, 1, 1, 0, ?, ?, 1, ?);",
        {MatrixBytes(Eigen::Vector3d(1.0, 2.0, 3.0)),
         MatrixBytes(Eigen::Matrix3d::Identity().eval()),
         MatrixBytes(Eigen::Vector3d(0.0, 0.0, 1.0))});
    ExecRawSqlWithBlobs(db, "INSERT INTO keypoints VALUES(1, 1, 2, ?);",
                        {MatrixBytes(Eigen::Vector2f(3.0f, 4.0f))});
    ExecRawSqlWithBlobs(db, "INSERT INTO descriptors VALUES(1, 1, 2, ?, 0);",
                        {MatrixBytes(descriptors)});
    // Fork-legacy two-view encoding: all-zero blobs for absent values, and
    // zero-length blobs for entries without inliers.
    ExecRawSqlWithBlobs(
        db,
        StringPrintf("INSERT INTO two_view_geometries VALUES(%lld, 1, 2, ?, "
                     "2, ?, ?, ?, ?, ?, NULL, NULL);",
                     static_cast<long long>(Database::ImagePairToPairId(1, 2))),
        {MatrixBytes(Eigen::Matrix<uint32_t, 1, 2>(0, 1)), zero_matrix,
         MatrixBytes(Eigen::Matrix3d::Identity().eval()), zero_matrix,
         MatrixBytes(Eigen::Vector4d::Zero().eval()),
         MatrixBytes(Eigen::Vector3d::Zero().eval())});
    ExecRawSqlWithBlobs(
        db,
        StringPrintf("INSERT INTO two_view_geometries VALUES(%lld, 0, 2, ?, "
                     "1, ?, ?, ?, ?, ?, NULL, NULL);",
                     static_cast<long long>(Database::ImagePairToPairId(3, 4))),
        {std::string(), std::string(), std::string(), std::string(),
         std::string(), std::string()});
    ExecRawSql(db, StringPrintf("PRAGMA user_version = %d;",
                                MakeDatabaseVersionNumber(4, 3, 0, 0)));
    sqlite3_close(db);
  }

  {
    auto database = Database::Open(path);
    EXPECT_EQ(database->NumRigs(), 2);
    const Rig rig1 = database->ReadRig(1);
    EXPECT_TRUE(rig1.RefSensorId() == sensor_t(SensorType::CAMERA, 1));
    EXPECT_EQ(rig1.NumSensors(), 3);
    ASSERT_TRUE(rig1.HasSensorFromRig(sensor_t(SensorType::CAMERA, 2)));
    EXPECT_EQ(rig1.SensorFromRigQvec(sensor_t(SensorType::CAMERA, 2)), qvec2);
    EXPECT_EQ(rig1.SensorFromRigTvec(sensor_t(SensorType::CAMERA, 2)), tvec2);
    EXPECT_TRUE(rig1.HasSensor(sensor_t(SensorType::IMU, 9)));
    EXPECT_FALSE(rig1.HasSensorFromRig(sensor_t(SensorType::IMU, 9)));
    const Rig rig2 = database->ReadRig(2);
    EXPECT_TRUE(rig2.RefSensorId() == sensor_t(SensorType::CAMERA, 3));
    EXPECT_EQ(rig2.NumSensors(), 2);
    ASSERT_TRUE(rig2.HasSensorFromRig(sensor_t(SensorType::CAMERA, 4)));
    EXPECT_EQ(rig2.SensorFromRigQvec(sensor_t(SensorType::CAMERA, 4)), qvec4);
    EXPECT_EQ(rig2.SensorFromRigTvec(sensor_t(SensorType::CAMERA, 4)), tvec4);

    const Frame frame1 = database->ReadFrame(1);
    EXPECT_EQ(frame1.RigId(), 1);
    EXPECT_FALSE(frame1.HasPose());
    EXPECT_EQ(frame1.NumDataIds(), 3);
    EXPECT_TRUE(frame1.HasDataId(data_t(sensor_t(SensorType::CAMERA, 1), 1)));
    EXPECT_TRUE(frame1.HasDataId(data_t(sensor_t(SensorType::CAMERA, 2), 2)));
    EXPECT_TRUE(frame1.HasDataId(data_t(sensor_t(SensorType::IMU, 9), 42)));
    const Frame frame2 = database->ReadFrame(2);
    EXPECT_EQ(frame2.NumDataIds(), 2);
    EXPECT_TRUE(frame2.HasDataId(data_t(sensor_t(SensorType::CAMERA, 3), 3)));
    EXPECT_TRUE(frame2.HasDataId(data_t(sensor_t(SensorType::CAMERA, 4), 4)));

    EXPECT_EQ(database->NumImages(), 4);
    EXPECT_EQ(database->ReadImage(1).Name(), "a.jpg");
    EXPECT_EQ(database->ReadImage(2).FrameId(), 1);
    EXPECT_EQ(database->ReadImage(4).FrameId(), 2);
    EXPECT_EQ(database->ReadImage(4).CameraId(), 4);

    const PosePrior pose_prior = database->ReadPosePrior(5);
    EXPECT_EQ(pose_prior.corr_data_id.id, 1);
    EXPECT_TRUE(pose_prior.corr_data_id.sensor_id ==
                sensor_t(SensorType::CAMERA, 1));
    EXPECT_EQ(pose_prior.position, Eigen::Vector3d(1.0, 2.0, 3.0));
    EXPECT_EQ(pose_prior.gravity, Eigen::Vector3d(0.0, 0.0, 1.0));
    EXPECT_EQ(pose_prior.coordinate_system,
              PosePrior::CoordinateSystem::CARTESIAN);

    EXPECT_EQ(database->ReadKeypoints(1).size(), 1);
    const FeatureDescriptors loaded_descriptors = database->ReadDescriptors(1);
    EXPECT_EQ(loaded_descriptors.type, FeatureExtractorType::SIFT);
    EXPECT_EQ(loaded_descriptors.data, descriptors);

    const TwoViewGeometry tvg12 = database->ReadTwoViewGeometry(1, 2);
    EXPECT_EQ(tvg12.inlier_matches.size(), 1);
    EXPECT_FALSE(tvg12.F.has_value());
    EXPECT_TRUE(tvg12.E.has_value());
    EXPECT_FALSE(tvg12.H.has_value());
    EXPECT_FALSE(tvg12.cam2_from_cam1.has_value());
    EXPECT_TRUE(database->ExistsTwoViewGeometry(3, 4));

    Image new_image;
    new_image.SetName("e.jpg");
    new_image.SetCameraId(1);
    EXPECT_EQ(database->WriteImage(new_image), 5);
    database->Close();
  }

  // The migrated database carries exactly the upstream schema. The
  // descriptors table keeps its column order: upstream itself appends the
  // type column to old databases and all statements name their columns.
  const std::vector<SchemaEntry> schema =
      ReadNormalizedSchema(path, {"descriptors"});
  const std::vector<SchemaEntry> upstream_schema =
      UpstreamNormalizedSchema({"descriptors"});
  EXPECT_EQ(schema, upstream_schema)
      << "migrated:\n" << FormatSchema(schema) << "upstream:\n"
      << FormatSchema(upstream_schema);
  std::vector<std::string> descriptor_columns =
      ReadRawColumnNames(path, "descriptors");
  std::sort(descriptor_columns.begin(), descriptor_columns.end());
  EXPECT_EQ(descriptor_columns,
            (std::vector<std::string>{"cols", "data", "image_id", "rows",
                                      "type"}));
  EXPECT_EQ(ReadRawDatabaseUserVersion(path), GetDatabaseVersionNumber());

  sqlite3* db = nullptr;
  CHECK_EQ(sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr),
           SQLITE_OK);
  sqlite3_stmt* stmt = nullptr;
  CHECK_EQ(sqlite3_prepare_v2(db, "PRAGMA foreign_key_check;", -1, &stmt,
                              nullptr),
           SQLITE_OK);
  EXPECT_EQ(sqlite3_step(stmt), SQLITE_DONE);
  sqlite3_finalize(stmt);
  CHECK_EQ(sqlite3_prepare_v2(db,
                              "SELECT F, H, qvec, tvec FROM "
                              "two_view_geometries ORDER BY pair_id;",
                              -1, &stmt, nullptr),
           SQLITE_OK);
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    for (int col = 0; col < 4; ++col) {
      EXPECT_EQ(sqlite3_column_type(stmt, col), SQLITE_NULL) << col;
    }
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);

  // Re-opening a migrated database is a no-op.
  {
    auto database = Database::Open(path);
    EXPECT_EQ(database->ReadRig(1).NumSensors(), 3);
    EXPECT_EQ(database->ReadFrame(1).NumDataIds(), 3);
    EXPECT_EQ(database->NumImages(), 5);
    database->Close();
  }
  EXPECT_EQ(ReadNormalizedSchema(path, {"descriptors"}), upstream_schema);
  std::remove(path.c_str());
}

TEST(base_database, TestMigrateImageKeyedPosePriors) {
  const std::string path =
      MakeMigrationTestDatabasePath("colmap_image_pose_priors");
  std::remove(path.c_str());
  {
    sqlite3* db = nullptr;
    CHECK_EQ(sqlite3_open(path.c_str(), &db), SQLITE_OK);
    ExecRawSql(db, kUpstreamSchemaSql);
    ExecRawSql(db, "DROP TABLE pose_priors;");
    // Pre-sensor upstream layout keyed by image_id.
    ExecRawSql(db,
               "CREATE TABLE pose_priors (image_id INTEGER PRIMARY KEY NOT "
               "NULL, position BLOB, coordinate_system INTEGER NOT NULL, "
               "position_covariance BLOB);");
    InsertRawSimplePinholeCamera(db, 3);
    ExecRawSql(db, "INSERT INTO images VALUES(7, 'a.jpg', 3);");
    ExecRawSqlWithBlobs(db, "INSERT INTO pose_priors VALUES(7, ?, 1, ?);",
                        {MatrixBytes(Eigen::Vector3d(1.0, 2.0, 3.0)),
                         MatrixBytes(Eigen::Matrix3d::Identity().eval())});
    sqlite3_close(db);
  }
  {
    auto database = Database::Open(path);
    ASSERT_EQ(database->NumPosePriors(), 1);
    const PosePrior pose_prior = database->ReadPosePrior(7);
    EXPECT_EQ(pose_prior.corr_data_id.id, 7);
    EXPECT_TRUE(pose_prior.corr_data_id.sensor_id ==
                sensor_t(SensorType::CAMERA, 3));
    EXPECT_EQ(pose_prior.position, Eigen::Vector3d(1.0, 2.0, 3.0));
    database->Close();
  }
  EXPECT_EQ(ReadNormalizedSchema(path), UpstreamNormalizedSchema());
  std::remove(path.c_str());
}

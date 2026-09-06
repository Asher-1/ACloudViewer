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

#include "base/database.h"
#include "geometry/rigid3.h"
#include "base/camera_models.h"
#include "base/pose.h"
#include "util/string.h"
#include "util/version.h"

#include <cstdio>
#include <filesystem>
#include <stdexcept>

#include "SQLite/sqlite3.h"

using namespace colmap;

const static std::string kMemoryDatabasePath = ":memory:";

TEST(base_database, TestOpenCloseConstructorDestructor) {
  Database database(kMemoryDatabasePath);
}

TEST(base_database, TestOpenClose) {
  Database database(kMemoryDatabasePath);
  database.Close();
}

TEST(base_database, TestRigFrameSqliteRoundTrip) {
  Database database(kMemoryDatabasePath);

  Camera camera1;
  camera1.InitializeWithId(SimplePinholeCameraModel::model_id, 800.0, 640, 480);
  camera1.SetCameraId(1);
  database.WriteCamera(camera1, true);
  Camera camera2 = camera1;
  camera2.SetCameraId(2);
  database.WriteCamera(camera2, true);

  Image image1;
  image1.SetImageId(1);
  image1.SetCameraId(1);
  image1.SetName("rig-frame-1");
  database.WriteImage(image1, true);
  Image image2;
  image2.SetImageId(2);
  image2.SetCameraId(2);
  image2.SetName("rig-frame-2");
  database.WriteImage(image2, true);

  Rig rig;
  rig.SetRigId(7);
  rig.AddRefCamera(2);
  rig.AddCamera(1, Eigen::Vector4d(0.0, 0.0, 0.0, 1.0),
                Eigen::Vector3d(0.2, -0.1, 0.05));
  rig.AddSensor(sensor_t(SensorType::IMU, 9),
                Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
                Eigen::Vector3d(0.01, 0.02, 0.03));
  EXPECT_EQ(database.WriteRig(rig, true), 7);
  EXPECT_TRUE(database.ExistsRig(7));

  Frame frame;
  frame.SetFrameId(11);
  frame.SetRigId(7);
  frame.AddImageId(1);
  frame.AddImageId(2);
  frame.AddDataId(data_t(sensor_t(SensorType::IMU, 9), 42));
  frame.SetRigFromWorld(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
                        Eigen::Vector3d(1.0, 2.0, 3.0));
  EXPECT_EQ(database.WriteFrame(frame, true), 11);
  EXPECT_TRUE(database.ExistsFrame(11));

  const Rig loaded_rig = database.ReadRig(7);
  EXPECT_EQ(loaded_rig.RefCameraId(), 2);
  EXPECT_TRUE(loaded_rig.HasCamera(1));
  EXPECT_TRUE(loaded_rig.HasSensor(sensor_t(SensorType::IMU, 9)));
  EXPECT_TRUE(loaded_rig.HasSensorFromRig(sensor_t(SensorType::IMU, 9)));
  ASSERT_LE(std::abs((loaded_rig.CamFromRigTvec(1) -
                     Eigen::Vector3d(0.2, -0.1, 0.05)).norm()), 1e-12);
  const Frame loaded_frame = database.ReadFrame(11);
  EXPECT_EQ(loaded_frame.RigId(), 7);
  EXPECT_TRUE(loaded_frame.HasImageId(1));
  EXPECT_TRUE(loaded_frame.HasImageId(2));
  EXPECT_TRUE(loaded_frame.HasDataId(data_t(sensor_t(SensorType::IMU, 9), 42)));
  EXPECT_TRUE(loaded_frame.HasPose());
  ASSERT_LE(std::abs((loaded_frame.RigFromWorldTvec() -
                     Eigen::Vector3d(1.0, 2.0, 3.0)).norm()), 1e-12);
  EXPECT_EQ(database.NumRigs(), 1);
  EXPECT_EQ(database.NumFrames(), 1);
}

TEST(base_database, TestGenericRigReferenceSensorSqliteRoundTrip) {
  Database database(kMemoryDatabasePath);
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 800.0, 640, 480);
  camera.SetCameraId(1);
  database.WriteCamera(camera, true);

  Rig rig;
  rig.SetRigId(12);
  rig.AddRefSensor(sensor_t(SensorType::IMU, 5));
  rig.AddCamera(1, Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
                Eigen::Vector3d(0.0, 0.0, 0.1));
  EXPECT_EQ(database.WriteRig(rig, true), 12);
  const Rig loaded = database.ReadRig(12);
  EXPECT_TRUE(loaded.RefSensorId() == sensor_t(SensorType::IMU, 5));
  EXPECT_TRUE(loaded.HasCamera(1));
  EXPECT_FALSE(loaded.HasSensorFromRig(sensor_t(SensorType::IMU, 5)));
}

TEST(base_database, TestGenericDataOnlyFrameSqliteRoundTrip) {
  Database database(kMemoryDatabasePath);
  Rig rig;
  rig.SetRigId(13);
  rig.AddRefSensor(sensor_t(SensorType::IMU, 8));
  EXPECT_EQ(database.WriteRig(rig, true), 13);

  Frame frame;
  frame.SetFrameId(14);
  frame.SetRigId(13);
  const data_t imu_data(sensor_t(SensorType::IMU, 8), 99);
  frame.AddDataId(imu_data);
  EXPECT_EQ(database.WriteFrame(frame, true), 14);

  const Frame loaded = database.ReadFrame(14);
  EXPECT_TRUE(loaded.HasDataId(imu_data));
  EXPECT_TRUE(loaded.ImageIds().empty());
}

TEST(base_database, TestFloatDescriptorSqliteRoundTrip) {
  Database database(kMemoryDatabasePath);
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 800.0, 640, 480);
  camera.SetCameraId(1);
  database.WriteCamera(camera, true);
  Image image;
  image.SetImageId(1);
  image.SetCameraId(1);
  image.SetName("loma-float");
  database.WriteImage(image, true);

  FeatureKeypoints keypoints = {FeatureKeypoint(10.5f, 20.25f),
                                FeatureKeypoint(30.0f, 40.0f)};
  FeatureDescriptorsFloat descriptors(2, 3);
  descriptors << 0.125f, -3.5f, 1.0e-7f, 1.0f, 2.5f, -9.25f;
  database.WriteKeypoints(1, keypoints);
  database.WriteFloatDescriptors(1, descriptors,
                                 FeatureDescriptorType::kLomaG);

  EXPECT_TRUE(database.ExistsFloatDescriptors(1));
  const auto loaded = database.ReadFloatDescriptors(1);
  EXPECT_EQ(loaded.rows(), 2);
  EXPECT_EQ(loaded.cols(), 3);
  EXPECT_TRUE((loaded - descriptors).cwiseAbs().maxCoeff() == 0.0f);
  EXPECT_TRUE(database.ReadDescriptorType(1) == FeatureDescriptorType::kLomaG);
  EXPECT_FALSE(database.ExistsDescriptors(1));
}

TEST(base_database, TestTransaction) {
  Database database(kMemoryDatabasePath);
  DatabaseTransaction database_transaction(&database);
}

TEST(base_database, TestTransactionMultiThreaded) {
  Database database(kMemoryDatabasePath);

  std::thread thread1([&database]() {
    DatabaseTransaction database_transaction(&database);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  });

  std::thread thread2([&database]() {
    DatabaseTransaction database_transaction(&database);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  });

  thread1.join();
  thread2.join();
}

TEST(base_database, TestEmpty) {
  Database database(kMemoryDatabasePath);
  EXPECT_EQ(database.NumCameras(), 0);
  EXPECT_EQ(database.NumImages(), 0);
  EXPECT_EQ(database.NumKeypoints(), 0);
  EXPECT_EQ(database.MaxNumKeypoints(), 0);
  EXPECT_EQ(database.NumDescriptors(), 0);
  EXPECT_EQ(database.MaxNumDescriptors(), 0);
  EXPECT_EQ(database.NumMatches(), 0);
  EXPECT_EQ(database.NumMatchedImagePairs(), 0);
  EXPECT_EQ(database.NumVerifiedImagePairs(), 0);
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

TEST(base_database, TestSwapImagePair) {
  EXPECT_FALSE(Database::SwapImagePair(0, 0));
  EXPECT_FALSE(Database::SwapImagePair(0, 1));
  EXPECT_TRUE(Database::SwapImagePair(1, 0));
  EXPECT_FALSE(Database::SwapImagePair(1, 1));
}

TEST(base_database, TestCamera) {
  Database database(kMemoryDatabasePath);
  EXPECT_EQ(database.NumCameras(), 0);
  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.SetCameraId(database.WriteCamera(camera));
  EXPECT_EQ(database.NumCameras(), 1);
  EXPECT_EQ(database.ExistsCamera(camera.CameraId()), true);
  EXPECT_EQ(database.ReadCamera(camera.CameraId()).CameraId(),
                    camera.CameraId());
  EXPECT_EQ(database.ReadCamera(camera.CameraId()).ModelId(),
                    camera.ModelId());
  EXPECT_EQ(database.ReadCamera(camera.CameraId()).FocalLength(),
                    camera.FocalLength());
  EXPECT_EQ(database.ReadCamera(camera.CameraId()).PrincipalPointX(),
                    camera.PrincipalPointX());
  EXPECT_EQ(database.ReadCamera(camera.CameraId()).PrincipalPointY(),
                    camera.PrincipalPointY());
  camera.SetFocalLength(2.0);
  database.UpdateCamera(camera);
  EXPECT_EQ(database.ReadCamera(camera.CameraId()).FocalLength(),
                    camera.FocalLength());
  Camera camera2 = camera;
  camera2.SetCameraId(camera.CameraId() + 1);
  database.WriteCamera(camera2, true);
  EXPECT_EQ(database.NumCameras(), 2);
  EXPECT_EQ(database.ExistsCamera(camera.CameraId()), true);
  EXPECT_EQ(database.ExistsCamera(camera2.CameraId()), true);
  EXPECT_EQ(database.ReadAllCameras().size(), 2);
  EXPECT_EQ(database.ReadAllCameras()[0].CameraId(), camera.CameraId());
  EXPECT_EQ(database.ReadAllCameras()[1].CameraId(),
                    camera2.CameraId());
  database.ClearCameras();
  EXPECT_EQ(database.NumCameras(), 0);
}

TEST(base_database, TestImage) {
  Database database(kMemoryDatabasePath);
  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.SetCameraId(database.WriteCamera(camera));
  EXPECT_EQ(database.NumImages(), 0);
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.CameraId());
  image.SetQvecPrior(Eigen::Vector4d(0.1, 0.2, 0.3, 0.4));
  image.SetTvecPrior(Eigen::Vector3d(0.1, 0.2, 0.3));
  image.SetImageId(database.WriteImage(image));
  EXPECT_EQ(database.NumImages(), 1);
  EXPECT_EQ(database.ExistsImage(image.ImageId()), true);
  EXPECT_EQ(database.ReadImage(image.ImageId()).ImageId(),
                    image.ImageId());
  EXPECT_EQ(database.ReadImage(image.ImageId()).CameraId(),
                    image.CameraId());
  EXPECT_EQ(database.ReadImage(image.ImageId()).QvecPrior(0),
                    image.QvecPrior(0));
  EXPECT_EQ(database.ReadImage(image.ImageId()).QvecPrior(1),
                    image.QvecPrior(1));
  EXPECT_EQ(database.ReadImage(image.ImageId()).QvecPrior(2),
                    image.QvecPrior(2));
  EXPECT_EQ(database.ReadImage(image.ImageId()).QvecPrior(3),
                    image.QvecPrior(3));
  EXPECT_EQ(database.ReadImage(image.ImageId()).TvecPrior(0),
                    image.TvecPrior(0));
  EXPECT_EQ(database.ReadImage(image.ImageId()).TvecPrior(1),
                    image.TvecPrior(1));
  EXPECT_EQ(database.ReadImage(image.ImageId()).TvecPrior(2),
                    image.TvecPrior(2));
  image.TvecPrior(0) += 2;
  database.UpdateImage(image);
  EXPECT_EQ(database.ReadImage(image.ImageId()).ImageId(),
                    image.ImageId());
  EXPECT_EQ(database.ReadImage(image.ImageId()).CameraId(),
                    image.CameraId());
  EXPECT_EQ(database.ReadImage(image.ImageId()).QvecPrior(0),
                    image.QvecPrior(0));
  EXPECT_EQ(database.ReadImage(image.ImageId()).QvecPrior(1),
                    image.QvecPrior(1));
  EXPECT_EQ(database.ReadImage(image.ImageId()).QvecPrior(2),
                    image.QvecPrior(2));
  EXPECT_EQ(database.ReadImage(image.ImageId()).QvecPrior(3),
                    image.QvecPrior(3));
  EXPECT_EQ(database.ReadImage(image.ImageId()).TvecPrior(0),
                    image.TvecPrior(0));
  EXPECT_EQ(database.ReadImage(image.ImageId()).TvecPrior(1),
                    image.TvecPrior(1));
  EXPECT_EQ(database.ReadImage(image.ImageId()).TvecPrior(2),
                    image.TvecPrior(2));
  Image image2 = image;
  image2.SetName("test2");
  image2.SetImageId(image.ImageId() + 1);
  database.WriteImage(image2, true);
  EXPECT_EQ(database.NumImages(), 2);
  EXPECT_EQ(database.ExistsImage(image.ImageId()), true);
  EXPECT_EQ(database.ExistsImage(image2.ImageId()), true);
  EXPECT_EQ(database.ReadAllImages().size(), 2);
  database.ClearImages();
  EXPECT_EQ(database.NumImages(), 0);
}

TEST(base_database, TestKeypoints) {
  Database database(kMemoryDatabasePath);
  Camera camera;
  camera.SetCameraId(database.WriteCamera(camera));
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.CameraId());
  image.SetImageId(database.WriteImage(image));
  EXPECT_EQ(database.NumKeypoints(), 0);
  EXPECT_EQ(database.NumKeypointsForImage(image.ImageId()), 0);
  const FeatureKeypoints keypoints = FeatureKeypoints(10);
  database.WriteKeypoints(image.ImageId(), keypoints);
  const FeatureKeypoints keypoints_read =
      database.ReadKeypoints(image.ImageId());
  EXPECT_EQ(keypoints.size(), keypoints_read.size());
  for (size_t i = 0; i < keypoints.size(); ++i) {
    EXPECT_EQ(keypoints[i].x, keypoints_read[i].x);
    EXPECT_EQ(keypoints[i].y, keypoints_read[i].y);
    EXPECT_EQ(keypoints[i].a11, keypoints_read[i].a11);
    EXPECT_EQ(keypoints[i].a12, keypoints_read[i].a12);
    EXPECT_EQ(keypoints[i].a21, keypoints_read[i].a21);
    EXPECT_EQ(keypoints[i].a22, keypoints_read[i].a22);
  }
  EXPECT_EQ(database.NumKeypoints(), 10);
  EXPECT_EQ(database.MaxNumKeypoints(), 10);
  EXPECT_EQ(database.NumKeypointsForImage(image.ImageId()), 10);
  const FeatureKeypoints keypoints2 = FeatureKeypoints(20);
  image.SetName("test2");
  image.SetImageId(database.WriteImage(image));
  database.WriteKeypoints(image.ImageId(), keypoints2);
  EXPECT_EQ(database.NumKeypoints(), 30);
  EXPECT_EQ(database.MaxNumKeypoints(), 20);
  EXPECT_EQ(database.NumKeypointsForImage(image.ImageId()), 20);
  database.ClearKeypoints();
  EXPECT_EQ(database.NumKeypoints(), 0);
  EXPECT_EQ(database.MaxNumKeypoints(), 0);
  EXPECT_EQ(database.NumKeypointsForImage(image.ImageId()), 0);
}

TEST(base_database, TestDescriptors) {
  Database database(kMemoryDatabasePath);
  Camera camera;
  camera.SetCameraId(database.WriteCamera(camera));
  Image image;
  image.SetName("test");
  image.SetCameraId(camera.CameraId());
  image.SetImageId(database.WriteImage(image));
  EXPECT_EQ(database.NumDescriptors(), 0);
  EXPECT_EQ(database.NumDescriptorsForImage(image.ImageId()), 0);
  const FeatureDescriptors descriptors = FeatureDescriptors::Random(10, 128);
  database.WriteDescriptors(image.ImageId(), descriptors);
  const FeatureDescriptors descriptors_read =
      database.ReadDescriptors(image.ImageId());
  EXPECT_EQ(descriptors.rows(), descriptors_read.rows());
  EXPECT_EQ(descriptors.cols(), descriptors_read.cols());
  for (FeatureDescriptors::Index r = 0; r < descriptors.rows(); ++r) {
    for (FeatureDescriptors::Index c = 0; c < descriptors.cols(); ++c) {
      EXPECT_EQ(descriptors(r, c), descriptors_read(r, c));
    }
  }
  EXPECT_EQ(database.NumDescriptors(), 10);
  EXPECT_EQ(database.MaxNumDescriptors(), 10);
  EXPECT_EQ(database.NumDescriptorsForImage(image.ImageId()), 10);
  const FeatureDescriptors descriptors2 = FeatureDescriptors(20, 128);
  image.SetName("test2");
  image.SetImageId(database.WriteImage(image));
  database.WriteDescriptors(image.ImageId(), descriptors2);
  EXPECT_EQ(database.NumDescriptors(), 30);
  EXPECT_EQ(database.MaxNumDescriptors(), 20);
  EXPECT_EQ(database.NumDescriptorsForImage(image.ImageId()), 20);
  database.ClearDescriptors();
  EXPECT_EQ(database.NumDescriptors(), 0);
  EXPECT_EQ(database.MaxNumDescriptors(), 0);
  EXPECT_EQ(database.NumDescriptorsForImage(image.ImageId()), 0);
}

TEST(base_database, TestMatches) {
  Database database(kMemoryDatabasePath);
  const image_t image_id1 = 1;
  const image_t image_id2 = 2;
  const FeatureMatches matches = FeatureMatches(1000);
  database.WriteMatches(image_id1, image_id2, matches);
  const FeatureMatches matches_read =
      database.ReadMatches(image_id1, image_id2);
  EXPECT_EQ(matches.size(), matches_read.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    EXPECT_EQ(matches[i].point2D_idx1, matches_read[i].point2D_idx1);
    EXPECT_EQ(matches[i].point2D_idx2, matches_read[i].point2D_idx2);
  }
  EXPECT_EQ(database.ReadAllMatches().size(), 1);
  EXPECT_EQ(database.ReadAllMatches()[0].first,
                    Database::ImagePairToPairId(image_id1, image_id2));
  EXPECT_EQ(database.NumMatches(), 1000);
  database.DeleteMatches(image_id1, image_id2);
  EXPECT_EQ(database.NumMatches(), 0);
  database.WriteMatches(image_id1, image_id2, matches);
  EXPECT_EQ(database.NumMatches(), 1000);
  database.ClearMatches();
  EXPECT_EQ(database.NumMatches(), 0);
}

TEST(base_database, TestTwoViewGeometry) {
  Database database(kMemoryDatabasePath);
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
  database.WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  const TwoViewGeometry two_view_geometry_read =
      database.ReadTwoViewGeometry(image_id1, image_id2);
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
      database.ReadTwoViewGeometry(image_id2, image_id1);
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
  database.ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);
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
  database.ReadTwoViewGeometryNumInliers(&image_pairs, &num_inliers);
  EXPECT_EQ(image_pairs.size(), 1);
  EXPECT_EQ(num_inliers.size(), 1);
  EXPECT_EQ(image_pairs[0].first, image_id1);
  EXPECT_EQ(image_pairs[0].second, image_id2);
  EXPECT_EQ(num_inliers[0], two_view_geometry.inlier_matches.size());
  EXPECT_EQ(database.NumInlierMatches(), 1000);
  database.DeleteInlierMatches(image_id1, image_id2);
  EXPECT_EQ(database.NumInlierMatches(), 0);
  database.WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  EXPECT_EQ(database.NumInlierMatches(), 1000);
}

TEST(base_database, TestMerge) {
  Database database1(kMemoryDatabasePath);
  Database database2(kMemoryDatabasePath);

  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.SetCameraId(database1.WriteCamera(camera));
  camera.SetCameraId(database2.WriteCamera(camera));

  Image image;
  image.SetCameraId(camera.CameraId());
  image.SetQvecPrior(Eigen::Vector4d(0.1, 0.2, 0.3, 0.4));
  image.SetTvecPrior(Eigen::Vector3d(0.1, 0.2, 0.3));

  image.SetName("test1");
  const image_t image_id1 = database1.WriteImage(image);
  image.SetName("test2");
  const image_t image_id2 = database1.WriteImage(image);
  image.SetName("test3");
  const image_t image_id3 = database2.WriteImage(image);
  image.SetName("test4");
  const image_t image_id4 = database2.WriteImage(image);

  auto keypoints1 = FeatureKeypoints(10);
  keypoints1[0].x = 100;
  auto keypoints2 = FeatureKeypoints(20);
  keypoints2[0].x = 200;
  auto keypoints3 = FeatureKeypoints(30);
  keypoints3[0].x = 300;
  auto keypoints4 = FeatureKeypoints(40);
  keypoints4[0].x = 400;

  const auto descriptors1 = FeatureDescriptors::Random(10, 128);
  const auto descriptors2 = FeatureDescriptors::Random(20, 128);
  const auto descriptors3 = FeatureDescriptors::Random(30, 128);
  const auto descriptors4 = FeatureDescriptors::Random(40, 128);

  database1.WriteKeypoints(image_id1, keypoints1);
  database1.WriteKeypoints(image_id2, keypoints2);
  database2.WriteKeypoints(image_id3, keypoints3);
  database2.WriteKeypoints(image_id4, keypoints4);
  database1.WriteDescriptors(image_id1, descriptors1);
  database1.WriteDescriptors(image_id2, descriptors2);
  database2.WriteDescriptors(image_id3, descriptors3);
  database2.WriteDescriptors(image_id4, descriptors4);
  database1.WriteMatches(image_id1, image_id2, FeatureMatches(10));
  database2.WriteMatches(image_id3, image_id4, FeatureMatches(10));
  database1.WriteTwoViewGeometry(image_id1, image_id2, TwoViewGeometry());
  database2.WriteTwoViewGeometry(image_id3, image_id4, TwoViewGeometry());

  Database merged_database(kMemoryDatabasePath);
  Database::Merge(database1, database2, &merged_database);
  EXPECT_EQ(merged_database.NumCameras(), 2);
  EXPECT_EQ(merged_database.NumImages(), 4);
  EXPECT_EQ(merged_database.NumKeypoints(), 100);
  EXPECT_EQ(merged_database.NumDescriptors(), 100);
  EXPECT_EQ(merged_database.NumMatches(), 20);
  EXPECT_EQ(merged_database.NumInlierMatches(), 0);
  EXPECT_EQ(merged_database.ReadAllImages()[0].CameraId(), 1);
  EXPECT_EQ(merged_database.ReadAllImages()[1].CameraId(), 1);
  EXPECT_EQ(merged_database.ReadAllImages()[2].CameraId(), 2);
  EXPECT_EQ(merged_database.ReadAllImages()[3].CameraId(), 2);
  EXPECT_EQ(merged_database.ReadKeypoints(1).size(), 10);
  EXPECT_EQ(merged_database.ReadKeypoints(2).size(), 20);
  EXPECT_EQ(merged_database.ReadKeypoints(3).size(), 30);
  EXPECT_EQ(merged_database.ReadKeypoints(4).size(), 40);
  EXPECT_EQ(merged_database.ReadKeypoints(1)[0].x, 100);
  EXPECT_EQ(merged_database.ReadKeypoints(2)[0].x, 200);
  EXPECT_EQ(merged_database.ReadKeypoints(3)[0].x, 300);
  EXPECT_EQ(merged_database.ReadKeypoints(4)[0].x, 400);
  EXPECT_EQ(merged_database.ReadDescriptors(1).size(),
                    descriptors1.size());
  EXPECT_EQ(merged_database.ReadDescriptors(2).size(),
                    descriptors2.size());
  EXPECT_EQ(merged_database.ReadDescriptors(3).size(),
                    descriptors3.size());
  EXPECT_EQ(merged_database.ReadDescriptors(4).size(),
                    descriptors4.size());
  EXPECT_TRUE(merged_database.ExistsMatches(1, 2));
  EXPECT_FALSE(merged_database.ExistsMatches(2, 3));
  EXPECT_FALSE(merged_database.ExistsMatches(2, 4));
  EXPECT_TRUE(merged_database.ExistsMatches(3, 4));
  merged_database.ClearAllTables();
  EXPECT_EQ(merged_database.NumCameras(), 0);
  EXPECT_EQ(merged_database.NumImages(), 0);
  EXPECT_EQ(merged_database.NumKeypoints(), 0);
  EXPECT_EQ(merged_database.NumDescriptors(), 0);
  EXPECT_EQ(merged_database.NumMatches(), 0);
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
    Database database(path);
    database.Close();
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
    Database database(path);
    database.Close();
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
    Database database(path);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
  std::remove(path.c_str());
}

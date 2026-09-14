// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------

#define TEST_NAME "base/rig"
#include "scene/database_sqlite.h"
#include "util/testing.h"

#include <sstream>

#include "scene/rig.h"

using namespace colmap;

TEST(base_rig, TestTextAndBinaryRoundTrip) {
  Rig rig;
  rig.SetRigId(7);
  rig.AddRefCamera(1);
  rig.AddCamera(2,
                Eigen::Vector4d(2.0, 0.0, 0.0, 0.0),
                Eigen::Vector3d(0.25, -0.5, 1.0));

  std::stringstream text;
  rig.WriteText(&text);
  Rig from_text;
  ASSERT_TRUE(from_text.ReadText(&text));
  EXPECT_EQ(from_text.RigId(), 7);
  EXPECT_EQ(from_text.RefCameraId(), 1);
  EXPECT_EQ(from_text.NumCameras(), 2);
  EXPECT_TRUE(from_text.CamFromRigQvec(2).isApprox(
      Eigen::Vector4d(1.0, 0.0, 0.0, 0.0)));
  EXPECT_TRUE(from_text.CamFromRigTvec(2).isApprox(
      Eigen::Vector3d(0.25, -0.5, 1.0)));

  std::stringstream binary;
  rig.WriteBinary(&binary);
  Rig from_binary;
  ASSERT_TRUE(from_binary.ReadBinary(&binary));
  EXPECT_EQ(from_binary.RigId(), 7);
  EXPECT_EQ(from_binary.RefCameraId(), 1);
  EXPECT_EQ(from_binary.NumCameras(), 2);
  EXPECT_TRUE(from_binary.CamFromRigTvec(2).isApprox(
      Eigen::Vector3d(0.25, -0.5, 1.0)));
}

TEST(base_rig, TestGenericReferenceSensorRoundTrip) {
  Rig rig;
  rig.SetRigId(8);
  rig.AddRefSensor(sensor_t(SensorType::IMU, 3));
  rig.AddCamera(2, Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
                Eigen::Vector3d(0.25, -0.5, 1.0));

  std::stringstream text;
  rig.WriteText(&text);
  Rig from_text;
  ASSERT_TRUE(from_text.ReadText(&text));
  EXPECT_TRUE(from_text.RefSensorId() == sensor_t(SensorType::IMU, 3));
  EXPECT_TRUE(from_text.HasCamera(2));
  EXPECT_FALSE(from_text.HasSensorFromRig(sensor_t(SensorType::IMU, 3)));

  std::stringstream binary;
  rig.WriteBinary(&binary);
  Rig from_binary;
  ASSERT_TRUE(from_binary.ReadBinary(&binary));
  EXPECT_TRUE(from_binary.RefSensorId() == sensor_t(SensorType::IMU, 3));
  EXPECT_TRUE(from_binary.CamFromRigTvec(2).isApprox(
      Eigen::Vector3d(0.25, -0.5, 1.0)));
}

// ---------------------------------------------------------------------------
// Upstream COLMAP 4.x scene/rig_test.cc rig-config suites (ReadRigConfig /
// ApplyRigConfig), ported with fork adaptations: concrete Database(":memory:")
// (no Open factory until W17.2b) and OpenCVCameraModel::model_id instead of
// the upstream CameraModelId enum spelling.
// ---------------------------------------------------------------------------
#include <fstream>

#include "scene/database.h"
#include "math/random_eigen.h"
#include "scene/synthetic.h"

#include <gmock/gmock.h>


namespace {

std::filesystem::path WriteTestConfig(const std::string& config) {
  const auto file_path = CreateTestDir() / "config.json";
  std::ofstream file(file_path);
  file << config << '\n';
  return file_path;
}

TEST(ReadRigConfig, Empty) {
  EXPECT_THAT(ReadRigConfig(WriteTestConfig("[]")), testing::IsEmpty());
}

TEST(ReadRigConfig, InvalidJson) {
  EXPECT_ANY_THROW(ReadRigConfig(WriteTestConfig("")));
  EXPECT_ANY_THROW(ReadRigConfig(WriteTestConfig("[{")));
}

TEST(ReadRigConfig, MissingImagePrefix) {
  EXPECT_ANY_THROW(ReadRigConfig(WriteTestConfig(R"(
          [
            {
              "cameras": [
                {
                    "ref_sensor": true,
                }
              ]
            }
          ]
          )")));
}

TEST(ReadRigConfig, InvalidRefSensor) {
  EXPECT_ANY_THROW(ReadRigConfig(WriteTestConfig(R"(
          [
            {
              "cameras": [
                {
                    "image_prefix": "rig1/camera1/",
                    "ref_sensor": true,
                },
                {
                    "image_prefix": "rig1/camera2/",
                    "ref_sensor": true,
                }
              ]
            }
          ]
          )")));
  EXPECT_ANY_THROW(ReadRigConfig(WriteTestConfig(R"(
              [
                {
                  "cameras": [
                    {
                        "image_prefix": "rig1/camera1/",
                    },
                    {
                        "image_prefix": "rig1/camera2/",
                    }
                  ]
                }
              ]
              )")));
}

TEST(ReadRigConfig, Nominal) {
  const std::vector<RigConfig> configs = ReadRigConfig(WriteTestConfig(R"(
[
  {
    "cameras": [
      {
          "image_prefix": "rig1/camera1/",
          "ref_sensor": true,
          "camera_model_name": "OPENCV",
          "camera_params": [640, 480, 320, 240, 0.1, 0.2, 0.3, 0.4]
      },
      {
          "image_prefix": "rig1/camera2/",
          "cam_from_rig_rotation": [0, 1, 0, 0],
          "cam_from_rig_translation": [1, 2, 3]
      }
    ]
  },
  {
    "cameras": [
      {
          "image_prefix": "rig2/camera1/",
          "ref_sensor": true
      },
      {
          "image_prefix": "rig2/camera2/"
      }
    ]
  }
]
)"));
  ASSERT_EQ(configs.size(), 2);
  ASSERT_EQ(configs[0].cameras.size(), 2);
  ASSERT_EQ(configs[1].cameras.size(), 2);

  EXPECT_EQ(configs[0].cameras[0].image_prefix, "rig1/camera1/");
  EXPECT_TRUE(configs[0].cameras[0].ref_sensor);
  ASSERT_FALSE(configs[0].cameras[0].cam_from_rig.has_value());
  ASSERT_TRUE(configs[0].cameras[0].camera.has_value());
  EXPECT_EQ(configs[0].cameras[0].camera->ModelId(), OpenCVCameraModel::model_id);
  EXPECT_TRUE(configs[0].cameras[0].camera->HasPriorFocalLength());
  EXPECT_THAT(configs[0].cameras[0].camera->Params(),
              testing::ElementsAre(640, 480, 320, 240, 0.1, 0.2, 0.3, 0.4));

  EXPECT_EQ(configs[0].cameras[1].image_prefix, "rig1/camera2/");
  EXPECT_FALSE(configs[0].cameras[1].ref_sensor);
  ASSERT_TRUE(configs[0].cameras[1].cam_from_rig.has_value());
  EXPECT_EQ(configs[0].cameras[1].cam_from_rig->rotation().coeffs(),
            Eigen::Vector4d(1, 0, 0, 0));
  EXPECT_EQ(configs[0].cameras[1].cam_from_rig->translation(),
            Eigen::Vector3d(1, 2, 3));
  ASSERT_FALSE(configs[0].cameras[1].camera.has_value());

  EXPECT_EQ(configs[1].cameras[0].image_prefix, "rig2/camera1/");
  EXPECT_TRUE(configs[1].cameras[0].ref_sensor);
  ASSERT_FALSE(configs[1].cameras[0].cam_from_rig.has_value());
  ASSERT_FALSE(configs[1].cameras[0].camera.has_value());

  EXPECT_EQ(configs[1].cameras[1].image_prefix, "rig2/camera2/");
  EXPECT_FALSE(configs[1].cameras[1].ref_sensor);
  ASSERT_FALSE(configs[1].cameras[1].cam_from_rig.has_value());
  ASSERT_FALSE(configs[1].cameras[1].camera.has_value());
}

void CreateTestData(int num_frames,
                    int num_cameras_per_rig,
                    Database* database,
                    Reconstruction* reconstruction) {
  // Create a synthetic dataset with a single rig and trivial frames.
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = num_frames * num_cameras_per_rig;
  SynthesizeDataset(options, reconstruction, database);

  // Manually overwrite the image names to match the expected rig config.
  int image_idx = 0;
  int frame_idx = 0;
  for (auto& image : database->ReadAllImages()) {
    if (image_idx++ % num_cameras_per_rig == 0) {
      ++frame_idx;
    }
    image.SetName("camera" + std::to_string(image_idx % num_cameras_per_rig) +
                  "_frame" + std::to_string(frame_idx));
    reconstruction->Image(image.ImageId()).SetName(image.Name());
    database->UpdateImage(image);
  }
}

TEST(ApplyRigConfig, WithReconstruction) {
  auto database = OpenSqliteDatabase(":memory:");
  Reconstruction reconstruction;
  CreateTestData(
      /*num_frames=*/5,
      /*num_cameras_per_rig=*/2,
      database.get(),
      &reconstruction);

  std::vector<RigConfig> configs;
  auto& config = configs.emplace_back();
  auto& camera1 = config.cameras.emplace_back();
  camera1.image_prefix = "camera0_";
  camera1.ref_sensor = true;
  auto& camera2 = config.cameras.emplace_back();
  camera2.image_prefix = "camera1_";

  ApplyRigConfig(configs, *database, &reconstruction);
  EXPECT_EQ(database->NumRigs(), 1);
  EXPECT_EQ(database->NumFrames(), 5);
  EXPECT_EQ(reconstruction.NumRigs(), 1);
  EXPECT_EQ(reconstruction.NumFrames(), 5);
  EXPECT_EQ(reconstruction.NumRegFrames(), 5);
}

TEST(ApplyRigConfig, WithDifferingDatabaseAndReconstructionIds) {
  auto database = OpenSqliteDatabase(":memory:");
  Reconstruction reconstruction;
  CreateTestData(
      /*num_frames=*/5,
      /*num_cameras_per_rig=*/2,
      database.get(),
      &reconstruction);

  // Create a reconstruction with differing rig/camera/frame/image ids.
  Reconstruction differing_reconstruction;
  for (const auto& [_, camera] : reconstruction.Cameras()) {
    auto differing_camera = camera;
    differing_camera.SetCameraId(camera.CameraId() + 1);
    differing_reconstruction.AddCamera(differing_camera);
  }
  for (const auto& [_, rig] : reconstruction.Rigs()) {
    Rig differing_rig;
    differing_rig.SetRigId(rig.RigId() + 1);
    differing_rig.AddRefSensor(
        sensor_t(rig.RefSensorId().type, rig.RefSensorId().id + 1));
    for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
      differing_rig.AddSensor(sensor_t(sensor_id.type, sensor_id.id + 1),
                              sensor_from_rig);
    }
    differing_reconstruction.AddRig(differing_rig);
  }
  for (const auto& [_, frame] : reconstruction.Frames()) {
    auto differing_frame = frame;
    differing_frame.ResetRigPtr();
    differing_frame.SetFrameId(frame.FrameId() + 1);
    differing_frame.SetRigId(frame.RigId() + 1);
    differing_frame.ClearDataIds();
    for (auto& data_id : frame.DataIds()) {
      differing_frame.AddDataId(
          data_t(sensor_t(data_id.sensor_id.type, data_id.sensor_id.id + 1),
                 data_id.id + 1));
    }
    differing_reconstruction.AddFrame(differing_frame);
  }
  for (const auto& [_, image] : reconstruction.Images()) {
    auto differing_image = image;
    differing_image.ResetCameraPtr();
    differing_image.ResetFramePtr();
    differing_image.SetImageId(image.ImageId() + 1);
    differing_image.SetCameraId(image.CameraId() + 1);
    differing_image.SetFrameId(image.FrameId() + 1);
    differing_reconstruction.AddImage(differing_image);
  }

  std::vector<RigConfig> configs;
  auto& config = configs.emplace_back();
  auto& camera1 = config.cameras.emplace_back();
  camera1.image_prefix = "camera0_";
  camera1.ref_sensor = true;
  auto& camera2 = config.cameras.emplace_back();
  camera2.image_prefix = "camera1_";

  ApplyRigConfig(configs, *database, &differing_reconstruction);
  EXPECT_EQ(database->NumRigs(), 1);
  EXPECT_EQ(database->NumFrames(), 5);
  EXPECT_EQ(differing_reconstruction.NumRigs(), 1);
  EXPECT_EQ(differing_reconstruction.NumFrames(), 5);
  EXPECT_EQ(differing_reconstruction.NumRegFrames(), 5);
}

TEST(ApplyRigConfig, WithPartialReconstruction) {
  auto database = OpenSqliteDatabase(":memory:");
  Reconstruction reconstruction;
  CreateTestData(
      /*num_frames=*/5,
      /*num_cameras_per_rig=*/2,
      database.get(),
      &reconstruction);

  reconstruction.DeRegisterFrame(1);
  reconstruction.DeRegisterFrame(2);
  // De-register only one image in the frame.
  // The other image allows us to register the frame.
  reconstruction.DeRegisterFrame(3);

  std::vector<RigConfig> configs;
  auto& config = configs.emplace_back();
  auto& camera1 = config.cameras.emplace_back();
  camera1.image_prefix = "camera0_";
  camera1.ref_sensor = true;
  auto& camera2 = config.cameras.emplace_back();
  camera2.image_prefix = "camera1_";

  ApplyRigConfig(configs, *database, &reconstruction);
  EXPECT_EQ(database->NumRigs(), 1);
  EXPECT_EQ(database->NumFrames(), 5);
  EXPECT_EQ(reconstruction.NumRigs(), 1);
  EXPECT_EQ(reconstruction.NumFrames(), 5);
  EXPECT_EQ(reconstruction.NumRegFrames(), 4);
}

TEST(ApplyRigConfig, WithoutReconstruction) {
  auto database = OpenSqliteDatabase(":memory:");
  Reconstruction reconstruction;
  CreateTestData(
      /*num_frames=*/5,
      /*num_cameras_per_rig=*/2,
      database.get(),
      &reconstruction);

  std::vector<RigConfig> configs;
  auto& config = configs.emplace_back();
  auto& camera1 = config.cameras.emplace_back();
  camera1.image_prefix = "camera0_";
  camera1.ref_sensor = true;
  auto& camera2 = config.cameras.emplace_back();
  camera2.image_prefix = "camera1_";
  camera2.camera =
      Camera::CreateFromModelId(2, OpenCVCameraModel::model_id, 2.0, 1024, 768);
  camera2.cam_from_rig =
      Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>());

  ApplyRigConfig(configs, *database);
  EXPECT_EQ(database->NumRigs(), 1);
  EXPECT_EQ(database->NumFrames(), 5);
  const auto [sensor_id2, sensor2_from_rig] =
      *database->ReadAllRigs().at(0).NonRefSensors().begin();
  EXPECT_EQ(sensor2_from_rig.value().rotation().coeffs(),
            camera2.cam_from_rig.value().rotation().coeffs());
  EXPECT_EQ(sensor2_from_rig.value().translation(),
            camera2.cam_from_rig.value().translation());
  EXPECT_EQ(database->ReadCamera(sensor_id2.id), camera2.camera);
}

TEST(ApplyRigConfig, WithUnconfiguredSingleAndConfiguredMultiCameraRigs) {
  auto database = OpenSqliteDatabase(":memory:");
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 3;
  options.num_frames_per_rig = 5;
  SynthesizeDataset(options, &reconstruction, database.get());

  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 3;
  SynthesizeDataset(options, &reconstruction, database.get());

  std::vector<RigConfig> configs;
  auto& config = configs.emplace_back();
  auto& camera1 = config.cameras.emplace_back();
  camera1.image_prefix = "camera000001_";
  camera1.ref_sensor = true;
  auto& camera2 = config.cameras.emplace_back();
  camera2.image_prefix = "camera000002_";

  ApplyRigConfig(configs, *database, &reconstruction);
  EXPECT_EQ(database->NumRigs(), 3);
  EXPECT_EQ(database->NumFrames(), 13);
  EXPECT_EQ(reconstruction.NumRigs(), 3);
  EXPECT_EQ(reconstruction.NumFrames(), 13);
  EXPECT_EQ(reconstruction.NumRegFrames(), 13);

  int num_non_trivial_rigs = 0;
  for (const auto& rig : database->ReadAllRigs()) {
    num_non_trivial_rigs += rig.NumSensors() > 1;
  }
  EXPECT_EQ(num_non_trivial_rigs, 1);

  int num_non_trivial_frames = 0;
  for (const auto& rig : database->ReadAllFrames()) {
    num_non_trivial_frames += rig.NumDataIds() > 1;
  }
  EXPECT_EQ(num_non_trivial_frames, 5);
}

}  // namespace

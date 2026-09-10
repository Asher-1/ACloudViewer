// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Upstream COLMAP dbb41680 (sfm/global_mapper_test.cc) port. Fork
// adaptations: concrete Database construction instead of Database::Open, the
// Database::ImagePairToPairId static method, and include paths.

#include "sfm/global_mapper.h"

#include "base/database.h"
#include "base/reconstruction.h"
#include "base/triangulation.h"
#include "scene/reconstruction_matchers.h"
#include "scene/synthetic.h"
#include "util/testing.h"

namespace colmap {
namespace {

std::shared_ptr<DatabaseCache> CreateDatabaseCache(const Database& database) {
  DatabaseCache::Options options;
  return DatabaseCache::Create(database, options);
}

TEST(GlobalMapper, WithoutNoise) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = std::make_shared<Database>(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  // GlobalMapperOptions.random_seed >= 0 pins the deterministic seeding of
  // the rotation-averaging and global-positioning initializations (the
  // upstream pipeline contract); without it the GP solve may land in a
  // mirrored basin on toolchains whose floating-point trajectory differs.
  GlobalMapperOptions options;
  options.random_seed = 42;
  global_mapper.Solve(options);

  // num_obs_tolerance: the synthetic track geometry is toolchain-dependent
  // (libstdc++ std::shuffle / uniform_int_distribution sequences differ
  // across gcc versions), and a track whose GT minimum triangulation angle
  // falls under min_tri_angle_deg may legitimately be filtered on some
  // toolchains. Upstream provides num_obs_tolerance for exactly this.
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.03));
}

TEST(GlobalMapper, WithoutNoiseWithNonTrivialKnownRig) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = std::make_shared<Database>(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_translation_stddev =
      0.1;                                                         // No noise
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 5.;  // No noise
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  GlobalMapperOptions options;
  options.random_seed = 42;
  global_mapper.Solve(options);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.03));
}

TEST(GlobalMapper, WithoutNoiseWithNonTrivialUnknownRig) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = std::make_shared<Database>(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_translation_stddev =
      0.1;                                                         // No noise
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 5.;  // No noise

  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  // Set the rig sensors to be unknown
  for (const auto& [rig_id, rig] : reconstruction->Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        reconstruction->Rig(rig_id).ResetSensorFromRig(sensor_id);
      }
    }
  }

  GlobalMapperOptions options;
  options.random_seed = 42;
  global_mapper.Solve(options);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithNoiseAndOutliers) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = std::make_shared<Database>(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.7;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  GlobalMapperOptions options;
  options.random_seed = 42;
  global_mapper.Solve(options);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(GlobalMapperOptions, RefineSensorFromRigPropagatesToSubOptions) {
  GlobalMapperOptions options;
  options.refine_sensor_from_rig = false;
  // Sub-options keep their own defaults (true) until accessed. The fork's
  // BundleAdjustmentOptions has no refine_sensor_from_rig field yet (W3-2b
  // step 4), so only the rotation-averaging and global-positioning
  // sub-options are asserted here.
  EXPECT_TRUE(options.rotation_averaging.refine_sensor_from_rig);
  EXPECT_TRUE(options.global_positioning.refine_sensor_from_rig);
  // Accessors return resolved sub-options with the top-level flag applied.
  EXPECT_FALSE(options.RotationAveraging().refine_sensor_from_rig);
  EXPECT_FALSE(options.GlobalPositioning().refine_sensor_from_rig);
}

}  // namespace
}  // namespace colmap

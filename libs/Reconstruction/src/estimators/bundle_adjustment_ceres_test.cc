// SPDX-License-Identifier: BSD-3-Clause
//
// Port of upstream MVS/colmap d3ccaf35
// src/colmap/estimators/bundle_adjustment_ceres_test.cc (batch B alignment):
// representative cases exercising the upstream-parity Ceres surface
// (CreateDefaultCeresBundleAdjuster + argument-less Solve returning the
// shared summary). The remaining upstream cases are incremental variants of
// the same rig/gauge/constant-recipe matrix that the fork's
// estimators/bundle_adjustment_test.cc already covers over the shared
// BundleAdjuster contract; they port mechanically on top of this surface as
// needed.

#include "estimators/bundle_adjustment_ceres.h"

#include <gtest/gtest.h>

#include <memory>

#include "scene/database.h"
#include "scene/reconstruction_matchers.h"
#include "scene/synthetic.h"
#include "util/testing.h"

namespace colmap {
namespace {

TEST(DefaultBundleAdjuster, NominalMultiCameraRig) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 200;
  SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction);
  Reconstruction reconstruction = gt_reconstruction;
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  synthetic_noise_options.point3D_stddev = 0.1;
  synthetic_noise_options.rig_from_world_rotation_stddev = 0.5;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.1;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);
  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  BundleAdjustmentOptions options;
  std::unique_ptr<CeresBundleAdjuster> bundle_adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  const auto summary = bundle_adjuster->Solve();
  ASSERT_NE(summary->termination_type,
            BundleAdjustmentTerminationType::FAILURE);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

TEST(DefaultBundleAdjuster, Cancellation) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 20;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  int num_checks = 0;
  BundleAdjustmentOptions options;
  options.check_if_stopped = [&num_checks]() {
    ++num_checks;
    return true;
  };
  const auto summary =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction)
          ->Solve();
  EXPECT_EQ(num_checks, 1);
  EXPECT_EQ(summary->termination_type,
            BundleAdjustmentTerminationType::USER_SUCCESS);
}

// Upstream parity (a2bd15c2 bundle_adjustment_ceres_test.cc,
// PosePriorBundleAdjuster): the fork's pose-prior adjuster is the Ceres
// adjuster with SetPosePriors(); prior filtering, alignment, normalization and
// the gauge fallback happen when the problem is assembled in Solve().
std::unique_ptr<CeresBundleAdjuster> CreatePosePriorCeresBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const PosePriorBundleAdjustmentOptions& prior_options,
    const BundleAdjustmentConfig& config, std::vector<PosePrior> pose_priors,
    Reconstruction& reconstruction) {
  auto adjuster =
      CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
  adjuster->SetPosePriors(prior_options, std::move(pose_priors));
  return adjuster;
}

BundleAdjustmentConfig ConfigWithAllRegImages(
    const Reconstruction& reconstruction) {
  BundleAdjustmentConfig config;
  for (const frame_t frame_id : reconstruction.RegFrameIds()) {
    for (const image_t image_id : reconstruction.Frame(frame_id).ImageIds()) {
      config.AddImage(image_id);
    }
  }
  return config;
}

TEST(PosePriorBundleAdjuster, AlignmentRobustToOutliers) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 7;
  synthetic_options.num_points3D = 50;
  synthetic_options.prior_position = true;
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction = gt_reconstruction;

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point3D_stddev = 0.2;
  synthetic_noise_options.rig_from_world_rotation_stddev = 1.0;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.2;
  synthetic_noise_options.prior_position_stddev = 0.05;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  // Add 2 outlier priors with very large covariance
  pose_priors.at(0).position += Eigen::Vector3d::Constant(10);
  pose_priors.at(0).position_covariance = Eigen::Matrix3d::Identity() * 1e6;
  pose_priors.at(1).position += Eigen::Vector3d::Constant(1);
  pose_priors.at(1).position_covariance = Eigen::Matrix3d::Identity() * 1e2;

  PosePriorBundleAdjustmentOptions prior_ba_options;
  prior_ba_options.alignment_ransac_options.random_seed = 0;
  prior_ba_options.alignment_ransac_options.max_error = 0.0;

  auto adjuster = CreatePosePriorCeresBundleAdjuster(
      BundleAdjustmentOptions(), prior_ba_options,
      ConfigWithAllRegImages(reconstruction), pose_priors, reconstruction);
  auto summary = adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(PosePriorBundleAdjuster, InsufficientPriorsUseTwoCameraGauge) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 50;
  synthetic_options.prior_position = true;
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  BundleAdjustmentConfig config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    config.AddImage(image_id);
  }

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  pose_priors.resize(2);
  auto adjuster = CreatePosePriorCeresBundleAdjuster(
      BundleAdjustmentOptions(), PosePriorBundleAdjustmentOptions(), config,
      std::move(pose_priors), reconstruction);

  // Fork note: the gauge fallback is decided while assembling the problem.
  EXPECT_TRUE(adjuster->Solve()->IsSolutionUsable());
  EXPECT_EQ(adjuster->Config().FixedGauge(),
            BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
}

TEST(PosePriorBundleAdjuster, MissingPositionCov) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 7;
  synthetic_options.num_points3D = 100;
  synthetic_options.prior_position = true;
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction = gt_reconstruction;

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  for (PosePrior& pose_prior : pose_priors) {
    EXPECT_FALSE(pose_prior.HasPositionCov());
  }

  PosePriorBundleAdjustmentOptions prior_ba_options;
  prior_ba_options.alignment_ransac_options.random_seed = 0;
  prior_ba_options.ceres->prior_position_loss_function_type =
      CeresLossFunctionType::CAUCHY;

  auto adjuster = CreatePosePriorCeresBundleAdjuster(
      BundleAdjustmentOptions(), prior_ba_options,
      ConfigWithAllRegImages(reconstruction), pose_priors, reconstruction);
  auto summary = adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(PosePriorBundleAdjuster, ConstantSensorFromRigWithMissingPositionCov) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 2;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 50;
  synthetic_options.prior_position = true;
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  for (const PosePrior& pose_prior : pose_priors) {
    EXPECT_FALSE(pose_prior.HasPositionCov());
  }

  BundleAdjustmentOptions ba_options;
  ba_options.refine_sensor_from_rig = false;
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    ba_config.AddImage(image_id);
  }

  auto adjuster = CreatePosePriorCeresBundleAdjuster(
      ba_options, PosePriorBundleAdjustmentOptions(), ba_config,
      std::move(pose_priors), reconstruction);
  ASSERT_TRUE(adjuster->Solve()->IsSolutionUsable());

  // Fork note: the pose parameter blocks are the adjuster's Rigid3d shadow
  // blocks, so the frozen sensor_from_rig of the non-reference camera is
  // identified as the constant 7-dim block. The reprojection residuals bake
  // it into RigReprojErrorConstantRig; only the non-reference camera's
  // position prior adds it to the problem, and it must stay constant.
  const ceres::Problem& problem = adjuster->Problem();
  std::vector<double*> parameter_blocks;
  problem.GetParameterBlocks(&parameter_blocks);
  size_t num_constant_pose_blocks = 0;
  size_t num_variable_pose_blocks = 0;
  for (double* block : parameter_blocks) {
    if (problem.ParameterBlockSize(block) != 7) {
      continue;
    }
    if (problem.IsParameterBlockConstant(block)) {
      ++num_constant_pose_blocks;
    } else {
      ++num_variable_pose_blocks;
    }
  }
  size_t num_non_ref_sensors = 0;
  for (const auto& [_, rig] : reconstruction.Rigs()) {
    num_non_ref_sensors += rig.NumSensors() - 1;
  }
  EXPECT_EQ(num_non_ref_sensors, 1);
  EXPECT_EQ(num_constant_pose_blocks, num_non_ref_sensors);
  EXPECT_EQ(num_variable_pose_blocks, reconstruction.NumRegFrames());
}

TEST(PosePriorBundleAdjuster, RejectsCasparBackend) {
  Reconstruction reconstruction;
  BundleAdjustmentOptions options;
  options.backend = BundleAdjustmentBackend::CASPAR;
  CeresBundleAdjuster adjuster(options, BundleAdjustmentConfig());
  adjuster.SetPosePriors(PosePriorBundleAdjustmentOptions(), {});
  EXPECT_THROW(adjuster.Solve(&reconstruction), std::invalid_argument);
}

TEST(PosePriorBundleAdjuster, OptimizationRobustToOutliers) {
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 7;
  synthetic_options.num_points3D = 100;
  synthetic_options.prior_position = true;
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction = gt_reconstruction;

  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point3D_stddev = 0.2;
  synthetic_noise_options.rig_from_world_rotation_stddev = 1.0;
  synthetic_noise_options.rig_from_world_translation_stddev = 0.2;
  synthetic_noise_options.prior_position_stddev = 0.05;
  SynthesizeNoise(synthetic_noise_options, &reconstruction);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  // Add 2 confident but wrong priors.
  pose_priors[0].position_covariance = Eigen::Matrix3d::Identity() * 0.01;
  pose_priors[0].position += Eigen::Vector3d::Constant(10);
  pose_priors[1].position_covariance = Eigen::Matrix3d::Identity() * 1.01;
  pose_priors[1].position += Eigen::Vector3d::Constant(10);

  PosePriorBundleAdjustmentOptions prior_ba_options;
  prior_ba_options.alignment_ransac_options.random_seed = 0;
  prior_ba_options.ceres->prior_position_loss_function_type =
      CeresLossFunctionType::CAUCHY;

  auto adjuster = CreatePosePriorCeresBundleAdjuster(
      BundleAdjustmentOptions(), prior_ba_options,
      ConfigWithAllRegImages(reconstruction), pose_priors, reconstruction);
  auto summary = adjuster->Solve();
  ASSERT_TRUE(summary->IsSolutionUsable());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

}  // namespace
}  // namespace colmap

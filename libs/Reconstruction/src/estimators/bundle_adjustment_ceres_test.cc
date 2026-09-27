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

#include <memory>

#include "scene/reconstruction_matchers.h"
#include "scene/synthetic.h"
#include "util/testing.h"

#include <gtest/gtest.h>

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

}  // namespace
}  // namespace colmap

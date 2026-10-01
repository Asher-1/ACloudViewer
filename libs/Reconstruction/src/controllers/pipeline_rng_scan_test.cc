// SPDX-License-Identifier: BSD-3-Clause
//
// TEMP-DIAG (batch F follow-up): scan global PRNG seeds to reproduce the
// seed-dependent observation deficit on the non-trivial-rig pipeline case.
// DISABLED_ on purpose - not part of the gate; run manually with
// --gtest_also_run_disabled_tests. Remove once the variance root cause is
// fixed.

#include "controllers/incremental_pipeline.h"

#include "math/random.h"
#include "scene/database.h"
#include "scene/synthetic.h"
#include "util/testing.h"

#include <gtest/gtest.h>

#include <iostream>

namespace colmap {
namespace {

TEST(PipelineRngScan, DISABLED_ScanSeeds) {
  for (unsigned k = 1; k <= 30; ++k) {
    SetPRNGSeed(k);
    const auto database_path = CreateTestDir() / ("db_" + std::to_string(k) + ".db");
    auto database = Database::Open(database_path);
    Reconstruction gt_reconstruction;
    SyntheticDatasetOptions synthetic_dataset_options;
    synthetic_dataset_options.num_rigs = 2;
    synthetic_dataset_options.num_cameras_per_rig = 2;
    synthetic_dataset_options.num_frames_per_rig = 7;
    synthetic_dataset_options.num_points3D = 100;
    synthetic_dataset_options.camera_has_prior_focal_length = false;
    synthetic_dataset_options.sensor_from_rig_translation_stddev = 0.05;
    synthetic_dataset_options.sensor_from_rig_rotation_stddev = 30;
    SynthesizeDataset(
        synthetic_dataset_options, &gt_reconstruction, database.get());

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto options = std::make_shared<IncrementalMapperOptions>();
    options->random_seed = static_cast<int>(k);
    IncrementalMapperController mapper(options.get(), "", database_path,
                                       reconstruction_manager.get());
    mapper.Start();
    mapper.Wait();

    const size_t num_models = reconstruction_manager->Size();
    size_t num_obs = 0;
    size_t num_frames = 0;
    size_t num_points = 0;
    if (num_models > 0) {
      const Reconstruction& rec = reconstruction_manager->Get(0);
      num_obs = rec.ComputeNumObservations();
      num_frames = rec.NumRegFrames();
      num_points = rec.NumPoints3D();
    }
    std::cout << "[rng-scan] seed=" << k << " models=" << num_models
              << " reg_frames=" << num_frames << " points3D=" << num_points
              << " observations=" << num_obs
              << " (gt=2800)" << std::endl;
  }
}

}  // namespace
}  // namespace colmap

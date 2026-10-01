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

// Deep-dive a single seed: pair fork point3D tracks with GT tracks via the
// shared 2D observations and report split/merge/XYZ-distance statistics.
TEST(PipelineRngScan, DISABLED_SeedDetail) {
  constexpr unsigned kSeed = 4;
  SetPRNGSeed(kSeed);
  const auto database_path = CreateTestDir() / ("db_detail_" + std::to_string(kSeed) + ".db");
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
  options->random_seed = static_cast<int>(kSeed);
  IncrementalMapperController mapper(options.get(), "", database_path,
                                     reconstruction_manager.get());
  mapper.Start();
  mapper.Wait();
  ASSERT_EQ(reconstruction_manager->Size(), 1);
  const Reconstruction& rec = reconstruction_manager->Get(0);

  // Index GT tracks by their observation set signature: map each (image,
  // point2D) observation to its GT point3D id, then count how many fork
  // points claim each GT point's observations.
  std::map<point3D_t, std::map<std::pair<image_t, point2D_t>, point3D_t>>
      gt_obs;
  for (const auto& [image_id, image] : gt_reconstruction.Images()) {
    for (point2D_t idx = 0; idx < image.Points2D().size(); ++idx) {
      const auto& p2d = image.Points2D()[idx];
      if (p2d.HasPoint3D()) {
        gt_obs[p2d.Point3DId()][{image_id, idx}] = p2d.Point3DId();
      }
    }
  }
  // Fork side: for each GT point, how many distinct fork points cover its
  // observations, and the max XYZ distance among them.
  std::map<std::pair<image_t, point2D_t>, point3D_t> fork_obs;
  for (const auto& [image_id, image] : rec.Images()) {
    for (point2D_t idx = 0; idx < image.Points2D().size(); ++idx) {
      const auto& p2d = image.Points2D()[idx];
      if (p2d.HasPoint3D()) {
        fork_obs[{image_id, idx}] = p2d.Point3DId();
      }
    }
  }
  int gt_covered_fully = 0, gt_split = 0, gt_missing_obs = 0;
  double max_split_xyz_dist = 0.0;
  std::map<size_t, int> fork_track_len_hist;
  for (const auto& [gt_pid, obs_map] : gt_obs) {
    std::map<point3D_t, int> fork_pid_count;
    size_t covered = 0;
    double min_dist = 1e30, max_dist = 0.0;
    for (const auto& [obs, _] : obs_map) {
      auto it = fork_obs.find(obs);
      if (it == fork_obs.end()) {
        continue;
      }
      ++covered;
      ++fork_pid_count[it->second];
      const double d = (rec.Point3D(it->second).XYZ() -
                        gt_reconstruction.Point3D(gt_pid).XYZ()).norm();
      min_dist = std::min(min_dist, d);
      max_dist = std::max(max_dist, d);
    }
    gt_missing_obs += static_cast<int>(obs_map.size() - covered);
    max_split_xyz_dist = std::max(max_split_xyz_dist, max_dist);
    if (fork_pid_count.size() > 1) {
      ++gt_split;
      std::cout << "[seed-detail] GT point " << gt_pid << " split into "
                << fork_pid_count.size() << " fork points; obs covered "
                << covered << "/" << obs_map.size()
                << "; xyz dist range [" << min_dist << ", " << max_dist
                << "]" << std::endl;
    } else if (fork_pid_count.size() == 1) {
      ++gt_covered_fully;
    }
  }
  for (const auto& [pid, p3] : rec.Points3D()) {
    ++fork_track_len_hist[p3.Track().Length()];
  }
  std::cout << "[seed-detail] summary: gt_fully_single=" << gt_covered_fully
            << " gt_split=" << gt_split
            << " gt_missing_obs=" << gt_missing_obs
            << " max_split_xyz_dist=" << max_split_xyz_dist << std::endl;
  std::cout << "[seed-detail] fork track length histogram:" << std::endl;
  for (const auto& [len, count] : fork_track_len_hist) {
    std::cout << "  len=" << len << " count=" << count << std::endl;
  }
}

}  // namespace
}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause
//
// Upstream parity (a2bd15c2 controllers/automatic_reconstruction_test.cc) with
// include paths mapped to the fork layout. The fork controller takes a raw
// reconstruction manager and configures itself in the constructor (no
// Setup()); the native COLMAP sparse mode is selected explicitly because AICore
// builds without CUDA default to the DA3 sparse mode.

#include "controllers/automatic_reconstruction.h"

#include <gtest/gtest.h>

#include <filesystem>

#include "scene/reconstruction_manager.h"
#include "scene/reconstruction_matchers.h"
#include "scene/synthetic.h"
#include "util/file.h"
#include "util/testing.h"

namespace colmap {
namespace {

AutomaticReconstructionController::Options CreateOptions(
    const std::filesystem::path& workspace_path,
    const std::filesystem::path& image_path,
    const AutomaticReconstructionController::Mapper mapper) {
  AutomaticReconstructionController::Options options;
  options.workspace_path = workspace_path;
  options.image_path = image_path;
  options.data_type = AutomaticReconstructionController::DataType::INDIVIDUAL;
  options.quality = AutomaticReconstructionController::Quality::LOW;
  options.single_camera = false;
  options.dense = false;  // Disable dense reconstruction to avoid GPU
  options.meshing = false;
  options.texturing = false;
  options.use_gpu = false;
  options.random_seed = 1;
  options.mapper = mapper;
  options.sparse_mode = SparseModelMode::COLMAP_NATIVE;
  return options;
}

void SynthesizeScene(const std::filesystem::path& image_path,
                     Reconstruction* gt_reconstruction) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 5;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 200;
  synthetic_dataset_options.num_points2D_without_point3D = 10;
  SynthesizeDataset(synthetic_dataset_options, gt_reconstruction);
  SynthesizeImages(SyntheticImageOptions(), *gt_reconstruction, image_path);
}

class ParameterizedAutomaticReconstructionTests
    : public ::testing::TestWithParam<
          AutomaticReconstructionController::Mapper> {};

TEST_P(ParameterizedAutomaticReconstructionTests, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  Reconstruction gt_reconstruction;
  SynthesizeScene(image_path, &gt_reconstruction);

  ReconstructionManager reconstruction_manager;
  AutomaticReconstructionController controller(
      CreateOptions(workspace_path, image_path, GetParam()),
      &reconstruction_manager);
  controller.Start();
  controller.Wait();

  EXPECT_EQ(reconstruction_manager.Size(), 1);
  EXPECT_THAT(reconstruction_manager.Get(0),
              ReconstructionNear(gt_reconstruction,
                                 /*max_rotation_error_deg=*/0.6,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.9,
                                 /*align=*/true));
}

// TODO: Add GLOBAL mapper test. Currently excluded because the test produces
// fewer observations than expected. The global pipeline is tested separately
// in global_pipeline_test.cc.
INSTANTIATE_TEST_SUITE_P(
    AutomaticReconstructionTests,
    ParameterizedAutomaticReconstructionTests,
    ::testing::Values(AutomaticReconstructionController::Mapper::INCREMENTAL,
                      AutomaticReconstructionController::Mapper::HIERARCHICAL));

// Fork addition: verify the GLOBAL dispatch produces a non-empty model and
// writes it. Upstream excludes GLOBAL from the exact scene-similarity gate
// because this synthetic graph may legitimately filter one frame.
TEST(AutomaticReconstructionController, GlobalMapperProducesModel) {
  const auto test_dir = CreateTestDir();
  const auto workspace_path = test_dir / "workspace";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(workspace_path);
  CreateDirIfNotExists(image_path);

  Reconstruction gt_reconstruction;
  SynthesizeScene(image_path, &gt_reconstruction);

  ReconstructionManager reconstruction_manager;
  AutomaticReconstructionController controller(
      CreateOptions(workspace_path,
                    image_path,
                    AutomaticReconstructionController::Mapper::GLOBAL),
      &reconstruction_manager);
  controller.Start();
  controller.Wait();

  ASSERT_EQ(reconstruction_manager.Size(), 1);
  EXPECT_GT(reconstruction_manager.Get(0).NumRegImages(), 0);
  EXPECT_LE(reconstruction_manager.Get(0).NumRegImages(),
            gt_reconstruction.NumRegImages());
  EXPECT_TRUE(ExistsDir(workspace_path / "sparse" / "0"));
}

}  // namespace
}  // namespace colmap

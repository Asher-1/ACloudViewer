// SPDX-License-Identifier: BSD-3-Clause
//
// Port of upstream MVS/colmap d3ccaf35 src/colmap/exe/sfm_test.cc (batch 24
// test alignment); include paths mapped to the fork layout and the fork's
// IncrementalMapperOptions / controller surfaces applied.

#include "exe/sfm.h"

#include <filesystem>
#include <memory>

#include "scene/reconstruction_matchers.h"
#include "scene/synthetic.h"
#include "scene/database.h"
#include "util/file.h"
#include "util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(PointTriangulator, IncludesRegisteredImagesOutsideImageList) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  auto database = Database::Open(database_path);

  auto reconstruction = std::make_shared<Reconstruction>();
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 10;
  SynthesizeDataset(synthetic_options, reconstruction.get(), database.get());

  ASSERT_EQ(reconstruction->NumRegImages(), 3);
  IncrementalMapperOptions options;
  options.image_names = {reconstruction->Image(1).Name(),
                         reconstruction->Image(2).Name()};
  options.ba_global_max_refinements = 0;
  options.num_threads = 1;

  database.reset();
  const auto output_path = test_path / "output";
  CreateDirIfNotExists(output_path);
  EXPECT_NO_THROW(RunPointTriangulatorImpl(reconstruction,
                                           database_path,
                                           test_path,
                                           output_path,
                                           options,
                                           /*clear_points=*/false,
                                           /*refine_intrinsics=*/false));
  EXPECT_EQ(reconstruction->NumRegImages(), 3);
}

TEST(PointTriangulator, WritesPartialReconstructionWhenCancelled) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  auto database = Database::Open(database_path);

  auto reconstruction = std::make_shared<Reconstruction>();
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, reconstruction.get(), database.get());
  database.reset();

  const size_t num_points3D = reconstruction->NumPoints3D();
  const auto output_path = test_path / "output";
  CreateDirIfNotExists(output_path);
  IncrementalMapperOptions options;
  options.extract_colors = false;
  bool stop_checked = false;
  RunPointTriangulatorImpl(reconstruction,
                           database_path,
                           test_path,
                           output_path,
                           options,
                           /*clear_points=*/false,
                           /*refine_intrinsics=*/false,
                           [&stop_checked]() {
                             stop_checked = true;
                             return true;
                           });

  EXPECT_TRUE(stop_checked);
  EXPECT_EQ(reconstruction->NumPoints3D(), num_points3D);
  EXPECT_FALSE(ExistsFile(output_path / "cameras.bin"));
  EXPECT_FALSE(ExistsFile(output_path / "images.bin"));
  EXPECT_FALSE(ExistsFile(output_path / "points3D.bin"));
  const auto partial_output_path = test_path / "output.partial";
  EXPECT_TRUE(ExistsFile(partial_output_path / "cameras.bin"));
  EXPECT_TRUE(ExistsFile(partial_output_path / "images.bin"));
  EXPECT_TRUE(ExistsFile(partial_output_path / "points3D.bin"));
}

TEST(PointTriangulator, DoesNotOverwriteInputWhenCancelledAfterClearingPoints) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  auto database = Database::Open(database_path);

  auto reconstruction = std::make_shared<Reconstruction>();
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, reconstruction.get(), database.get());
  database.reset();

  const Reconstruction expected_reconstruction = *reconstruction;
  const auto model_path = test_path / "model";
  CreateDirIfNotExists(model_path);
  reconstruction->Write(model_path);

  IncrementalMapperOptions options;
  options.extract_colors = false;
  RunPointTriangulatorImpl(reconstruction,
                           database_path,
                           test_path,
                           model_path,
                           options,
                           /*clear_points=*/true,
                           /*refine_intrinsics=*/false,
                           []() { return true; });

  Reconstruction written_reconstruction;
  written_reconstruction.Read(model_path);
  EXPECT_THAT(written_reconstruction,
              ReconstructionEq(expected_reconstruction));

  const auto partial_output_path = test_path / "model.partial";
  EXPECT_TRUE(ExistsFile(partial_output_path / "cameras.bin"));
  EXPECT_TRUE(ExistsFile(partial_output_path / "images.bin"));
  EXPECT_TRUE(ExistsFile(partial_output_path / "points3D.bin"));
}

TEST(IncrementalMapper, WritesInterruptedReconstruction) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 100;
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());
  database.reset();

  const auto output_path = test_path / "output";
  CreateDirIfNotExists(output_path);
  auto options = std::make_shared<IncrementalMapperOptions>();
  options->extract_colors = false;
  options->multiple_models = false;
  options->num_threads = 1;
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();

  bool stop_requested = false;
  EXPECT_TRUE(RunIncrementalMapperImpl(
      database_path,
      test_path,
      output_path,
      options,
      reconstruction_manager,
      [&stop_requested]() { stop_requested = true; },
      {},
      [&stop_requested]() { return stop_requested; }));

  EXPECT_TRUE(stop_requested);
  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_TRUE(ExistsFile(output_path / "0" / "cameras.bin"));
  EXPECT_TRUE(ExistsFile(output_path / "0" / "images.bin"));
  EXPECT_TRUE(ExistsFile(output_path / "0" / "points3D.bin"));
}

TEST(IncrementalMapper, DiscardsInterruptBeforeInitialPair) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 100;
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());
  database.reset();

  const auto output_path = test_path / "output";
  CreateDirIfNotExists(output_path);
  auto options = std::make_shared<IncrementalMapperOptions>();
  options->extract_colors = false;
  options->multiple_models = false;
  options->num_threads = 1;
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();

  int num_stop_checks = 0;
  EXPECT_FALSE(RunIncrementalMapperImpl(
      database_path,
      test_path,
      output_path,
      options,
      reconstruction_manager,
      {},
      {},
      [&num_stop_checks]() { return ++num_stop_checks >= 2; }));

  EXPECT_GE(num_stop_checks, 2);
  EXPECT_EQ(reconstruction_manager->Size(), 0);
  EXPECT_FALSE(ExistsDir(output_path / "0"));
}

}  // namespace
}  // namespace colmap

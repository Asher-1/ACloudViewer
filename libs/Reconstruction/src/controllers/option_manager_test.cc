// SPDX-License-Identifier: BSD-3-Clause
//
// Ported from upstream MVS/colmap d3ccaf35 (batch 24 test alignment);
// include paths mapped to the fork layout.

#include "controllers/option_manager.h"

#include "controllers/image_reader.h"
#include "controllers/incremental_pipeline.h"
#include "feature/sift.h"
#include "mvs/patch_match_options.h"
#include "util/file.h"
#include "util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(OptionManager, Reset) {
  OptionManager options;
  *options.database_path = "/test/path";
  *options.image_path = "/test/images";
  options.AddDatabaseOptions();
  options.AddImageOptions();
  EXPECT_EQ(*options.database_path, "/test/path");
  EXPECT_EQ(*options.image_path, "/test/images");

  options.Reset();

  EXPECT_EQ(*options.database_path, "");
  EXPECT_EQ(*options.image_path, "");
}

TEST(OptionManager, ResetOptions) {
  OptionManager options;
  *options.database_path = "/test/path";
  *options.image_path = "/test/images";
  const int original_num_threads = options.sift_extraction->num_threads;
  options.sift_extraction->num_threads = original_num_threads + 42;

  options.ResetOptions(/*reset_paths=*/true);
  EXPECT_EQ(*options.database_path, "");
  EXPECT_EQ(*options.image_path, "");
  EXPECT_EQ(options.sift_extraction->num_threads, original_num_threads);

  *options.database_path = "/test/path";
  *options.image_path = "/test/images";
  options.sift_extraction->num_threads = original_num_threads + 42;
  options.ResetOptions(/*reset_paths=*/false);
  EXPECT_EQ(*options.database_path, "/test/path");
  EXPECT_EQ(*options.image_path, "/test/images");
  EXPECT_EQ(options.sift_extraction->num_threads, original_num_threads);
}

TEST(OptionManager, AddOptionsIdempotent) {
  OptionManager options;

  // Adding options multiple times should not cause issues
  options.AddLogOptions();
  options.AddLogOptions();

  options.AddRandomOptions();
  options.AddRandomOptions();

  options.AddExtractionOptions();
  options.AddExtractionOptions();

  options.AddMatchingOptions();
  options.AddMatchingOptions();

  options.AddMapperOptions();
  options.AddMapperOptions();

  // If idempotency is not maintained, the above would cause errors
  SUCCEED();
}

TEST(OptionManager, AddAllOptions) {
  OptionManager options;
  options.AddAllOptions();

  // Verify that at least some key options are initialized
  EXPECT_NE(options.image_reader, nullptr);
  EXPECT_NE(options.sift_extraction, nullptr);
  EXPECT_NE(options.sift_matching, nullptr);
  EXPECT_NE(options.bundle_adjustment, nullptr);
  EXPECT_NE(options.mapper, nullptr);
#if defined(COLMAP_MVS_ENABLED)
  EXPECT_NE(options.patch_match_stereo, nullptr);
#endif
}

TEST(OptionManager, WriteAndRead) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  // Create necessary directories
  CreateDirIfNotExists(test_dir / "images");

  // Create and configure an OptionManager
  OptionManager options_write;
  options_write.AddDatabaseOptions();
  options_write.AddImageOptions();
  options_write.AddExtractionOptions();
  options_write.AddMapperOptions();
  options_write.AddGlobalMapperOptions();

  *options_write.database_path = test_dir / "database.db";
  *options_write.image_path = test_dir / "images";
  // Fork adaptation: empty path values in a written config fail to parse on
  // the fork's boost::program_options version, so give the mask paths real
  // values to keep the round-trip equivalence check meaningful.
  options_write.image_reader->mask_path = test_dir / "masks";
  options_write.image_reader->camera_mask_path = test_dir / "camera_masks";
  options_write.sift_extraction->max_image_size = 2048;
  options_write.sift_extraction->max_num_features = 4096;
  options_write.mapper->min_num_matches = 20;

  // Write to file
  options_write.Write(config_path);
  EXPECT_TRUE(ExistsFile(config_path));

  // Read from file
  OptionManager options_read;
  options_read.AddDatabaseOptions();
  options_read.AddImageOptions();
  options_read.AddExtractionOptions();
  options_read.AddMapperOptions();
  options_read.AddGlobalMapperOptions();

  EXPECT_TRUE(options_read.Read(config_path));

  // Verify that values were read correctly
  EXPECT_EQ(*options_read.database_path, *options_write.database_path);
  EXPECT_EQ(*options_read.image_path, *options_write.image_path);
  EXPECT_EQ(options_read.sift_extraction->max_image_size,
            options_write.sift_extraction->max_image_size);
  EXPECT_EQ(options_read.sift_extraction->max_num_features,
            options_write.sift_extraction->max_num_features);
  EXPECT_EQ(options_read.mapper->min_num_matches,
            options_write.mapper->min_num_matches);
}

TEST(OptionManager, ReRead) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  // Create necessary directories
  CreateDirIfNotExists(test_dir / "images");

  // Create and write initial config
  OptionManager options_write;
  options_write.AddAllOptions();
  *options_write.database_path = test_dir / "database.db";
  *options_write.image_path = test_dir / "images";
  // Fork adaptation: empty path values in a written config fail to parse on
  // the fork's boost::program_options version, so give the mask paths real
  // values to keep the round-trip equivalence check meaningful.
  options_write.image_reader->mask_path = test_dir / "masks";
  options_write.image_reader->camera_mask_path = test_dir / "camera_masks";
  options_write.sift_extraction->max_image_size = 2048;
  options_write.Write(config_path);

  // Read with ReRead
  OptionManager options_read;
  EXPECT_TRUE(options_read.ReRead(config_path));

  // Verify values
  EXPECT_EQ(*options_read.database_path, *options_write.database_path);
  EXPECT_EQ(*options_read.image_path, *options_write.image_path);
  EXPECT_EQ(options_read.sift_extraction->max_image_size, 2048);
}

TEST(OptionManager, ReadNonExistentFile) {
  OptionManager options;
  options.AddAllOptions();

  EXPECT_FALSE(options.Read("/path/that/does/not/exist.ini"));
}

TEST(OptionManager, Check) {
  const auto test_dir = CreateTestDir();

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();

  // Should fail with non-existent paths
  *options.database_path = test_dir / "database.db";
  *options.image_path = "/path/that/does/not/exist";
  EXPECT_FALSE(options.Check());

  // Should succeed with valid paths
  CreateDirIfNotExists(test_dir / "images");
  *options.image_path = test_dir / "images";
  EXPECT_TRUE(options.Check());
}

TEST(OptionManager, CheckDatabaseParentDir) {
  const auto test_dir = CreateTestDir();

  OptionManager options;
  options.AddDatabaseOptions();

  // Should succeed when database parent dir exists
  *options.database_path = test_dir / "database.db";
  EXPECT_TRUE(options.Check());

  // Should fail when database path is a directory
  CreateDirIfNotExists(test_dir / "bad_database");
  *options.database_path = test_dir / "bad_database";
  EXPECT_FALSE(options.Check());
}

TEST(OptionManager, ParseWithOptions) {
  const auto test_dir = CreateTestDir();
  CreateDirIfNotExists(test_dir / "images");

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddExtractionOptions();

  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";

  // Create argv with additional options
  const std::vector<std::string> args = {
      "colmap",
      "--database_path",
      database_path.string(),
      "--image_path",
      image_path.string(),
      "--SiftExtraction.max_image_size",
      "1024",
      "--SiftExtraction.max_num_features",
      "2048",
  };

  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  options.Parse(argv.size(), argv.data());

  // Verify parsed values
  EXPECT_EQ(*options.database_path, database_path);
  EXPECT_EQ(*options.image_path, image_path);
  EXPECT_EQ(options.sift_extraction->max_image_size, 1024);
  EXPECT_EQ(options.sift_extraction->max_num_features, 2048);
}

TEST(OptionManager, ParseLatestUpstreamFeatureOptionAliases) {
  OptionManager options;
  options.AddExtractionOptions();
  options.AddMatchingOptions();

  const std::vector<std::string> args = {
      "colmap",
      "--FeatureExtraction.use_gpu",
      "0",
      "--FeatureExtraction.max_image_size",
      "1024",
      "--FeatureMatching.use_gpu",
      "0",
      "--FeatureMatching.max_num_matches",
      "2048",
  };
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));
  EXPECT_FALSE(options.sift_extraction->use_gpu);
  EXPECT_EQ(options.sift_extraction->max_image_size, 1024);
  EXPECT_FALSE(options.sift_matching->use_gpu);
  EXPECT_EQ(options.sift_matching->max_num_matches, 2048);
}

TEST(OptionManager, ParseCasparSolverOptions) {
  OptionManager options;
  options.AddBundleAdjustmentOptions();

  const std::vector<std::string> args = {
      "colmap",
      "--BundleAdjustmentCaspar.solver_iter_max",
      "37",
      "--BundleAdjustmentCaspar.pcg_iter_max",
      "11",
      "--BundleAdjustmentCaspar.diag_scaling_down",
      "0.25",
      "--BundleAdjustmentCaspar.solver_rel_decrease_min",
      "0.5",
  };
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));
  EXPECT_EQ(options.bundle_adjustment->caspar_max_num_iterations, 37);
  EXPECT_EQ(options.bundle_adjustment->caspar_pcg_iter_max, 11);
  EXPECT_DOUBLE_EQ(options.bundle_adjustment->caspar_diag_scaling_down, 0.25);
  EXPECT_DOUBLE_EQ(options.bundle_adjustment->caspar_solver_rel_decrease_min,
                   0.5);
}

TEST(OptionManager, ParseWithProjectPath) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";
  CreateDirIfNotExists(test_dir / "images");

  // Create and write a config file
  OptionManager options_write;
  options_write.AddDatabaseOptions();
  options_write.AddImageOptions();
  options_write.AddExtractionOptions();

  *options_write.database_path = test_dir / "database.db";
  *options_write.image_path = test_dir / "images";
  // Fork adaptation: empty path values in a written config fail to parse on
  // the fork's boost::program_options version, so give the mask paths real
  // values to keep the round-trip equivalence check meaningful.
  options_write.image_reader->mask_path = test_dir / "masks";
  options_write.image_reader->camera_mask_path = test_dir / "camera_masks";
  options_write.sift_extraction->max_image_size = 3000;
  options_write.Write(config_path);

  // Parse using project_path
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddExtractionOptions();

  const std::vector<std::string> args = {
      "colmap",
      "--project_path",
      config_path.string(),
  };

  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  options.Parse(argv.size(), argv.data());

  // Verify values were loaded from config file
  EXPECT_EQ(*options.database_path, *options_write.database_path);
  EXPECT_EQ(*options.image_path, *options_write.image_path);
  EXPECT_EQ(options.sift_extraction->max_image_size, 3000);
}

TEST(OptionManager, ParseEmptyArguments) {
  OptionManager options;

  const std::vector<std::string> args = {"colmap"};
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }

  // Should succeed with no required options
  options.Parse(argv.size(), argv.data());
}

TEST(OptionManager, ParseUnknownArgumentsFails) {
  // Batch C5: Parse moved to BaseOptionManager with upstream's bool-return
  // contract, so the unknown-argument case is verifiable in-process again.
  const auto test_dir = CreateTestDir();
  OptionManager options(/*add_project_options=*/false);
  options.AddDatabaseOptions();
  const auto database_path = test_dir / "database.db";
  const std::vector<std::string> args = {
      "colmap",
      "--database_path",
      database_path.string(),
      "--unknown_option",
      "value",
  };
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  EXPECT_FALSE(options.Parse(argv.size(), argv.data()));
}


TEST(OptionManager, WriteAfterResetOptions) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  OptionManager options;
  options.AddAllOptions();
  *options.database_path = test_dir / "database.db";
  CreateDirIfNotExists(test_dir / "images");
  *options.image_path = test_dir / "images";

  // ResetOptions reassigns option structs, which reallocates sub-objects
  // (e.g., feature_matching->sift, bundle_adjustment->ceres). This must not
  // invalidate the raw pointers registered by AddAllOptions, otherwise
  // Write() will dereference dangling pointers.
  options.ResetOptions(/*reset_paths=*/false);

  EXPECT_NO_THROW(options.Write(config_path));
  EXPECT_TRUE(ExistsFile(config_path));
}

}  // namespace
}  // namespace colmap

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Ported from upstream MVS/colmap a2bd15c2 controllers/feature_matching_test.cc.
// The fork's matcher threads consume the legacy SiftMatchingOptions, so only
// the CreateGeometricVerifier cases (plain + rig verification) are ported
// here; the Create*FeatureMatcher cases remain covered by
// feature_matching_utils_test.cc.

#include "controllers/feature_matching.h"

#include "scene/synthetic.h"
#include "util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

void CreateTestDatabase(int num_images, Database& database) {
  Reconstruction unused_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = num_images;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 20;
  synthetic_dataset_options.num_points2D_without_point3D = 3;
  synthetic_dataset_options.prior_position = true;
  SynthesizeDataset(
      synthetic_dataset_options, &unused_reconstruction, &database);
}

TEST(CreateGeometricVerifier, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearTwoViewGeometries();

  ExistingMatchedPairingOptions pairing_options;

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto verifier = CreateGeometricVerifier(
      verifier_options, pairing_options, geometry_options, database_path);
  ASSERT_NE(verifier, nullptr);
  verifier->Start();
  verifier->Wait();

  EXPECT_GE(database->ReadAllMatches().size(), 3);
  EXPECT_GE(database->ReadTwoViewGeometries().size(), 3);
}

void ExpectRigVerificationResults(const Database& database,
                                  int num_expected_matches,
                                  int num_expected_calibrated,
                                  int num_expected_calibrated_rig) {
  // Verify that two-view geometries were created.
  int num_calibrated = 0;
  int num_calibrated_rig = 0;
  int num_others = 0;
  for (const auto& [pair_id, two_view_geometry] :
       database.ReadTwoViewGeometries()) {
    EXPECT_EQ(two_view_geometry.inlier_matches.size(), num_expected_matches);
    switch (two_view_geometry.config) {
      case TwoViewGeometry::CALIBRATED:
        ++num_calibrated;
        break;
      case TwoViewGeometry::CALIBRATED_RIG:
        ++num_calibrated_rig;
        break;
      default:
        ++num_others;
    }
  }
  // Two calibrated pairs between images in the same frames.
  EXPECT_EQ(num_calibrated, num_expected_calibrated);
  // Four calibrated pairs between images in different frames.
  EXPECT_EQ(num_calibrated_rig, num_expected_calibrated_rig);
  EXPECT_EQ(num_others, 0);
}

TEST(CreateGeometricVerifier, RigVerificationWithNonTrivialFrames) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 25;
  synthetic_dataset_options.match_config =
      SyntheticDatasetOptions::MatchConfig::EXHAUSTIVE;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction, database.get());

  ExistingMatchedPairingOptions pairing_options;

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = -1;
  verifier_options.rig_verification = true;

  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 5;

  auto verifier = CreateGeometricVerifier(
      verifier_options, pairing_options, geometry_options, database_path);
  ASSERT_NE(verifier, nullptr);
  verifier->Start();
  verifier->Wait();

  // All pairs should be overwritten with calibrated rig pairs.
  ExpectRigVerificationResults(*database,
                               synthetic_dataset_options.num_points3D,
                               /*num_expected_calibrated=*/0,
                               /*num_expected_calibrated_rig=*/15);
}

TEST(CreateGeometricVerifier, RigVerificationWithTrivialFrames) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 25;
  synthetic_dataset_options.match_config =
      SyntheticDatasetOptions::MatchConfig::EXHAUSTIVE;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction, database.get());

  ExistingMatchedPairingOptions pairing_options;

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  verifier_options.rig_verification = true;

  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 5;

  auto verifier = CreateGeometricVerifier(
      verifier_options, pairing_options, geometry_options, database_path);
  ASSERT_NE(verifier, nullptr);
  verifier->Start();
  verifier->Wait();

  // Trivial frames should be skipped and unmodified.
  ExpectRigVerificationResults(*database,
                               synthetic_dataset_options.num_points3D,
                               /*num_expected_calibrated=*/1,
                               /*num_expected_calibrated_rig=*/0);
}

}  // namespace
}  // namespace colmap

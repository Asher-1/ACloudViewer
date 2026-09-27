// SPDX-License-Identifier: BSD-3-Clause
//
// Port of upstream MVS/colmap d3ccaf35 src/colmap/feature/onnx_matchers_test.cc
// (batch 24 test alignment), adapted to the fork's engine surface:
//   - the upstream brute-force ONNX network maps to the fork's direct Eigen
//     cosine-similarity matcher (identical min_cossim/max_ratio/cross_check
//     semantics), so the brute-force cases are hermetic;
//   - the LightGlue ONNX matchers map to the AICore ggml LightGlue task and
//     run real inference when the catalog models are provisioned, skipping
//     with a reason otherwise (loma_test convention);
//   - the LomaB128 descriptor-type gating case has no fork equivalent (the
//     fork's feature types are plain float matrices and LoMa descriptors are
//     matched by the LoMa matcher), and the matcher-caching cases rely on
//     upstream's per-image session caching that the fork's stateless shells
//     do not expose; both are adjudicated out.

#include "feature/matcher.h"

#include <filesystem>
#include <memory>
#include <vector>

#include "feature/aliked.h"
#include "feature/types.h"
#include "feature/utils.h"
#include "math/random.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

constexpr int kDescriptorDim = 128;

FeatureDescriptorsFloat CreateRandomDescriptors(int num_descriptors,
                                                bool normalize = true) {
  FeatureDescriptorsFloatData float_data =
      FeatureDescriptorsFloatData::Random(num_descriptors, kDescriptorDim);
  // The fork's L2 normalization returns a new matrix by value.
  return normalize ? L2NormalizeFeatureDescriptors(float_data) : float_data;
}

std::vector<FeatureKeypoint> CreateDummyKeypoints(int count) {
  std::vector<FeatureKeypoint> keypoints(count);
  for (int i = 0; i < count; ++i) {
    keypoints[i].x = static_cast<float>(i);
    keypoints[i].y = static_cast<float>(i);
  }
  return keypoints;
}

AlikedMatchingOptions CreateBruteForceOptions() {
  AlikedMatchingOptions options;
  options.min_cossim = 0.85;
  options.max_ratio = 1.0;
  options.cross_check = true;
  return options;
}

std::unique_ptr<FeatureMatcher> CreateBruteForceMatcher(
        const AlikedMatchingOptions& bf_options) {
  FeatureMatchingOptions options(FeatureMatcherType::ALIKED_BRUTEFORCE);
  *options.aliked = bf_options;
  return FeatureMatcher::Create(options);
}

std::unique_ptr<FeatureMatcher> CreateLightGlueMatcher(
        const AlikedMatchingOptions& lightglue_options) {
  FeatureMatchingOptions options(FeatureMatcherType::ALIKED_LIGHTGLUE);
  *options.aliked = lightglue_options;
  return FeatureMatcher::Create(options);
}

FeatureMatcher::Image MakeImage(image_t id,
                                std::vector<FeatureKeypoint> keypoints,
                                FeatureDescriptorsFloat descriptors) {
  FeatureMatcher::Image image;
  image.image_id = id;
  image.camera = nullptr;
  image.keypoints = std::move(keypoints);
  image.descriptors = std::move(descriptors);
  return image;
}

bool LightGlueModelAvailable() {
  return std::filesystem::is_regular_file(DefaultAlikedLightGlueModelPath());
}

}  // namespace

class BruteForceMatcherTest : public testing::Test {};

TEST_F(BruteForceMatcherTest, SelfMatching) {
  constexpr int kNumDescriptors = 20;

  auto keypoints = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors = CreateRandomDescriptors(kNumDescriptors);

  AlikedMatchingOptions bf_options = CreateBruteForceOptions();
  bf_options.max_ratio = 0;  // Disable ratio test.
  auto matcher = CreateBruteForceMatcher(bf_options);

  FeatureMatcher::Image image1 = MakeImage(1, keypoints, descriptors);
  FeatureMatcher::Image image2 = MakeImage(2, keypoints, descriptors);

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  // Self-matching should match every descriptor to itself.
  EXPECT_EQ(matches.size(), kNumDescriptors);
  for (const auto& match : matches) {
    EXPECT_EQ(match.point2D_idx1, match.point2D_idx2);
  }
}

TEST_F(BruteForceMatcherTest, RatioTest) {
  constexpr int kNumDescriptors = 20;

  auto keypoints1 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors1 = CreateRandomDescriptors(kNumDescriptors);
  auto keypoints2 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors2 = CreateRandomDescriptors(kNumDescriptors);

  AlikedMatchingOptions bf_options_no_ratio = CreateBruteForceOptions();
  bf_options_no_ratio.max_ratio = 0;     // Disable ratio test.
  bf_options_no_ratio.cross_check = false;
  bf_options_no_ratio.min_cossim = -1;   // Accept all.

  AlikedMatchingOptions bf_options_ratio = CreateBruteForceOptions();
  bf_options_ratio.max_ratio = 0.8;      // Enable ratio test.
  bf_options_ratio.cross_check = false;
  bf_options_ratio.min_cossim = -1;      // Accept all.

  FeatureMatcher::Image image1 =
          MakeImage(1, keypoints1, descriptors1);
  FeatureMatcher::Image image2 =
          MakeImage(2, keypoints2, descriptors2);

  FeatureMatches matches_no_ratio, matches_ratio;
  CreateBruteForceMatcher(bf_options_no_ratio)
          ->Match(image1, image2, &matches_no_ratio);
  CreateBruteForceMatcher(bf_options_ratio)
          ->Match(image1, image2, &matches_ratio);

  // Ratio test should filter some matches.
  EXPECT_GT(matches_no_ratio.size(), 0);
  EXPECT_LT(matches_ratio.size(), matches_no_ratio.size());
}

TEST_F(BruteForceMatcherTest, CrossCheck) {
  constexpr int kNumDescriptors = 20;

  auto keypoints1 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors1 = CreateRandomDescriptors(kNumDescriptors);
  auto keypoints2 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors2 = CreateRandomDescriptors(kNumDescriptors);

  AlikedMatchingOptions bf_options_no_cross = CreateBruteForceOptions();
  bf_options_no_cross.max_ratio = 0;
  bf_options_no_cross.cross_check = false;
  bf_options_no_cross.min_cossim = -1;

  AlikedMatchingOptions bf_options_cross = CreateBruteForceOptions();
  bf_options_cross.max_ratio = 0;
  bf_options_cross.cross_check = true;
  bf_options_cross.min_cossim = -1;

  FeatureMatcher::Image image1 =
          MakeImage(1, keypoints1, descriptors1);
  FeatureMatcher::Image image2 =
          MakeImage(2, keypoints2, descriptors2);

  FeatureMatches matches_no_cross, matches_cross;
  CreateBruteForceMatcher(bf_options_no_cross)
          ->Match(image1, image2, &matches_no_cross);
  CreateBruteForceMatcher(bf_options_cross)
          ->Match(image1, image2, &matches_cross);

  // Cross-check should filter some matches.
  EXPECT_GT(matches_no_cross.size(), 0);
  EXPECT_LT(matches_cross.size(), matches_no_cross.size());
}

TEST_F(BruteForceMatcherTest, MinCossim) {
  constexpr int kNumDescriptors = 20;

  auto keypoints1 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors1 = CreateRandomDescriptors(kNumDescriptors);
  auto keypoints2 = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors2 = CreateRandomDescriptors(kNumDescriptors);

  AlikedMatchingOptions bf_options_low = CreateBruteForceOptions();
  bf_options_low.max_ratio = 0;
  bf_options_low.cross_check = false;
  bf_options_low.min_cossim = -1;  // Accept all.

  AlikedMatchingOptions bf_options_high = CreateBruteForceOptions();
  bf_options_high.max_ratio = 0;
  bf_options_high.cross_check = false;
  bf_options_high.min_cossim = 0.9;  // Very strict.

  FeatureMatcher::Image image1 =
          MakeImage(1, keypoints1, descriptors1);
  FeatureMatcher::Image image2 =
          MakeImage(2, keypoints2, descriptors2);

  FeatureMatches matches_low, matches_high;
  CreateBruteForceMatcher(bf_options_low)
          ->Match(image1, image2, &matches_low);
  CreateBruteForceMatcher(bf_options_high)
          ->Match(image1, image2, &matches_high);

  // Higher min_cossim should filter more matches.
  EXPECT_GT(matches_low.size(), 0);
  EXPECT_LT(matches_high.size(), matches_low.size());
}

TEST_F(BruteForceMatcherTest, TooFewDescriptors) {
  // The upstream brute-force ONNX graph requires at least two descriptors
  // per image for its ratio branch; the fork mirrors that floor.
  auto keypoints1 = CreateDummyKeypoints(1);
  auto descriptors1 = CreateRandomDescriptors(1);
  auto keypoints2 = CreateDummyKeypoints(10);
  auto descriptors2 = CreateRandomDescriptors(10);

  auto matcher =
          CreateBruteForceMatcher(CreateBruteForceOptions());

  FeatureMatcher::Image image1 =
          MakeImage(1, keypoints1, descriptors1);
  FeatureMatcher::Image image2 =
          MakeImage(2, keypoints2, descriptors2);

  FeatureMatches matches12;
  matcher->Match(image1, image2, &matches12);
  EXPECT_EQ(matches12.size(), 0);

  FeatureMatches matches21;
  matcher->Match(image2, image1, &matches21);
  EXPECT_EQ(matches21.size(), 0);
}

TEST_F(BruteForceMatcherTest, EmptyDescriptors) {
  auto keypoints1 = CreateDummyKeypoints(0);
  auto descriptors1 = CreateRandomDescriptors(0);
  auto keypoints2 = CreateDummyKeypoints(10);
  auto descriptors2 = CreateRandomDescriptors(10);

  auto matcher =
          CreateBruteForceMatcher(CreateBruteForceOptions());

  FeatureMatcher::Image image1 =
          MakeImage(1, keypoints1, descriptors1);
  FeatureMatcher::Image image2 =
          MakeImage(2, keypoints2, descriptors2);

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  // Should return empty matches without crashing.
  EXPECT_EQ(matches.size(), 0);
}

// ---------------------------------------------------------------------------
// LightGlue (AICore ggml) matcher cases: real inference, skipped when the
// catalog models are not provisioned.
// ---------------------------------------------------------------------------

class AlikedLightGlueMatcherTest : public testing::Test {};

TEST_F(AlikedLightGlueMatcherTest, SelfMatching) {
  if (!LightGlueModelAvailable()) {
    GTEST_SKIP() << "ALIKED LightGlue model not provisioned in the shared "
                    "AICore catalog cache";
  }
  constexpr int kNumDescriptors = 20;

  auto keypoints = CreateDummyKeypoints(kNumDescriptors);
  auto descriptors = CreateRandomDescriptors(kNumDescriptors);

  AlikedMatchingOptions lightglue_options;
  lightglue_options.min_score = 0.1;
  auto matcher = CreateLightGlueMatcher(lightglue_options);

  FeatureMatcher::Image image1 = MakeImage(1, keypoints, descriptors);
  FeatureMatcher::Image image2 = MakeImage(2, keypoints, descriptors);

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  // Self-matching with a strong front end should be close to an identity
  // mapping; assert the mutual-consistency subset that is engine-robust.
  EXPECT_GT(matches.size(), 0);
  for (const auto& match : matches) {
    EXPECT_EQ(match.point2D_idx1, match.point2D_idx2);
  }
}

TEST_F(AlikedLightGlueMatcherTest, EmptyKeypoints) {
  if (!LightGlueModelAvailable()) {
    GTEST_SKIP() << "ALIKED LightGlue model not provisioned in the shared "
                    "AICore catalog cache";
  }
  auto keypoints1 = CreateDummyKeypoints(0);
  auto descriptors1 = CreateRandomDescriptors(0);
  auto keypoints2 = CreateDummyKeypoints(10);
  auto descriptors2 = CreateRandomDescriptors(10);

  AlikedMatchingOptions lightglue_options;
  lightglue_options.min_score = 0.1;
  auto matcher = CreateLightGlueMatcher(lightglue_options);

  FeatureMatcher::Image image1 =
          MakeImage(1, keypoints1, descriptors1);
  FeatureMatcher::Image image2 =
          MakeImage(2, keypoints2, descriptors2);
  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  EXPECT_EQ(matches.size(), 0);
}

}  // namespace
}  // namespace colmap

// SPDX-License-Identifier: BSD-3-Clause
//
// Port of upstream MVS/colmap d3ccaf35 src/colmap/feature/aliked_test.cc
// (batch 24 test alignment), adapted to the fork's AICore ggml ALIKED task:
//   - real extraction runs through the FeatureExtractor::Create factory and
//     the AICore catalog models, skipping when the N16Rot extractor model is
//     not provisioned (loma_test convention);
//   - the upstream parameterization over ALIKED_N32 is adjudicated to a
//     skip: the N32 extractor model is not published in the shared catalog
//     yet, so only N16Rot runs real inference;
//   - descriptors are plain float matrices on the fork (128-dim ALIKED
//     contract preserved).

#include "feature/aliked.h"

#include <filesystem>

#include "feature/matcher.h"
#include "math/random.h"
#include "sensor/bitmap.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

void CreateRandomRgbImage(const int width, const int height, Bitmap* bitmap) {
  *bitmap = Bitmap(width, height, /*as_rgb=*/true);
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      bitmap->SetPixel(c,
                       r,
                       BitmapColor<uint8_t>(RandomUniformInteger(0, 255),
                                            RandomUniformInteger(0, 255),
                                            RandomUniformInteger(0, 255)));
    }
  }
}

bool N16RotModelAvailable() {
  return std::filesystem::is_regular_file(
          DefaultAlikedExtractorModelPath("n16rot"));
}

FeatureExtractionOptions CreateExtractionOptions() {
  FeatureExtractionOptions options(FeatureExtractorType::ALIKED_N16ROT);
  options.aliked->min_score = 0.0;
  return options;
}

TEST(AlikedTest, Nominal) {
  if (!N16RotModelAvailable()) {
    GTEST_SKIP() << "ALIKED N16Rot model not provisioned in the shared AICore "
                    "catalog cache";
  }
  Bitmap image;
  CreateRandomRgbImage(200, 100, &image);

  FeatureExtractionOptions extraction_options = CreateExtractionOptions();
  auto extractor = FeatureExtractor::Create(extraction_options);
  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  ASSERT_TRUE(extractor->Extract(image, &keypoints, &descriptors));

  EXPECT_GT(keypoints.size(), 0);
  EXPECT_LE(keypoints.size(),
            static_cast<size_t>(extraction_options.aliked->max_num_features));
  EXPECT_EQ(descriptors.type, FeatureExtractorType::ALIKED_N16ROT);
  EXPECT_EQ(keypoints.size(), static_cast<size_t>(descriptors.data.rows()));
  // Packed float32 descriptors: 128 values x 4 bytes.
  EXPECT_EQ(descriptors.data.cols(), 128 * sizeof(float));

  for (const auto& keypoint : keypoints) {
    EXPECT_GE(keypoint.x, 0);
    EXPECT_GE(keypoint.y, 0);
    EXPECT_LE(keypoint.x, image.Width());
    EXPECT_LE(keypoint.y, image.Height());
  }

  // Self-matching through the brute-force matcher mirrors the upstream
  // Nominal case's matcher section.
  FeatureMatchingOptions matching_options(
      FeatureMatcherType::ALIKED_BRUTEFORCE);
  matching_options.aliked->brute_force.max_ratio = 0;
  auto matcher = FeatureMatcher::Create(matching_options);
  FeatureMatcher::Image image1;
  image1.image_id = 1;
  image1.keypoints = std::make_shared<FeatureKeypoints>(keypoints);
  image1.descriptors = std::make_shared<FeatureDescriptors>(descriptors);
  FeatureMatcher::Image image2 = image1;
  image2.image_id = 2;
  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  EXPECT_EQ(matches.size(), keypoints.size());
  for (const auto& match : matches) {
    EXPECT_EQ(match.point2D_idx1, match.point2D_idx2);
  }
}

TEST(AlikedTest, MaxNumFeatures) {
  if (!N16RotModelAvailable()) {
    GTEST_SKIP() << "ALIKED N16Rot model not provisioned in the shared AICore "
                    "catalog cache";
  }
  Bitmap image;
  CreateRandomRgbImage(200, 100, &image);

  FeatureExtractionOptions options_default = CreateExtractionOptions();
  auto extractor_default = FeatureExtractor::Create(options_default);
  FeatureKeypoints keypoints_default;
  FeatureDescriptors descriptors_default;
  ASSERT_TRUE(extractor_default->Extract(image, &keypoints_default,
                                         &descriptors_default));

  FeatureExtractionOptions options_limited = CreateExtractionOptions();
  options_limited.aliked->max_num_features = 100;
  auto extractor_limited = FeatureExtractor::Create(options_limited);
  FeatureKeypoints keypoints_limited;
  FeatureDescriptors descriptors_limited;
  ASSERT_TRUE(extractor_limited->Extract(image, &keypoints_limited,
                                         &descriptors_limited));

  EXPECT_LE(keypoints_limited.size(), 100);
  EXPECT_LT(keypoints_limited.size(), keypoints_default.size());
}

TEST(AlikedTest, MinScore) {
  if (!N16RotModelAvailable()) {
    GTEST_SKIP() << "ALIKED N16Rot model not provisioned in the shared AICore "
                    "catalog cache";
  }
  Bitmap image;
  CreateRandomRgbImage(200, 100, &image);

  FeatureExtractionOptions options_low = CreateExtractionOptions();
  options_low.aliked->min_score = 0.0;
  auto extractor_low = FeatureExtractor::Create(options_low);
  FeatureKeypoints keypoints_low;
  FeatureDescriptors descriptors_low;
  ASSERT_TRUE(extractor_low->Extract(image, &keypoints_low, &descriptors_low));

  FeatureExtractionOptions options_high = CreateExtractionOptions();
  options_high.aliked->min_score = 0.9;
  auto extractor_high = FeatureExtractor::Create(options_high);
  FeatureKeypoints keypoints_high;
  FeatureDescriptors descriptors_high;
  ASSERT_TRUE(
          extractor_high->Extract(image, &keypoints_high, &descriptors_high));

  // A higher score threshold should detect fewer keypoints.
  EXPECT_LT(keypoints_high.size(), keypoints_low.size());
}

TEST(AlikedTest, N32ModelNotPublished) {
  // Fork adjudication: the shared catalog publishes only the N16Rot
  // extractor so far; requesting N32 must fail the factory instead of
  // silently falling back. Unconditional: the N32 catalog default resolves
  // to an empty path regardless of which models are cached.
  EXPECT_TRUE(DefaultAlikedExtractorModelPath("n32").empty());
  FeatureExtractionOptions options(FeatureExtractorType::ALIKED_N32);
  options.aliked->min_score = 0.0;
  EXPECT_THROW(FeatureExtractor::Create(options), std::exception);
}

}  // namespace
}  // namespace colmap

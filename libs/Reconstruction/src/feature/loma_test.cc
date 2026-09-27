// SPDX-License-Identifier: BSD-3-Clause
//
// Port of upstream MVS/colmap d3ccaf35 src/colmap/feature/loma_test.cc
// (batch 24 test alignment), adapted to the fork's ggml AICore LoMa engine:
// - the upstream ONNX factories resolve through FeatureExtractor/FeatureMatcher::Create;
// - descriptors are plain float matrices (fork feature types);
// - upstream's bf16 matcher-creation test maps to ggml variant matcher creation;
// - when the catalog models are not provisioned (headless/offline gate), the
//   real-inference cases skip with a reason instead of failing, mirroring the
//   doc's engine-guardrail convention.

#include "feature/extractor.h"
#include "feature/loma.h"
#include "feature/matcher.h"
#include "feature/types.h"
#include "math/random.h"
#include "scene/camera.h"
#include "sensor/bitmap.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

void CreateRandomRgbImage(int width, int height, Bitmap* bitmap) {
  SetPRNGSeed(42);
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

// Skips the calling test when the ggml LoMa models cannot be provisioned
// (offline/headless gate); keeps the real-inference coverage active whenever
// the shared AICore cache has the catalog models.
void SkipUnlessLomaModelsAvailable() {
  LomaExtractionOptions probe;
  if (!ProvisionDefaultLomaModels(&probe)) {
    GTEST_SKIP() << "LoMa ggml models not provisioned (offline gate)";
  }
}

TEST(LomaTest, Nominal) {
  SkipUnlessLomaModelsAvailable();
  Bitmap image;
  CreateRandomRgbImage(512, 512, &image);

  FeatureExtractionOptions extraction_options(FeatureExtractorType::LOMA_B);
  extraction_options.use_gpu = false;
  extraction_options.loma->min_score = 0.0;
  auto extractor = FeatureExtractor::Create(extraction_options);
  FeatureKeypoints keypoints;
  FeatureDescriptorsFloat descriptors;
  ASSERT_TRUE(extractor->Extract(image, &keypoints, &descriptors));

  EXPECT_GT(keypoints.size(), 0);
  EXPECT_EQ(keypoints.size(), descriptors.rows());
  // Fork adaptation: the ggml LOMA_B engine outputs 256-dim descriptors
  // (upstream ONNX exports 128-dim).
  EXPECT_EQ(descriptors.cols(), 256);

  for (const auto& keypoint : keypoints) {
    EXPECT_GE(keypoint.x, 0);
    EXPECT_GE(keypoint.y, 0);
    EXPECT_LE(keypoint.x, image.Width());
    EXPECT_LE(keypoint.y, image.Height());
  }

  auto camera = std::make_shared<Camera>();
  camera->SetWidth(image.Width());
  camera->SetHeight(image.Height());

  FeatureMatchingOptions matching_options(FeatureMatcherType::LOMA_B);
  matching_options.use_gpu = false;
  matching_options.loma->min_score = 0.0;
  auto matcher = FeatureMatcher::Create(matching_options);

  FeatureMatcher::Image image1;
  image1.image_id = 1;
  image1.camera = camera;
  image1.keypoints = keypoints;
  image1.descriptors = descriptors;
  FeatureMatcher::Image image2 = image1;
  image2.image_id = 2;

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);

  ASSERT_GT(matches.size(), 0);
  int num_self_matches = 0;
  for (const auto& match : matches) {
    EXPECT_GE(match.point2D_idx1, 0);
    EXPECT_LT(match.point2D_idx1, keypoints.size());
    EXPECT_GE(match.point2D_idx2, 0);
    EXPECT_LT(match.point2D_idx2, keypoints.size());
    if (match.point2D_idx1 == match.point2D_idx2) ++num_self_matches;
  }
  EXPECT_GT(num_self_matches, 0.5 * matches.size());
}

TEST(LomaTest, MinScore) {
  SkipUnlessLomaModelsAvailable();
  Bitmap image;
  CreateRandomRgbImage(512, 512, &image);

  FeatureExtractionOptions options_low(FeatureExtractorType::LOMA_B);
  options_low.use_gpu = false;
  options_low.loma->min_score = 0.0;
  auto extractor_low = FeatureExtractor::Create(options_low);
  FeatureKeypoints keypoints_low;
  FeatureDescriptorsFloat descriptors_low;
  ASSERT_TRUE(extractor_low->Extract(image, &keypoints_low, &descriptors_low));

  FeatureExtractionOptions options_high(FeatureExtractorType::LOMA_B);
  options_high.use_gpu = false;
  options_high.loma->min_score = 0.9;
  auto extractor_high = FeatureExtractor::Create(options_high);
  FeatureKeypoints keypoints_high;
  FeatureDescriptorsFloat descriptors_high;
  ASSERT_TRUE(
      extractor_high->Extract(image, &keypoints_high, &descriptors_high));

  EXPECT_LE(keypoints_high.size(), keypoints_low.size());
}

TEST(LomaTest, DynamicNumKeypoints) {
  SkipUnlessLomaModelsAvailable();
  Bitmap image;
  CreateRandomRgbImage(512, 512, &image);

  FeatureExtractionOptions options_512(FeatureExtractorType::LOMA_B);
  options_512.use_gpu = false;
  options_512.loma->min_score = 0.0;
  options_512.loma->max_num_features = 512;
  auto extractor_512 = FeatureExtractor::Create(options_512);
  FeatureKeypoints keypoints_512;
  FeatureDescriptorsFloat descriptors_512;
  ASSERT_TRUE(extractor_512->Extract(image, &keypoints_512, &descriptors_512));

  FeatureExtractionOptions options_2048(FeatureExtractorType::LOMA_B);
  options_2048.use_gpu = false;
  options_2048.loma->min_score = 0.0;
  options_2048.loma->max_num_features = 2048;
  auto extractor_2048 = FeatureExtractor::Create(options_2048);
  FeatureKeypoints keypoints_2048;
  FeatureDescriptorsFloat descriptors_2048;
  ASSERT_TRUE(
      extractor_2048->Extract(image, &keypoints_2048, &descriptors_2048));

  // Upstream asserts the exact requested counts (dynamic ONNX detector
  // input). The ggml engine keeps max_num_features as a real runtime input;
  // assert the respected upper bounds, which is the guarantee the option
  // contract makes on this backend.
  EXPECT_LE(keypoints_512.size(), 512u);
  EXPECT_LE(keypoints_2048.size(), 2048u);
}

TEST(LomaTest, MatcherCreationAcrossVariants) {
  // Upstream: Bf16MatcherCpuFallback creates a bf16-option matcher without a
  // GPU. The ggml engine has no bf16 switch; the equivalent contract is that
  // every LoMa variant matcher resolves (model-path resolution only, no
  // download at creation time).
  for (const auto type : {FeatureMatcherType::LOMA_B,
                          FeatureMatcherType::LOMA_G}) {
    FeatureMatchingOptions options(type);
    options.use_gpu = false;
    EXPECT_NE(FeatureMatcher::Create(options), nullptr);
  }
}

}  // namespace
}  // namespace colmap

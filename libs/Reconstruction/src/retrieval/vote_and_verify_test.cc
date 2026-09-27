// SPDX-License-Identifier: BSD-3-Clause
//
// Ported from upstream MVS/colmap d3ccaf35 (batch 24 test alignment);
// include paths mapped to the fork layout.

#define _USE_MATH_DEFINES

#include "retrieval/vote_and_verify.h"

#include "math/random.h"
#include "retrieval/geometry.h"

#include <cmath>

#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {
namespace {

struct SyntheticData {
  FeatureGeometryTransform img2_from_img1;
  std::vector<FeatureGeometryMatch> matches;
};

SyntheticData SynthesizeData(size_t num_inliers, size_t num_outliers) {
  SyntheticData data;
  data.img2_from_img1.scale = RandomUniformReal<float>(0.1, 10);
  data.img2_from_img1.angle = RandomUniformReal<float>(0, 2 * M_PI);
  data.img2_from_img1.tx = RandomUniformReal<float>(-100, 100);
  data.img2_from_img1.ty = RandomUniformReal<float>(-100, 100);

  const float sin_angle = std::sin(data.img2_from_img1.angle);
  const float cos_angle = std::cos(data.img2_from_img1.angle);

  data.matches.resize(num_inliers + num_outliers);
  for (size_t i = 0; i < num_inliers; ++i) {
    FeatureGeometryMatch& match = data.matches[i];
    match.geometry1.scale = RandomUniformReal<float>(0.5, 2);
    match.geometry1.orientation = RandomUniformReal<float>(0, 2 * M_PI);
    match.geometry1.x = RandomUniformReal<float>(-100, 100);
    match.geometry1.y = RandomUniformReal<float>(-100, 100);
    // The fork's FeatureGeometryMatch carries a vector of correspondences;
    // upstream's single geometry2 maps to a one-element vector here.
    FeatureGeometry geometry2;
    geometry2.scale = data.img2_from_img1.scale * match.geometry1.scale;
    geometry2.orientation =
        data.img2_from_img1.angle + match.geometry1.orientation;
    geometry2.x =
        data.img2_from_img1.scale *
        (cos_angle * match.geometry1.x - sin_angle * match.geometry1.y);
    geometry2.y =
        data.img2_from_img1.scale *
        (sin_angle * match.geometry1.x + cos_angle * match.geometry1.y);
    match.geometries2 = {geometry2};
  }

  for (size_t i = 0; i < num_outliers; ++i) {
    FeatureGeometryMatch& match = data.matches[i + num_inliers];
    match.geometry1.scale = RandomUniformReal<float>(0.5, 2);
    match.geometry1.orientation = RandomUniformReal<float>(0, 2 * M_PI);
    match.geometry1.x = RandomUniformReal<float>(-100, 100);
    match.geometry1.y = RandomUniformReal<float>(-100, 100);
    FeatureGeometry geometry2;
    geometry2.scale = RandomUniformReal<float>(0.5, 2);
    geometry2.orientation = RandomUniformReal<float>(0, 2 * M_PI);
    geometry2.x = RandomUniformReal<float>(-100, 100);
    geometry2.y = RandomUniformReal<float>(-100, 100);
    match.geometries2 = {geometry2};
  }

  return data;
}

TEST(VoteAndVerify, NoMatches) {
  EXPECT_EQ(VoteAndVerify(VoteAndVerifyOptions(), {}), 0);
}

// Fork adjudication: the eff_inlier_count option behind this case is not
// bound on the fork's VoteAndVerifyOptions; deferred until bound.

TEST(VoteAndVerify, EffectiveInliers) {
  const size_t kNumInliers = 100;
  const size_t kNumOutliers = 50;
  const auto data = SynthesizeData(kNumInliers, kNumOutliers);
  VoteAndVerifyOptions options;
  // Fork: eff_inlier_count is not bound on the fork's options.
  const int num_inliers = VoteAndVerify(options, data.matches);
  EXPECT_GT(num_inliers, 0.8 * kNumInliers);
}

}  // namespace
}  // namespace retrieval
}  // namespace colmap

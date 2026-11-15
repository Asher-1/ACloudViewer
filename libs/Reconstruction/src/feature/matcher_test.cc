// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Port of upstream MVS/colmap d3ccaf35 src/colmap/feature/matcher_test.cc
// (batch 24 test alignment). Include paths mapped to the fork layout; the
// upstream onnx_matchers.h contract is carried by learned_matchers.h and the
// nested BruteForceONNXMatchingOptions group is flattened onto
// AlikedMatchingOptions (ONNX -> ggml substitution), so the cossim assertions
// target aliked->min_cossim.
// SPDX-License-Identifier: BSD-3-Clause
// ----------------------------------------------------------------------------

#include "feature/matcher.h"

#include "feature/aliked.h"
#include "feature/extractor.h"
#include "feature/loma.h"
#include "feature/sift.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(FeatureMatchingOptions, Copy) {
  FeatureMatchingOptions options;
  options.max_num_matches += 100;
  options.sift->max_ratio *= 0.1;
  options.aliked->brute_force.min_cossim *= 0.1;

  FeatureMatchingOptions copy = options;

  // Verify fields are copied
  EXPECT_EQ(copy.max_num_matches, options.max_num_matches);
  EXPECT_EQ(copy.sift->max_ratio, options.sift->max_ratio);
  EXPECT_EQ(copy.aliked->brute_force.min_cossim, options.aliked->brute_force.min_cossim);

  // Verify deep copy of shared_ptr (different pointer instances)
  EXPECT_NE(options.sift.get(), copy.sift.get());
  EXPECT_NE(options.aliked.get(), copy.aliked.get());
}

TEST(FeatureMatchingOptions, CopyAssignment) {
  FeatureMatchingOptions options;
  options.max_num_matches += 100;
  options.sift->max_ratio *= 0.1;
  options.aliked->brute_force.min_cossim *= 0.1;

  FeatureMatchingOptions assigned;
  assigned = options;

  // Verify fields are copied.
  EXPECT_EQ(assigned.max_num_matches, options.max_num_matches);
  EXPECT_EQ(assigned.sift->max_ratio, options.sift->max_ratio);
  EXPECT_EQ(assigned.aliked->brute_force.min_cossim, options.aliked->brute_force.min_cossim);

  // Verify deep copy (different pointer instances).
  EXPECT_NE(options.sift.get(), assigned.sift.get());
  EXPECT_NE(options.aliked.get(), assigned.aliked.get());

  // Test self-assignment (assign via const ref to avoid -Wself-assign).
  const auto* sift_ptr = options.sift.get();
  const auto* aliked_ptr = options.aliked.get();
  const auto prev_max_num_matches = options.max_num_matches;
  const auto& self_ref = options;
  options = self_ref;
  EXPECT_EQ(options.sift.get(), sift_ptr);
  EXPECT_EQ(options.aliked.get(), aliked_ptr);
  EXPECT_EQ(options.max_num_matches, prev_max_num_matches);
}

TEST(FeatureMatchingOptions, RequiresOpenGL) {
  FeatureMatchingOptions options;
  options.use_gpu = false;

  const FeatureMatcherType types[] = {FeatureMatcherType::SIFT_BRUTEFORCE,
                                      FeatureMatcherType::SIFT_LIGHTGLUE,
                                      FeatureMatcherType::ALIKED_BRUTEFORCE,
                                      FeatureMatcherType::ALIKED_LIGHTGLUE,
                                      FeatureMatcherType::LOMA_BRUTEFORCE,
                                      FeatureMatcherType::LOMA_B,
                                      FeatureMatcherType::LOMA_B128,
                                      FeatureMatcherType::LOMA_R,
                                      FeatureMatcherType::LOMA_L,
                                      FeatureMatcherType::LOMA_G};

  for (const auto type : types) {
    options.type = type;
    EXPECT_FALSE(options.RequiresOpenGL());
  }
}

TEST(FeatureMatchingOptions, Check) {
  FeatureMatchingOptions options;
  options.use_gpu = false;

  const FeatureMatcherType types[] = {FeatureMatcherType::SIFT_BRUTEFORCE,
                                      FeatureMatcherType::SIFT_LIGHTGLUE,
                                      FeatureMatcherType::ALIKED_BRUTEFORCE,
                                      FeatureMatcherType::ALIKED_LIGHTGLUE,
                                      FeatureMatcherType::LOMA_BRUTEFORCE,
                                      FeatureMatcherType::LOMA_B,
                                      FeatureMatcherType::LOMA_B128,
                                      FeatureMatcherType::LOMA_R,
                                      FeatureMatcherType::LOMA_L,
                                      FeatureMatcherType::LOMA_G};

  for (const auto type : types) {
    options.type = type;
    EXPECT_TRUE(options.Check());
  }
}

}  // namespace
}  // namespace colmap

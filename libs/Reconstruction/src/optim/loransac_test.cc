// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#define TEST_NAME "optim/ransac"
#include "util/testing.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "geometry/pose.h"
#include "geometry/similarity_transform.h"
#include "estimators/solvers/similarity_transform.h"
#include "optim/loransac.h"
#include "math/random.h"

#include <algorithm>

using namespace colmap;

namespace {

struct SimilarityTransformTestData {
  SimilarityTransform3 expected_tform;
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> tgt;
  size_t num_samples;
  size_t num_outliers;
};

SimilarityTransformTestData GenerateTestData(const size_t num_samples = 1000,
                                             const size_t num_outliers = 400) {
  SimilarityTransformTestData data{
      SimilarityTransform3(2, ComposeIdentityQuaternion(),
                           Eigen::Vector3d(100, 10, 10)),
      {},
      {},
      num_samples,
      num_outliers};
  for (size_t i = 0; i < num_samples; ++i) {
    data.src.emplace_back(i, std::sqrt(i) + 2, std::sqrt(2 * i + 2));
    data.tgt.push_back(data.src.back());
    data.expected_tform.TransformPoint(&data.tgt.back());
  }
  for (size_t i = 0; i < num_outliers; ++i) {
    data.tgt[i] = Eigen::Vector3d(RandomReal(-3000.0, -2000.0),
                                  RandomReal(-4000.0, -3000.0),
                                  RandomReal(-5000.0, -4000.0));
  }
  return data;
}

template <typename Report>
void ValidateReport(const Report& report,
                    const SimilarityTransformTestData& data) {
  EXPECT_TRUE(report.success);
  EXPECT_GT(report.num_trials, 0);
  EXPECT_EQ(report.support.num_inliers, data.num_samples - data.num_outliers);
  for (size_t i = 0; i < data.num_samples; ++i) {
    EXPECT_EQ(static_cast<bool>(report.inlier_mask[i]), i >= data.num_outliers);
  }
  EXPECT_LT((data.expected_tform.Matrix().topLeftCorner<3, 4>() - report.model)
                .norm(),
            1e-6);
}

// Local estimator whose refit shrinks the inlier set below its own minimal
// sample count; recursive refinement must stop instead of calling Refine.
class ShrinkingInlierEstimator {
 public:
  using X_t = double;
  using Y_t = double;
  using M_t = int;

  static const int kMinNumSamples = 5;

  static void Estimate(const std::vector<X_t>& X,
                       const std::vector<Y_t>& Y,
                       std::vector<M_t>* models) {
    CHECK_EQ(X.size(), Y.size());
    CHECK_GE(X.size(), kMinNumSamples);
    models->assign(1, 0);
  }

  static bool Refine(const std::vector<X_t>& X,
                     const std::vector<Y_t>& Y,
                     M_t* model) {
    CHECK_EQ(X.size(), Y.size());
    CHECK_GE(X.size(), kMinNumSamples);
    *model = 1;
    return true;
  }

  static void Residuals(const std::vector<X_t>& X,
                        const std::vector<Y_t>& Y,
                        const M_t model,
                        std::vector<double>* residuals) {
    CHECK_EQ(X.size(), Y.size());
    residuals->assign(X.size(), 0.9);
    if (model == 1) {
      residuals->assign(X.size(), 2.0);
      std::fill_n(residuals->begin(), 4, 0.0);
    }
  }
};

// Local estimator exposing both a value-returning Estimate (the fork adapter
// shape) and Refine. Upstream LO-RANSAC must refine the current best model,
// so the reported model has to be the refined one (2), not the refit (0).
class RefineAndEstimateEstimator {
 public:
  using X_t = double;
  using Y_t = double;
  using M_t = int;

  static const int kMinNumSamples = 1;

  static std::vector<M_t> Estimate(const std::vector<X_t>& X,
                                   const std::vector<Y_t>& Y) {
    CHECK_EQ(X.size(), Y.size());
    return {0};
  }

  static bool Refine(const std::vector<X_t>& X,
                     const std::vector<Y_t>& Y,
                     M_t* model) {
    CHECK_EQ(X.size(), Y.size());
    *model = 2;
    return true;
  }

  static void Residuals(const std::vector<X_t>& X,
                        const std::vector<Y_t>& Y,
                        const M_t model,
                        std::vector<double>* residuals) {
    CHECK_EQ(X.size(), Y.size());
    residuals->assign(X.size(), model == 2 ? 0.0 : 0.5);
  }
};

}  // namespace

TEST(optim_ransac, TestReport) {
  LORANSAC<SimilarityTransformEstimator<3>,
           SimilarityTransformEstimator<3>>::Report report;
  EXPECT_EQ(report.success, false);
  EXPECT_EQ(report.num_trials, 0);
  EXPECT_EQ(report.support.num_inliers, 0);
  EXPECT_EQ(report.support.residual_sum,
                    std::numeric_limits<double>::max());
  EXPECT_EQ(report.inlier_mask.size(), 0);
}

TEST(optim_ransac, TestSimilarityTransform) {
  SetPRNGSeed(0);

  const size_t num_samples = 1000;
  const size_t num_outliers = 400;

  // Create some arbitrary transformation.
  const SimilarityTransform3 orig_tform(2, ComposeIdentityQuaternion(),
                                        Eigen::Vector3d(100, 10, 10));

  // Generate exact data
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;
  for (size_t i = 0; i < num_samples; ++i) {
    src.emplace_back(i, std::sqrt(i) + 2, std::sqrt(2 * i + 2));
    dst.push_back(src.back());
    orig_tform.TransformPoint(&dst.back());
  }

  // Add some faulty data.
  for (size_t i = 0; i < num_outliers; ++i) {
    dst[i] = Eigen::Vector3d(RandomReal(-3000.0, -2000.0),
                             RandomReal(-4000.0, -3000.0),
                             RandomReal(-5000.0, -4000.0));
  }

  // Robustly estimate transformation using RANSAC.
  RANSACOptions options;
  options.max_error = 10;
  LORANSAC<SimilarityTransformEstimator<3>, SimilarityTransformEstimator<3>>
      ransac(options);
  const auto report = ransac.Estimate(src, dst);

  EXPECT_EQ(report.success, true);
  EXPECT_GT(report.num_trials, 0);

  // Make sure outliers were detected correctly.
  EXPECT_EQ(report.support.num_inliers, num_samples - num_outliers);
  for (size_t i = 0; i < num_samples; ++i) {
    if (i < num_outliers) {
      EXPECT_FALSE(report.inlier_mask[i]);
    } else {
      EXPECT_TRUE(report.inlier_mask[i]);
    }
  }

  // Make sure original transformation is estimated correctly.
  const double matrix_diff =
      (orig_tform.Matrix().topLeftCorner<3, 4>() - report.model).norm();
  EXPECT_TRUE(std::abs(matrix_diff) < 1e-6);
}

TEST(LORANSAC, ParallelSimilarityTransform) {
  SetPRNGSeed(0);
  const auto data = GenerateTestData();

  RANSACOptions options;
  options.max_error = 10;
  options.random_seed = kDefaultPRNGSeed;
  options.num_threads = 4;
  LORANSAC<SimilarityTransformEstimator<3>, SimilarityTransformEstimator<3>>
      loransac(options);
  ValidateReport(loransac.Estimate(data.src, data.tgt), data);
}

// LORANSAC overrides RANSAC::Estimate, so it must apply random_seed itself:
// the same seed has to reproduce the identical trial sequence.
TEST(LORANSAC, ReproducibilityWithRandomSeed) {
  SetPRNGSeed(0);
  const auto data = GenerateTestData();

  RANSACOptions options;
  options.max_error = 10;
  options.random_seed = 42;
  using Estimator = LORANSAC<SimilarityTransformEstimator<3>,
                             SimilarityTransformEstimator<3>>;
  const auto report1 = Estimator(options).Estimate(data.src, data.tgt);
  // Perturb the global PRNG state between calls.
  SetPRNGSeed(12345);
  const auto report2 = Estimator(options).Estimate(data.src, data.tgt);
  ValidateReport(report1, data);
  EXPECT_EQ(report1.num_trials, report2.num_trials);
  EXPECT_EQ(report1.inlier_mask, report2.inlier_mask);
  EXPECT_EQ(report1.model, report2.model);
}

TEST(LORANSAC, RecursiveRefinementStopsBelowMinimumSampleCount) {
  RANSACOptions options;
  options.max_error = 1.0;
  options.min_num_trials = 1;
  options.max_num_trials = 1;
  options.random_seed = kDefaultPRNGSeed;

  using Estimator = LORANSAC<ShrinkingInlierEstimator,
                             ShrinkingInlierEstimator,
                             MEstimatorSupportMeasurer>;
  Estimator loransac(options);
  const std::vector<double> samples(6, 0.0);

  Estimator::Report report;
  EXPECT_NO_THROW(report = loransac.Estimate(samples, samples));
  EXPECT_FALSE(report.success);
  EXPECT_EQ(report.support.num_inliers, 4);
}

TEST(LORANSAC, LocalOptimizationPrefersRefineOverEstimate) {
  RANSACOptions options;
  options.max_error = 1.0;
  options.min_num_trials = 1;
  options.max_num_trials = 1;
  options.random_seed = kDefaultPRNGSeed;

  using Estimator =
      LORANSAC<RefineAndEstimateEstimator, RefineAndEstimateEstimator>;
  const std::vector<double> samples(6, 0.0);
  const auto report = Estimator(options).Estimate(samples, samples);
  ASSERT_TRUE(report.success);
  EXPECT_EQ(report.model, 2);
  EXPECT_EQ(report.support.num_inliers, samples.size());
}

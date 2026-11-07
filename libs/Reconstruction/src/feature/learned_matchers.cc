// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "feature/learned_matchers.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "feature/utils.h"
#include "util/logging.h"

namespace colmap {
namespace {

class BruteForceGGMLFeatureMatcher : public FeatureMatcher {
 public:
  BruteForceGGMLFeatureMatcher(
          const FeatureMatchingOptions& options,
          const BruteForceGGMLMatchingOptions& brute_force_options,
          const bool normalize_descriptors)
      : options_(options),
        brute_force_options_(brute_force_options),
        normalize_descriptors_(normalize_descriptors) {
    THROW_CHECK(options_.Check());
    THROW_CHECK(brute_force_options_.Check());
  }

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    matches->clear();
    THROW_CHECK_NOTNULL(image1.descriptors);
    THROW_CHECK_NOTNULL(image2.descriptors);

    const FeatureDescriptorsFloatData d1 = Prepare(*image1.descriptors);
    const FeatureDescriptorsFloatData d2 = Prepare(*image2.descriptors);
    // Upstream's brute-force graph requires at least two descriptors per
    // image for its ratio branch; mirror that floor and return no matches.
    if (d1.rows() < 2 || d2.rows() < 2) {
      return;
    }
    THROW_CHECK_EQ(d1.cols(), d2.cols());

    // O(K1*K2*D) similarity; this dominates and is what the upstream graph
    // evaluates too, so no index structure is used.
    const Eigen::MatrixXf sim = d1 * d2.transpose();
    const Eigen::Index num1 = sim.rows();
    const Eigen::Index num2 = sim.cols();

    std::vector<Eigen::Index> best_2to1;
    if (brute_force_options_.cross_check) {
      best_2to1.resize(static_cast<size_t>(num2));
      for (Eigen::Index j = 0; j < num2; ++j) {
        sim.col(j).maxCoeff(&best_2to1[static_cast<size_t>(j)]);
      }
    }

    const float min_cossim = static_cast<float>(brute_force_options_.min_cossim);
    const float max_ratio = static_cast<float>(brute_force_options_.max_ratio);
    const bool use_ratio_test = max_ratio > 0.0f && max_ratio < 1.0f;
    for (Eigen::Index i = 0; i < num1; ++i) {
      Eigen::Index best_j = 0;
      const float best = sim.row(i).maxCoeff(&best_j);
      if (best < min_cossim) {
        continue;
      }
      if (use_ratio_test) {
        float second = -2.0f;
        for (Eigen::Index j = 0; j < num2; ++j) {
          if (j != best_j) second = std::max(second, sim(i, j));
        }
        // Lowe's ratio test in Euclidean space of unit vectors:
        // d = sqrt(2 - 2 * cossim).
        const float d_best = std::sqrt(std::max(0.0f, 2.0f - 2.0f * best));
        const float d_second = std::sqrt(std::max(0.0f, 2.0f - 2.0f * second));
        if (d_second > 0.0f && d_best > max_ratio * d_second) {
          continue;
        }
      }
      if (brute_force_options_.cross_check &&
          best_2to1[static_cast<size_t>(best_j)] != i) {
        continue;
      }
      matches->emplace_back(static_cast<point2D_t>(i),
                            static_cast<point2D_t>(best_j));
    }

    if (options_.max_num_matches > 0 &&
        matches->size() > static_cast<size_t>(options_.max_num_matches)) {
      matches->resize(options_.max_num_matches);
    }
  }

  void MatchGuided(double,
                   const Image&,
                   const Image&,
                   TwoViewGeometry*) override {
    LOG_FATAL_THROW(std::runtime_error)
        << "Guided matching not supported for brute-force matching.";
  }

 private:
  FeatureDescriptorsFloatData Prepare(
          const FeatureDescriptors& descriptors) const {
    FeatureDescriptorsFloatData float_descriptors =
            descriptors.ToFloat().data;
    if (normalize_descriptors_) {
      L2NormalizeFeatureDescriptors(&float_descriptors);
    }
    return float_descriptors;
  }

  const FeatureMatchingOptions options_;
  const BruteForceGGMLMatchingOptions brute_force_options_;
  const bool normalize_descriptors_;
};

}  // namespace

bool BruteForceGGMLMatchingOptions::Check() const {
  CHECK_OPTION_IN(min_cossim, -1, 1);
  CHECK_OPTION_IN(max_ratio, 0, 1);
  return true;
}

std::unique_ptr<FeatureMatcher> CreateBruteForceGGMLFeatureMatcher(
        const FeatureMatchingOptions& options,
        const BruteForceGGMLMatchingOptions& brute_force_options,
        const bool normalize_descriptors) {
  return std::make_unique<BruteForceGGMLFeatureMatcher>(
          options, brute_force_options, normalize_descriptors);
}

}  // namespace colmap

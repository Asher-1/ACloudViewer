// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <vector>

#include "controllers/feature_matching.h"
#include "estimators/two_view_geometry.h"
#include "feature/matcher.h"
#include "feature/matching.h"
#include "util/threading.h"

namespace colmap {

// Upstream parity (d3ccaf35 controllers/feature_matching_utils.h): the
// multi-threaded matching pipeline on top of the FeatureMatcherCache. The
// fork reuses its internal::FeatureMatcherData payload and the engine-backed
// FeatureMatcher::Create shells (batch A); the fork's FeatureMatcher::Image
// is value-semantic without the upstream pose-prior slot, so the workers
// carry the same data minus that optional field.

using FeatureMatcherData = internal::FeatureMatcherData;

class FeatureMatcherWorker : public Thread {
 public:
  using Input = FeatureMatcherData;
  using Output = FeatureMatcherData;

  FeatureMatcherWorker(const FeatureMatchingOptions& matching_options,
                       const TwoViewGeometryOptions& geometry_options,
                       const std::shared_ptr<FeatureMatcherCache>& cache,
                       JobQueue<Input>* input_queue,
                       JobQueue<Output>* output_queue);

 private:
  void Run() override;

  FeatureMatchingOptions matching_options_;
  TwoViewGeometryOptions geometry_options_;
  std::shared_ptr<FeatureMatcherCache> cache_;
  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;
};

class FeatureMatcherController {
 public:
  FeatureMatcherController(const FeatureMatchingOptions& matching_options,
                           const TwoViewGeometryOptions& geometry_options,
                           std::shared_ptr<FeatureMatcherCache> cache);

  ~FeatureMatcherController();

  bool Setup();

  void Match(const std::vector<std::pair<image_t, image_t>>& image_pairs);

 private:
  FeatureMatchingOptions matching_options_;
  TwoViewGeometryOptions geometry_options_;
  std::shared_ptr<FeatureMatcherCache> cache_;
  bool skip_geometric_verification_;
  bool is_setup_;
  std::vector<std::unique_ptr<FeatureMatcherWorker>> matchers_;
  std::vector<std::unique_ptr<FeatureMatcherWorker>> guided_matchers_;
  std::vector<std::unique_ptr<Thread>> verifiers_;
  JobQueue<FeatureMatcherData> matcher_queue_;
  JobQueue<FeatureMatcherData> verifier_queue_;
  JobQueue<FeatureMatcherData> guided_matcher_queue_;
  JobQueue<FeatureMatcherData> output_queue_;
};

class GeometricVerifierController {
 public:
  GeometricVerifierController(const GeometricVerifierOptions& verifier_options,
                              const TwoViewGeometryOptions& geometry_options,
                              std::shared_ptr<FeatureMatcherCache> cache);

  const GeometricVerifierOptions& Options() const;
  GeometricVerifierOptions& Options();

  ~GeometricVerifierController();

  bool Setup();

  void Verify(const std::vector<std::pair<image_t, image_t>>& image_pairs);

 private:
  TwoViewGeometryOptions geometry_options_;
  std::shared_ptr<FeatureMatcherCache> cache_;
  GeometricVerifierOptions options_;
  bool is_setup_;
  std::vector<std::unique_ptr<Thread>> verifiers_;
  JobQueue<FeatureMatcherData> verifier_queue_;
  JobQueue<FeatureMatcherData> output_queue_;
};

}  // namespace colmap

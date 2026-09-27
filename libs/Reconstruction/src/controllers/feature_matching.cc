// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "controllers/feature_matching.h"

#include "feature/matching.h"
#include "feature/utils.h"
#include "scene/database.h"
#include "util/misc.h"
#include "util/timer.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace colmap {
namespace {

// Mirrors the verification-relevant TwoViewGeometryOptions into the fork's
// flat SiftMatchingOptions, which is what the SiftFeatureMatcher verifier
// derives its TwoViewGeometryOptions from. Upstream hands the geometry
// options to the verifier directly; the fork keeps a single match-option
// struct, so the explicit mirror is the equivalent injection point. Users
// that set the same knob in both places get the verification_options value,
// matching upstream where geometry_options drives the estimator.
SiftMatchingOptions MergeGeometryOptions(
        const SiftMatchingOptions& match_options,
        const TwoViewGeometryOptions& geometry_options) {
  SiftMatchingOptions options = match_options;
  options.min_num_inliers = geometry_options.min_num_inliers;
  options.max_error = geometry_options.ransac_options.max_error;
  options.confidence = geometry_options.ransac_options.confidence;
  options.min_num_trials =
      static_cast<int>(geometry_options.ransac_options.min_num_trials);
  options.max_num_trials =
      static_cast<int>(geometry_options.ransac_options.max_num_trials);
  options.min_inlier_ratio = geometry_options.ransac_options.min_inlier_ratio;
  return options;
}

// Upstream parity (d3ccaf35 controllers/feature_matching.cc): pairing-driven
// matcher thread. Consumes batches from a PairGenerator and runs them through
// the fork's SiftFeatureMatcher (matching + two-view verification + database
// writes) inside per-batch database transactions.
class PairingFeatureMatcher : public Thread {
 public:
  PairingFeatureMatcher(std::unique_ptr<PairGenerator> pair_generator,
                        const SiftMatchingOptions& match_options,
                        const TwoViewGeometryOptions& geometry_options,
                        std::shared_ptr<Database> database,
                        std::shared_ptr<FeatureMatcherCache> cache)
      : options_(MergeGeometryOptions(match_options, geometry_options)),
        database_(THROW_CHECK_NOTNULL(database)),
        pair_generator_(std::move(THROW_CHECK_NOTNULL(pair_generator))),
        cache_(THROW_CHECK_NOTNULL(cache)) {}

 private:
  void Run() override {
    PrintHeading1("Feature matching");

    cache_->Setup();

    SiftFeatureMatcher matcher(options_, database_.get(), cache_.get());
    if (!matcher.Setup()) {
      return;
    }

    Timer timer;
    timer.Start();

    while (!pair_generator_->HasFinished()) {
      if (IsStopped()) {
        timer.PrintMinutes();
        return;
      }

      std::vector<std::pair<image_t, image_t>> image_pairs =
          pair_generator_->Next();
      if (image_pairs.empty()) {
        continue;
      }

      DatabaseTransaction database_transaction(database_.get());
      matcher.Match(image_pairs);
    }

    timer.PrintMinutes();
  }

  SiftMatchingOptions options_;
  std::shared_ptr<Database> database_;
  std::unique_ptr<PairGenerator> pair_generator_;
  std::shared_ptr<FeatureMatcherCache> cache_;
};

}  // namespace

std::unique_ptr<Thread> CreateExhaustiveFeatureMatcher(
        const ExhaustivePairingOptions& pairing_options,
        const SiftMatchingOptions& match_options,
        const TwoViewGeometryOptions& geometry_options,
        const std::filesystem::path& database_path) {
  THROW_CHECK(pairing_options.Check());
  auto database = Database::Open(database_path);
  auto cache = std::make_shared<FeatureMatcherCache>(
          pairing_options.CacheSize(), database.get());
  auto pair_generator = std::make_unique<ExhaustivePairGenerator>(
          pairing_options, cache);
  return std::make_unique<PairingFeatureMatcher>(
          std::move(pair_generator), match_options, geometry_options,
          std::move(database), std::move(cache));
}

std::unique_ptr<Thread> CreateVocabTreeFeatureMatcher(
        const VocabTreePairingOptions& pairing_options,
        const SiftMatchingOptions& match_options,
        const TwoViewGeometryOptions& geometry_options,
        const std::filesystem::path& database_path) {
  THROW_CHECK(pairing_options.Check());
  auto database = Database::Open(database_path);
  auto cache = std::make_shared<FeatureMatcherCache>(
          pairing_options.CacheSize(), database.get());
  auto pair_generator = std::make_unique<VocabTreePairGenerator>(
          pairing_options, cache);
  return std::make_unique<PairingFeatureMatcher>(
          std::move(pair_generator), match_options, geometry_options,
          std::move(database), std::move(cache));
}

std::unique_ptr<Thread> CreateSequentialFeatureMatcher(
        const SequentialPairingOptions& pairing_options,
        const SiftMatchingOptions& match_options,
        const TwoViewGeometryOptions& geometry_options,
        const std::filesystem::path& database_path) {
  THROW_CHECK(pairing_options.Check());
  auto database = Database::Open(database_path);
  auto cache = std::make_shared<FeatureMatcherCache>(
          pairing_options.CacheSize(), database.get());
  auto pair_generator = std::make_unique<SequentialPairGenerator>(
          pairing_options, cache);
  return std::make_unique<PairingFeatureMatcher>(
          std::move(pair_generator), match_options, geometry_options,
          std::move(database), std::move(cache));
}

std::unique_ptr<Thread> CreateSpatialFeatureMatcher(
        const SpatialPairingOptions& pairing_options,
        const SiftMatchingOptions& match_options,
        const TwoViewGeometryOptions& geometry_options,
        const std::filesystem::path& database_path) {
  THROW_CHECK(pairing_options.Check());
  auto database = Database::Open(database_path);
  auto cache = std::make_shared<FeatureMatcherCache>(
          pairing_options.CacheSize(), database.get());
  auto pair_generator = std::make_unique<SpatialPairGenerator>(pairing_options,
                                                               cache);
  return std::make_unique<PairingFeatureMatcher>(
          std::move(pair_generator), match_options, geometry_options,
          std::move(database), std::move(cache));
}

std::unique_ptr<Thread> CreateTransitiveFeatureMatcher(
        const TransitivePairingOptions& pairing_options,
        const SiftMatchingOptions& match_options,
        const TwoViewGeometryOptions& geometry_options,
        const std::filesystem::path& database_path) {
  THROW_CHECK(pairing_options.Check());
  auto database = Database::Open(database_path);
  auto cache = std::make_shared<FeatureMatcherCache>(
          pairing_options.CacheSize(), database.get());
  auto pair_generator = std::make_unique<TransitivePairGenerator>(
          pairing_options, cache);
  return std::make_unique<PairingFeatureMatcher>(
          std::move(pair_generator), match_options, geometry_options,
          std::move(database), std::move(cache));
}

std::unique_ptr<Thread> CreateImagePairsFeatureMatcher(
        const ImportedPairingOptions& pairing_options,
        const SiftMatchingOptions& match_options,
        const TwoViewGeometryOptions& geometry_options,
        const std::filesystem::path& database_path) {
  THROW_CHECK(pairing_options.Check());
  auto database = Database::Open(database_path);
  auto cache = std::make_shared<FeatureMatcherCache>(
          pairing_options.CacheSize(), database.get());
  auto pair_generator = std::make_unique<ImportedPairGenerator>(pairing_options,
                                                                cache);
  return std::make_unique<PairingFeatureMatcher>(
          std::move(pair_generator), match_options, geometry_options,
          std::move(database), std::move(cache));
}

std::unique_ptr<Thread> CreateFeaturePairsFeatureMatcher(
        const FeaturePairsMatchingOptions& pairing_options,
        const SiftMatchingOptions& match_options,
        const TwoViewGeometryOptions& geometry_options,
        const std::filesystem::path& database_path) {
  THROW_CHECK(pairing_options.Check());
  // The legacy matcher owns the feature-pairs file format end-to-end
  // (import + optional verification); the geometry options are mirrored the
  // same way as in the pairing-driven factories.
  return std::make_unique<FeaturePairsFeatureMatcher>(
          pairing_options, MergeGeometryOptions(match_options, geometry_options),
          database_path);
}

namespace {

// Upstream parity (d3ccaf35 controllers/feature_matching.cc GeometricVerifier):
// re-runs two-view geometric verification on all already-matched image pairs
// with a thread pool, writing the resulting TwoViewGeometry back. The fork
// verifies plain per-pair geometry; upstream rig verification and
// use-existing-relative-pose modes are not supported.
class GeometricVerifier : public Thread {
 public:
  GeometricVerifier(const int num_threads,
                    const ExistingMatchedPairingOptions& pairing_options,
                    const TwoViewGeometryOptions& geometry_options,
                    std::shared_ptr<Database> database,
                    std::shared_ptr<FeatureMatcherCache> cache)
      : num_threads_(num_threads),
        pairing_options_(pairing_options),
        geometry_options_(geometry_options),
        database_(THROW_CHECK_NOTNULL(database)),
        cache_(THROW_CHECK_NOTNULL(cache)) {}

 private:
  void Run() override {
    PrintHeading1("Geometric verification");

    cache_->Setup();

    ExistingMatchedPairGenerator generator(pairing_options_, cache_);

    const int num_threads =
            num_threads_ > 0
                    ? num_threads_
                    : static_cast<int>(std::thread::hardware_concurrency());
    ThreadPool thread_pool(num_threads);

    while (!generator.HasFinished()) {
      if (IsStopped()) {
        break;
      }

      std::vector<std::pair<image_t, image_t>> image_pairs =
              generator.Next();
      if (image_pairs.empty()) {
        continue;
      }

      // Writes go through the cache's database mutex (same serialization
      // as the legacy SiftFeatureMatcher verifier pool), so no explicit
      // database transaction is held across asynchronous tasks.
      for (const auto& [image_id1, image_id2] : image_pairs) {
        thread_pool.AddTask([this, image_id1, image_id2]() {
          VerifyPair(image_id1, image_id2);
        });
      }
    }

    thread_pool.Wait();
    GetTimer().PrintMinutes();
  }

  void VerifyPair(const image_t image_id1, const image_t image_id2) {
    if (IsStopped()) {
      return;
    }

    const FeatureMatches matches =
            cache_->GetMatches(image_id1, image_id2);
    if (matches.size() <
        static_cast<size_t>(geometry_options_.min_num_inliers)) {
      return;
    }

    try {
      const Image& image1 = cache_->GetImage(image_id1);
      const Image& image2 = cache_->GetImage(image_id2);
      const Camera& camera1 = cache_->GetCamera(image1.CameraId());
      const Camera& camera2 = cache_->GetCamera(image2.CameraId());
      const auto keypoints1 = cache_->GetKeypoints(image_id1);
      const auto keypoints2 = cache_->GetKeypoints(image_id2);
      const auto points1 = FeatureKeypointsToPointsVector(*keypoints1);
      const auto points2 = FeatureKeypointsToPointsVector(*keypoints2);

      const TwoViewGeometry two_view_geometry = EstimateTwoViewGeometry(
              camera1, points1, camera2, points2, matches,
              geometry_options_);
      // Upstream parity (feature_matching_utils.cc
      // WriteFilteredTwoViewGeometry): skip UNDEFINED results, normalize
      // degenerate/weak results to an empty DEGENERATE geometry, and delete
      // any existing row first - writing over an existing row violates the
      // database primary-key constraint.
      if (two_view_geometry.config ==
          TwoViewGeometry::ConfigurationType::UNDEFINED) {
        return;
      }
      TwoViewGeometry filtered = two_view_geometry;
      if (filtered.config ==
                  TwoViewGeometry::ConfigurationType::DEGENERATE ||
              filtered.inlier_matches.size() <
                  static_cast<size_t>(geometry_options_.min_num_inliers)) {
        filtered = TwoViewGeometry();
        filtered.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
      }
      cache_->DeleteTwoViewGeometry(image_id1, image_id2);
      cache_->WriteTwoViewGeometry(image_id1, image_id2, filtered);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Geometric verification failed for image pair ("
                 << image_id1 << ", " << image_id2 << "): " << e.what();
    }
  }

  int num_threads_;
  ExistingMatchedPairingOptions pairing_options_;
  TwoViewGeometryOptions geometry_options_;
  std::shared_ptr<Database> database_;
  std::shared_ptr<FeatureMatcherCache> cache_;
};

}  // namespace

std::unique_ptr<Thread> CreateGeometricVerifier(
        const GeometricVerifierOptions& verifier_options,
        const ExistingMatchedPairingOptions& pairing_options,
        const TwoViewGeometryOptions& geometry_options,
        const std::filesystem::path& database_path) {
  THROW_CHECK(pairing_options.Check());
  auto database = Database::Open(database_path);
  auto cache = std::make_shared<FeatureMatcherCache>(
          pairing_options.CacheSize(), database.get());
  return std::make_unique<GeometricVerifier>(
          verifier_options.num_threads, pairing_options, geometry_options,
          std::move(database), std::move(cache));
}

}  // namespace colmap

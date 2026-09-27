// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Upstream parity (d3ccaf35 colmap/controllers/feature_matching.h): factory
// entry points that combine a PairGenerator (controllers/pairing.h) with the
// fork's SIFT/LoMa matching pipeline. The upstream FeatureMatchingOptions
// parameter is represented by SiftMatchingOptions, the fork's single
// matcher-family option set (use_gpu / gpu_index / LoMa dispatch / guided
// matching / flat two-view-verification fields); matcher-type selection, rig
// verification, and frame-pair skipping have no fork counterpart. The
// verification-relevant TwoViewGeometryOptions fields are mirrored into the
// match options by the factories. Differences are adjudicated in
// colmap_alignment_manifest.json.

#include "controllers/pairing.h"
#include "estimators/two_view_geometry.h"
#include "feature/sift.h"
#include "util/threading.h"

#include <filesystem>
#include <memory>

namespace colmap {

// Upstream parity (d3ccaf35 controllers/feature_matching.h): the fork
// verifies plain per-pair two-view geometry, so the upstream rig-related
// flags (rig_verification, use_existing_relative_pose) have no fork
// counterpart and only the thread count remains.
struct GeometricVerifierOptions {
  // Number of threads for geometric verification.
  int num_threads = -1;
};

// Exhaustively match images using an ExhaustivePairGenerator.
std::unique_ptr<Thread> CreateExhaustiveFeatureMatcher(
        const ExhaustivePairingOptions& pairing_options,
        const SiftMatchingOptions& match_options,
        const TwoViewGeometryOptions& geometry_options,
        const std::filesystem::path& database_path);

// Match each image against its nearest neighbors using a vocabulary tree.
std::unique_ptr<Thread> CreateVocabTreeFeatureMatcher(
        const VocabTreePairingOptions& pairing_options,
        const SiftMatchingOptions& match_options,
        const TwoViewGeometryOptions& geometry_options,
        const std::filesystem::path& database_path);

// Sequentially match images within their sequential neighborhood, with
// optional vocabulary-tree based loop detection.
std::unique_ptr<Thread> CreateSequentialFeatureMatcher(
        const SequentialPairingOptions& pairing_options,
        const SiftMatchingOptions& match_options,
        const TwoViewGeometryOptions& geometry_options,
        const std::filesystem::path& database_path);

// Match images against spatial nearest neighbors using prior location
// information, e.g. provided manually or extracted from EXIF.
std::unique_ptr<Thread> CreateSpatialFeatureMatcher(
        const SpatialPairingOptions& pairing_options,
        const SiftMatchingOptions& match_options,
        const TwoViewGeometryOptions& geometry_options,
        const std::filesystem::path& database_path);

// Match transitive image pairs in a database with existing feature matches.
std::unique_ptr<Thread> CreateTransitiveFeatureMatcher(
        const TransitivePairingOptions& pairing_options,
        const SiftMatchingOptions& match_options,
        const TwoViewGeometryOptions& geometry_options,
        const std::filesystem::path& database_path);

// Match images manually specified in a list of image pairs.
std::unique_ptr<Thread> CreateImagePairsFeatureMatcher(
        const ImportedPairingOptions& pairing_options,
        const SiftMatchingOptions& match_options,
        const TwoViewGeometryOptions& geometry_options,
        const std::filesystem::path& database_path);

// Import feature matches from a text file, optionally verifying them
// geometrically. Reuses the legacy FeaturePairsFeatureMatcher, which owns
// this file format end-to-end.
std::unique_ptr<Thread> CreateFeaturePairsFeatureMatcher(
        const FeaturePairsMatchingOptions& pairing_options,
        const SiftMatchingOptions& match_options,
        const TwoViewGeometryOptions& geometry_options,
        const std::filesystem::path& database_path);

// Re-run plain two-view geometric verification on already-matched image
// pairs in the database. Upstream CreateGeometricVerifier additionally
// supports rig-level verification and reusing existing relative poses;
// those modes have no fork counterpart and the standalone
// GeometricVerifierOptions therefore only carries num_threads.
std::unique_ptr<Thread> CreateGeometricVerifier(
        const GeometricVerifierOptions& verifier_options,
        const ExistingMatchedPairingOptions& pairing_options,
        const TwoViewGeometryOptions& geometry_options,
        const std::filesystem::path& database_path);

}  // namespace colmap

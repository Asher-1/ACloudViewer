// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "controllers/image_reader.h"
#include "controllers/pairing.h"
#include "estimators/two_view_geometry.h"
#include "feature/extractor.h"
#include "feature/matcher.h"

namespace cloudViewer {

// camera_mode: -1 keeps the ImageReaderOptions as given; otherwise
// { AUTO = 0, SINGLE = 1, PER_FOLDER = 2, PER_IMAGE = 3 } overrides the
// single_camera* flags (colmap::CameraMode).
int ExtractFeature(const std::string& database_path,
                   const std::string& image_path,
                   const std::string& image_list_path = "",
                   int camera_mode = -1,
                   const colmap::ImageReaderOptions& image_reader_options =
                           colmap::ImageReaderOptions(),
                   const colmap::FeatureExtractionOptions& extraction_options =
                           colmap::FeatureExtractionOptions());

int ImportFeature(const std::string& database_path,
                  const std::string& image_path,
                  const std::string& import_path,
                  const std::string& image_list_path = "",
                  int camera_mode = -1,
                  const colmap::ImageReaderOptions& image_reader_options =
                          colmap::ImageReaderOptions(),
                  const colmap::FeatureExtractionOptions& extraction_options =
                          colmap::FeatureExtractionOptions());

// match_type: {'pairs', 'raw', 'inliers'}.
int ImportMatches(const std::string& database_path,
                  const std::string& match_list_path,
                  const std::string& match_type = "pairs",
                  const colmap::FeatureMatchingOptions& matching_options =
                          colmap::FeatureMatchingOptions(),
                  const colmap::TwoViewGeometryOptions& geometry_options =
                          colmap::TwoViewGeometryOptions());

int ExhaustiveMatch(const std::string& database_path,
                    const colmap::FeatureMatchingOptions& matching_options =
                            colmap::FeatureMatchingOptions(),
                    const colmap::TwoViewGeometryOptions& geometry_options =
                            colmap::TwoViewGeometryOptions(),
                    const colmap::ExhaustivePairingOptions& pairing_options =
                            colmap::ExhaustivePairingOptions());

int SequentialMatch(const std::string& database_path,
                    const colmap::FeatureMatchingOptions& matching_options =
                            colmap::FeatureMatchingOptions(),
                    const colmap::TwoViewGeometryOptions& geometry_options =
                            colmap::TwoViewGeometryOptions(),
                    const colmap::SequentialPairingOptions& pairing_options =
                            colmap::SequentialPairingOptions());

int SpatialMatch(const std::string& database_path,
                 const colmap::FeatureMatchingOptions& matching_options =
                         colmap::FeatureMatchingOptions(),
                 const colmap::TwoViewGeometryOptions& geometry_options =
                         colmap::TwoViewGeometryOptions(),
                 const colmap::SpatialPairingOptions& pairing_options =
                         colmap::SpatialPairingOptions());

int TransitiveMatch(const std::string& database_path,
                    const colmap::FeatureMatchingOptions& matching_options =
                            colmap::FeatureMatchingOptions(),
                    const colmap::TwoViewGeometryOptions& geometry_options =
                            colmap::TwoViewGeometryOptions(),
                    const colmap::TransitivePairingOptions& pairing_options =
                            colmap::TransitivePairingOptions());

int VocabTreeMatch(const std::string& database_path,
                   const colmap::FeatureMatchingOptions& matching_options =
                           colmap::FeatureMatchingOptions(),
                   const colmap::TwoViewGeometryOptions& geometry_options =
                           colmap::TwoViewGeometryOptions(),
                   const colmap::VocabTreePairingOptions& pairing_options =
                           colmap::VocabTreePairingOptions());

// Re-runs geometric verification on the matched pairs already stored in the
// database (colmap geometric_verifier).
int VerifyGeometry(const std::string& database_path,
                   const colmap::TwoViewGeometryOptions& geometry_options =
                           colmap::TwoViewGeometryOptions(),
                   int batch_size = 1000,
                   int num_threads = -1,
                   bool rig_verification = false);

// Guided geometric verification against the sparse model in `input_path`
// (colmap guided_geometric_verifier).
int VerifyGuidedGeometry(
        const std::string& database_path,
        const std::string& input_path,
        const colmap::TwoViewGeometryOptions& geometry_options =
                colmap::TwoViewGeometryOptions(),
        int batch_size = 1000,
        int num_threads = -1);

}  // namespace cloudViewer

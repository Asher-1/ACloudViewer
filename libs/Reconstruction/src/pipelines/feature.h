// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "base/image_reader.h"
#include "feature/extraction.h"
#include "feature/matching.h"
#include "feature/sift.h"

namespace cloudViewer {
int ExtractFeature(
        const std::string& database_path,
        const std::string& image_path,
        const std::string& image_list_path = "",
        int camera_mode = 0,
        const colmap::ImageReaderOptions& image_reader_options =
                colmap::ImageReaderOptions(),
        const colmap::SiftExtractionOptions& sift_extraction_options =
                colmap::SiftExtractionOptions());

int ImportFeature(const std::string& database_path,
                  const std::string& image_path,
                  const std::string& import_path,
                  const std::string& image_list_path = "",
                  int camera_mode = 0,
                  const colmap::ImageReaderOptions& image_reader_options =
                          colmap::ImageReaderOptions(),
                  const colmap::SiftExtractionOptions& sift_extraction_options =
                          colmap::SiftExtractionOptions());

int ImportMatches(const std::string& database_path,
                  const std::string& match_list_path,
                  const std::string& match_type = "pairs",
                  const colmap::SiftMatchingOptions& sift_matching_options =
                          colmap::SiftMatchingOptions());

int ExhaustiveMatch(
        const std::string& database_path,
        const colmap::SiftMatchingOptions& sift_matching_options =
                colmap::SiftMatchingOptions(),
        const colmap::ExhaustiveMatchingOptions& exhaustive_matching_options =
                colmap::ExhaustiveMatchingOptions());

int SequentialMatch(
        const std::string& database_path,
        const colmap::SiftMatchingOptions& sift_matching_options =
                colmap::SiftMatchingOptions(),
        const colmap::SequentialMatchingOptions& sequential_matching_options =
                colmap::SequentialMatchingOptions());

int SpatialMatch(
        const std::string& database_path,
        const colmap::SiftMatchingOptions& sift_matching_options =
                colmap::SiftMatchingOptions(),
        const colmap::SpatialMatchingOptions& spatial_matching_options =
                colmap::SpatialMatchingOptions());

int TransitiveMatch(
        const std::string& database_path,
        const colmap::SiftMatchingOptions& sift_matching_options =
                colmap::SiftMatchingOptions(),
        const colmap::TransitiveMatchingOptions& transitive_matching_options =
                colmap::TransitiveMatchingOptions());

int VocabTreeMatch(
        const std::string& database_path,
        const colmap::SiftMatchingOptions& sift_matching_options =
                colmap::SiftMatchingOptions(),
        const colmap::VocabTreeMatchingOptions& vocab_tree_matching_options =
                colmap::VocabTreeMatchingOptions());

}  // namespace cloudViewer

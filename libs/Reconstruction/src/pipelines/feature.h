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
// Author: Asher (Dahai Lu)

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

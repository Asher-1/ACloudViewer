// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "exe/feature.h"

#include "pipelines/feature.h"
#include "pipelines/option_utils.h"

namespace cloudViewer {

int ExtractFeature(
        const std::string& database_path,
        const std::string& image_path,
        const std::string& image_list_path,
        int camera_mode,
        const colmap::ImageReaderOptions& image_reader_options,
        const colmap::SiftExtractionOptions& sift_extraction_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("image_path", &image_path);
    parser.registerOption("image_list_path", &image_list_path);
    // supported camera model { AUTO=0, SINGLE=1, PER_FOLDER=2, PER_IMAGE=3 }
    parser.registerOption("camera_mode", &camera_mode);
    parser.addExtractionOptions(image_reader_options, sift_extraction_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunFeatureExtractor(parser.getArgc(), parser.getArgv());
}

int ImportFeature(
        const std::string& database_path,
        const std::string& image_path,
        const std::string& import_path,
        const std::string& image_list_path,
        int camera_mode,
        const colmap::ImageReaderOptions& image_reader_options,
        const colmap::SiftExtractionOptions& sift_extraction_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("image_path", &image_path);
    parser.registerOption("import_path", &import_path);
    parser.registerOption("image_list_path", &image_list_path);
    parser.registerOption("camera_mode", &camera_mode);
    parser.addExtractionOptions(image_reader_options, sift_extraction_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunFeatureImporter(parser.getArgc(), parser.getArgv());
}

int ImportMatches(const std::string& database_path,
                  const std::string& match_list_path,
                  const std::string& match_type,
                  const colmap::SiftMatchingOptions& sift_matching_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("match_list_path", &match_list_path);
    // supported match_type {'pairs', 'raw', 'inliers'}
    parser.registerOption("match_type", &match_type);
    parser.addMatchingOptions(sift_matching_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunMatchesImporter(parser.getArgc(), parser.getArgv());
}

int ExhaustiveMatch(
        const std::string& database_path,
        const colmap::SiftMatchingOptions& sift_matching_options,
        const colmap::ExhaustiveMatchingOptions& exhaustive_matching_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.addExhaustiveMatchingOptions(sift_matching_options,
                                        exhaustive_matching_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunExhaustiveMatcher(parser.getArgc(), parser.getArgv());
}

int SequentialMatch(
        const std::string& database_path,
        const colmap::SiftMatchingOptions& sift_matching_options,
        const colmap::SequentialMatchingOptions& sequential_matching_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.addSequentialMatchingOptions(sift_matching_options,
                                        sequential_matching_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunSequentialMatcher(parser.getArgc(), parser.getArgv());
}

int SpatialMatch(
        const std::string& database_path,
        const colmap::SiftMatchingOptions& sift_matching_options,
        const colmap::SpatialMatchingOptions& spatial_matching_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.addSpatialMatchingOptions(sift_matching_options,
                                     spatial_matching_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunSpatialMatcher(parser.getArgc(), parser.getArgv());
}

int TransitiveMatch(
        const std::string& database_path,
        const colmap::SiftMatchingOptions& sift_matching_options,
        const colmap::TransitiveMatchingOptions& transitive_matching_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.addTransitiveMatchingOptions(sift_matching_options,
                                        transitive_matching_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunTransitiveMatcher(parser.getArgc(), parser.getArgv());
}

int VocabTreeMatch(
        const std::string& database_path,
        const colmap::SiftMatchingOptions& sift_matching_options,
        const colmap::VocabTreeMatchingOptions& vocab_tree_matching_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.addVocabTreeMatchingOptions(sift_matching_options,
                                       vocab_tree_matching_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunVocabTreeMatcher(parser.getArgc(), parser.getArgv());
}

}  // namespace cloudViewer

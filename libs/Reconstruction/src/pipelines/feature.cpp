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

int ExtractFeature(const std::string& database_path,
                   const std::string& image_path,
                   const std::string& image_list_path,
                   int camera_mode,
                   const colmap::ImageReaderOptions& image_reader_options,
                   const colmap::FeatureExtractionOptions& extraction_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("image_path", &image_path);
    parser.registerOption("image_list_path", &image_list_path);
    parser.registerOption("camera_mode", &camera_mode);
    parser.addFeatureExtractionOptions(image_reader_options,
                                       extraction_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunFeatureExtractor(parser.getArgc(), parser.getArgv());
}

int ImportFeature(const std::string& database_path,
                  const std::string& image_path,
                  const std::string& import_path,
                  const std::string& image_list_path,
                  int camera_mode,
                  const colmap::ImageReaderOptions& image_reader_options,
                  const colmap::FeatureExtractionOptions& extraction_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("image_path", &image_path);
    parser.registerOption("import_path", &import_path);
    parser.registerOption("image_list_path", &image_list_path);
    parser.registerOption("camera_mode", &camera_mode);
    parser.addFeatureExtractionOptions(image_reader_options,
                                       extraction_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunFeatureImporter(parser.getArgc(), parser.getArgv());
}

int ImportMatches(const std::string& database_path,
                  const std::string& match_list_path,
                  const std::string& match_type,
                  const colmap::FeatureMatchingOptions& matching_options,
                  const colmap::TwoViewGeometryOptions& geometry_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("match_list_path", &match_list_path);
    parser.registerOption("match_type", &match_type);
    parser.addFeatureMatchingOptions(matching_options);
    parser.addTwoViewGeometryOptions(geometry_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunMatchesImporter(parser.getArgc(), parser.getArgv());
}

int ExhaustiveMatch(const std::string& database_path,
                    const colmap::FeatureMatchingOptions& matching_options,
                    const colmap::TwoViewGeometryOptions& geometry_options,
                    const colmap::ExhaustivePairingOptions& pairing_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.addExhaustivePairingOptions(matching_options, geometry_options,
                                       pairing_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunExhaustiveMatcher(parser.getArgc(), parser.getArgv());
}

int SequentialMatch(const std::string& database_path,
                    const colmap::FeatureMatchingOptions& matching_options,
                    const colmap::TwoViewGeometryOptions& geometry_options,
                    const colmap::SequentialPairingOptions& pairing_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.addSequentialPairingOptions(matching_options, geometry_options,
                                       pairing_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunSequentialMatcher(parser.getArgc(), parser.getArgv());
}

int SpatialMatch(const std::string& database_path,
                 const colmap::FeatureMatchingOptions& matching_options,
                 const colmap::TwoViewGeometryOptions& geometry_options,
                 const colmap::SpatialPairingOptions& pairing_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.addSpatialPairingOptions(matching_options, geometry_options,
                                    pairing_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunSpatialMatcher(parser.getArgc(), parser.getArgv());
}

int TransitiveMatch(const std::string& database_path,
                    const colmap::FeatureMatchingOptions& matching_options,
                    const colmap::TwoViewGeometryOptions& geometry_options,
                    const colmap::TransitivePairingOptions& pairing_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.addTransitivePairingOptions(matching_options, geometry_options,
                                       pairing_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunTransitiveMatcher(parser.getArgc(), parser.getArgv());
}

int VocabTreeMatch(const std::string& database_path,
                   const colmap::FeatureMatchingOptions& matching_options,
                   const colmap::TwoViewGeometryOptions& geometry_options,
                   const colmap::VocabTreePairingOptions& pairing_options) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.addVocabTreePairingOptions(matching_options, geometry_options,
                                      pairing_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunVocabTreeMatcher(parser.getArgc(), parser.getArgv());
}

int VerifyGeometry(const std::string& database_path,
                   const colmap::TwoViewGeometryOptions& geometry_options,
                   int batch_size,
                   int num_threads,
                   bool rig_verification) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("batch_size", &batch_size);
    parser.registerOption("num_threads", &num_threads);
    parser.registerOption("rig_verification", &rig_verification);
    parser.addTwoViewGeometryOptions(geometry_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunGeometricVerifier(parser.getArgc(), parser.getArgv());
}

int VerifyGuidedGeometry(const std::string& database_path,
                         const std::string& input_path,
                         const colmap::TwoViewGeometryOptions& geometry_options,
                         int batch_size,
                         int num_threads) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("input_path", &input_path);
    parser.registerOption("batch_size", &batch_size);
    parser.registerOption("num_threads", &num_threads);
    parser.addTwoViewGeometryOptions(geometry_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunGuidedGeometricVerifier(parser.getArgc(),
                                              parser.getArgv());
}

}  // namespace cloudViewer

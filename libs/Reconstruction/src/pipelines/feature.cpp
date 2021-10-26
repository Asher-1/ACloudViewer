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

#include "pipelines/feature.h"
#include "pipelines/option_utils.hpp"

#include "exe/feature.h"

namespace cloudViewer {

int ExtractFeature(const std::string& database_path,
                        const std::string& image_path,
                        const std::string& image_list_path,
                        int camera_mode) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("image_path", &image_path);
    parser.registerOption("image_list_path", &image_list_path);
    // supported camera model { AUTO = 0, SINGLE = 1, PER_FOLDER = 2, PER_IMAGE
    // = 3 }
    parser.registerOption("camera_mode", &camera_mode);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunFeatureExtractor(parser.getArgc(), parser.getArgv());
}

int ImportFeature(const std::string& database_path,
                       const std::string& image_path,
                       const std::string& import_path,
                       const std::string& image_list_path,
                       int camera_mode) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("image_path", &image_path);
    parser.registerOption("import_path", &import_path);
    parser.registerOption("image_list_path", &image_list_path);
    parser.registerOption("camera_mode", &camera_mode);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunFeatureImporter(parser.getArgc(), parser.getArgv());
}

int ImportMatches(const std::string& database_path,
                       const std::string& match_list_path,
                       const std::string& match_type) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("match_list_path", &match_list_path);
    // supported match_type {'pairs', 'raw', 'inliers'}
    parser.registerOption("match_type", &match_type);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunMatchesImporter(parser.getArgc(), parser.getArgv());
}

int ExhaustiveMatch(const std::string& database_path) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunExhaustiveMatcher(parser.getArgc(), parser.getArgv());
}

int SequentialMatch(const std::string& database_path) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunSequentialMatcher(parser.getArgc(), parser.getArgv());
}

int SpatialMatch(const std::string& database_path) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunSpatialMatcher(parser.getArgc(), parser.getArgv());
}

int TransitiveMatch(const std::string& database_path) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunTransitiveMatcher(parser.getArgc(), parser.getArgv());
}

int VocabTreeMatch(const std::string& database_path) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunVocabTreeMatcher(parser.getArgc(), parser.getArgv());
}

}  // namespace cloudViewer

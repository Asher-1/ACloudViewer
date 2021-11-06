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
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "exe/vocab_tree.h"

#include "pipelines/option_utils.h"
#include "pipelines/vocab_tree.h"

namespace cloudViewer {

int BuildVocabTree(const std::string& database_path,
                   const std::string& vocab_tree_path,
                   int num_visual_words /*= 256 * 256*/,
                   int num_checks /*= 256*/,
                   int branching /*= 256*/,
                   int num_iterations /*= 11*/,
                   int max_num_images /*= -1*/) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("vocab_tree_path", &vocab_tree_path);
    parser.registerOption("num_visual_words", &num_visual_words);
    parser.registerOption("num_checks", &num_checks);
    parser.registerOption("branching", &branching);
    parser.registerOption("num_iterations", &num_iterations);
    parser.registerOption("max_num_images", &max_num_images);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunVocabTreeBuilder(parser.getArgc(), parser.getArgv());
}

int RetrieveVocabTree(const std::string& database_path,
                      const std::string& vocab_tree_path,
                      const std::string& output_index_path /*= ""*/,
                      const std::string& query_image_list_path /*= ""*/,
                      const std::string& database_image_list_path /*= ""*/,
                      int max_num_images /*= -1*/,
                      int num_neighbors /*= 5*/,
                      int num_checks /*= 256*/,
                      int num_images_after_verification /*= 0*/,
                      int max_num_features /*= -1*/) {
    OptionsParser parser;
    parser.registerOption("database_path", &database_path);
    parser.registerOption("vocab_tree_path", &vocab_tree_path);
    parser.registerOption("output_index_path", &output_index_path);
    parser.registerOption("query_image_list_path", &query_image_list_path);
    parser.registerOption("database_image_list_path",
                          &database_image_list_path);
    parser.registerOption("max_num_images", &max_num_images);
    parser.registerOption("num_neighbors", &num_neighbors);
    parser.registerOption("num_checks", &num_checks);
    parser.registerOption("num_images_after_verification",
                          &num_images_after_verification);
    parser.registerOption("max_num_features", &max_num_features);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunVocabTreeRetriever(parser.getArgc(), parser.getArgv());
}

}  // namespace cloudViewer

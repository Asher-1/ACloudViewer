// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

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

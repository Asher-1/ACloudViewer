// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

namespace cloudViewer {

int BuildVocabTree(const std::string& database_path,
                   const std::string& vocab_tree_path,
                   int num_visual_words = 256 * 256,
                   int num_checks = 256,
                   int branching = 256,
                   int num_iterations = 11,
                   int max_num_images = -1);

int RetrieveVocabTree(const std::string& database_path,
                      const std::string& vocab_tree_path,
                      const std::string& output_index_path = "",
                      const std::string& query_image_list_path = "",
                      const std::string& database_image_list_path = "",
                      int max_num_images = -1,
                      int num_neighbors = 5,
                      int num_checks = 256,
                      int num_images_after_verification = 0,
                      int max_num_features = -1);

}  // namespace cloudViewer

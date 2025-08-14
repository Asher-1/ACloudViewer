// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/vocab_tree/vocab_tree.h"

#include "pipelines/vocab_tree.h"
#include "pybind/docstring.h"
#include "pybind/reconstruction/reconstruction_options.h"

namespace cloudViewer {
namespace reconstruction {
namespace vocab_tree {

// Reconstruction vocabulary tree functions have similar arguments, sharing arg
// docstrings
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"database_path",
                 "Path to database in which to store the extracted data"},
                {"vocab_tree_path", "The vocabulary tree path."},
                {"num_visual_words",
                 "The desired number of visual words, i.e. the number of leaf "
                 "node clusters. Note that the actual number of visual words "
                 "might be less."},
                {"num_checks",
                 "The number of checks in the nearest neighbor search."},
                {"branching",
                 "The branching factor of the hierarchical k-means tree."},
                {"num_iterations",
                 "The number of iterations for the clustering."},
                {"max_num_images", "The maximum number of images."},
                {"database_image_list_path", "The database image list path."},
                {"query_image_list_path", "The query image list path."},
                {"output_index_path", "The output index path."},
                {"max_num_images",
                 "The maximum number of most similar images to retrieve."},
                {"num_neighbors",
                 "The number of nearest neighbor visual words that each "
                 "feature descriptor is assigned to."},
                {"num_images_after_verification",
                 "Whether to perform spatial verification after image "
                 "retrieval."},
                {"max_num_features", "The maximum number of features."}};

void pybind_vocab_tree_methods(py::module &m) {
    m.def("build_vocab_tree", &BuildVocabTree,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the building of vocabulary tree", "database_path"_a,
          "vocab_tree_path"_a, "num_visual_words"_a = 256 * 256,
          "num_checks"_a = 256, "branching"_a = 256, "num_iterations"_a = 11,
          "max_num_images"_a = -1);
    docstring::FunctionDocInject(m, "build_vocab_tree",
                                 map_shared_argument_docstrings);

    m.def("retrieve_vocab_tree", &RetrieveVocabTree,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the retrieve of vocabulary tree", "database_path"_a,
          "vocab_tree_path"_a, "output_index_path"_a = "",
          "query_image_list_path"_a = "", "database_image_list_path"_a = "",
          "max_num_images"_a = -1, "num_neighbors"_a = 5, "num_checks"_a = 256,
          "num_images_after_verification"_a = 0, "max_num_features"_a = -1);
    docstring::FunctionDocInject(m, "retrieve_vocab_tree",
                                 map_shared_argument_docstrings);
}

void pybind_vocab_tree(py::module &m) {
    py::module m_submodule =
            m.def_submodule("vocab_tree", "Reconstruction vocabulary tree.");
    pybind_vocab_tree_methods(m_submodule);
}

}  // namespace vocab_tree
}  // namespace reconstruction
}  // namespace cloudViewer

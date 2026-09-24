// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/vocab_tree/vocab_tree.h"

#include "feature/types.h"
#include "pipelines/vocab_tree.h"
#include "pybind/docstring.h"
#include "pybind/reconstruction/reconstruction_options.h"
#include "retrieval/utils.h"
#include "retrieval/visual_index.h"

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

void pybind_vocab_tree_methods(py::module& m) {
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

void pybind_vocab_tree(py::module& m) {
    py::module m_submodule =
            m.def_submodule("vocab_tree", "Reconstruction vocabulary tree.");
    pybind_vocab_tree_methods(m_submodule);

    // Upstream pycolmap parity (src/pycolmap/retrieval/visual_index.cc): the
    // faiss-backed visual index for image retrieval.
    using colmap::FeatureDescriptorsFloat;
    using colmap::FeatureKeypoints;
    using colmap::retrieval::ImageScore;
    using colmap::retrieval::VisualIndex;

    py::class_<ImageScore>(m_submodule, "ImageScore",
                           "A retrieval candidate with its similarity score.")
            .def(py::init<>())
            .def_readwrite("image_id", &ImageScore::image_id)
            .def_readwrite("score", &ImageScore::score)
            .def("__repr__", [](const ImageScore& self) {
                return "ImageScore(image_id=" + std::to_string(self.image_id) +
                       ", score=" + std::to_string(self.score) + ")";
            });

    py::class_<VisualIndex, std::unique_ptr<VisualIndex>> visual_index(
            m_submodule, "VisualIndex",
            "A faiss-backed visual index for image retrieval.");
    py::class_<VisualIndex::IndexOptions> index_options(
            visual_index, "IndexOptions",
            "Options for adding images to the index.");
    index_options.def(py::init<>())
            .def_readwrite("num_neighbors",
                           &VisualIndex::IndexOptions::num_neighbors)
            .def_readwrite("num_checks", &VisualIndex::IndexOptions::num_checks)
            .def_readwrite("num_threads",
                           &VisualIndex::IndexOptions::num_threads);

    py::class_<VisualIndex::QueryOptions> query_options(
            visual_index, "QueryOptions", "Options for querying the index.");
    query_options.def(py::init<>())
            .def_readwrite("max_num_images",
                           &VisualIndex::QueryOptions::max_num_images)
            .def_readwrite("num_neighbors",
                           &VisualIndex::QueryOptions::num_neighbors)
            .def_readwrite(
                    "num_images_after_verification",
                    &VisualIndex::QueryOptions::num_images_after_verification)
            .def_readwrite("num_checks", &VisualIndex::QueryOptions::num_checks)
            .def_readwrite("num_threads",
                           &VisualIndex::QueryOptions::num_threads);

    py::class_<VisualIndex::BuildOptions> build_options(
            visual_index, "BuildOptions",
            "Options for building the index from training descriptors.");
    build_options.def(py::init<>())
            .def_readwrite("num_visual_words",
                           &VisualIndex::BuildOptions::num_visual_words)
            .def_readwrite("num_iterations",
                           &VisualIndex::BuildOptions::num_iterations)
            .def_readwrite("num_rounds", &VisualIndex::BuildOptions::num_rounds)
            .def_readwrite("num_checks", &VisualIndex::BuildOptions::num_checks)
            .def_readwrite("num_threads",
                           &VisualIndex::BuildOptions::num_threads);

    visual_index
            .def_static(
                    "create",
                    [](int desc_dim, int embedding_dim) {
                        return VisualIndex::Create(desc_dim, embedding_dim);
                    },
                    "desc_dim"_a = 128, "embedding_dim"_a = 64,
                    "Create an empty visual index for the given descriptor "
                    "dimensions.")
            .def_static(
                    "read",
                    [](const std::string& path) {
                        return VisualIndex::Read(path);
                    },
                    "path"_a,
                    "Read a visual index from the given path (auto-downloads "
                    "known vocab tree URIs).")
            .def(
                    "write",
                    [](const VisualIndex& self, const std::string& path) {
                        self.Write(path);
                    },
                    "path"_a, "Write the visual index to the given path.")
            .def_property_readonly("num_visual_words",
                                   &VisualIndex::NumVisualWords)
            .def_property_readonly("num_images", &VisualIndex::NumImages)
            .def_property_readonly("desc_dim", &VisualIndex::DescDim)
            .def_property_readonly("embedding_dim", &VisualIndex::EmbeddingDim)
            .def("is_image_indexed", &VisualIndex::IsImageIndexed, "image_id"_a)
            .def(
                    "add",
                    [](VisualIndex& self,
                       const VisualIndex::IndexOptions& options, int image_id,
                       const FeatureKeypoints& keypoints,
                       const FeatureDescriptorsFloat& descriptors) {
                        self.Add(options, image_id, keypoints, descriptors);
                    },
                    "options"_a, "image_id"_a, "keypoints"_a, "descriptors"_a,
                    "Add an image to the index (descriptors as an (N, D) "
                    "float32 array).")
            .def(
                    "query",
                    [](const VisualIndex& self,
                       const VisualIndex::QueryOptions& options,
                       const FeatureKeypoints& keypoints,
                       const FeatureDescriptorsFloat& descriptors) {
                        std::vector<ImageScore> image_scores;
                        self.Query(options, keypoints, descriptors,
                                   &image_scores);
                        return image_scores;
                    },
                    "options"_a, "keypoints"_a, "descriptors"_a,
                    "Query for the most similar images; returns a list of "
                    "ImageScore.")
            .def("prepare", &VisualIndex::Prepare,
                 "Prepare the index after adding images and before "
                 "querying.")
            .def(
                    "build",
                    [](VisualIndex& self,
                       const VisualIndex::BuildOptions& options,
                       const FeatureDescriptorsFloat& descriptors) {
                        self.Build(options, descriptors);
                    },
                    "options"_a, "descriptors"_a,
                    "Build the visual words from training descriptors (an "
                    "(N, D) float32 array).")
            .def("__repr__", [](const VisualIndex& self) {
                return "VisualIndex(num_visual_words=" +
                       std::to_string(self.NumVisualWords()) + ")";
            });
}

}  // namespace vocab_tree
}  // namespace reconstruction
}  // namespace cloudViewer

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/feature/feature.h"

#include "pipelines/feature.h"
#include "pybind/docstring.h"
#include "pybind/reconstruction/reconstruction_options.h"

namespace cloudViewer {
namespace reconstruction {
namespace feature {

// Reconstruction feature functions have similar arguments, sharing arg
// docstrings
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"database_path",
                 "Path to database in which to store the extracted data"},
                {"image_path",
                 "Root path to folder which contains the images."},
                {"image_list_path", "The images list file path with ."},
                {"import_path",
                 "Optional list of images to read. "
                 "The list must contain the relative path of the images with "
                 "respect to the image_path"},
                {"camera_mode",
                 "The camera mode like { AUTO = 0, SINGLE = 1, PER_FOLDER = 2, "
                 "PER_IMAGE = 3 }"},
                {"match_list_path", "The matches list directories"},
                {"match_type",
                 "The match type supported {'pairs', 'raw', 'inliers'}"}};

void pybind_feature_methods(py::module &m) {
    m.def("extract_feature", &ExtractFeature,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the extraction of images feature", "database_path"_a,
          "image_path"_a, "image_list_path"_a = "", "camera_mode"_a = 0,
          "image_reader_options"_a = colmap::ImageReaderOptions(),
          "sift_extraction_options"_a = colmap::SiftExtractionOptions());
    docstring::FunctionDocInject(m, "extract_feature",
                                 map_shared_argument_docstrings);

    m.def("import_feature", &ImportFeature,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the importation of images feature", "database_path"_a,
          "image_path"_a, "import_path"_a, "image_list_path"_a = "",
          "camera_mode"_a = 0,
          "image_reader_options"_a = colmap::ImageReaderOptions(),
          "sift_extraction_options"_a = colmap::SiftExtractionOptions());
    docstring::FunctionDocInject(m, "import_feature",
                                 map_shared_argument_docstrings);

    m.def("import_matches", &ImportMatches,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the importation of image matches", "database_path"_a,
          "match_list_path"_a, "match_type"_a = "pairs",
          "sift_matching_options"_a = colmap::SiftMatchingOptions());
    docstring::FunctionDocInject(m, "import_matches",
                                 map_shared_argument_docstrings);

    m.def("exhaustive_match", &ExhaustiveMatch,
          py::call_guard<py::gil_scoped_release>(),
          "Function for exhaustive image matches", "database_path"_a,
          "sift_matching_options"_a = colmap::SiftMatchingOptions(),
          "exhaustive_matching_options"_a =
                  colmap::ExhaustiveMatchingOptions());
    docstring::FunctionDocInject(m, "exhaustive_match",
                                 map_shared_argument_docstrings);

    m.def("sequential_match", &SequentialMatch,
          py::call_guard<py::gil_scoped_release>(),
          "Function for sequential image matches", "database_path"_a,
          "sift_matching_options"_a = colmap::SiftMatchingOptions(),
          "sequential_matching_options"_a =
                  colmap::SequentialMatchingOptions());
    docstring::FunctionDocInject(m, "sequential_match",
                                 map_shared_argument_docstrings);

    m.def("spatial_match", &SpatialMatch,
          py::call_guard<py::gil_scoped_release>(),
          "Function for spatial image matches", "database_path"_a,
          "sift_matching_options"_a = colmap::SiftMatchingOptions(),
          "spatial_matching_options"_a = colmap::SpatialMatchingOptions());
    docstring::FunctionDocInject(m, "spatial_match",
                                 map_shared_argument_docstrings);

    m.def("transitive_match", &TransitiveMatch,
          py::call_guard<py::gil_scoped_release>(),
          "Function for transitive image matches", "database_path"_a,
          "sift_matching_options"_a = colmap::SiftMatchingOptions(),
          "transitive_matching_options"_a =
                  colmap::TransitiveMatchingOptions());
    docstring::FunctionDocInject(m, "transitive_match",
                                 map_shared_argument_docstrings);

    m.def("vocab_tree_match", &VocabTreeMatch,
          py::call_guard<py::gil_scoped_release>(),
          "Function for vocab_tree image matches", "database_path"_a,
          "sift_matching_options"_a = colmap::SiftMatchingOptions(),
          "vocab_tree_matching_options"_a = colmap::VocabTreeMatchingOptions());
    docstring::FunctionDocInject(m, "vocab_tree_match",
                                 map_shared_argument_docstrings);
}

void pybind_feature(py::module &m) {
    py::module m_submodule =
            m.def_submodule("feature", "Reconstruction Images Feature.");
    pybind_feature_methods(m_submodule);
}

}  // namespace feature
}  // namespace reconstruction
}  // namespace cloudViewer

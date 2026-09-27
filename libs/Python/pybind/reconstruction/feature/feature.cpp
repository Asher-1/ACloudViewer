// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/feature/feature.h"

#include "feature/aliked.h"
#include "feature/extractor.h"
#include "feature/loma.h"
#include "feature/learned_matchers.h"
#include "feature/matcher.h"
#include "feature/types.h"
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

void pybind_feature_methods(py::module& m) {
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

void pybind_feature(py::module& m) {
    py::module m_submodule =
            m.def_submodule("feature", "Reconstruction Images Feature.");
    pybind_feature_methods(m_submodule);

    // Upstream pycolmap parity (src/pycolmap/feature/types.cc): the feature
    // value types used by the retrieval and database surfaces.
    using colmap::FeatureKeypoint;
    using colmap::FeatureMatch;

    // Upstream pycolmap parity (src/pycolmap/feature/types.cc): the
    // descriptor block types. Upstream groups the matrix with its extractor
    // type in a struct; the fork engine stores plain row-major Eigen
    // matrices, whose transparent numpy caster would swallow a direct class
    // registration. These binding-level wrappers preserve the upstream class
    // surface over the fork storage.
    struct PyFeatureDescriptorsBlock {
        colmap::FeatureDescriptors data;
    };
    struct PyFeatureDescriptorsFloatBlock {
        colmap::FeatureDescriptorsFloat data;
    };

    py::class_<PyFeatureDescriptorsBlock> feature_descriptors(
            m_submodule, "FeatureDescriptors",
            "A row-major uint8 descriptor block (rows = features).");
    feature_descriptors.def(py::init<>())
            .def_property(
                    "data",
                    [](const PyFeatureDescriptorsBlock& self) {
                        return py::cast(self.data);
                    },
                    [](PyFeatureDescriptorsBlock& self,
                       const py::array_t<uint8_t, py::array::c_style |
                                                          py::array::forcecast>&
                               arr) {
                        if (arr.ndim() != 2) {
                            THROW_CHECK_EQ(arr.ndim(), 2);
                        }
                        self.data.resize(arr.shape(0), arr.shape(1));
                        std::memcpy(self.data.data(), arr.data(),
                                    sizeof(uint8_t) * arr.size());
                    },
                    "The uint8 matrix as a numpy view.")
            .def_property_readonly(
                    "shape",
                    [](const PyFeatureDescriptorsBlock& self) {
                        return py::make_tuple(self.data.rows(),
                                              self.data.cols());
                    },
                    "(rows, cols) of the descriptor block.")
            .def_static(
                    "from_float",
                    [](const PyFeatureDescriptorsFloatBlock& float_desc) {
                        // Upstream semantics: reinterpret float32 bytes as
                        // uint8 (row count unchanged, cols * 4).
                        PyFeatureDescriptorsBlock out;
                        out.data.resize(float_desc.data.rows(),
                                        float_desc.data.cols() * 4);
                        std::memcpy(out.data.data(), float_desc.data.data(),
                                    sizeof(float) * float_desc.data.rows() *
                                            float_desc.data.cols());
                        return out;
                    },
                    "float_desc"_a,
                    "Create from float descriptors by reinterpreting "
                    "float32 data as uint8 bytes.")
            .def(
                    "to_float",
                    [](const PyFeatureDescriptorsBlock& self) {
                        if (self.data.cols() % 4 != 0) {
                            THROW_CHECK_EQ(self.data.cols() % 4, 0);
                        }
                        PyFeatureDescriptorsFloatBlock out;
                        out.data.resize(self.data.rows(), self.data.cols() / 4);
                        std::memcpy(out.data.data(), self.data.data(),
                                    sizeof(float) * self.data.rows() *
                                            (self.data.cols() / 4));
                        return out;
                    },
                    "Convert to float descriptors by reinterpreting uint8 "
                    "data as float32.")
            .def("__repr__", [](const PyFeatureDescriptorsBlock& self) {
                return "FeatureDescriptors(shape=(" +
                       std::to_string(self.data.rows()) + ", " +
                       std::to_string(self.data.cols()) + "))";
            });

    py::class_<PyFeatureDescriptorsFloatBlock> feature_descriptors_float(
            m_submodule, "FeatureDescriptorsFloat",
            "A row-major float32 descriptor block (rows = features).");
    feature_descriptors_float.def(py::init<>())
            .def_property(
                    "data",
                    [](const PyFeatureDescriptorsFloatBlock& self) {
                        return py::cast(self.data);
                    },
                    [](PyFeatureDescriptorsFloatBlock& self,
                       const py::array_t<float, py::array::c_style |
                                                        py::array::forcecast>&
                               arr) {
                        if (arr.ndim() != 2) {
                            THROW_CHECK_EQ(arr.ndim(), 2);
                        }
                        self.data.resize(arr.shape(0), arr.shape(1));
                        std::memcpy(self.data.data(), arr.data(),
                                    sizeof(float) * arr.size());
                    },
                    "The float32 matrix as a numpy view.")
            .def_property_readonly(
                    "shape",
                    [](const PyFeatureDescriptorsFloatBlock& self) {
                        return py::make_tuple(self.data.rows(),
                                              self.data.cols());
                    },
                    "(rows, cols) of the descriptor block.")
            .def_static(
                    "from_bytes",
                    [](const PyFeatureDescriptorsBlock& byte_desc) {
                        if (byte_desc.data.cols() % 4 != 0) {
                            THROW_CHECK_EQ(byte_desc.data.cols() % 4, 0);
                        }
                        PyFeatureDescriptorsFloatBlock out;
                        out.data.resize(byte_desc.data.rows(),
                                        byte_desc.data.cols() / 4);
                        std::memcpy(out.data.data(), byte_desc.data.data(),
                                    sizeof(float) * byte_desc.data.rows() *
                                            (byte_desc.data.cols() / 4));
                        return out;
                    },
                    "byte_desc"_a,
                    "Create from byte descriptors by reinterpreting uint8 "
                    "data as float32.")
            .def(
                    "to_bytes",
                    [](const PyFeatureDescriptorsFloatBlock& self) {
                        PyFeatureDescriptorsBlock out;
                        out.data.resize(self.data.rows(), self.data.cols() * 4);
                        std::memcpy(out.data.data(), self.data.data(),
                                    sizeof(float) * self.data.rows() *
                                            self.data.cols());
                        return out;
                    },
                    "Convert to byte descriptors by reinterpreting float32 "
                    "data as uint8 bytes.")
            .def("__repr__", [](const PyFeatureDescriptorsFloatBlock& self) {
                return "FeatureDescriptorsFloat(shape=(" +
                       std::to_string(self.data.rows()) + ", " +
                       std::to_string(self.data.cols()) + "))";
            });

    py::class_<FeatureKeypoint>(m_submodule, "FeatureKeypoint",
                                "A feature location with its affine shape.")
            .def(py::init<>())
            .def(py::init<float, float>(), "x"_a, "y"_a)
            .def(py::init<float, float, float, float>(), "x"_a, "y"_a,
                 "scale"_a, "orientation"_a)
            .def(py::init<float, float, float, float, float, float>(), "x"_a,
                 "y"_a, "a11"_a, "a12"_a, "a21"_a, "a22"_a)
            .def_readwrite("x", &FeatureKeypoint::x)
            .def_readwrite("y", &FeatureKeypoint::y)
            .def_readwrite("a11", &FeatureKeypoint::a11)
            .def_readwrite("a12", &FeatureKeypoint::a12)
            .def_readwrite("a21", &FeatureKeypoint::a21)
            .def_readwrite("a22", &FeatureKeypoint::a22)
            .def("compute_scale", &FeatureKeypoint::ComputeScale)
            .def("compute_scale_x", &FeatureKeypoint::ComputeScaleX)
            .def("compute_scale_y", &FeatureKeypoint::ComputeScaleY)
            .def("compute_orientation", &FeatureKeypoint::ComputeOrientation)
            .def("compute_shear", &FeatureKeypoint::ComputeShear)
            .def("rescale", [](FeatureKeypoint& self,
                               float scale) { self.Rescale(scale); })
            .def("__repr__", [](const FeatureKeypoint& self) {
                return "FeatureKeypoint(x=" + std::to_string(self.x) +
                       ", y=" + std::to_string(self.y) + ")";
            });

    py::class_<FeatureMatch>(m_submodule, "FeatureMatch",
                             "A feature match between two images (point2D "
                             "indices).")
            .def(py::init<>())
            .def(py::init<colmap::point2D_t, colmap::point2D_t>(),
                 "point2D_idx1"_a, "point2D_idx2"_a)
            .def_readwrite("point2D_idx1", &FeatureMatch::point2D_idx1)
            .def_readwrite("point2D_idx2", &FeatureMatch::point2D_idx2)
            .def("__repr__", [](const FeatureMatch& self) {
                return "FeatureMatch(" + std::to_string(self.point2D_idx1) +
                       ", " + std::to_string(self.point2D_idx2) + ")";
            });

    // Upstream pycolmap parity (src/pycolmap/feature/types.cc): matrix
    // conversion helpers, implemented at the binding layer over the
    // FeatureKeypoint/FeatureMatch value types.
    m_submodule
            .def(
                    "keypoints_to_matrix",
                    [](const std::vector<colmap::FeatureKeypoint>& keypoints) {
                        py::array_t<double> out(
                                {static_cast<py::ssize_t>(keypoints.size()),
                                 py::ssize_t(4)});
                        auto buf = out.mutable_unchecked<2>();
                        for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
                            buf(i, 0) = keypoints[i].x;
                            buf(i, 1) = keypoints[i].y;
                            buf(i, 2) = keypoints[i].ComputeScale();
                            buf(i, 3) = keypoints[i].ComputeOrientation();
                        }
                        return out;
                    },
                    "keypoints"_a,
                    "Convert FeatureKeypoints to an Nx4 matrix [x, y, "
                    "scale, orientation].")
            .def(
                    "keypoints_from_matrix",
                    [](const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               matrix) {
                        THROW_CHECK_EQ(matrix.ndim(), 2);
                        THROW_CHECK_EQ(matrix.shape(1), 4);
                        auto buf = matrix.unchecked<2>();
                        std::vector<colmap::FeatureKeypoint> keypoints;
                        keypoints.reserve(buf.shape(0));
                        for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
                            keypoints.emplace_back(buf(i, 0), buf(i, 1),
                                                   buf(i, 2), buf(i, 3));
                        }
                        return keypoints;
                    },
                    "matrix"_a,
                    "Convert an Nx4 matrix [x, y, scale, orientation] to "
                    "FeatureKeypoints.")
            .def(
                    "matches_to_matrix",
                    [](const std::vector<colmap::FeatureMatch>& matches) {
                        py::array_t<colmap::point2D_t> out(
                                {static_cast<py::ssize_t>(matches.size()),
                                 py::ssize_t(2)});
                        auto buf = out.mutable_unchecked<2>();
                        for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
                            buf(i, 0) = matches[i].point2D_idx1;
                            buf(i, 1) = matches[i].point2D_idx2;
                        }
                        return out;
                    },
                    "matches"_a,
                    "Convert FeatureMatches to an Nx2 matrix of point2D "
                    "indices.")
            .def(
                    "matches_from_matrix",
                    [](const py::array_t<colmap::point2D_t,
                                         py::array::c_style |
                                                 py::array::forcecast>&
                               matrix) {
                        THROW_CHECK_EQ(matrix.ndim(), 2);
                        THROW_CHECK_EQ(matrix.shape(1), 2);
                        auto buf = matrix.unchecked<2>();
                        std::vector<colmap::FeatureMatch> matches;
                        matches.reserve(buf.shape(0));
                        for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
                            matches.emplace_back(buf(i, 0), buf(i, 1));
                        }
                        return matches;
                    },
                    "matrix"_a,
                    "Convert an Nx2 matrix of point2D indices to "
                    "FeatureMatches.");

    // ------------------------------------------------------------------
    // Upstream pycolmap parity (src/pycolmap/feature/extraction.cc):
    // top-level enums and the LoMa option structs (fork engine runs LoMa
    // on the ggml AICore backend instead of onnxruntime - the model paths
    // are provisioned through the shared AICore catalog).
    // ------------------------------------------------------------------
    py::enum_<colmap::FeatureExtractorType>(m, "FeatureExtractorType",
                                            "The feature extractor type.")
            .value("UNDEFINED", colmap::FeatureExtractorType::UNDEFINED)
            .value("SIFT", colmap::FeatureExtractorType::SIFT)
            .value("ALIKED_N16ROT", colmap::FeatureExtractorType::ALIKED_N16ROT)
            .value("ALIKED_N32", colmap::FeatureExtractorType::ALIKED_N32)
            .value("LOMA_B", colmap::FeatureExtractorType::LOMA_B)
            .value("LOMA_B128", colmap::FeatureExtractorType::LOMA_B128)
            .export_values();

    // Upstream parity (d3ccaf35 feature/types.h): matcher-type dispatch for
    // the new-style FeatureMatcher abstraction. SIFT_*/LOMA_* resolve on the
    // SIFT/LoMa engines; ALIKED_* land with the AICore ggml task family.
    py::enum_<colmap::FeatureMatcherType>(m, "FeatureMatcherType",
                                          "The feature matcher type.")
            .value("UNDEFINED", colmap::FeatureMatcherType::UNDEFINED)
            .value("SIFT_BRUTEFORCE",
                   colmap::FeatureMatcherType::SIFT_BRUTEFORCE)
            .value("SIFT_LIGHTGLUE", colmap::FeatureMatcherType::SIFT_LIGHTGLUE)
            .value("ALIKED_BRUTEFORCE",
                   colmap::FeatureMatcherType::ALIKED_BRUTEFORCE)
            .value("ALIKED_LIGHTGLUE",
                   colmap::FeatureMatcherType::ALIKED_LIGHTGLUE)
            .value("LOMA_BRUTEFORCE",
                   colmap::FeatureMatcherType::LOMA_BRUTEFORCE)
            .value("LOMA_B", colmap::FeatureMatcherType::LOMA_B)
            .value("LOMA_B128", colmap::FeatureMatcherType::LOMA_B128)
            .value("LOMA_R", colmap::FeatureMatcherType::LOMA_R)
            .value("LOMA_L", colmap::FeatureMatcherType::LOMA_L)
            .value("LOMA_G", colmap::FeatureMatcherType::LOMA_G)
            .export_values();

    // ------------------------------------------------------------------
    // New-style abstraction shells (upstream feature/extractor.h and
    // feature/matcher.h): thin factories over the fork SIFT/LoMa engines;
    // the ALIKED branches throw until the AICore ggml task lands.
    // ------------------------------------------------------------------
    py::class_<colmap::FeatureExtractionOptions>(
            m, "FeatureExtractionOptions")
            .def(py::init<colmap::FeatureExtractorType>(),
                 "type"_a = colmap::FeatureExtractorType::SIFT)
            .def_readwrite("type",
                           &colmap::FeatureExtractionOptions::type)
            .def_readwrite("max_image_size",
                           &colmap::FeatureExtractionOptions::max_image_size)
            .def_readwrite("num_threads",
                           &colmap::FeatureExtractionOptions::num_threads)
            .def_readwrite("use_gpu",
                           &colmap::FeatureExtractionOptions::use_gpu)
            .def_readwrite("gpu_index",
                           &colmap::FeatureExtractionOptions::gpu_index)
            .def_readwrite("sift",
                           &colmap::FeatureExtractionOptions::sift)
            .def_readwrite("loma",
                           &colmap::FeatureExtractionOptions::loma)
            .def_readwrite("aliked",
                           &colmap::FeatureExtractionOptions::aliked)
            .def("requires_rgb",
                 &colmap::FeatureExtractionOptions::RequiresRGB)
            .def("requires_open_gl",
                 &colmap::FeatureExtractionOptions::RequiresOpenGL)
            .def("eff_max_image_size",
                 &colmap::FeatureExtractionOptions::EffMaxImageSize)
            .def("check", &colmap::FeatureExtractionOptions::Check);

    py::class_<colmap::FeatureExtractor>(
            m, "FeatureExtractor",
            "The new-style feature extraction abstraction; Create() "
            "dispatches to the fork SIFT (CPU) or LoMa (ggml) engine.")
            .def_static("create", &colmap::FeatureExtractor::Create,
                        "options"_a)
            .def("extract",
                 [](colmap::FeatureExtractor& self,
                    const colmap::Bitmap& bitmap) {
                     colmap::FeatureKeypoints keypoints;
                     colmap::FeatureDescriptorsFloat descriptors;
                     THROW_CHECK(self.Extract(bitmap, &keypoints,
                                              &descriptors));
                     return py::make_tuple(keypoints, descriptors);
                 },
                 "bitmap"_a,
                 "Extract keypoints and float descriptors from the image.");

    py::class_<colmap::FeatureMatchingOptions>(m, "FeatureMatchingOptions")
            .def(py::init<colmap::FeatureMatcherType>(),
                 "type"_a = colmap::FeatureMatcherType::SIFT_BRUTEFORCE)
            .def_readwrite("type", &colmap::FeatureMatchingOptions::type)
            .def_readwrite("num_threads",
                           &colmap::FeatureMatchingOptions::num_threads)
            .def_readwrite("use_gpu",
                           &colmap::FeatureMatchingOptions::use_gpu)
            .def_readwrite("gpu_index",
                           &colmap::FeatureMatchingOptions::gpu_index)
            .def_readwrite("max_num_matches",
                           &colmap::FeatureMatchingOptions::max_num_matches)
            .def_readwrite("guided_matching",
                           &colmap::FeatureMatchingOptions::guided_matching)
            .def_readwrite("skip_geometric_verification",
                           &colmap::FeatureMatchingOptions::
                                   skip_geometric_verification)
            .def_readwrite("rig_verification",
                           &colmap::FeatureMatchingOptions::rig_verification)
            .def_readwrite("skip_image_pairs_in_same_frame",
                           &colmap::FeatureMatchingOptions::
                                   skip_image_pairs_in_same_frame)
            .def_readwrite("sift", &colmap::FeatureMatchingOptions::sift)
            .def_readwrite("loma", &colmap::FeatureMatchingOptions::loma)
            .def_readwrite("aliked", &colmap::FeatureMatchingOptions::aliked)
            .def("check", &colmap::FeatureMatchingOptions::Check);

    py::class_<colmap::FeatureMatcher::Image>(
            m, "FeatureMatcherImage",
            "The per-image inputs of the new-style matcher (upstream nests "
            "this as FeatureMatcher::Image).")
            .def(py::init<>())
            .def_readwrite("image_id",
                           &colmap::FeatureMatcher::Image::image_id)
            .def_readwrite("camera",
                           &colmap::FeatureMatcher::Image::camera)
            .def_readwrite("keypoints",
                           &colmap::FeatureMatcher::Image::keypoints)
            .def_readwrite("descriptors",
                           &colmap::FeatureMatcher::Image::descriptors);

    py::class_<colmap::FeatureMatcher>(m, "FeatureMatcher",
                                       "The new-style in-memory feature "
                                       "matching abstraction.")
            .def_static("create", &colmap::FeatureMatcher::Create,
                        "options"_a)
            .def("match",
                 [](colmap::FeatureMatcher& self,
                    const colmap::FeatureMatcher::Image& image1,
                    const colmap::FeatureMatcher::Image& image2) {
                     colmap::FeatureMatches matches;
                     self.Match(image1, image2, &matches);
                     return matches;
                 },
                 "image1"_a, "image2"_a,
                 "Match two images and return the feature matches.")
            .def("match_guided",
                 [](colmap::FeatureMatcher& self, double max_error,
                    const colmap::FeatureMatcher::Image& image1,
                    const colmap::FeatureMatcher::Image& image2) {
                     colmap::TwoViewGeometry two_view_geometry;
                     self.MatchGuided(max_error, image1, image2,
                                      &two_view_geometry);
                     return two_view_geometry;
                 },
                 "max_error"_a, "image1"_a, "image2"_a,
                 "Match two images with the guided matcher and return the "
                 "two-view geometry.");

    // Fork-specific descriptor-type dispatch (consumed by
    // LomaExtractionOptions and the database descriptor-type column).
    py::enum_<colmap::FeatureDescriptorType>(m, "FeatureDescriptorType",
                                             "The feature descriptor family "
                                             "stored for an image.")
            .value("kSift", colmap::FeatureDescriptorType::kSift)
            .value("kLomaB", colmap::FeatureDescriptorType::kLomaB)
            .value("kLomaB128", colmap::FeatureDescriptorType::kLomaB128)
            .value("kLomaR", colmap::FeatureDescriptorType::kLomaR)
            .value("kLomaL", colmap::FeatureDescriptorType::kLomaL)
            .value("kLomaG", colmap::FeatureDescriptorType::kLomaG)
            .export_values();


    py::class_<colmap::LomaExtractionOptions>(m, "LomaExtractionOptions")
            .def(py::init<>())
            .def_readwrite("detector_model_path",
                           &colmap::LomaExtractionOptions::detector_model_path,
                           "Detector model path (ggml AICore catalog default "
                           "when empty).")
            .def_readwrite(
                    "descriptor_model_path",
                    &colmap::LomaExtractionOptions::descriptor_model_path,
                    "Descriptor model path (ggml AICore catalog default when "
                    "empty).")
            .def_readwrite("max_num_features",
                           &colmap::LomaExtractionOptions::max_num_features)
            .def_readwrite("min_score",
                           &colmap::LomaExtractionOptions::min_score)
            .def_readwrite("device", &colmap::LomaExtractionOptions::device,
                           "ggml backend device selector (auto/cpu/gpu).")
            .def_readwrite("num_threads",
                           &colmap::LomaExtractionOptions::num_threads)
            .def_readwrite("descriptor_type",
                           &colmap::LomaExtractionOptions::descriptor_type)
            .def("check", &colmap::LomaExtractionOptions::Check);

    py::class_<colmap::AlikedExtractionOptions>(m, "AlikedExtractionOptions")
            .def(py::init<>())
            .def_readwrite("max_num_features",
                           &colmap::AlikedExtractionOptions::max_num_features)
            .def_readwrite("min_score",
                           &colmap::AlikedExtractionOptions::min_score)
            .def_readwrite("n16rot_model_path",
                           &colmap::AlikedExtractionOptions::n16rot_model_path)
            .def_readwrite("n32_model_path",
                           &colmap::AlikedExtractionOptions::n32_model_path)
            .def("check", &colmap::AlikedExtractionOptions::Check);

    py::class_<colmap::AlikedMatchingOptions>(m, "AlikedMatchingOptions")
            .def(py::init<>())
            .def_readwrite("min_cossim",
                           &colmap::AlikedMatchingOptions::min_cossim)
            .def_readwrite("min_score",
                           &colmap::AlikedMatchingOptions::min_score)
            .def("check", &colmap::AlikedMatchingOptions::Check);

    // Upstream Python names; the C++ types carry the GGML backend name.
    py::class_<colmap::BruteForceGGMLMatchingOptions>(
            m, "BruteForceONNXMatchingOptions")
            .def(py::init<>())
            .def_readwrite("min_cossim",
                           &colmap::BruteForceGGMLMatchingOptions::min_cossim)
            .def_readwrite("max_ratio",
                           &colmap::BruteForceGGMLMatchingOptions::max_ratio)
            .def_readwrite("cross_check",
                           &colmap::BruteForceGGMLMatchingOptions::cross_check)
            .def_readwrite("model_path",
                           &colmap::BruteForceGGMLMatchingOptions::model_path)
            .def("check", &colmap::BruteForceGGMLMatchingOptions::Check);

    py::class_<colmap::LightGlueGGMLMatchingOptions>(
            m, "LightGlueONNXMatchingOptions")
            .def(py::init<>())
            .def_readwrite("min_score",
                           &colmap::LightGlueGGMLMatchingOptions::min_score)
            .def_readwrite("model_path",
                           &colmap::LightGlueGGMLMatchingOptions::model_path)
            .def("check", &colmap::LightGlueGGMLMatchingOptions::Check);

    py::class_<colmap::LomaVariantMatcherOptions>(
            m, "LomaVariantMatcherOptions")
            .def(py::init<>())
            .def_readwrite("model_path",
                           &colmap::LomaVariantMatcherOptions::model_path)
            .def_readwrite("model_path_bf16",
                           &colmap::LomaVariantMatcherOptions::model_path_bf16)
            .def("check", &colmap::LomaVariantMatcherOptions::Check);

    py::class_<colmap::LomaMatchingOptions>(m, "LomaMatchingOptions")
            .def(py::init<>())
            .def_readwrite(
                    "matcher_model_path",
                    &colmap::LomaMatchingOptions::matcher_model_path,
                    "Matcher model path (ggml AICore catalog default when "
                    "empty).")
            .def_readwrite("min_score",
                           &colmap::LomaMatchingOptions::min_score)
            .def_readwrite("device", &colmap::LomaMatchingOptions::device)
            .def_readwrite("matcher_variant",
                           &colmap::LomaMatchingOptions::matcher_variant)
            .def("check", &colmap::LomaMatchingOptions::Check);
}

}  // namespace feature
}  // namespace reconstruction
}  // namespace cloudViewer

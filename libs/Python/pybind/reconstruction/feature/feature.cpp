// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/feature/feature.h"

#include "feature/aliked.h"
#include "feature/extractor.h"
#include "feature/learned_matchers.h"
#include "feature/lightglue.h"
#include "feature/loma.h"
#include "feature/matcher.h"
#include "feature/sift.h"
#include "feature/types.h"
#include "pipelines/feature.h"
#include "pybind/docstring.h"
#include "pybind/reconstruction/hash_container_casters.h"
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
                 "The match type supported {'pairs', 'raw', 'inliers'}"},
                {"input_path",
                 "Path to the reconstruction whose poses guide the guided "
                 "geometric verification"},
                {"batch_size", "The number of image pairs verified per batch"},
                {"num_threads", "The number of threads (-1 = all cores)"},
                {"rig_verification",
                 "Whether to verify image pairs jointly per rig frame"}};

void pybind_feature_methods(py::module& m) {
    // Default-argument values are converted at registration time, so this
    // binder runs after the feature/estimators/pipeline option classes exist
    // (see reconstruction.cpp).
    m.def("extract_feature", &ExtractFeature,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the extraction of images feature", "database_path"_a,
          "image_path"_a, "image_list_path"_a = "", "camera_mode"_a = -1,
          "image_reader_options"_a = colmap::ImageReaderOptions(),
          "extraction_options"_a = colmap::FeatureExtractionOptions());
    docstring::FunctionDocInject(m, "extract_feature",
                                 map_shared_argument_docstrings);

    m.def("import_feature", &ImportFeature,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the importation of images feature", "database_path"_a,
          "image_path"_a, "import_path"_a, "image_list_path"_a = "",
          "camera_mode"_a = -1,
          "image_reader_options"_a = colmap::ImageReaderOptions(),
          "extraction_options"_a = colmap::FeatureExtractionOptions());
    docstring::FunctionDocInject(m, "import_feature",
                                 map_shared_argument_docstrings);

    m.def("import_matches", &ImportMatches,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the importation of image matches", "database_path"_a,
          "match_list_path"_a, "match_type"_a = "pairs",
          "matching_options"_a = colmap::FeatureMatchingOptions(),
          "geometry_options"_a = colmap::TwoViewGeometryOptions());
    docstring::FunctionDocInject(m, "import_matches",
                                 map_shared_argument_docstrings);

    m.def("exhaustive_match", &ExhaustiveMatch,
          py::call_guard<py::gil_scoped_release>(),
          "Function for exhaustive image matches", "database_path"_a,
          "matching_options"_a = colmap::FeatureMatchingOptions(),
          "geometry_options"_a = colmap::TwoViewGeometryOptions(),
          "pairing_options"_a = colmap::ExhaustivePairingOptions());
    docstring::FunctionDocInject(m, "exhaustive_match",
                                 map_shared_argument_docstrings);

    m.def("sequential_match", &SequentialMatch,
          py::call_guard<py::gil_scoped_release>(),
          "Function for sequential image matches", "database_path"_a,
          "matching_options"_a = colmap::FeatureMatchingOptions(),
          "geometry_options"_a = colmap::TwoViewGeometryOptions(),
          "pairing_options"_a = colmap::SequentialPairingOptions());
    docstring::FunctionDocInject(m, "sequential_match",
                                 map_shared_argument_docstrings);

    m.def("spatial_match", &SpatialMatch,
          py::call_guard<py::gil_scoped_release>(),
          "Function for spatial image matches", "database_path"_a,
          "matching_options"_a = colmap::FeatureMatchingOptions(),
          "geometry_options"_a = colmap::TwoViewGeometryOptions(),
          "pairing_options"_a = colmap::SpatialPairingOptions());
    docstring::FunctionDocInject(m, "spatial_match",
                                 map_shared_argument_docstrings);

    m.def("transitive_match", &TransitiveMatch,
          py::call_guard<py::gil_scoped_release>(),
          "Function for transitive image matches", "database_path"_a,
          "matching_options"_a = colmap::FeatureMatchingOptions(),
          "geometry_options"_a = colmap::TwoViewGeometryOptions(),
          "pairing_options"_a = colmap::TransitivePairingOptions());
    docstring::FunctionDocInject(m, "transitive_match",
                                 map_shared_argument_docstrings);

    m.def("vocab_tree_match", &VocabTreeMatch,
          py::call_guard<py::gil_scoped_release>(),
          "Function for vocab_tree image matches", "database_path"_a,
          "matching_options"_a = colmap::FeatureMatchingOptions(),
          "geometry_options"_a = colmap::TwoViewGeometryOptions(),
          "pairing_options"_a = colmap::VocabTreePairingOptions());
    docstring::FunctionDocInject(m, "vocab_tree_match",
                                 map_shared_argument_docstrings);

    m.def("verify_geometry", &VerifyGeometry,
          py::call_guard<py::gil_scoped_release>(),
          "Geometrically verify the raw matches already stored in the "
          "database (upstream ``geometric_verifier``)",
          "database_path"_a,
          "geometry_options"_a = colmap::TwoViewGeometryOptions(),
          "batch_size"_a = 1000, "num_threads"_a = -1,
          "rig_verification"_a = false);
    docstring::FunctionDocInject(m, "verify_geometry",
                                 map_shared_argument_docstrings);

    m.def("verify_guided_geometry", &VerifyGuidedGeometry,
          py::call_guard<py::gil_scoped_release>(),
          "Geometrically verify the raw matches guided by the poses of an "
          "existing reconstruction (upstream ``guided_geometric_verifier``)",
          "database_path"_a, "input_path"_a,
          "geometry_options"_a = colmap::TwoViewGeometryOptions(),
          "batch_size"_a = 1000, "num_threads"_a = -1);
    docstring::FunctionDocInject(m, "verify_guided_geometry",
                                 map_shared_argument_docstrings);
}

void pybind_feature(py::module& m) {
    py::module m_submodule =
            m.def_submodule("feature", "Reconstruction Images Feature.");

    // Upstream pycolmap parity (src/pycolmap/feature/types.cc): the feature
    // value types used by the retrieval and database surfaces.
    using colmap::FeatureKeypoint;
    using colmap::FeatureMatch;

    // Upstream pycolmap parity (src/pycolmap/feature/types.cc): the
    // descriptor block types. Upstream groups the matrix with its extractor
    // type in a struct; the fork engine stores plain row-major Eigen
    // matrices, whose transparent numpy caster would swallow a direct class
    py::class_<colmap::FeatureDescriptors> feature_descriptors(
            m_submodule, "FeatureDescriptors",
            "Descriptor bytes together with their extractor type.");
    feature_descriptors.def(py::init<>())
            .def(py::init<colmap::FeatureExtractorType,
                          colmap::FeatureDescriptorsData>(),
                 "type"_a, "data"_a)
            .def_readwrite("type", &colmap::FeatureDescriptors::type)
            .def_readwrite("data", &colmap::FeatureDescriptors::data)
            .def_property_readonly("shape",
                                   [](const colmap::FeatureDescriptors& self) {
                                       return py::make_tuple(self.data.rows(),
                                                             self.data.cols());
                                   })
            .def_static("from_float", &colmap::FeatureDescriptors::FromFloat,
                        "float_desc"_a)
            .def("to_float", &colmap::FeatureDescriptors::ToFloat)
            .def("__repr__", [](const colmap::FeatureDescriptors& self) {
                return "FeatureDescriptors(type=" +
                       std::string(colmap::FeatureExtractorTypeToString(
                               self.type)) +
                       ", shape=(" + std::to_string(self.data.rows()) + ", " +
                       std::to_string(self.data.cols()) + "))";
            });

    py::class_<colmap::FeatureDescriptorsFloat> feature_descriptors_float(
            m_submodule, "FeatureDescriptorsFloat",
            "Float descriptor values together with their extractor type.");
    feature_descriptors_float.def(py::init<>())
            .def(py::init<colmap::FeatureExtractorType,
                          colmap::FeatureDescriptorsFloatData>(),
                 "type"_a, "data"_a)
            .def_readwrite("type", &colmap::FeatureDescriptorsFloat::type)
            .def_readwrite("data", &colmap::FeatureDescriptorsFloat::data)
            .def_property_readonly(
                    "shape",
                    [](const colmap::FeatureDescriptorsFloat& self) {
                        return py::make_tuple(self.data.rows(),
                                              self.data.cols());
                    })
            .def_static("from_bytes",
                        &colmap::FeatureDescriptorsFloat::FromBytes,
                        "byte_desc"_a)
            .def("to_bytes", &colmap::FeatureDescriptorsFloat::ToBytes)
            .def("__repr__", [](const colmap::FeatureDescriptorsFloat& self) {
                return "FeatureDescriptorsFloat(type=" +
                       std::string(colmap::FeatureExtractorTypeToString(
                               self.type)) +
                       ", shape=(" + std::to_string(self.data.rows()) + ", " +
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
    py::class_<colmap::FeatureExtractionOptions>(m, "FeatureExtractionOptions")
            .def(py::init<colmap::FeatureExtractorType>(),
                 "type"_a = colmap::FeatureExtractorType::SIFT)
            .def_readwrite("type", &colmap::FeatureExtractionOptions::type)
            .def_readwrite("max_image_size",
                           &colmap::FeatureExtractionOptions::max_image_size)
            .def_readwrite("num_threads",
                           &colmap::FeatureExtractionOptions::num_threads)
            .def_readwrite("use_gpu",
                           &colmap::FeatureExtractionOptions::use_gpu)
            .def_readwrite("gpu_index",
                           &colmap::FeatureExtractionOptions::gpu_index)
            .def_readwrite("sift", &colmap::FeatureExtractionOptions::sift)
            .def_readwrite("loma", &colmap::FeatureExtractionOptions::loma)
            .def_readwrite("aliked", &colmap::FeatureExtractionOptions::aliked)
            .def("requires_rgb", &colmap::FeatureExtractionOptions::RequiresRGB)
            .def("requires_opengl",
                 &colmap::FeatureExtractionOptions::RequiresOpenGL)
            .def("eff_max_image_size",
                 &colmap::FeatureExtractionOptions::EffMaxImageSize)
            .def("check", &colmap::FeatureExtractionOptions::Check);

    py::class_<colmap::FeatureExtractor>(
            m, "FeatureExtractor",
            "The feature extraction abstraction; Create() dispatches to the "
            "SIFT (CPU/GPU) or the ggml ALIKED / LoMa engine.")
            .def_static("create", &colmap::FeatureExtractor::Create,
                        "options"_a)
            .def(
                    "extract",
                    [](colmap::FeatureExtractor& self,
                       const colmap::Bitmap& bitmap) {
                        colmap::FeatureKeypoints keypoints;
                        colmap::FeatureDescriptors descriptors;
                        THROW_CHECK(
                                self.Extract(bitmap, &keypoints, &descriptors));
                        return py::make_tuple(keypoints, descriptors);
                    },
                    "bitmap"_a,
                    "Extract keypoints and descriptors from the image.");

    py::class_<colmap::FeatureMatchingOptions>(m, "FeatureMatchingOptions")
            .def(py::init<colmap::FeatureMatcherType>(),
                 "type"_a = colmap::FeatureMatcherType::SIFT_BRUTEFORCE)
            .def_readwrite("type", &colmap::FeatureMatchingOptions::type)
            .def_readwrite("num_threads",
                           &colmap::FeatureMatchingOptions::num_threads)
            .def_readwrite("use_gpu", &colmap::FeatureMatchingOptions::use_gpu)
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

    // FeatureMatcher::Image requires shared ownership-shaped inputs, while
    // pybind invokes Match synchronously with Python-owned typed descriptors.
    // Non-owning aliases preserve both type and data without an unnecessary
    // copy; their lifetime is bounded by the call below.
    const auto make_matcher_image =
            [](const colmap::Camera* camera,
               const colmap::FeatureKeypoints* keypoints,
               const colmap::FeatureDescriptors& descriptors) {
                colmap::FeatureMatcher::Image image;
                image.camera = camera;
                if (keypoints != nullptr) {
                    image.keypoints =
                            std::shared_ptr<const colmap::FeatureKeypoints>(
                                    std::shared_ptr<void>(), keypoints);
                }
                image.descriptors =
                        std::shared_ptr<const colmap::FeatureDescriptors>(
                                std::shared_ptr<void>(), &descriptors);
                return image;
            };

    py::class_<colmap::FeatureMatcher>(m, "FeatureMatcher",
                                       "The in-memory feature matching "
                                       "abstraction.")
            .def_static("create", &colmap::FeatureMatcher::Create, "options"_a)
            .def(
                    "match",
                    [make_matcher_image](
                            colmap::FeatureMatcher& self,
                            const colmap::FeatureKeypoints* keypoints1,
                            const colmap::FeatureDescriptors& descriptors1,
                            const colmap::FeatureKeypoints* keypoints2,
                            const colmap::FeatureDescriptors& descriptors2) {
                        const colmap::FeatureMatcher::Image image1 =
                                make_matcher_image(nullptr, keypoints1,
                                                   descriptors1);
                        const colmap::FeatureMatcher::Image image2 =
                                make_matcher_image(nullptr, keypoints2,
                                                   descriptors2);
                        colmap::FeatureMatches matches;
                        self.Match(image1, image2, &matches);
                        return matches;
                    },
                    "keypoints1"_a, "descriptors1"_a, "keypoints2"_a,
                    "descriptors2"_a,
                    "Match features between two images. Keypoints are "
                    "optional. Returns the feature matches.")
            .def(
                    "match_guided",
                    [make_matcher_image](
                            colmap::FeatureMatcher& self, double max_error,
                            const colmap::FeatureKeypoints& keypoints1,
                            const colmap::FeatureDescriptors& descriptors1,
                            const colmap::Camera& camera1,
                            const colmap::FeatureKeypoints& keypoints2,
                            const colmap::FeatureDescriptors& descriptors2,
                            const colmap::Camera& camera2,
                            colmap::TwoViewGeometry& two_view_geometry) {
                        const colmap::FeatureMatcher::Image image1 =
                                make_matcher_image(&camera1, &keypoints1,
                                                   descriptors1);
                        const colmap::FeatureMatcher::Image image2 =
                                make_matcher_image(&camera2, &keypoints2,
                                                   descriptors2);
                        self.MatchGuided(max_error, image1, image2,
                                         &two_view_geometry);
                    },
                    "max_error"_a, "keypoints1"_a, "descriptors1"_a,
                    "camera1"_a, "keypoints2"_a, "descriptors2"_a, "camera2"_a,
                    "two_view_geometry"_a,
                    "Perform guided matching using existing two-view "
                    "geometry. Updates the two_view_geometry in-place.");

    // The per-engine option structs are shared_ptr members of
    // FeatureExtractionOptions / FeatureMatchingOptions, hence the
    // shared_ptr holders. The fork runs ALIKED / LoMa / LightGlue on the
    // ggml AICore backend; ``device`` replaces the upstream ONNX execution
    // provider switches (use_bf16 / use_fast_resize).
    py::class_<colmap::LomaExtractionOptions,
               std::shared_ptr<colmap::LomaExtractionOptions>>(
            m, "LomaExtractionOptions")
            .def(py::init<>())
            .def_readwrite("max_num_features",
                           &colmap::LomaExtractionOptions::max_num_features)
            .def_readwrite("min_score",
                           &colmap::LomaExtractionOptions::min_score)
            .def_readwrite("detector_model_path",
                           &colmap::LomaExtractionOptions::detector_model_path,
                           "Detector model path (ggml AICore catalog default "
                           "when empty).")
            .def_readwrite(
                    "descriptor_model_path",
                    &colmap::LomaExtractionOptions::descriptor_model_path,
                    "LOMA_B descriptor model path (ggml AICore catalog "
                    "default when empty).")
            .def_readwrite(
                    "descriptor_b128_model_path",
                    &colmap::LomaExtractionOptions::descriptor_b128_model_path,
                    "LOMA_B128 descriptor model path (ggml AICore catalog "
                    "default when empty).")
            .def_readwrite("device", &colmap::LomaExtractionOptions::device,
                           "ggml backend device selector (auto/cpu/gpu).")
            .def("check", &colmap::LomaExtractionOptions::Check);

    py::class_<colmap::AlikedExtractionOptions,
               std::shared_ptr<colmap::AlikedExtractionOptions>>(
            m, "AlikedExtractionOptions")
            .def(py::init<>())
            .def_readwrite("max_num_features",
                           &colmap::AlikedExtractionOptions::max_num_features)
            .def_readwrite("min_score",
                           &colmap::AlikedExtractionOptions::min_score)
            .def_readwrite("n16rot_model_path",
                           &colmap::AlikedExtractionOptions::n16rot_model_path)
            .def_readwrite("n32_model_path",
                           &colmap::AlikedExtractionOptions::n32_model_path)
            .def_readwrite("device", &colmap::AlikedExtractionOptions::device,
                           "ggml backend device selector (auto/cpu/gpu).")
            .def("check", &colmap::AlikedExtractionOptions::Check);

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
            .def_readwrite("device",
                           &colmap::LightGlueGGMLMatchingOptions::device,
                           "ggml backend device selector (auto/cpu/gpu).")
            .def("check", &colmap::LightGlueGGMLMatchingOptions::Check);

    py::class_<colmap::AlikedMatchingOptions,
               std::shared_ptr<colmap::AlikedMatchingOptions>>(
            m, "AlikedMatchingOptions")
            .def(py::init<>())
            .def_readwrite("brute_force",
                           &colmap::AlikedMatchingOptions::brute_force,
                           "Brute-force matching options.")
            .def_readwrite("lightglue",
                           &colmap::AlikedMatchingOptions::lightglue,
                           "LightGlue matching options.")
            .def("check", &colmap::AlikedMatchingOptions::Check);

    py::class_<colmap::LomaMatchingOptions::Variant>(
            m, "LomaVariantMatcherOptions")
            .def(py::init<>())
            .def_readwrite("model_path",
                           &colmap::LomaMatchingOptions::Variant::model_path,
                           "Matcher model path for this variant (ggml AICore "
                           "catalog default when empty).");

    py::class_<colmap::LomaMatchingOptions,
               std::shared_ptr<colmap::LomaMatchingOptions>>(
            m, "LomaMatchingOptions")
            .def(py::init<>())
            .def_readwrite("min_score", &colmap::LomaMatchingOptions::min_score,
                           "Matching filter, matches LG.")
            .def_readwrite("b", &colmap::LomaMatchingOptions::b,
                           "LOMA_B model paths.")
            .def_readwrite("b128", &colmap::LomaMatchingOptions::b128,
                           "LOMA_B128 model paths.")
            .def_readwrite("r", &colmap::LomaMatchingOptions::r,
                           "LOMA_R model paths.")
            .def_readwrite("l", &colmap::LomaMatchingOptions::l,
                           "LOMA_L model paths.")
            .def_readwrite("g", &colmap::LomaMatchingOptions::g,
                           "LOMA_G model paths.")
            .def_readwrite("brute_force",
                           &colmap::LomaMatchingOptions::brute_force,
                           "Brute-force matching options.")
            .def_readwrite("device", &colmap::LomaMatchingOptions::device,
                           "ggml backend device selector (auto/cpu/gpu).")
            .def("check", &colmap::LomaMatchingOptions::Check);
}

}  // namespace feature
}  // namespace reconstruction
}  // namespace cloudViewer

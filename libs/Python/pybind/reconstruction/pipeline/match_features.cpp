// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Upstream pycolmap parity (src/pycolmap/pipeline/match_features.cc): the
// pairing-driven feature matching surface over the fork controllers
// (controllers/pairing.h, controllers/feature_matching.h, exe/feature.h).

#include "pybind/reconstruction/pipeline/match_features.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <thread>

#include "controllers/feature_matching.h"
#include "controllers/pairing.h"
#include "estimators/two_view_geometry.h"
#include "exe/feature.h"
#include "feature/matcher.h"
#include "pybind/docstring.h"
#include "scene/database.h"
#include "scene/reconstruction.h"
#include "util/cancellation.h"
#include "util/file.h"
#include "util/logging.h"
#include "util/threading.h"

namespace cloudViewer {
namespace reconstruction {
namespace pipeline {

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

// Upstream parity (src/pycolmap/utils.h): the device selector for
// feature matching - a plain enum. AUTO uses the GPU when CUDA is
// available; CUDA/CPU force the respective device.
enum class Device { AUTO = -1, CPU = 0, CUDA = 1 };

inline bool IsGPU(const Device device) {
    if (device == Device::AUTO) {
#ifdef CUDA_ENABLED
        return true;
#else
        return false;
#endif
    }
    return static_cast<bool>(device);
}

[[noreturn]] void ThrowCancelled() { throw std::runtime_error("Cancelled"); }

// Upstream parity (src/pycolmap/helpers.h PyWait): wait for a running thread
// while servicing Python signals and the cancellation token. The GIL is
// released while polling and re-acquired periodically for PyErr_CheckSignals.
void PyWait(
        colmap::Thread* thread,
        const std::shared_ptr<colmap::CancellationToken>& cancellation_token) {
    while (thread->IsRunning()) {
        if (cancellation_token && cancellation_token->IsCancelled()) {
            thread->Stop();
            thread->Wait();
            ThrowCancelled();
        }
        {
            py::gil_scoped_acquire acquire;
            if (PyErr_CheckSignals() != 0) {
                thread->Stop();
                thread->Wait();
                throw py::error_already_set();
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // After finishing, join the thread to avoid abort on destruction.
    thread->Wait();
}

void CheckDatabasePath(const std::filesystem::path& database_path) {
    THROW_CHECK(std::filesystem::exists(database_path))
            << "Database does not exist: " << database_path;
}

template <typename PairingOptions,
          std::unique_ptr<colmap::Thread> MatcherFactory(
                  const PairingOptions&,
                  const colmap::FeatureMatchingOptions&,
                  const colmap::TwoViewGeometryOptions&,
                  const std::filesystem::path&)>
void MatchFeatures(
        const std::string& database_path_str,
        colmap::FeatureMatchingOptions matching_options,
        const PairingOptions& pairing_options,
        const colmap::TwoViewGeometryOptions& verification_options,
        const Device device,
        const std::shared_ptr<colmap::CancellationToken>& cancellation_token) {
    // The fork's pybind_filesystem.h caster only binds the experimental
    // filesystem path, so the binding layer takes std::string (the same
    // convention as the other fork binders) and converts here.
    std::filesystem::path database_path(database_path_str);
    CheckDatabasePath(database_path);
    // Run the optional Python-side `check()` hook of the pairing options
    // (upstream behavior; pass when no check method exists).
    try {
        py::cast(pairing_options).attr("check").attr("__call__")();
    } catch (py::error_already_set& ex) {
        if (!ex.matches(PyExc_AttributeError)) {
            throw;
        }
    }
    if (matching_options.use_gpu != IsGPU(device)) {
        LOG(WARNING) << "FeatureMatchingOptions::use_gpu does not match "
                        "device. FeatureMatchingOptions::use_gpu is ignored.";
    }
    matching_options.use_gpu = IsGPU(device);
    THROW_CHECK(matching_options.Check());
    py::gil_scoped_release release;
    std::unique_ptr<colmap::Thread> matcher =
            MatcherFactory(pairing_options, matching_options,
                           verification_options, database_path);
    matcher->Start();
    PyWait(matcher.get(), cancellation_token);
}

void VerifyMatches(
        const std::string& database_path_str,
        const std::string& pairs_path_str,
        const colmap::TwoViewGeometryOptions& verification_options,
        const std::shared_ptr<colmap::CancellationToken>& cancellation_token) {
    std::filesystem::path database_path(database_path_str);
    std::filesystem::path pairs_path(pairs_path_str);
    CheckDatabasePath(database_path);
    THROW_CHECK(std::filesystem::exists(pairs_path))
            << "Pairs file does not exist: " << pairs_path;
    py::gil_scoped_release release;  // matching is multi-threaded

    colmap::FeatureMatchingOptions matching_options;
    matching_options.use_gpu = false;

    colmap::ImportedPairingOptions pairing_options;
    pairing_options.match_list_path = pairs_path;

    std::unique_ptr<colmap::Thread> matcher =
            colmap::CreateImagePairsFeatureMatcher(
                    pairing_options, matching_options, verification_options,
                    database_path);
    matcher->Start();
    PyWait(matcher.get(), cancellation_token);
}

void GeometricVerification(
        const std::string& database_path_str,
        const colmap::GeometricVerifierOptions& verifier_options,
        const colmap::ExistingMatchedPairingOptions& pairing_options,
        const colmap::TwoViewGeometryOptions& geometry_options,
        const std::shared_ptr<colmap::CancellationToken>& cancellation_token) {
    std::filesystem::path database_path(database_path_str);
    CheckDatabasePath(database_path);

    py::gil_scoped_release release;  // verification is multi-threaded
    std::unique_ptr<colmap::Thread> verifier = colmap::CreateGeometricVerifier(
            verifier_options, pairing_options, geometry_options, database_path);
    verifier->Start();
    PyWait(verifier.get(), cancellation_token);
}

void GuidedGeometricVerification(
        const colmap::Reconstruction& reconstruction,
        const std::string& database_path_str,
        const colmap::ExistingMatchedPairingOptions& pairing_options,
        const colmap::TwoViewGeometryOptions& geometry_options,
        int num_threads,
        const std::shared_ptr<colmap::CancellationToken>& cancellation_token) {
    std::filesystem::path database_path(database_path_str);
    CheckDatabasePath(database_path);

    py::gil_scoped_release release;  // verification is multi-threaded
    bool interrupted = false;
    colmap::RunGuidedGeometricVerifierImpl(
            reconstruction, database_path, pairing_options, geometry_options,
            num_threads, [&]() {
                if (cancellation_token && cancellation_token->IsCancelled()) {
                    interrupted = true;
                    return true;
                }
                py::gil_scoped_acquire acquire;
                if (PyErr_CheckSignals() != 0) {
                    PyErr_Clear();
                    interrupted = true;
                    return true;
                }
                return false;
            });
    if (interrupted) {
        ThrowCancelled();
    }
}

}  // namespace

void pybind_match_features(py::module& m, py::module& m_root) {
    // ------------------------------------------------------------------
    // Device (upstream pycolmap/utils.h)
    // ------------------------------------------------------------------
    py::enum_<Device>(
            m, "Device",
            "The device selector for feature matching: AUTO uses the GPU "
            "when CUDA is available; CUDA/CPU force the device.")
            .value("AUTO", Device::AUTO)
            .value("CPU", Device::CPU)
            .value("CUDA", Device::CUDA)
            .export_values();

    // ------------------------------------------------------------------
    // PairingOptions (upstream pipeline/match_features.cc)
    // ------------------------------------------------------------------
    auto PyExhaustivePairingOptions =
            py::class_<colmap::ExhaustivePairingOptions>(
                    m, "ExhaustivePairingOptions")
                    .def(py::init<>())
                    .def_readwrite(
                            "block_size",
                            &colmap::ExhaustivePairingOptions::block_size)
                    .def("check", &colmap::ExhaustivePairingOptions::Check);

    auto PySpatialPairingOptions =
            py::class_<colmap::SpatialPairingOptions>(m,
                                                      "SpatialPairingOptions")
                    .def(py::init<>())
                    .def_readwrite("ignore_z",
                                   &colmap::SpatialPairingOptions::ignore_z,
                                   "Whether to ignore the Z-component of the "
                                   "location prior.")
                    .def_readwrite(
                            "max_num_neighbors",
                            &colmap::SpatialPairingOptions::max_num_neighbors,
                            "The maximum number of nearest neighbors to "
                            "match.")
                    .def_readwrite(
                            "min_num_neighbors",
                            &colmap::SpatialPairingOptions::min_num_neighbors,
                            "The minimum number of nearest neighbors to "
                            "match. Neighbors include those within "
                            "max_distance or to satisfy min_num_neighbors.")
                    .def_readwrite("max_distance",
                                   &colmap::SpatialPairingOptions::max_distance,
                                   "The maximum distance between the query and "
                                   "nearest neighbor [meters].")
                    .def_readwrite("num_threads",
                                   &colmap::SpatialPairingOptions::num_threads)
                    .def("check", &colmap::SpatialPairingOptions::Check);

    auto PyVocabTreePairingOptions =
            py::class_<colmap::VocabTreePairingOptions>(
                    m, "VocabTreePairingOptions")
                    .def(py::init<>())
                    .def_readwrite(
                            "num_images",
                            &colmap::VocabTreePairingOptions::num_images,
                            "Number of images to retrieve for each query "
                            "image.")
                    .def_readwrite(
                            "num_nearest_neighbors",
                            &colmap::VocabTreePairingOptions::
                                    num_nearest_neighbors,
                            "Number of nearest neighbors to retrieve per "
                            "query feature.")
                    .def_readwrite(
                            "num_checks",
                            &colmap::VocabTreePairingOptions::num_checks,
                            "Number of nearest-neighbor checks to use in "
                            "retrieval.")
                    .def_readwrite("num_images_after_verification",
                                   &colmap::VocabTreePairingOptions::
                                           num_images_after_verification,
                                   "How many images to return after spatial "
                                   "verification. Set to 0 to turn off spatial "
                                   "verification.")
                    .def_readwrite(
                            "max_num_features",
                            &colmap::VocabTreePairingOptions::max_num_features,
                            "The maximum number of features to use for "
                            "indexing an image.")
                    .def_readwrite(
                            "vocab_tree_path",
                            &colmap::VocabTreePairingOptions::vocab_tree_path,
                            "Path to the vocabulary tree.")
                    .def_readwrite(
                            "match_list_path",
                            &colmap::VocabTreePairingOptions::match_list_path,
                            "Optional path to file with specific image names "
                            "to match.")
                    .def_readwrite(
                            "num_threads",
                            &colmap::VocabTreePairingOptions::num_threads)
                    .def("check", &colmap::VocabTreePairingOptions::Check);

    auto PySequentialPairingOptions =
            py::class_<colmap::SequentialPairingOptions>(
                    m, "SequentialPairingOptions")
                    .def(py::init<>())
                    .def_readwrite("overlap",
                                   &colmap::SequentialPairingOptions::overlap,
                                   "Number of overlapping image pairs.")
                    .def_readwrite(
                            "quadratic_overlap",
                            &colmap::SequentialPairingOptions::
                                    quadratic_overlap,
                            "Whether to match images against their quadratic "
                            "neighbors.")
                    .def_readwrite(
                            "expand_rig_images",
                            &colmap::SequentialPairingOptions::
                                    expand_rig_images,
                            "Whether to match an image against all images in "
                            "neighboring rig frames. If no rigs/frames are "
                            "configured in the database, this option is "
                            "ignored.")
                    .def_readwrite(
                            "loop_detection",
                            &colmap::SequentialPairingOptions::loop_detection,
                            "Loop detection is invoked every "
                            "`loop_detection_period` images.")
                    .def_readwrite("loop_detection_period",
                                   &colmap::SequentialPairingOptions::
                                           loop_detection_period,
                                   "The frequency at which loop detection is "
                                   "triggered, in number of images.")
                    .def_readwrite(
                            "loop_detection_num_images",
                            &colmap::SequentialPairingOptions::
                                    loop_detection_num_images,
                            "The number of images to retrieve in loop "
                            "detection. This number should be significantly "
                            "larger than the sequential matching overlap.")
                    .def_readwrite(
                            "loop_detection_min_index_distance",
                            &colmap::SequentialPairingOptions::
                                    loop_detection_min_index_distance,
                            "The minimum image index distance between a loop "
                            "detection query and a retrieved image. The index "
                            "is determined by the sequential image order. "
                            "Set to 0 to disable this restriction.")
                    .def_readwrite(
                            "loop_detection_num_nearest_neighbors",
                            &colmap::SequentialPairingOptions::
                                    loop_detection_num_nearest_neighbors,
                            "Number of nearest neighbors to retrieve per "
                            "query feature.")
                    .def_readwrite(
                            "loop_detection_num_checks",
                            &colmap::SequentialPairingOptions::
                                    loop_detection_num_checks,
                            "Number of nearest-neighbor checks to use in "
                            "retrieval.")
                    .def_readwrite(
                            "loop_detection_num_images_after_verification",
                            &colmap::SequentialPairingOptions::
                                    loop_detection_num_images_after_verification,
                            "How many images to return after spatial "
                            "verification. Set to 0 to turn off spatial "
                            "verification.")
                    .def_readwrite(
                            "loop_detection_max_num_features",
                            &colmap::SequentialPairingOptions::
                                    loop_detection_max_num_features,
                            "The maximum number of features to use for "
                            "indexing an image. If an image has more "
                            "features, only the largest-scale features will "
                            "be indexed.")
                    .def_readwrite(
                            "vocab_tree_path",
                            &colmap::SequentialPairingOptions::vocab_tree_path,
                            "Path to the vocabulary tree.")
                    .def_readwrite(
                            "num_threads",
                            &colmap::SequentialPairingOptions::num_threads,
                            "Number of threads for loop detection indexing "
                            "and retrieval.")
                    .def("vocab_tree_options",
                         &colmap::SequentialPairingOptions::VocabTreeOptions)
                    .def("check", &colmap::SequentialPairingOptions::Check);

    auto PyImportedPairingOptions =
            py::class_<colmap::ImportedPairingOptions>(m,
                                                       "ImportedPairingOptions")
                    .def(py::init<>())
                    .def_readwrite(
                            "block_size",
                            &colmap::ImportedPairingOptions::block_size,
                            "Number of image pairs to match in one batch.")
                    .def_readwrite(
                            "match_list_path",
                            &colmap::ImportedPairingOptions::match_list_path,
                            "Path to the file with the matches.")
                    .def("check", &colmap::ImportedPairingOptions::Check);

    auto PyExistingMatchedPairingOptions =
            py::class_<colmap::ExistingMatchedPairingOptions>(
                    m, "ExistingMatchedPairingOptions")
                    .def(py::init<>())
                    .def_readwrite(
                            "batch_size",
                            &colmap::ExistingMatchedPairingOptions::batch_size)
                    .def("check",
                         &colmap::ExistingMatchedPairingOptions::Check);

    // Fork-only (upstream has no pycolmap transitive binding): the pairing
    // options of the transitive matcher consumed by feature.transitive_match.
    py::class_<colmap::TransitivePairingOptions>(m, "TransitivePairingOptions")
            .def(py::init<>())
            .def_readwrite("batch_size",
                           &colmap::TransitivePairingOptions::batch_size,
                           "The number of image pairs to match in one batch.")
            .def_readwrite("num_iterations",
                           &colmap::TransitivePairingOptions::num_iterations,
                           "The number of transitive closure iterations.")
            .def("check", &colmap::TransitivePairingOptions::Check);

    auto PyGeometricVerifierOptions =
            py::class_<colmap::GeometricVerifierOptions>(
                    m, "GeometricVerifierOptions",
                    "Options for geometric verification of existing matches.")
                    .def(py::init<>())
                    .def_readwrite(
                            "num_threads",
                            &colmap::GeometricVerifierOptions::num_threads)
                    .def_readwrite(
                            "rig_verification",
                            &colmap::GeometricVerifierOptions::rig_verification,
                            "Whether to perform rig verification at the end. "
                            "Unnecessary when we have existing relative poses "
                            "for each matched pair.")
                    .def_readwrite(
                            "use_existing_relative_pose",
                            &colmap::GeometricVerifierOptions::
                                    use_existing_relative_pose,
                            "Whether to use the existing relative pose stored "
                            "in TwoViewGeometry in the database. If no "
                            "TwoViewGeometry is found we will fall back to "
                            "geometric verification with RANSAC.");

    // ------------------------------------------------------------------
    // PairGenerator family (upstream pipeline/match_features.cc)
    // ------------------------------------------------------------------
    py::class_<colmap::PairGenerator>(m, "PairGenerator")
            .def("reset", &colmap::PairGenerator::Reset)
            .def("has_finished", &colmap::PairGenerator::HasFinished)
            .def("next", &colmap::PairGenerator::Next)
            .def("all_pairs", &colmap::PairGenerator::AllPairs);

    py::class_<colmap::ExhaustivePairGenerator, colmap::PairGenerator>(
            m, "ExhaustivePairGenerator")
            .def(py::init<const colmap::ExhaustivePairingOptions&,
                          const std::shared_ptr<colmap::Database>&>(),
                 "options"_a, "database"_a);

    py::class_<colmap::VocabTreePairGenerator, colmap::PairGenerator>(
            m, "VocabTreePairGenerator")
            .def(py::init<const colmap::VocabTreePairingOptions&,
                          const std::shared_ptr<colmap::Database>&,
                          const std::vector<colmap::image_t>&>(),
                 "options"_a, "database"_a,
                 "query_image_ids"_a = std::vector<colmap::image_t>());

    py::class_<colmap::SequentialPairGenerator, colmap::PairGenerator>(
            m, "SequentialPairGenerator")
            .def(py::init<const colmap::SequentialPairingOptions&,
                          const std::shared_ptr<colmap::Database>&>(),
                 "options"_a, "database"_a);

    py::class_<colmap::SpatialPairGenerator, colmap::PairGenerator>(
            m, "SpatialPairGenerator")
            .def(py::init<const colmap::SpatialPairingOptions&,
                          const std::shared_ptr<colmap::Database>&>(),
                 "options"_a, "database"_a);

    py::class_<colmap::ImportedPairGenerator, colmap::PairGenerator>(
            m, "ImportedPairGenerator")
            .def(py::init<const colmap::ImportedPairingOptions&,
                          const std::shared_ptr<colmap::Database>&>(),
                 "options"_a, "database"_a);

    // ------------------------------------------------------------------
    // Entry points (upstream pipeline/match_features.cc)
    // ------------------------------------------------------------------
    m.def("match_exhaustive",
          &MatchFeatures<colmap::ExhaustivePairingOptions,
                         colmap::CreateExhaustiveFeatureMatcher>,
          "database_path"_a,
          py::arg_v("matching_options", colmap::FeatureMatchingOptions(),
                    "FeatureMatchingOptions()"),
          py::arg_v("pairing_options", colmap::ExhaustivePairingOptions(),
                    "ExhaustivePairingOptions()"),
          py::arg_v("verification_options", colmap::TwoViewGeometryOptions(),
                    "TwoViewGeometryOptions()"),
          "device"_a = Device::AUTO, "cancellation_token"_a = py::none(),
          "Exhaustive feature matching");

    m.def("match_spatial",
          &MatchFeatures<colmap::SpatialPairingOptions,
                         colmap::CreateSpatialFeatureMatcher>,
          "database_path"_a,
          py::arg_v("matching_options", colmap::FeatureMatchingOptions(),
                    "FeatureMatchingOptions()"),
          py::arg_v("pairing_options", colmap::SpatialPairingOptions(),
                    "SpatialPairingOptions()"),
          py::arg_v("verification_options", colmap::TwoViewGeometryOptions(),
                    "TwoViewGeometryOptions()"),
          "device"_a = Device::AUTO, "cancellation_token"_a = py::none(),
          "Spatial feature matching");

    m.def("match_vocabtree",
          &MatchFeatures<colmap::VocabTreePairingOptions,
                         colmap::CreateVocabTreeFeatureMatcher>,
          "database_path"_a,
          py::arg_v("matching_options", colmap::FeatureMatchingOptions(),
                    "FeatureMatchingOptions()"),
          py::arg_v("pairing_options", colmap::VocabTreePairingOptions(),
                    "VocabTreePairingOptions()"),
          py::arg_v("verification_options", colmap::TwoViewGeometryOptions(),
                    "TwoViewGeometryOptions()"),
          "device"_a = Device::AUTO, "cancellation_token"_a = py::none(),
          "Vocab tree feature matching");

    m.def("match_sequential",
          &MatchFeatures<colmap::SequentialPairingOptions,
                         colmap::CreateSequentialFeatureMatcher>,
          "database_path"_a,
          py::arg_v("matching_options", colmap::FeatureMatchingOptions(),
                    "FeatureMatchingOptions()"),
          py::arg_v("pairing_options", colmap::SequentialPairingOptions(),
                    "SequentialPairingOptions()"),
          py::arg_v("verification_options", colmap::TwoViewGeometryOptions(),
                    "TwoViewGeometryOptions()"),
          "device"_a = Device::AUTO, "cancellation_token"_a = py::none(),
          "Sequential feature matching");

    m.def("match_image_pairs",
          &MatchFeatures<colmap::ImportedPairingOptions,
                         colmap::CreateImagePairsFeatureMatcher>,
          "database_path"_a,
          py::arg_v("matching_options", colmap::FeatureMatchingOptions(),
                    "FeatureMatchingOptions()"),
          py::arg_v("pairing_options", colmap::ImportedPairingOptions(),
                    "ImportedPairingOptions()"),
          py::arg_v("verification_options", colmap::TwoViewGeometryOptions(),
                    "TwoViewGeometryOptions()"),
          "device"_a = Device::AUTO, "cancellation_token"_a = py::none(),
          "Match features between image pairs specified in a file");

    m.def("verify_matches", &VerifyMatches, "database_path"_a, "pairs_path"_a,
          py::arg_v("options", colmap::TwoViewGeometryOptions(),
                    "TwoViewGeometryOptions()"),
          "cancellation_token"_a = py::none(),
          "Run geometric verification of the matches");

    m.def("geometric_verification", &GeometricVerification, "database_path"_a,
          py::arg_v("verifier_options", colmap::GeometricVerifierOptions(),
                    "GeometricVerifierOptions()"),
          py::arg_v("pairing_options", colmap::ExistingMatchedPairingOptions(),
                    "ExistingMatchedPairingOptions()"),
          py::arg_v("two_view_geometry_options",
                    colmap::TwoViewGeometryOptions(),
                    "TwoViewGeometryOptions()"),
          "cancellation_token"_a = py::none(),
          "Run geometric verification on all image pairs in the database");

    m.def("guided_geometric_verification", &GuidedGeometricVerification,
          "reconstruction"_a, "database_path"_a,
          py::arg_v("pairing_options", colmap::ExistingMatchedPairingOptions(),
                    "ExistingMatchedPairingOptions()"),
          py::arg_v("two_view_geometry_options",
                    colmap::TwoViewGeometryOptions(),
                    "TwoViewGeometryOptions()"),
          py::arg("num_threads") = -1, "cancellation_token"_a = py::none(),
          "Run geometric verification given an existing colmap reconstruction "
          "on all image pairs in the database");
}

}  // namespace pipeline
}  // namespace reconstruction
}  // namespace cloudViewer

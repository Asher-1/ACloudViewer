// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/sfm/mappers.h"

#include <sstream>
#include <unordered_map>

#include "estimators/bundle_adjustment.h"
#include "pybind/docstring.h"
#include "scene/correspondence_graph.h"
#include "scene/database_cache.h"
#include "scene/reconstruction.h"
#include "sfm/incremental_mapper.h"
#include "sfm/incremental_triangulator.h"
#include "sfm/observation_manager.h"
#include "util/hash_containers.h"

namespace cloudViewer {
namespace reconstruction {
namespace sfm {

// Upstream pycolmap parity (src/pycolmap/sfm/incremental_mapper.cc,
// incremental_triangulator.cc, observation_manager.cc): Open3D-style class
// bindings over the fork engine types (namespace colmap). The engine holds
// raw back-pointers (DatabaseCache/Reconstruction), so the bindings pin the
// Python-side lifetimes with keep_alive (verified by the smoke test).
using colmap::BundleAdjustmentOptions;
using colmap::DatabaseCache;
using colmap::IncrementalMapper;
using colmap::IncrementalTriangulator;
using colmap::ObservationManager;
using colmap::Reconstruction;
using colmap::Rigid3d;

// Owner wrappers: the engine types hold raw back-pointers
// (DatabaseCache/CorrespondenceGraph/Reconstruction), and py::init
// keep_alive does not pin Python-side lifetimes on this pybind build, so
// each wrapper holds py::object references that keep the Python owners
// alive for as long as the bound object lives.
struct PyObservationManager {
    py::object reconstruction_ref;
    std::unique_ptr<ObservationManager> manager;
};

struct PyTriangulator {
    py::object graph_ref;
    py::object reconstruction_ref;
    std::unique_ptr<IncrementalTriangulator> triangulator;
};

struct PyMapper {
    py::object cache_ref;
    py::object reconstruction_ref;
    std::unique_ptr<IncrementalMapper> mapper;
};

static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"options", "The incremental mapper options."},
                {"tri_options", "The incremental triangulator options."},
                {"ba_options", "The bundle adjustment options."},
                {"image_id", "The image identifier."},
};

void pybind_mappers(py::module& m) {
    py::class_<IncrementalMapper::LocalBundleAdjustmentReport> local_report(
            m, "LocalBundleAdjustmentReport",
            "Statistics of a local bundle adjustment round.");
    local_report.def(py::init<>())
            .def_readwrite("num_merged_observations",
                           &IncrementalMapper::LocalBundleAdjustmentReport::
                                   num_merged_observations)
            .def_readwrite("num_completed_observations",
                           &IncrementalMapper::LocalBundleAdjustmentReport::
                                   num_completed_observations)
            .def_readwrite("num_filtered_observations",
                           &IncrementalMapper::LocalBundleAdjustmentReport::
                                   num_filtered_observations)
            .def_readwrite("num_adjusted_observations",
                           &IncrementalMapper::LocalBundleAdjustmentReport::
                                   num_adjusted_observations);

    // The engine types hold raw back-pointers (DatabaseCache,
    // CorrespondenceGraph, Reconstruction); pin the Python-side lifetimes via
    // keep_alive (nurse index = the Python owner, patient = the borrowed
    // argument).
    py::class_<PyObservationManager> obs_manager(
            m, "ObservationManager",
            "Book-keeping for 3D point observations on top of a "
            "reconstruction and its correspondence graph.");
    obs_manager
            .def(py::init([](py::object reconstruction_obj,
                             py::object graph_obj) {
                     auto* reconstruction =
                             reconstruction_obj.cast<Reconstruction*>();
                     std::shared_ptr<const colmap::CorrespondenceGraph> graph =
                             graph_obj.is_none()
                                     ? nullptr
                                     : graph_obj.cast<std::shared_ptr<
                                               const colmap::
                                                       CorrespondenceGraph>>();
                     auto holder = new PyObservationManager();
                     holder->reconstruction_ref = reconstruction_obj;
                     holder->manager.reset(new ObservationManager(
                             *reconstruction, std::move(graph)));
                     return holder;
                 }),
                 "reconstruction"_a, "correspondence_graph"_a = py::none())
            .def_property_readonly(
                    "reconstruction",
                    [](PyObservationManager& self) -> Reconstruction& {
                        return self.manager->Reconstruction();
                    },
                    py::return_value_policy::reference_internal)
            .def(
                    "add_image",
                    [](PyObservationManager& self, colmap::image_t image_id) {
                        self.manager->AddImage(image_id);
                    },
                    "image_id"_a, "Add image stats for streaming/online SfM.")
            .def(
                    "add_point3D",
                    [](PyObservationManager& self, const Eigen::Vector3d& xyz,
                       const colmap::Track& track) {
                        return self.manager->AddPoint3D(xyz, track);
                    },
                    "xyz"_a, "track"_a,
                    "Add a new 3D point and return its unique ID.")
            .def(
                    "add_observation",
                    [](PyObservationManager& self, colmap::point3D_t point3D_id,
                       const colmap::TrackElement& track_el) {
                        self.manager->AddObservation(point3D_id, track_el);
                    },
                    "point3D_id"_a, "track_el"_a,
                    "Add an observation to an existing 3D point.")
            .def(
                    "delete_point3D",
                    [](PyObservationManager& self, colmap::point3D_t id) {
                        self.manager->DeletePoint3D(id);
                    },
                    "point3D_id"_a)
            .def(
                    "delete_observation",
                    [](PyObservationManager& self, colmap::image_t image_id,
                       colmap::point2D_t point2D_idx) {
                        self.manager->DeleteObservation(image_id, point2D_idx);
                    },
                    "image_id"_a, "point2D_idx"_a)
            .def(
                    "merge_points3D",
                    [](PyObservationManager& self, colmap::point3D_t id1,
                       colmap::point3D_t id2) {
                        return self.manager->MergePoints3D(id1, id2);
                    },
                    "point3D_id1"_a, "point3D_id2"_a)
            .def(
                    "filter_points3D",
                    [](PyObservationManager& self, double max_reproj_error,
                       double min_tri_angle,
                       const std::vector<colmap::point3D_t>& point3D_ids) {
                        return self.manager->FilterPoints3D(
                                max_reproj_error, min_tri_angle,
                                colmap::FlatHashSet<colmap::point3D_t>(
                                        point3D_ids.begin(),
                                        point3D_ids.end()));
                    },
                    "max_reproj_error"_a, "min_tri_angle"_a, "point3D_ids"_a)
            .def(
                    "filter_points3D_in_images",
                    [](PyObservationManager& self, double max_reproj_error,
                       double min_tri_angle,
                       const std::vector<colmap::image_t>& image_ids) {
                        return self.manager->FilterPoints3DInImages(
                                max_reproj_error, min_tri_angle,
                                colmap::FlatHashSet<colmap::image_t>(
                                        image_ids.begin(), image_ids.end()));
                    },
                    "max_reproj_error"_a, "min_tri_angle"_a, "image_ids"_a)
            .def(
                    "filter_all_points3D",
                    [](PyObservationManager& self, double max_reproj_error,
                       double min_tri_angle) {
                        return self.manager->FilterAllPoints3D(max_reproj_error,
                                                               min_tri_angle);
                    },
                    "max_reproj_error"_a, "min_tri_angle"_a)
            .def(
                    "filter_points3D_with_short_tracks",
                    [](PyObservationManager& self, size_t min_track_length) {
                        return self.manager->FilterPoints3DWithShortTracks(
                                min_track_length);
                    },
                    "min_track_length"_a)
            .def("filter_observations_with_negative_depth",
                 [](PyObservationManager& self) {
                     return self.manager->FilterObservationsWithNegativeDepth();
                 })
            .def(
                    "filter_points3D_with_small_triangulation_angle",
                    [](PyObservationManager& self, double min_tri_angle,
                       const std::vector<colmap::point3D_t>& point3D_ids) {
                        return self.manager
                                ->FilterPoints3DWithSmallTriangulationAngle(
                                        min_tri_angle,
                                        colmap::FlatHashSet<colmap::point3D_t>(
                                                point3D_ids.begin(),
                                                point3D_ids.end()));
                    },
                    "min_tri_angle"_a, "point3D_ids"_a);

    py::class_<PyTriangulator> triangulator(
            m, "IncrementalTriangulator",
            "Incremental triangulator on top of a correspondence graph and a "
            "reconstruction (both must outlive the triangulator).");
    triangulator
            .def(py::init([](py::object graph_obj,
                             py::object reconstruction_obj) {
                     auto* corr_graph = graph_obj.cast<
                             const colmap::CorrespondenceGraph*>();
                     auto* reconstruction =
                             reconstruction_obj.cast<Reconstruction*>();
                     auto holder = new PyTriangulator();
                     holder->graph_ref = graph_obj;
                     holder->reconstruction_ref = reconstruction_obj;
                     holder->triangulator.reset(new IncrementalTriangulator(
                             corr_graph, reconstruction));
                     return holder;
                 }),
                 "correspondence_graph"_a, "reconstruction"_a)
            .def(
                    "triangulate_image",
                    [](PyTriangulator& self,
                       const IncrementalTriangulator::Options& options,
                       colmap::image_t image_id) {
                        return self.triangulator->TriangulateImage(options,
                                                                   image_id);
                    },
                    "options"_a, "image_id"_a,
                    "Triangulate the observations of an image (create, "
                    "continue "
                    "and merge tracks); returns the number of created "
                    "observations.")
            .def(
                    "complete_image",
                    [](PyTriangulator& self,
                       const IncrementalTriangulator::Options& options,
                       colmap::image_t image_id) {
                        return self.triangulator->CompleteImage(options,
                                                                image_id);
                    },
                    "options"_a, "image_id"_a,
                    "Complete tracks for the observations of an image.")
            .def(
                    "complete_tracks",
                    [](PyTriangulator& self,
                       const IncrementalTriangulator::Options& options,
                       const std::vector<colmap::point3D_t>& point3D_ids) {
                        return self.triangulator->CompleteTracks(
                                options, std::unordered_set<colmap::point3D_t>(
                                                 point3D_ids.begin(),
                                                 point3D_ids.end()));
                    },
                    "options"_a, "point3D_ids"_a)
            .def(
                    "complete_all_tracks",
                    [](PyTriangulator& self,
                       const IncrementalTriangulator::Options& options) {
                        return self.triangulator->CompleteAllTracks(options);
                    },
                    "options"_a)
            .def(
                    "merge_tracks",
                    [](PyTriangulator& self,
                       const IncrementalTriangulator::Options& options,
                       const std::vector<colmap::point3D_t>& point3D_ids) {
                        return self.triangulator->MergeTracks(
                                options, std::unordered_set<colmap::point3D_t>(
                                                 point3D_ids.begin(),
                                                 point3D_ids.end()));
                    },
                    "options"_a, "point3D_ids"_a)
            .def(
                    "merge_all_tracks",
                    [](PyTriangulator& self,
                       const IncrementalTriangulator::Options& options) {
                        return self.triangulator->MergeAllTracks(options);
                    },
                    "options"_a)
            .def(
                    "retriangulate",
                    [](PyTriangulator& self,
                       const IncrementalTriangulator::Options& options) {
                        return self.triangulator->Retriangulate(options);
                    },
                    "options"_a,
                    "Retriangulate under-reconstructed image pairs.")
            .def(
                    "add_modified_point3D",
                    [](PyTriangulator& self, colmap::point3D_t point3D_id) {
                        self.triangulator->AddModifiedPoint3D(point3D_id);
                    },
                    "point3D_id"_a)
            .def("get_modified_points3D",
                 [](PyTriangulator& self) {
                     const auto& modified =
                             self.triangulator->GetModifiedPoints3D();
                     return std::vector<colmap::point3D_t>(modified.begin(),
                                                           modified.end());
                 })
            .def("clear_modified_points3D", [](PyTriangulator& self) {
                self.triangulator->ClearModifiedPoints3D();
            });

    py::class_<PyMapper> mapper(
            m, "IncrementalMapper",
            "All functionality for the incremental reconstruction "
            "procedure. The database cache must live for the entire "
            "life-time of the mapper.");
    mapper.def(py::init([](std::shared_ptr<const DatabaseCache> cache) {
                   auto holder = new PyMapper();
                   holder->cache_ref = py::cast(cache);
                   holder->mapper.reset(new IncrementalMapper(cache.get()));
                   return holder;
               }),
               "database_cache"_a,
               "Create the mapper; the database cache is retained for the "
               "mapper's life-time.")
            .def(
                    "begin_reconstruction",
                    [](PyMapper& self, py::object reconstruction_obj) {
                        self.reconstruction_ref = reconstruction_obj;
                        self.mapper->BeginReconstruction(
                                reconstruction_obj.cast<Reconstruction*>());
                    },
                    "reconstruction"_a,
                    "Prepare the mapper for a new reconstruction (the "
                    "reconstruction must outlive the mapper).")
            .def(
                    "end_reconstruction",
                    [](PyMapper& self, bool discard) {
                        self.mapper->EndReconstruction(discard);
                    },
                    "discard"_a,
                    "Cleanup the mapper after the current reconstruction.")
            .def(
                    "find_initial_image_pair",
                    [](PyMapper& self,
                       const IncrementalMapper::Options& options,
                       colmap::image_t image_id1,
                       colmap::image_t image_id2) -> py::object {
                        colmap::image_t id1 = image_id1;
                        colmap::image_t id2 = image_id2;
                        if (!self.mapper->FindInitialImagePair(options, &id1,
                                                               &id2)) {
                            return py::none();
                        }
                        return py::cast(std::make_pair(id1, id2));
                    },
                    "options"_a, "image_id1"_a, "image_id2"_a,
                    "Find the initial image pair; returns (image_id1, "
                    "image_id2) or None.")
            .def(
                    "register_initial_image_pair",
                    [](PyMapper& self,
                       const IncrementalMapper::Options& options,
                       colmap::image_t image_id1, colmap::image_t image_id2) {
                        return self.mapper->RegisterInitialImagePair(
                                options, image_id1, image_id2);
                    },
                    "options"_a, "image_id1"_a, "image_id2"_a,
                    "Attempt to seed the reconstruction from an image pair.")
            .def(
                    "find_next_images",
                    [](PyMapper& self,
                       const IncrementalMapper::Options& options,
                       bool structure_less) {
                        return self.mapper->FindNextImages(options,
                                                           structure_less);
                    },
                    "options"_a, "structure_less"_a = false,
                    "Find the best next images to register.")
            .def(
                    "register_next_image",
                    [](PyMapper& self,
                       const IncrementalMapper::Options& options,
                       colmap::image_t image_id) {
                        return self.mapper->RegisterNextImage(options,
                                                              image_id);
                    },
                    "options"_a, "image_id"_a,
                    "Attempt to register an image to the existing model.")
            .def(
                    "register_next_general_frame",
                    [](PyMapper& self,
                       const IncrementalMapper::Options& options,
                       colmap::Frame& frame) {
                        return self.mapper->RegisterNextGeneralFrame(options,
                                                                     frame);
                    },
                    "options"_a, "frame"_a,
                    "Attempt to register a multi-sensor frame via pooled "
                    "2D-3D correspondences and generalized absolute pose.")
            .def(
                    "register_next_structure_less_image",
                    [](PyMapper& self,
                       const IncrementalMapper::Options& options,
                       colmap::image_t image_id) {
                        return self.mapper->RegisterNextStructureLessImage(
                                options, image_id);
                    },
                    "options"_a, "image_id"_a,
                    "Attempt to register an image using structure-less "
                    "resectioning.")
            .def(
                    "triangulate_image",
                    [](PyMapper& self,
                       const IncrementalTriangulator::Options& tri_options,
                       colmap::image_t image_id) {
                        return self.mapper->TriangulateImage(tri_options,
                                                             image_id);
                    },
                    "tri_options"_a, "image_id"_a)
            .def(
                    "retriangulate",
                    [](PyMapper& self,
                       const IncrementalTriangulator::Options& tri_options) {
                        return self.mapper->Retriangulate(tri_options);
                    },
                    "tri_options"_a)
            .def(
                    "complete_tracks",
                    [](PyMapper& self,
                       const IncrementalTriangulator::Options& tri_options) {
                        return self.mapper->CompleteTracks(tri_options);
                    },
                    "tri_options"_a)
            .def(
                    "merge_tracks",
                    [](PyMapper& self,
                       const IncrementalTriangulator::Options& tri_options) {
                        return self.mapper->MergeTracks(tri_options);
                    },
                    "tri_options"_a)
            .def(
                    "complete_and_merge_tracks",
                    [](PyMapper& self,
                       const IncrementalTriangulator::Options& tri_options) {
                        return self.mapper->CompleteAndMergeTracks(tri_options);
                    },
                    "tri_options"_a)
            .def(
                    "adjust_local_bundle",
                    [](PyMapper& self,
                       const IncrementalMapper::Options& options,
                       const BundleAdjustmentOptions& ba_options,
                       const IncrementalTriangulator::Options& tri_options,
                       colmap::image_t image_id,
                       const std::vector<colmap::point3D_t>& point3D_ids) {
                        return self.mapper->AdjustLocalBundle(
                                options, ba_options, tri_options, image_id,
                                std::unordered_set<colmap::point3D_t>(
                                        point3D_ids.begin(),
                                        point3D_ids.end()));
                    },
                    "options"_a, "ba_options"_a, "tri_options"_a, "image_id"_a,
                    "point3D_ids"_a,
                    "Adjust locally connected images and points of a "
                    "reference image.")
            .def(
                    "adjust_global_bundle",
                    [](PyMapper& self,
                       const IncrementalMapper::Options& options,
                       const BundleAdjustmentOptions& ba_options) {
                        return self.mapper->AdjustGlobalBundle(options,
                                                               ba_options);
                    },
                    "options"_a, "ba_options"_a,
                    "Global bundle adjustment of the whole reconstruction.")
            .def(
                    "iterative_global_refinement",
                    [](PyMapper& self, int max_num_refinements,
                       double max_refinement_change,
                       const IncrementalMapper::Options& options,
                       const BundleAdjustmentOptions& ba_options,
                       const IncrementalTriangulator::Options& tri_options,
                       bool normalize_reconstruction) {
                        self.mapper->IterativeGlobalRefinement(
                                max_num_refinements, max_refinement_change,
                                options, ba_options, tri_options,
                                normalize_reconstruction);
                    },
                    "max_num_refinements"_a, "max_refinement_change"_a,
                    "options"_a, "ba_options"_a, "tri_options"_a,
                    "normalize_reconstruction"_a = true,
                    "Perform multiple rounds of global bundle adjustment.")
            .def(
                    "filter_images",
                    [](PyMapper& self,
                       const IncrementalMapper::Options& options) {
                        return self.mapper->FilterImages(options);
                    },
                    "options"_a,
                    "Filter images with degenerate camera parameters or no "
                    "observations.")
            .def(
                    "filter_points",
                    [](PyMapper& self,
                       const IncrementalMapper::Options& options) {
                        return self.mapper->FilterPoints(options);
                    },
                    "options"_a,
                    "Filter points with large reprojection errors or small "
                    "triangulation angles.")
            .def_property_readonly(
                    "reconstruction",
                    [](const PyMapper& self) -> const Reconstruction& {
                        return self.mapper->GetReconstruction();
                    },
                    py::return_value_policy::reference_internal)
            .def("num_total_reg_images",
                 [](const PyMapper& self) {
                     return self.mapper->NumTotalRegImages();
                 })
            .def("num_shared_reg_images",
                 [](const PyMapper& self) {
                     return self.mapper->NumSharedRegImages();
                 })
            .def_property_readonly("num_reg_frames_per_rig",
                                   [](const PyMapper& self) {
                                       std::unordered_map<uint32_t, size_t> out;
                                       for (const auto& [rig_id, num] :
                                            self.mapper->NumRegFramesPerRig()) {
                                           out.emplace(rig_id, num);
                                       }
                                       return out;
                                   })
            .def("get_modified_points3D",
                 [](PyMapper& self) {
                     const auto& modified = self.mapper->GetModifiedPoints3D();
                     return std::vector<colmap::point3D_t>(modified.begin(),
                                                           modified.end());
                 })
            .def("clear_modified_points3D",
                 [](PyMapper& self) { self.mapper->ClearModifiedPoints3D(); });

    m.attr("__docstring__") = map_shared_argument_docstrings;
}

}  // namespace sfm
}  // namespace reconstruction
}  // namespace cloudViewer

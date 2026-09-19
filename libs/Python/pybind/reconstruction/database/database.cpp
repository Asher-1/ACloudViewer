// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/database/database.h"

#include <memory>
#include <optional>
#include <sstream>
#include <unordered_map>

#include "geometry/pose_prior.h"
#include "pipelines/database.h"
#include "pybind/docstring.h"
#include "scene/camera.h"
#include "scene/database.h"
#include "scene/frame.h"
#include "scene/image.h"
#include "scene/rig.h"
#include "scene/two_view_geometry.h"

namespace cloudViewer {
namespace reconstruction {
namespace database {

// Reconstruction feature functions have similar arguments, sharing arg
// docstrings
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"database_path",
                 "Path to database in which to store the extracted data"},
                {"first_database_path",
                 "The first imported database directory."},
                {"second_database_path",
                 "The other imported database directory"},
                {"merged_database_path", "The merged database directory"},
                {"type", "supported type {all, images, features, matches}"}};

void pybind_database_methods(py::module& m) {
    m.def("clean_database", &CleanDatabase,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the clearance of database", "database_path"_a,
          "type"_a);
    docstring::FunctionDocInject(m, "clean_database",
                                 map_shared_argument_docstrings);

    m.def("create_database", &CreateDatabase,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the creation of database", "database_path"_a);
    docstring::FunctionDocInject(m, "create_database",
                                 map_shared_argument_docstrings);

    m.def("merge_database", &MergeDatabase,
          py::call_guard<py::gil_scoped_release>(),
          "Function for the merge between two databases",
          "first_database_path"_a, "second_database_path"_a,
          "merged_database_path"_a);
    docstring::FunctionDocInject(m, "merge_database",
                                 map_shared_argument_docstrings);
}

void pybind_database(py::module& m) {
    py::module m_submodule =
            m.def_submodule("database", "Reconstruction Database.");
    pybind_database_methods(m_submodule);

    // Upstream pycolmap parity (src/pycolmap/scene/database.cc): the Database
    // class as the standard programmatic interface. The fork's abstract
    // Database (W17.2b) is opened through the registered factories.
    using colmap::Camera;
    using colmap::camera_t;
    using colmap::Database;
    using colmap::FeatureDescriptors;
    using colmap::FeatureKeypoint;
    using colmap::FeatureKeypoints;
    using colmap::FeatureMatch;
    using colmap::FeatureMatches;
    using colmap::Frame;
    using colmap::frame_t;
    using colmap::Image;
    using colmap::image_t;
    using colmap::pose_prior_t;
    using colmap::PosePrior;
    using colmap::Rig;
    using colmap::rig_t;
    using colmap::TwoViewGeometry;

    // RAII transaction wrapper: the engine's DatabaseTransaction commits in
    // its destructor, which Python's GC would delay - so commit() releases it
    // deterministically (there is no rollback surface on the engine side).
    struct PyDatabaseTransaction {
        explicit PyDatabaseTransaction(Database* db)
            : tx_(new colmap::DatabaseTransaction(db)) {}
        void commit() { tx_.reset(); }
        // Held by unique_ptr: the engine's DatabaseTransaction is neither
        // movable nor copyable, so the wrapper cannot be returned by value.
        std::unique_ptr<colmap::DatabaseTransaction> tx_;
    };

    py::class_<PyDatabaseTransaction>(
            m_submodule, "DatabaseTransaction",
            "A scoped write transaction. Use as a context manager; the "
            "transaction commits on commit() or when leaving the with-block "
            "(rollback is not available on the engine surface).")
            .def(py::init([](Database* db) {
                     return new PyDatabaseTransaction(db);
                 }),
                 "database"_a)
            .def("commit", &PyDatabaseTransaction::commit,
                 "Commit (end) the transaction.")
            .def("__enter__",
                 [](PyDatabaseTransaction& self) -> PyDatabaseTransaction& {
                     return self;
                 })
            .def("__exit__",
                 [](PyDatabaseTransaction& self, const py::object&,
                    const py::object&, const py::object&) { self.commit(); });

    py::class_<Database, std::shared_ptr<Database>> db(
            m_submodule, "Database",
            "The abstract COLMAP database interface (SQLite-backed). Holds "
            "cameras, rigs, frames, images, features, matches, two-view "
            "geometries and pose priors.");
    db.def_static(
              "open",
              [](const std::string& path) { return Database::Open(path); },
              "path"_a,
              "Open (or create) the database at the given path; throws "
              "runtime_error on failure.")
            .def("close", &Database::Close, "Close the database handle.")
            .def_property_readonly("num_cameras", &Database::NumCameras)
            .def_property_readonly("num_rigs", &Database::NumRigs)
            .def_property_readonly("num_frames", &Database::NumFrames)
            .def_property_readonly("num_images", &Database::NumImages)
            .def_property_readonly("num_pose_priors", &Database::NumPosePriors)
            .def_property_readonly("num_keypoints", &Database::NumKeypoints)
            .def_property_readonly("num_descriptors", &Database::NumDescriptors)
            .def_property_readonly("num_matches", &Database::NumMatches)
            .def_property_readonly("num_inlier_matches",
                                   &Database::NumInlierMatches)
            .def_property_readonly("num_matched_image_pairs",
                                   &Database::NumMatchedImagePairs)
            .def_property_readonly("num_verified_image_pairs",
                                   &Database::NumVerifiedImagePairs)
            .def("exists_camera", &Database::ExistsCamera, "camera_id"_a)
            .def("exists_rig", &Database::ExistsRig, "rig_id"_a)
            .def("exists_frame", &Database::ExistsFrame, "frame_id"_a)
            .def("exists_image", &Database::ExistsImage, "image_id"_a)
            .def("exists_image_with_name", &Database::ExistsImageWithName,
                 "name"_a)
            .def("exists_pose_prior", &Database::ExistsPosePrior,
                 "pose_prior_id"_a)
            .def("read_camera", &Database::ReadCamera, "camera_id"_a)
            .def("read_all_cameras", &Database::ReadAllCameras)
            .def("read_rig", &Database::ReadRig, "rig_id"_a)
            .def("read_all_rigs", &Database::ReadAllRigs)
            .def("read_frame", &Database::ReadFrame, "frame_id"_a)
            .def("read_all_frames", &Database::ReadAllFrames)
            .def("read_image", &Database::ReadImage, "image_id"_a)
            .def("read_image_with_name", &Database::ReadImageWithName, "name"_a)
            .def("read_all_images", &Database::ReadAllImages)
            .def("read_pose_prior", &Database::ReadPosePrior, "pose_prior_id"_a)
            .def("read_all_pose_priors", &Database::ReadAllPosePriors)
            .def(
                    "read_keypoints",
                    [](const Database& self, image_t image_id) {
                        const FeatureKeypoints keypoints =
                                self.ReadKeypoints(image_id);
                        py::array_t<float> arr(
                                {static_cast<py::ssize_t>(keypoints.size()),
                                 py::ssize_t(6)});
                        auto buf = arr.mutable_unchecked<2>();
                        for (size_t i = 0; i < keypoints.size(); ++i) {
                            const FeatureKeypoint& kp = keypoints[i];
                            buf(i, 0) = kp.x;
                            buf(i, 1) = kp.y;
                            buf(i, 2) = kp.a11;
                            buf(i, 3) = kp.a12;
                            buf(i, 4) = kp.a21;
                            buf(i, 5) = kp.a22;
                        }
                        return arr;
                    },
                    "image_id"_a,
                    "Read keypoints as an (N, 6) float32 array (x, y, a11, "
                    "a12, a21, a22).")
            .def(
                    "write_keypoints",
                    [](const Database& self, image_t image_id,
                       const py::array_t<float, py::array::c_style |
                                                        py::array::forcecast>&
                               arr) {
                        THROW_CHECK_EQ(arr.ndim(), 2);
                        THROW_CHECK(arr.shape(1) == 2 || arr.shape(1) == 6);
                        auto buf = arr.unchecked<2>();
                        FeatureKeypoints keypoints(buf.shape(0));
                        for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
                            if (buf.shape(1) == 2) {
                                keypoints[i] =
                                        FeatureKeypoint(buf(i, 0), buf(i, 1));
                            } else {
                                keypoints[i] = FeatureKeypoint(
                                        buf(i, 0), buf(i, 1), buf(i, 2),
                                        buf(i, 3), buf(i, 4), buf(i, 5));
                            }
                        }
                        self.WriteKeypoints(image_id, keypoints);
                    },
                    "image_id"_a, "keypoints"_a,
                    "Write keypoints from an (N, 2) or (N, 6) float32 "
                    "array.")
            .def(
                    "read_descriptors",
                    [](const Database& self, image_t image_id) {
                        const FeatureDescriptors descriptors =
                                self.ReadDescriptors(image_id);
                        return descriptors;
                    },
                    "image_id"_a, "Read descriptors as an (N, D) uint8 array.")
            .def(
                    "write_descriptors",
                    [](const Database& self, image_t image_id,
                       const FeatureDescriptors& descriptors) {
                        self.WriteDescriptors(image_id, descriptors);
                    },
                    "image_id"_a, "descriptors"_a,
                    "Write descriptors from an (N, D) uint8 array.")
            .def(
                    "read_matches",
                    [](const Database& self, image_t image_id1,
                       image_t image_id2) {
                        const FeatureMatches matches =
                                self.ReadMatches(image_id1, image_id2);
                        py::array_t<uint32_t> arr(
                                {static_cast<py::ssize_t>(matches.size()),
                                 py::ssize_t(2)});
                        auto buf = arr.mutable_unchecked<2>();
                        for (size_t i = 0; i < matches.size(); ++i) {
                            buf(i, 0) = matches[i].point2D_idx1;
                            buf(i, 1) = matches[i].point2D_idx2;
                        }
                        return arr;
                    },
                    "image_id1"_a, "image_id2"_a,
                    "Read raw matches as an (N, 2) uint32 array.")
            .def(
                    "write_matches",
                    [](const Database& self, image_t image_id1,
                       image_t image_id2,
                       const py::array_t<uint32_t,
                                         py::array::c_style |
                                                 py::array::forcecast>& arr) {
                        THROW_CHECK_EQ(arr.ndim(), 2);
                        THROW_CHECK_EQ(arr.shape(1), 2);
                        auto buf = arr.unchecked<2>();
                        FeatureMatches matches(buf.shape(0));
                        for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
                            matches[i] = FeatureMatch(buf(i, 0), buf(i, 1));
                        }
                        self.WriteMatches(image_id1, image_id2, matches);
                    },
                    "image_id1"_a, "image_id2"_a, "matches"_a,
                    "Write raw matches from an (N, 2) uint32 array.")
            .def("delete_matches", &Database::DeleteMatches, "image_id1"_a,
                 "image_id2"_a)
            .def("delete_inlier_matches", &Database::DeleteInlierMatches,
                 "image_id1"_a, "image_id2"_a)
            .def("delete_two_view_geometry", &Database::DeleteTwoViewGeometry,
                 "image_id1"_a, "image_id2"_a)
            .def("read_two_view_geometry", &Database::ReadTwoViewGeometry,
                 "image_id1"_a, "image_id2"_a)
            .def("write_two_view_geometry", &Database::WriteTwoViewGeometry,
                 "image_id1"_a, "image_id2"_a, "two_view_geometry"_a)
            .def("update_two_view_geometry", &Database::UpdateTwoViewGeometry,
                 "image_id1"_a, "image_id2"_a, "two_view_geometry"_a)
            .def("write_camera", &Database::WriteCamera, "camera"_a,
                 "use_camera_id"_a = false,
                 "Add a camera and return its database identifier.")
            .def("update_camera", &Database::UpdateCamera, "camera"_a)
            .def("write_rig", &Database::WriteRig, "rig"_a,
                 "use_rig_id"_a = false)
            .def("update_rig", &Database::UpdateRig, "rig"_a)
            .def("write_frame", &Database::WriteFrame, "frame"_a,
                 "use_frame_id"_a = false)
            .def("update_frame", &Database::UpdateFrame, "frame"_a)
            .def("write_image", &Database::WriteImage, "image"_a,
                 "use_image_id"_a = false,
                 "Add an image and return its database identifier.")
            .def("update_image", &Database::UpdateImage, "image"_a)
            .def("write_pose_prior", &Database::WritePosePrior, "pose_prior"_a,
                 "use_pose_prior_id"_a = false)
            .def("update_pose_prior", &Database::UpdatePosePrior,
                 "pose_prior"_a)
            .def("clear_all_tables", &Database::ClearAllTables,
                 "Clear all database tables.")
            .def(
                    "transaction",
                    [](Database& self) { return PyDatabaseTransaction(&self); },
                    "Begin a scoped write transaction (use as a context "
                    "manager).",
                    py::keep_alive<0, 1>());
}

}  // namespace database
}  // namespace reconstruction
}  // namespace cloudViewer

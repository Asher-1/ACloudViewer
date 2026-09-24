// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/scene/scene.h"

#include "feature/types.h"
#include "geometry/rigid3.h"
#include "geometry/sim3.h"
#include "pybind/docstring.h"
#include "scene/camera.h"
#include "scene/correspondence_graph.h"
#include "scene/database.h"
#include "scene/database_cache.h"
#include "scene/frame.h"
#include "scene/image.h"
#include "scene/point2d.h"
#include "scene/point3d.h"
#include "scene/pose_graph.h"
#include "scene/reconstruction.h"
#include "scene/reconstruction_manager.h"
#include "scene/rig.h"
#include "scene/synthetic.h"
#include "scene/track.h"
#include "scene/two_view_geometry.h"
#include "sensor/models.h"

namespace cloudViewer {
namespace reconstruction {
namespace scene {

// The COLMAP fork engine types live in namespace colmap.
using colmap::Camera;
using colmap::camera_t;
using colmap::Database;
using colmap::DatabaseCache;
using colmap::Frame;
using colmap::frame_t;
using colmap::Image;
using colmap::image_t;
using colmap::Point2D;
using colmap::point2D_t;
using colmap::Point3D;
using colmap::point3D_t;
using colmap::Reconstruction;
using colmap::Rig;
using colmap::rig_t;
using colmap::Rigid3d;
using colmap::sensor_t;
using colmap::SensorType;
using colmap::Sim3d;
using colmap::SyntheticDatasetOptions;
using colmap::Track;
using colmap::TrackElement;
using colmap::TwoViewGeometry;

// Upstream pycolmap parity (src/pycolmap/scene): the core scene objects as
// plain Open3D-style class bindings. Point2D/Point3D/Track/Frame/Rig/
// Database etc. follow in later batches of this integration.
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"path",
                 "Path to the model folder containing cameras, images and "
                 "points3D files (binary or text)."},
                {"camera_id", "Identifier of the camera."},
                {"image_id", "Identifier of the image."},
};

void pybind_scene(py::module& m) {
    py::module m_scene = m.def_submodule("scene");

    py::class_<Camera> camera(m_scene, "Camera",
                              "Intrinsic camera parameters.");
    camera.def(py::init<>())
            .def_static(
                    "create_from_model_name",
                    [](camera_t camera_id, const std::string& model_name,
                       double focal_length, size_t width, size_t height) {
                        return Camera::CreateFromModelName(
                                camera_id, model_name, focal_length, width,
                                height);
                    },
                    "camera_id"_a, "model_name"_a, "focal_length"_a, "width"_a,
                    "height"_a, "Create a camera from its model name.")
            .def_property_readonly("camera_id", &Camera::CameraId)
            .def_property_readonly("model_id", &Camera::ModelId)
            .def_property_readonly("model_name", &Camera::ModelName)
            .def_property_readonly("width", &Camera::Width)
            .def_property_readonly("height", &Camera::Height)
            .def_property_readonly(
                    "focal_length", &Camera::MeanFocalLength,
                    "Mean focal length (safe for multi-focal models).")
            // Params has const/non-const overloads; bind the const getter.
            .def_property_readonly(
                    "params", [](const Camera& self) { return self.Params(); })
            .def("has_prior_focal_length", &Camera::HasPriorFocalLength,
                 "Whether the focal length was set from a prior.")
            .def(
                    "cam_from_img",
                    [](const Camera& self, const Eigen::Vector2d& point2D) {
                        return self.CamFromImg(point2D);
                    },
                    "point2D"_a,
                    "Project a raw image point to (virtual) normalized "
                    "camera coordinates; returns None if unprojectable.")
            .def(
                    "img_from_cam",
                    [](const Camera& self, const Eigen::Vector3d& camera_point,
                       bool check_cheirality) {
                        return self.ImgFromCam(camera_point, check_cheirality);
                    },
                    "camera_point"_a, "check_cheirality"_a = true,
                    "Project a camera-frame point to raw image "
                    "coordinates; returns None if behind the camera.")
            .def("__repr__", [](const Camera& self) {
                return "Camera(id=" + std::to_string(self.CameraId()) +
                       ", model=" + self.ModelName() +
                       ", width=" + std::to_string(self.Width()) +
                       ", height=" + std::to_string(self.Height()) + ")";
            });

    py::class_<Image> image(m_scene, "Image",
                            "Image with 2D feature points and a "
                            "reference to its camera.");
    image.def(py::init<>())
            // ImageId()/Name() have const/non-const overloads.
            .def_property(
                    "image_id",
                    [](const Image& self) { return self.ImageId(); },
                    [](Image& self, image_t id) { self.SetImageId(id); },
                    "Unique identifier of the image.")
            .def_property(
                    "name", [](const Image& self) { return self.Name(); },
                    [](Image& self, const std::string& name) {
                        self.SetName(name);
                    },
                    "Name of the image.")
            .def_property(
                    "camera_id",
                    [](const Image& self) { return self.CameraId(); },
                    [](Image& self, camera_t id) { self.SetCameraId(id); },
                    "Unique identifier of the camera.")
            .def_property_readonly(
                    "num_points2D",
                    [](const Image& self) { return self.NumPoints2D(); })
            .def_property_readonly(
                    "points2D",
                    [](const Image& self) { return self.Points2D(); },
                    "The 2D feature points (copies; Point2D carries no "
                    "back-pointers).")
            .def_property_readonly(
                    "num_points3D",
                    [](const Image& self) { return self.NumPoints3D(); })
            .def_property_readonly("has_pose", &Image::HasPose)
            .def_property_readonly(
                    "cam_from_world",
                    [](const Image& self) { return self.CamFromWorld(); },
                    "The rig-aware camera-from-world pose.")
            .def("__repr__", [](const Image& self) {
                return "Image(id=" + std::to_string(self.ImageId()) +
                       ", name=" + self.Name() + ")";
            });

    py::class_<Point2D> point2d(m_scene, "Point2D",
                                "A 2D point with an optional 3D point "
                                "association.");
    point2d.def(py::init<>())
            .def_property_readonly(
                    "xy", [](const Point2D& self) { return self.XY(); },
                    "The raw image coordinates.")
            .def_property_readonly(
                    "point3D_id",
                    [](const Point2D& self) { return self.Point3DId(); },
                    "The associated 3D point identifier (invalid if none).")
            .def("has_point3D", &Point2D::HasPoint3D,
                 "Whether this observation is part of a 3D point track.");

    py::class_<TrackElement> track_element(
            m_scene, "TrackElement",
            "One observation of a 3D point: (image_id, point2D_idx).");
    track_element.def(py::init<>())
            .def_readwrite("image_id", &TrackElement::image_id)
            .def_readwrite("point2D_idx", &TrackElement::point2D_idx);

    py::class_<Track> track(m_scene, "Track",
                            "The set of observations supporting a 3D point.");
    track.def(py::init<>())
            .def_property_readonly(
                    "length", [](const Track& self) { return self.Length(); })
            .def_property_readonly("elements", [](const Track& self) {
                return self.Elements();
            });

    py::class_<Point3D> point3d(m_scene, "Point3D",
                                "A triangulated 3D point with color and "
                                "track.");
    point3d.def(py::init<>())
            .def_property_readonly(
                    "xyz", [](const Point3D& self) { return self.XYZ(); })
            .def_property_readonly(
                    "color", [](const Point3D& self) { return self.Color(); })
            .def_property_readonly(
                    "error", [](const Point3D& self) { return self.Error(); })
            .def_property_readonly(
                    "track", [](const Point3D& self) { return self.Track(); });

    // Sensor/data identifier primitives (upstream src/pycolmap/scene).
    py::enum_<SensorType> sensor_type(m_scene, "SensorType",
                                      "Type of a rig sensor.");
    sensor_type.value("INVALID", SensorType::INVALID)
            .value("CAMERA", SensorType::CAMERA)
            .value("IMU", SensorType::IMU);

    py::class_<sensor_t> sensor_id(m_scene, "SensorId",
                                   "Typed sensor identifier (type + id).");
    sensor_id.def(py::init<>())
            .def(py::init([](SensorType type, uint32_t id) {
                     return sensor_t(type, id);
                 }),
                 "type"_a, "id"_a)
            .def_readwrite("type", &sensor_t::type)
            .def_readwrite("id", &sensor_t::id)
            .def("__repr__", [](const sensor_t& self) {
                return "SensorId(type=" +
                       std::to_string(static_cast<int>(self.type)) +
                       ", id=" + std::to_string(self.id) + ")";
            });

    py::class_<colmap::data_t> data_id(
            m_scene, "DataId", "Typed data identifier (sensor + data).");
    data_id.def(py::init<>())
            .def(py::init([](SensorType type, uint32_t sensor_id_value,
                             uint64_t data_id_value) {
                     return colmap::data_t(sensor_t(type, sensor_id_value),
                                           data_id_value);
                 }),
                 "type"_a, "sensor_id"_a, "data_id"_a)
            .def_readwrite("sensor_id", &colmap::data_t::sensor_id)
            .def_readwrite("id", &colmap::data_t::id)
            .def("__repr__", [](const colmap::data_t& self) {
                return "DataId(sensor=(" +
                       std::to_string(static_cast<int>(self.sensor_id.type)) +
                       ", " + std::to_string(self.sensor_id.id) +
                       "), id=" + std::to_string(self.id) + ")";
            });

    py::class_<Frame> frame(m_scene, "Frame",
                            "One simultaneous capture of a rig, owning an "
                            "optional rig-from-world pose.");
    frame.def(py::init<>())
            .def_property(
                    "frame_id",
                    [](const Frame& self) { return self.FrameId(); },
                    [](Frame& self, frame_t id) { self.SetFrameId(id); },
                    "Unique identifier of the frame.")
            .def_property(
                    "rig_id", [](const Frame& self) { return self.RigId(); },
                    [](Frame& self, rig_t id) { self.SetRigId(id); },
                    "Unique identifier of the rig.")
            .def("has_rig_id", &Frame::HasRigId, "Whether the rig_id is set.")
            .def("add_data_id", &Frame::AddDataId, "data_id"_a,
                 "Associate data with the frame.")
            .def("has_data", &Frame::HasDataId, "data_id"_a,
                 "Whether the frame has associated data.")
            .def("num_data_ids", &Frame::NumDataIds,
                 "Number of associated data items.")
            .def("clear_data_ids", &Frame::ClearDataIds,
                 "Clear all associated data.")
            .def("add_image_id", &Frame::AddImageId, "image_id"_a,
                 "Associate an image with the frame (legacy camera-data "
                 "adapter).")
            .def("has_image_id", &Frame::HasImageId, "image_id"_a)
            .def_property_readonly(
                    "image_ids",
                    [](const Frame& self) { return self.ImageIds(); },
                    "The associated image identifiers.")
            .def_property_readonly(
                    "data_ids",
                    [](const Frame& self) { return self.DataIds(); },
                    "The associated data identifiers.")
            .def_property(
                    "rig_from_world",
                    [](const Frame& self) { return self.RigFromWorld(); },
                    [](Frame& self, const Rigid3d& pose) {
                        self.SetRigFromWorld(pose);
                    },
                    "The rig-from-world pose.")
            .def("has_pose", &Frame::HasPose, "Whether the pose is set.")
            .def("reset_pose", &Frame::ResetPose, "Clear the pose.")
            .def("sensor_from_world", &Frame::SensorFromWorld, "sensor_id"_a,
                 "The sensor-from-world pose of the given sensor (requires "
                 "the frame to be wired to its rig).")
            .def("__repr__", [](const Frame& self) {
                return "Frame(frame_id=" + std::to_string(self.FrameId()) +
                       ", rig_id=" + std::to_string(self.RigId()) + ")";
            });

    py::class_<Rig> rig(m_scene, "Rig",
                        "A rig of sensors with optional extrinsics.");
    rig.def(py::init<>())
            .def_property(
                    "rig_id", [](const Rig& self) { return self.RigId(); },
                    [](Rig& self, rig_t id) { self.SetRigId(id); },
                    "Unique identifier of the rig.")
            .def("add_ref_sensor", &Rig::AddRefSensor, "sensor_id"_a,
                 "Add a reference sensor whose pose defines the rig pose.")
            .def(
                    "add_sensor",
                    [](Rig& self, const sensor_t& sensor_id,
                       std::optional<Eigen::Vector4d> qvec,
                       std::optional<Eigen::Vector3d> tvec) {
                        self.AddSensor(sensor_id, qvec, tvec);
                    },
                    "sensor_id"_a, "qvec"_a = py::none(), "tvec"_a = py::none(),
                    "Add a non-reference sensor with an optional "
                    "sensor-from-rig pose given as (w, x, y, z) quaternion.")
            .def("has_sensor", &Rig::HasSensor, "sensor_id"_a)
            .def("num_sensors", &Rig::NumSensors,
                 "The total number of sensors.")
            .def("is_ref_sensor", &Rig::IsRefSensor, "sensor_id"_a,
                 "Whether the sensor is the reference sensor.")
            .def_property_readonly(
                    "ref_sensor_id",
                    [](const Rig& self) { return self.RefSensorId(); })
            .def("non_ref_sensors", &Rig::NonRefSensors,
                 "The non-reference sensors with their optional "
                 "sensor-from-rig poses.")
            .def("sensor_ids", &Rig::SensorIds, "All sensor identifiers.")
            .def("sensor_from_rig", &Rig::SensorFromRig, "sensor_id"_a,
                 "The sensor-from-rig pose of the given sensor.")
            .def(
                    "set_sensor_from_rig",
                    [](Rig& self, const sensor_t& sensor_id,
                       std::optional<Rigid3d> sensor_from_rig) {
                        self.SetSensorFromRig(sensor_id, sensor_from_rig);
                    },
                    "sensor_id"_a, "sensor_from_rig"_a = py::none(),
                    "Set (or reset with None) the sensor-from-rig pose of a "
                    "non-reference sensor.")
            .def("__repr__", [](const Rig& self) {
                return "Rig(rig_id=" + std::to_string(self.RigId()) +
                       ", num_sensors=" + std::to_string(self.NumSensors()) +
                       ")";
            });

    py::class_<TwoViewGeometry> two_view_geometry(
            m_scene, "TwoViewGeometry",
            "Two-view geometry between an image pair (E/F/H, relative pose, "
            "inlier matches).");
    py::enum_<TwoViewGeometry::ConfigurationType>(two_view_geometry,
                                                  "ConfigurationType")
            .value("UNDEFINED", TwoViewGeometry::ConfigurationType::UNDEFINED)
            .value("DEGENERATE", TwoViewGeometry::ConfigurationType::DEGENERATE)
            .value("CALIBRATED", TwoViewGeometry::ConfigurationType::CALIBRATED)
            .value("CALIBRATED_RIG",
                   TwoViewGeometry::ConfigurationType::CALIBRATED_RIG)
            .value("UNCALIBRATED",
                   TwoViewGeometry::ConfigurationType::UNCALIBRATED)
            .value("PLANAR", TwoViewGeometry::ConfigurationType::PLANAR)
            .value("PANORAMIC", TwoViewGeometry::ConfigurationType::PANORAMIC)
            .value("PLANAR_OR_PANORAMIC",
                   TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC)
            .value("WATERMARK", TwoViewGeometry::ConfigurationType::WATERMARK)
            .value("MULTIPLE", TwoViewGeometry::ConfigurationType::MULTIPLE)
            .export_values();
    two_view_geometry.def(py::init<>())
            .def_readwrite("config", &TwoViewGeometry::config)
            .def_property_readonly(
                    "E",
                    [](const TwoViewGeometry& self) -> py::object {
                        return self.E ? py::cast(*self.E) : py::none();
                    })
            .def_property_readonly(
                    "F",
                    [](const TwoViewGeometry& self) -> py::object {
                        return self.F ? py::cast(*self.F) : py::none();
                    })
            .def_property_readonly(
                    "H",
                    [](const TwoViewGeometry& self) -> py::object {
                        return self.H ? py::cast(*self.H) : py::none();
                    })
            .def_property_readonly(
                    "cam2_from_cam1",
                    [](const TwoViewGeometry& self) -> py::object {
                        return self.cam2_from_cam1
                                       ? py::cast(*self.cam2_from_cam1)
                                       : py::none();
                    })
            .def_property_readonly(
                    "camera1",
                    [](const TwoViewGeometry& self) -> py::object {
                        return self.camera1 ? py::cast(*self.camera1)
                                            : py::none();
                    })
            .def_property_readonly(
                    "camera2",
                    [](const TwoViewGeometry& self) -> py::object {
                        return self.camera2 ? py::cast(*self.camera2)
                                            : py::none();
                    })
            .def_property_readonly(
                    "tri_angle",
                    [](const TwoViewGeometry& self) { return self.tri_angle; })
            .def_property_readonly(
                    "inlier_matches",
                    [](const TwoViewGeometry& self) {
                        py::array_t<uint32_t> arr(
                                {static_cast<py::ssize_t>(
                                         self.inlier_matches.size()),
                                 py::ssize_t(2)});
                        auto buf = arr.mutable_unchecked<2>();
                        for (size_t i = 0; i < self.inlier_matches.size();
                             ++i) {
                            buf(i, 0) = self.inlier_matches[i].point2D_idx1;
                            buf(i, 1) = self.inlier_matches[i].point2D_idx2;
                        }
                        return arr;
                    },
                    "The inlier matches as an (N, 2) uint32 array of "
                    "point2D indices.")
            .def("invert", [](TwoViewGeometry& self) { self.Invert(); });

    py::class_<SyntheticDatasetOptions> synthetic_options(
            m_scene, "SyntheticDatasetOptions",
            "Options for synthetic dataset generation.");
    synthetic_options.def(py::init<>())
            .def_readwrite("num_rigs", &SyntheticDatasetOptions::num_rigs)
            .def_readwrite("num_cameras_per_rig",
                           &SyntheticDatasetOptions::num_cameras_per_rig)
            .def_readwrite("num_frames_per_rig",
                           &SyntheticDatasetOptions::num_frames_per_rig)
            .def_readwrite("num_points3D",
                           &SyntheticDatasetOptions::num_points3D)
            .def_readwrite("track_length",
                           &SyntheticDatasetOptions::track_length)
            .def_readwrite("sensor_from_rig_translation_stddev",
                           &SyntheticDatasetOptions::
                                   sensor_from_rig_translation_stddev)
            .def_readwrite(
                    "sensor_from_rig_rotation_stddev",
                    &SyntheticDatasetOptions::sensor_from_rig_rotation_stddev)
            .def_readwrite("camera_width",
                           &SyntheticDatasetOptions::camera_width)
            .def_readwrite("camera_height",
                           &SyntheticDatasetOptions::camera_height)
            .def_readwrite(
                    "camera_has_prior_focal_length",
                    &SyntheticDatasetOptions::camera_has_prior_focal_length)
            .def_readwrite(
                    "num_points2D_without_point3D",
                    &SyntheticDatasetOptions::num_points2D_without_point3D)
            .def_readwrite("inlier_match_ratio",
                           &SyntheticDatasetOptions::inlier_match_ratio)
            .def_readwrite("prior_position",
                           &SyntheticDatasetOptions::prior_position)
            .def_readwrite("prior_gravity",
                           &SyntheticDatasetOptions::prior_gravity);

    m_scene.def(
            "synthesize_dataset",
            [](const SyntheticDatasetOptions& options,
               Reconstruction& reconstruction, Database* database) {
                colmap::SynthesizeDataset(options, &reconstruction, database);
            },
            "options"_a, "reconstruction"_a, "database"_a = py::none(),
            "Synthesize a synthetic dataset into a reconstruction and "
            "optionally a database.");

    py::class_<DatabaseCache, std::shared_ptr<DatabaseCache>> database_cache(
            m_scene, "DatabaseCache",
            "An in-memory cache of the database for fast access during "
            "mapping.");
    py::class_<DatabaseCache::Options> cache_options(
            database_cache, "Options", "Database cache loading options.");
    cache_options.def(py::init<>())
            .def_readwrite("min_num_matches",
                           &DatabaseCache::Options::min_num_matches)
            .def_readwrite("ignore_watermarks",
                           &DatabaseCache::Options::ignore_watermarks)
            .def_readwrite("load_all_images",
                           &DatabaseCache::Options::load_all_images)
            .def_readwrite("convert_pose_priors_to_enu",
                           &DatabaseCache::Options::convert_pose_priors_to_enu)
            .def_property(
                    "image_names",
                    [](const DatabaseCache::Options& self) {
                        return std::vector<std::string>(
                                self.image_names.begin(),
                                self.image_names.end());
                    },
                    [](DatabaseCache::Options& self,
                       const std::vector<std::string>& names) {
                        self.image_names = colmap::FlatHashSet<std::string>(
                                names.begin(), names.end());
                    });
    database_cache
            .def_static(
                    "create",
                    [](const Database& database,
                       const DatabaseCache::Options& options) {
                        return DatabaseCache::Create(database, options);
                    },
                    "database"_a, "options"_a = DatabaseCache::Options(),
                    "Load a database cache with the given options.")
            .def_property_readonly("num_rigs", &DatabaseCache::NumRigs)
            .def_property_readonly("num_cameras", &DatabaseCache::NumCameras)
            .def_property_readonly("num_frames", &DatabaseCache::NumFrames)
            .def_property_readonly("num_images", &DatabaseCache::NumImages)
            .def_property_readonly("num_pose_priors",
                                   &DatabaseCache::NumPosePriors)
            .def("exists_image", &DatabaseCache::ExistsImage, "image_id"_a)
            .def("exists_camera", &DatabaseCache::ExistsCamera, "camera_id"_a)
            .def("exists_frame", &DatabaseCache::ExistsFrame, "frame_id"_a)
            .def_property_readonly(
                    "correspondence_graph",
                    [](DatabaseCache& self) {
                        return self.CorrespondenceGraph();
                    },
                    "The correspondence graph (shared with the cache).")
            .def(
                    "image",
                    [](DatabaseCache& self, image_t image_id) -> Image& {
                        return self.Image(image_id);
                    },
                    "image_id"_a,
                    "Get the image with the identifier (reference into the "
                    "cache).",
                    py::return_value_policy::reference_internal)
            .def(
                    "camera",
                    [](DatabaseCache& self, camera_t camera_id) -> Camera& {
                        return self.Camera(camera_id);
                    },
                    "camera_id"_a, "Get the camera with the identifier.",
                    py::return_value_policy::reference_internal);

    py::class_<Reconstruction> reconstruction(
            m_scene, "Reconstruction",
            "Incremental reconstruction container (cameras, images, 3D "
            "points).");
    reconstruction.def(py::init<>())
            .def(py::init<const Reconstruction&>(), "other"_a,
                 "Copy the reconstruction (registration state included).")
            .def(
                    "read",
                    [](Reconstruction& self, const std::string& path) {
                        self.Read(path);
                    },
                    "path"_a,
                    "Load a reconstruction from binary/text model files.")
            .def(
                    "write",
                    [](const Reconstruction& self, const std::string& path) {
                        self.Write(path);
                    },
                    "path"_a, "Write the reconstruction as binary model files.")
            .def(
                    "register_image",
                    [](Reconstruction& self, image_t image_id) {
                        self.RegisterImage(image_id);
                    },
                    "image_id"_a,
                    "Register an image (keeps the frame-level set in "
                    "lockstep).")
            .def(
                    "deregister_image",
                    [](Reconstruction& self, image_t image_id) {
                        self.DeRegisterImage(image_id);
                    },
                    "image_id"_a,
                    "De-register an image and clean up its observations.")
            .def(
                    "register_frame",
                    [](Reconstruction& self, frame_t frame_id) {
                        self.RegisterFrame(frame_id);
                    },
                    "frame_id"_a,
                    "Register a posed frame (upstream parity; idempotent).")
            .def(
                    "deregister_frame",
                    [](Reconstruction& self, frame_t frame_id) {
                        self.DeRegisterFrame(frame_id);
                    },
                    "frame_id"_a,
                    "De-register a frame: clean up observations of all its "
                    "images and reset the pose (the RA outlier-filter path).")
            .def_property_readonly("num_images", &Reconstruction::NumImages)
            .def_property_readonly("num_reg_images",
                                   &Reconstruction::NumRegImages)
            .def_property_readonly("num_cameras", &Reconstruction::NumCameras)
            .def_property_readonly("num_points3D", &Reconstruction::NumPoints3D)
            .def_property_readonly("num_rigs", &Reconstruction::NumRigs)
            .def_property_readonly("num_frames", &Reconstruction::NumFrames)
            .def_property_readonly("num_reg_frames",
                                   &Reconstruction::NumRegFrames)
            .def("exists_image", &Reconstruction::ExistsImage, "image_id"_a,
                 "Whether an image with the identifier exists.")
            .def("exists_camera", &Reconstruction::ExistsCamera, "camera_id"_a)
            .def("exists_rig", &Reconstruction::ExistsRig, "rig_id"_a)
            .def("exists_frame", &Reconstruction::ExistsFrame, "frame_id"_a)
            .def("exists_point3D", &Reconstruction::ExistsPoint3D,
                 "point3D_id"_a)
            .def(
                    "image",
                    [](Reconstruction& self, image_t image_id) -> Image& {
                        return self.Image(image_id);
                    },
                    "image_id"_a,
                    "Get the image with the identifier (reference into the "
                    "reconstruction; the copy would drop the frame "
                    "back-pointers that carry the pose).",
                    py::return_value_policy::reference_internal)
            .def(
                    "camera",
                    [](Reconstruction& self, camera_t camera_id) -> Camera& {
                        return self.Camera(camera_id);
                    },
                    "camera_id"_a,
                    "Get the camera with the identifier (reference into the "
                    "reconstruction).",
                    py::return_value_policy::reference_internal)
            .def(
                    "mean_track_length",
                    [](const Reconstruction& self) {
                        return self.ComputeMeanTrackLength();
                    },
                    "Mean length of the 3D point tracks.")
            .def(
                    "point3D",
                    [](const Reconstruction& self, point3D_t point3D_id) {
                        return self.Point3D(point3D_id);
                    },
                    "point3D_id"_a,
                    "Get a copy of the 3D point with the identifier.",
                    py::return_value_policy::copy)
            .def(
                    "frame",
                    [](Reconstruction& self, frame_t frame_id) -> Frame& {
                        return self.Frame(frame_id);
                    },
                    "frame_id"_a,
                    "Get the frame with the identifier (reference into the "
                    "reconstruction).",
                    py::return_value_policy::reference_internal)
            .def(
                    "rig",
                    [](Reconstruction& self, rig_t rig_id) -> Rig& {
                        return self.Rig(rig_id);
                    },
                    "rig_id"_a,
                    "Get the rig with the identifier (reference into the "
                    "reconstruction).",
                    py::return_value_policy::reference_internal)
            .def_property_readonly("reg_image_ids",
                                   &Reconstruction::RegImageIds,
                                   "The registered image identifiers.")
            .def_property_readonly(
                    "reg_frame_ids",
                    [](const Reconstruction& self) {
                        return self.RegFrameIds();
                    },
                    "The registered frame identifiers.")
            .def_property_readonly(
                    "images",
                    [](Reconstruction& self) {
                        py::dict out;
                        for (auto& [image_id, image] : self.Images()) {
                            // reference (not reference_internal): the
                            // keep_alive parent cannot be activated for dict
                            // elements; the reconstruction must outlive the
                            // dict, as documented.
                            out[py::cast(image_id)] = py::cast(
                                    &image, py::return_value_policy::reference);
                        }
                        return out;
                    },
                    "The images as a dict keyed by image id (references into "
                    "the reconstruction).",
                    py::return_value_policy::reference_internal)
            .def_property_readonly(
                    "cameras",
                    [](Reconstruction& self) {
                        py::dict out;
                        for (auto& [camera_id, camera] : self.Cameras()) {
                            out[py::cast(camera_id)] = py::cast(
                                    &camera,
                                    py::return_value_policy::reference);
                        }
                        return out;
                    },
                    "The cameras as a dict keyed by camera id (references "
                    "into the reconstruction).",
                    py::return_value_policy::reference_internal)
            .def_property_readonly(
                    "points3D_map",
                    [](const Reconstruction& self) { return self.Points3D(); },
                    "The 3D points as a dict keyed by point id (copies; "
                    "Point3D carries no back-pointers).")
            .def(
                    "sim3d_transform",
                    [](Reconstruction& self, const Sim3d& tform) {
                        self.Transform(tform);
                    },
                    "tform"_a,
                    "Apply a similarity transform to the reconstruction.");

    // Upstream pycolmap parity (src/pycolmap/scene/correspondence_graph.cc):
    // the scene graph used by incremental mapping. The shared_ptr holder is
    // required: DatabaseCache owns the graph through a shared_ptr, so any
    // Python view of it must share (not take) ownership.
    py::class_<colmap::CorrespondenceGraph,
               std::shared_ptr<colmap::CorrespondenceGraph>>
            corr_graph(m_scene, "CorrespondenceGraph",
                       "The correspondence graph between feature "
                       "observations.");
    py::class_<colmap::CorrespondenceGraph::Correspondence> correspondence(
            corr_graph, "Correspondence",
            "One correspondence: (image_id, point2D_idx).");
    correspondence.def(py::init<>())
            .def(py::init<colmap::image_t, colmap::point2D_t>(), "image_id"_a,
                 "point2D_idx"_a)
            .def_readwrite(
                    "image_id",
                    &colmap::CorrespondenceGraph::Correspondence::image_id)
            .def_readwrite(
                    "point2D_idx",
                    &colmap::CorrespondenceGraph::Correspondence::point2D_idx);
    corr_graph.def(py::init<>())
            .def("finalize", &colmap::CorrespondenceGraph::Finalize,
                 "Flatten the internal storage for fast queries; must be "
                 "called after all image pairs were added.")
            .def_property_readonly("num_images",
                                   &colmap::CorrespondenceGraph::NumImages)
            .def_property_readonly("num_image_pairs",
                                   &colmap::CorrespondenceGraph::NumImagePairs)
            .def("num_observations_for_image",
                 &colmap::CorrespondenceGraph::NumObservationsForImage,
                 "image_id"_a)
            .def("num_correspondences_for_image",
                 &colmap::CorrespondenceGraph::NumCorrespondencesForImage,
                 "image_id"_a)
            .def("num_matches_between_images",
                 &colmap::CorrespondenceGraph::NumMatchesBetweenImages,
                 "image_id1"_a, "image_id2"_a)
            .def("num_matches_between_all_images",
                 [](const colmap::CorrespondenceGraph& self) {
                     std::unordered_map<uint64_t, uint32_t> out;
                     for (const auto& [pair_id, num_matches] :
                          self.NumMatchesBetweenAllImages()) {
                         out.emplace(pair_id, num_matches);
                     }
                     return out;
                 })
            .def("exists_image", &colmap::CorrespondenceGraph::ExistsImage,
                 "image_id"_a)
            .def("image_pairs", &colmap::CorrespondenceGraph::ImagePairs)
            .def("add_image", &colmap::CorrespondenceGraph::AddImage,
                 "image_id"_a, "num_points2D"_a)
            .def(
                    "add_two_view_geometry",
                    [](colmap::CorrespondenceGraph& self,
                       colmap::image_t image_id1, colmap::image_t image_id2,
                       colmap::TwoViewGeometry two_view_geometry) {
                        self.AddTwoViewGeometry(image_id1, image_id2,
                                                std::move(two_view_geometry));
                    },
                    "image_id1"_a, "image_id2"_a, "two_view_geometry"_a,
                    "Add an image pair edge; invalid or duplicate matches are "
                    "ignored with a warning.")
            .def(
                    "find_correspondences",
                    [](const colmap::CorrespondenceGraph& self,
                       colmap::image_t image_id,
                       colmap::point2D_t point2D_idx) {
                        std::vector<colmap::CorrespondenceGraph::Correspondence>
                                corrs;
                        self.ExtractCorrespondences(image_id, point2D_idx,
                                                    &corrs);
                        return corrs;
                    },
                    "image_id"_a, "point2D_idx"_a)
            .def(
                    "extract_transitive_correspondences",
                    [](const colmap::CorrespondenceGraph& self,
                       colmap::image_t image_id, colmap::point2D_t point2D_idx,
                       size_t transitivity) {
                        std::vector<colmap::CorrespondenceGraph::Correspondence>
                                corrs;
                        self.ExtractTransitiveCorrespondences(
                                image_id, point2D_idx, transitivity, &corrs);
                        return corrs;
                    },
                    "image_id"_a, "point2D_idx"_a, "transitivity"_a)
            .def(
                    "extract_matches_between_images",
                    [](const colmap::CorrespondenceGraph& self,
                       colmap::image_t image_id1, colmap::image_t image_id2) {
                        colmap::FeatureMatches matches;
                        self.ExtractMatchesBetweenImages(image_id1, image_id2,
                                                         matches);
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
                    "Matches between two images as an (N, 2) uint32 array.")
            .def(
                    "extract_two_view_geometry",
                    [](const colmap::CorrespondenceGraph& self,
                       colmap::image_t image_id1, colmap::image_t image_id2,
                       bool extract_inlier_matches) {
                        return self.ExtractTwoViewGeometry(
                                image_id1, image_id2, extract_inlier_matches);
                    },
                    "image_id1"_a, "image_id2"_a,
                    "extract_inlier_matches"_a = true)
            .def("has_correspondences",
                 &colmap::CorrespondenceGraph::HasCorrespondences, "image_id"_a,
                 "point2D_idx"_a)
            .def("is_two_view_observation",
                 &colmap::CorrespondenceGraph::IsTwoViewObservation,
                 "image_id"_a, "point2D_idx"_a);

    // Upstream pycolmap parity (src/pycolmap/scene/pose_graph.cc).
    py::class_<colmap::PoseGraph> pose_graph(m_scene, "PoseGraph",
                                             "A graph of relative poses.");
    py::class_<colmap::PoseGraph::Edge> pose_edge(pose_graph, "Edge",
                                                  "A relative pose edge.");
    pose_edge.def(py::init<>())
            .def(py::init<const Rigid3d&>(), "cam2_from_cam1"_a)
            .def_readwrite("cam2_from_cam1",
                           &colmap::PoseGraph::Edge::cam2_from_cam1)
            .def_readwrite("num_matches", &colmap::PoseGraph::Edge::num_matches)
            .def_readwrite("valid", &colmap::PoseGraph::Edge::valid)
            .def("invert", &colmap::PoseGraph::Edge::Invert,
                 "Invert the edge to match swapped image order.");
    pose_graph.def(py::init<>())
            .def_property_readonly(
                    "edges",
                    [](colmap::PoseGraph& self) {
                        std::unordered_map<uint64_t, colmap::PoseGraph::Edge>
                                out;
                        for (const auto& [pair_id, edge] : self.Edges()) {
                            out.emplace(pair_id, edge);
                        }
                        return out;
                    })
            .def_property_readonly("num_edges", &colmap::PoseGraph::NumEdges)
            .def_property_readonly("empty", &colmap::PoseGraph::Empty)
            .def("clear", &colmap::PoseGraph::Clear)
            .def(
                    "add_edge",
                    [](colmap::PoseGraph& self, colmap::image_t image_id1,
                       colmap::image_t image_id2,
                       const colmap::PoseGraph::Edge& edge) {
                        self.AddEdge(image_id1, image_id2, edge);
                    },
                    "image_id1"_a, "image_id2"_a, "edge"_a)
            .def("has_edge", &colmap::PoseGraph::HasEdge, "image_id1"_a,
                 "image_id2"_a)
            .def("get_edge", &colmap::PoseGraph::GetEdge, "image_id1"_a,
                 "image_id2"_a)
            .def("delete_edge", &colmap::PoseGraph::DeleteEdge, "image_id1"_a,
                 "image_id2"_a)
            .def(
                    "update_edge",
                    [](colmap::PoseGraph& self, colmap::image_t image_id1,
                       colmap::image_t image_id2,
                       const colmap::PoseGraph::Edge& edge) {
                        self.UpdateEdge(image_id1, image_id2, edge);
                    },
                    "image_id1"_a, "image_id2"_a, "edge"_a)
            .def(
                    "load",
                    [](colmap::PoseGraph& self,
                       const colmap::CorrespondenceGraph& corr_graph) {
                        self.Load(corr_graph);
                    },
                    "corr_graph"_a,
                    "Load the pose graph from a correspondence graph (the "
                    "relative poses of verified two-view geometries).");

    // Upstream pycolmap parity (src/pycolmap/scene/reconstruction_manager.cc).
    py::class_<colmap::ReconstructionManager> reconstruction_manager(
            m_scene, "ReconstructionManager",
            "A container for reconstructions (e.g. multiple models).");
    reconstruction_manager.def(py::init<>())
            .def_property_readonly("size", &colmap::ReconstructionManager::Size)
            .def(
                    "get",
                    [](colmap::ReconstructionManager& self,
                       size_t idx) -> Reconstruction& { return self.Get(idx); },
                    "idx"_a,
                    "Get the reconstruction with the given index (reference "
                    "into the manager).",
                    py::return_value_policy::reference_internal)
            .def("add", &colmap::ReconstructionManager::Add,
                 "Add a new empty reconstruction and return its index.")
            .def("delete", &colmap::ReconstructionManager::Delete, "idx"_a)
            .def("clear", &colmap::ReconstructionManager::Clear)
            .def(
                    "read",
                    [](colmap::ReconstructionManager& self,
                       const std::string& path) { return self.Read(path); },
                    "path"_a,
                    "Read a reconstruction from the path and return its "
                    "index.")
            .def(
                    "write",
                    [](const colmap::ReconstructionManager& self,
                       const std::string& path) { self.Write(path, nullptr); },
                    "path"_a,
                    "Write all reconstructions into sub-folders 0, 1, 2, "
                    "...");

    m_scene.attr("__docstring__") = map_shared_argument_docstrings;
}

}  // namespace scene
}  // namespace reconstruction
}  // namespace cloudViewer

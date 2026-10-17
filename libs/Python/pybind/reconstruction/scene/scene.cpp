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
#include "pybind/reconstruction/enum_ctor.h"
#include "pybind/reconstruction/hash_container_casters.h"
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
#include "sfm/observation_manager.h"
#include "util/ply.h"

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

    // Upstream pycolmap parity (src/pycolmap/scene/camera.cc): the camera
    // model enumeration, constructible from its name and implicitly from
    // the integer model id.
    py::enum_<colmap::CameraModelId> camera_model_id(
            m_scene, "CameraModelId",
            "The camera model family. Values carry the model names; the "
            "integer values match the upstream CameraModelId ordering.");
    camera_model_id.value("INVALID", colmap::CameraModelId::kInvalid);
#define CAMERA_MODEL_CASE(CameraModel)                             \
    camera_model_id.value(colmap::CameraModel::model_name.c_str(), \
                          colmap::CameraModel::model_id);

    CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE
    AddStringToEnumConstructor(camera_model_id);
    py::implicitly_convertible<int, colmap::CameraModelId>();

    // Upstream pycolmap parity (f1ea91d0 src/pycolmap/scene/camera.cc): the
    // calibration-source enum, constructible from its name and implicitly
    // from the persisted integer value.
    py::enum_<colmap::CameraSource> py_camera_source(m_scene, "CameraSource");
    py_camera_source.value("BEST", colmap::CameraSource::BEST)
            .value("UNKNOWN", colmap::CameraSource::UNKNOWN)
            .value("USER", colmap::CameraSource::USER)
            .value("GUESS", colmap::CameraSource::GUESS)
            .value("EXIF", colmap::CameraSource::EXIF)
            .value("SINGLE_VIEW", colmap::CameraSource::SINGLE_VIEW)
            .value("VIEW_GRAPH", colmap::CameraSource::VIEW_GRAPH);
    AddStringToEnumConstructor(py_camera_source);
    py::implicitly_convertible<int, colmap::CameraSource>();

    py::class_<Camera> camera(m_scene, "Camera",
                              "Intrinsic camera parameters.");
    camera.def(py::init<>())
            .def_static(
                    "create_from_model_name",
                    [](camera_t camera_id, const std::string& model_name,
                       double focal_length, size_t width, size_t height,
                       colmap::CameraSource source) {
                        return Camera::CreateFromModelName(
                                camera_id, model_name, focal_length, width,
                                height, source);
                    },
                    "camera_id"_a, "model_name"_a, "focal_length"_a, "width"_a,
                    "height"_a, "source"_a = colmap::CameraSource::UNKNOWN,
                    "Create a camera from its model name.")
            .def_property(
                    "camera_id",
                    [](const Camera& self) { return self.CameraId(); },
                    [](Camera& self, camera_t id) { self.SetCameraId(id); },
                    "Unique identifier of the camera.")
            .def_property_readonly(
                    "model_id",
                    [](const Camera& self) {
                        return static_cast<int>(self.ModelId());
                    },
                    "The CameraModelId value as an integer.")
            .def_property_readonly("model_name", &Camera::ModelName)
            .def_property(
                    "model",
                    [](const Camera& self) {
                        return static_cast<int>(self.ModelId());
                    },
                    [](Camera& self, int model_id) {
                        self.SetModelId(
                                static_cast<colmap::CameraModelId>(model_id));
                    },
                    "The camera model as the integer CameraModelId value "
                    "(upstream binds this as an enum; prefer "
                    "create_from_model_name / model_name for symbolic use).")
            .def_property(
                    "width", [](const Camera& self) { return self.Width(); },
                    [](Camera& self, size_t width) { self.SetWidth(width); },
                    "Image width in pixels.")
            .def_property(
                    "height", [](const Camera& self) { return self.Height(); },
                    [](Camera& self, size_t height) { self.SetHeight(height); },
                    "Image height in pixels.")
            .def_property(
                    "focal_length",
                    [](const Camera& self) { return self.FocalLength(); },
                    [](Camera& self, double f) { self.SetFocalLength(f); },
                    "The model-canonical focal length (first focal "
                    "parameter).")
            .def("mean_focal_length", &Camera::MeanFocalLength,
                 "Mean focal length (safe for multi-focal models).")
            .def_property(
                    "focal_length_x",
                    [](const Camera& self) { return self.FocalLengthX(); },
                    [](Camera& self, double f) { self.SetFocalLengthX(f); },
                    "Focal length along x.")
            .def_property(
                    "focal_length_y",
                    [](const Camera& self) { return self.FocalLengthY(); },
                    [](Camera& self, double f) { self.SetFocalLengthY(f); },
                    "Focal length along y.")
            // Upstream pycolmap parity (f1ea91d0 src/pycolmap/scene/camera.cc):
            // the calibration source is the single source of truth; the
            // has_prior_focal_length setter is a deprecated compatibility
            // shim mapping to USER/GUESS.
            .def_property(
                    "source", [](const Camera& self) { return self.Source(); },
                    [](Camera& self, colmap::CameraSource source) {
                        if (source == colmap::CameraSource::BEST) {
                            throw py::value_error(
                                    "Camera.source cannot be BEST. BEST is a "
                                    "database query selector, not a valid "
                                    "camera source.");
                        }
                        self.SetSource(source);
                    },
                    "Origin/source of the camera calibration.")
            .def_property(
                    "has_prior_focal_length",
                    [](const Camera& self) {
                        return self.HasPriorFocalLength();
                    },
                    [](Camera& self, bool has_prior) {
                        PyErr_WarnEx(PyExc_DeprecationWarning,
                                     "Camera.has_prior_focal_length setter is "
                                     "deprecated, set Camera.source directly "
                                     "instead.",
                                     1);
                        self.SetSource(has_prior ? colmap::CameraSource::USER
                                                 : colmap::CameraSource::GUESS);
                    })
            // Params has const/non-const overloads; bind the const getter
            // and a setter.
            .def_property(
                    "params", [](const Camera& self) { return self.Params(); },
                    [](Camera& self, const std::vector<double>& params) {
                        self.SetParams(params);
                    },
                    "The raw intrinsic parameters vector.")
            .def_property(
                    "has_prior_focal_length",
                    [](const Camera& self) {
                        return self.HasPriorFocalLength();
                    },
                    [](Camera& self, bool prior) {
                        self.SetPriorFocalLength(prior);
                    },
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
            .def(
                    "cam_ray_from_img",
                    [](const Camera& self, const Eigen::Vector2d& point2D) {
                        return self.CamRayFromImg(point2D);
                    },
                    "point2D"_a,
                    "Unproject a pixel to a unit calibrated bearing; "
                    "returns None if unprojectable.")
            .def(
                    "cam_from_img_threshold",
                    [](const Camera& self, double threshold) {
                        return self.CamFromImgThreshold(threshold);
                    },
                    "threshold"_a,
                    "Convert a pixel-space threshold to camera space.")
            .def(
                    "calibration_matrix",
                    [](const Camera& self) { return self.CalibrationMatrix(); },
                    "The 3x3 calibration matrix K.")
            .def(
                    "focal_length_idxs",
                    [](const Camera& self) { return self.FocalLengthIdxs(); },
                    "Indices of the focal length parameters in params.")
            .def(
                    "principal_point_idxs",
                    [](const Camera& self) {
                        return self.PrincipalPointIdxs();
                    },
                    "Indices of the principal point parameters in params.")
            .def(
                    "extra_params_idxs",
                    [](const Camera& self) { return self.ExtraParamsIdxs(); },
                    "Indices of the extra (distortion) parameters in params.")
            // NOTE: upstream also exposes metadata_params_idxs; the fork
            // Camera does not carry metadata parameters yet (engine gap,
            // recorded in the alignment manifest).
            .def(
                    "metadata_params_idxs",
                    [](const Camera& self) {
                        const auto idxs = self.MetaDataParamsIdxs();
                        return std::vector<size_t>(idxs.begin(), idxs.end());
                    },
                    "Indices of metadata parameters in params (empty for "
                    "perspective models; only spherical upstream models "
                    "carry metadata, none of which are supported by this "
                    "fork).")
            .def("is_perspective", &Camera::IsPerspective,
                 "Whether the model is a perspective model.")
            .def("is_perspective_pinhole", &Camera::IsPerspectivePinhole,
                 "Whether the model is a perspective pinhole model.")
            .def("is_perspective_fisheye", &Camera::IsPerspectiveFisheye,
                 "Whether the model is a perspective fisheye model.")
            .def("is_spherical", &Camera::IsSpherical,
                 "Whether the model is a spherical model.")
            .def("is_undistorted", &Camera::IsUndistorted,
                 "Whether the model applies no distortion.")
            .def(
                    "has_bogus_params",
                    // Fork signature (upstream d3ccaf35 switched to
                    // min_focal_length_ratio + min_principal_point_ratio).
                    [](const Camera& self, double min_focal_length_ratio,
                       double max_focal_length_ratio, double max_extra_param) {
                        return self.HasBogusParams(min_focal_length_ratio,
                                                   max_focal_length_ratio,
                                                   max_extra_param);
                    },
                    "min_focal_length_ratio"_a, "max_focal_length_ratio"_a,
                    "max_extra_param"_a,
                    "Whether the parameters are bogus under the given "
                    "sanity bounds.")
            .def(
                    "rescale",
                    [](Camera& self, double scale) { self.Rescale(scale); },
                    "scale"_a, "Rescale the camera dimensions by a factor.")
            .def(
                    "rescale",
                    [](Camera& self, size_t width, size_t height) {
                        self.Rescale(width, height);
                    },
                    "width"_a, "height"_a,
                    "Rescale the camera to the given dimensions.")
            .def(
                    "params_to_string",
                    [](const Camera& self) { return self.ParamsToString(); },
                    "The parameters as a comma-separated string.")
            .def(
                    "set_params_from_string",
                    [](Camera& self, const std::string& params) {
                        return self.SetParamsFromString(params);
                    },
                    "params"_a,
                    "Set the parameters from a comma-separated string; "
                    "returns whether parsing succeeded.")
            .def("verify_params", &Camera::VerifyParams,
                 "Whether the parameters satisfy the model's validity "
                 "domain.")
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
            .def_property(
                    "cam_from_world",
                    [](const Image& self) { return self.CamFromWorld(); },
                    [](Image& self, const Rigid3d& pose) {
                        self.SetCamFromWorld(pose);
                    },
                    "The rig-aware camera-from-world pose.")
            .def("set_cam_from_world", &Image::SetCamFromWorld,
                 "cam_from_world"_a,
                 "Set the rig-aware camera-from-world pose.")
            .def_property_readonly(
                    "frame_id",
                    [](const Image& self) { return self.FrameId(); },
                    "Identifier of the owning frame (invalid if unset).")
            .def_property_readonly(
                    "data_id", [](const Image& self) { return self.DataId(); },
                    "The frame data identifier of this image.")
            .def("has_frame_id", &Image::HasFrameId,
                 "Whether the image is assigned to a frame.")
            .def("has_camera_id", &Image::HasCameraId,
                 "Whether the camera identifier has been set.")
            .def(
                    "has_frame_ptr",
                    [](const Image& self) { return self.HasFramePtr(); },
                    "Whether the back-pointer to the owning frame is wired.")
            .def(
                    "has_camera_ptr",
                    [](const Image& self) { return self.HasCameraPtr(); },
                    "Whether the back-pointer to the camera is wired.")
            .def("is_ref_in_frame", &Image::IsRefInFrame,
                 "Whether this image's camera is the frame's reference "
                 "sensor.")
            .def(
                    "point2D",
                    [](const Image& self,
                       point2D_t point2D_idx) -> const Point2D& {
                        return self.Point2D(point2D_idx);
                    },
                    "point2D_idx"_a, "Access the 2D point at this index.",
                    py::return_value_policy::reference_internal)
            .def(
                    "set_point3D_for_point2D",
                    [](Image& self, point2D_t point2D_idx,
                       point3D_t point3D_id) {
                        self.SetPoint3DForPoint2D(point2D_idx, point3D_id);
                    },
                    "point2D_idx"_a, "point3D_id"_a,
                    "Associate a 3D point with the 2D point at this index.")
            .def(
                    "reset_point3D_for_point2D",
                    [](Image& self, point2D_t point2D_idx) {
                        self.ResetPoint3DForPoint2D(point2D_idx);
                    },
                    "point2D_idx"_a,
                    "Clear the 3D point association of the 2D point at "
                    "this index.")
            .def("has_point3D", &Image::HasPoint3D, "point3D_id"_a,
                 "Whether the 3D point is observed in this image.")
            .def("projection_center", &Image::ProjectionCenter,
                 "The optical center in the world frame (requires pose).")
            .def("viewing_direction", &Image::ViewingDirection,
                 "The viewing direction in the world frame (requires "
                 "pose).")
            .def("reset_camera_ptr", &Image::ResetCameraPtr,
                 "Clear the back-pointer to the camera.")
            .def("reset_frame_ptr", &Image::ResetFramePtr,
                 "Clear the back-pointer to the owning frame.")
            .def("__repr__", [](const Image& self) {
                return "Image(id=" + std::to_string(self.ImageId()) +
                       ", name=" + self.Name() + ")";
            });

    py::class_<Point2D> point2d(m_scene, "Point2D",
                                "A 2D point with an optional 3D point "
                                "association.");
    point2d.def(py::init<>())
            .def(py::init([](const Eigen::Vector2d& xy, point3D_t point3D_id) {
                     Point2D p;
                     p.SetXY(xy);
                     p.SetPoint3DId(point3D_id);
                     return p;
                 }),
                 "xy"_a, "point3D_id"_a,
                 "Construct from image coordinates and an optional 3D "
                 "point association.")
            .def_property(
                    "xy", [](const Point2D& self) { return self.XY(); },
                    [](Point2D& self, const Eigen::Vector2d& xy) {
                        self.SetXY(xy);
                    },
                    "The raw image coordinates.")
            .def(
                    "x", [](const Point2D& self) { return self.XY()[0]; },
                    "Raw x coordinate.")
            .def(
                    "y", [](const Point2D& self) { return self.XY()[1]; },
                    "Raw y coordinate.")
            .def_property(
                    "point3D_id",
                    [](const Point2D& self) { return self.Point3DId(); },
                    [](Point2D& self, point3D_t id) { self.SetPoint3DId(id); },
                    "The associated 3D point identifier (invalid if none).")
            .def("has_point3D", &Point2D::HasPoint3D,
                 "Whether this observation is part of a 3D point track.");

    py::class_<TrackElement> track_element(
            m_scene, "TrackElement",
            "One observation of a 3D point: (image_id, point2D_idx).");
    track_element.def(py::init<>())
            .def(py::init<image_t, point2D_t>(), "image_id"_a, "point2D_idx"_a)
            .def_readwrite("image_id", &TrackElement::image_id)
            .def_readwrite("point2D_idx", &TrackElement::point2D_idx)
            .def("__repr__", [](const TrackElement& self) {
                return "TrackElement(image_id=" +
                       std::to_string(self.image_id) +
                       ", point2D_idx=" + std::to_string(self.point2D_idx) +
                       ")";
            });

    py::class_<Track> track(m_scene, "Track",
                            "The set of observations supporting a 3D point.");
    track.def(py::init<>())
            .def(py::init([](const std::vector<TrackElement>& elements) {
                     Track t;
                     t.SetElements(elements);
                     return t;
                 }),
                 "elements"_a, "Construct a track from observation elements.")
            .def_property(
                    "elements",
                    [](const Track& self) { return self.Elements(); },
                    [](Track& self, const std::vector<TrackElement>& elements) {
                        self.SetElements(elements);
                    },
                    "The observation elements (copies).")
            .def_property_readonly(
                    "length", [](const Track& self) { return self.Length(); })
            .def(
                    "add_element",
                    [](Track& self, image_t image_id, point2D_t point2D_idx) {
                        self.AddElement(image_id, point2D_idx);
                    },
                    "image_id"_a, "point2D_idx"_a,
                    "Add an observation to the track.")
            .def(
                    "add_element",
                    [](Track& self, const TrackElement& element) {
                        self.AddElement(element);
                    },
                    "element"_a, "Add an observation to the track.")
            .def(
                    "add_elements",
                    [](Track& self, const std::vector<TrackElement>& elements) {
                        self.AddElements(elements);
                    },
                    "elements"_a, "Add multiple elements.")
            .def(
                    "delete_element",
                    [](Track& self, image_t image_id, point2D_t point2D_idx) {
                        self.DeleteElement(image_id, point2D_idx);
                    },
                    "image_id"_a, "point2D_idx"_a,
                    "Delete an observation from the track.")
            .def(
                    "delete_element",
                    [](Track& self, size_t idx) { self.DeleteElement(idx); },
                    "index"_a, "Remove the TrackElement at index.")
            .def(
                    "element",
                    [](Track& self, size_t idx) -> TrackElement& {
                        return self.Element(idx);
                    },
                    "index"_a, "Access the element at index.",
                    py::return_value_policy::reference_internal)
            .def(
                    "set_element",
                    [](Track& self, size_t idx, const TrackElement& element) {
                        self.SetElement(idx, element);
                    },
                    "index"_a, "element"_a, "Set the element at index.")
            .def(
                    "reserve",
                    [](Track& self, size_t num_elements) {
                        self.Reserve(num_elements);
                    },
                    "num_elements"_a, "Reserve capacity for elements.")
            .def("compress", &Track::Compress,
                 "Shrink the element capacity to fit its size.");

    py::class_<Point3D> point3d(m_scene, "Point3D",
                                "A triangulated 3D point with color and "
                                "track.");
    point3d.def(py::init<>())
            .def_property(
                    "xyz", [](const Point3D& self) { return self.XYZ(); },
                    [](Point3D& self, const Eigen::Vector3d& xyz) {
                        self.SetXYZ(xyz);
                    },
                    "The 3D coordinates in the world frame.")
            .def_property(
                    "color", [](const Point3D& self) { return self.Color(); },
                    [](Point3D& self, const Eigen::Vector3ub& color) {
                        self.SetColor(color);
                    },
                    "The RGB color (0-255).")
            .def_property(
                    "error", [](const Point3D& self) { return self.Error(); },
                    [](Point3D& self, double error) { self.SetError(error); },
                    "The mean reprojection error in image space.")
            .def_property(
                    "track", [](const Point3D& self) { return self.Track(); },
                    [](Point3D& self, const Track& track) {
                        self.SetTrack(track);
                    },
                    "The supporting observations.")
            .def("has_error", &Point3D::HasError,
                 "Check if the reprojection error has been computed.");

    // Sensor/data identifier primitives (upstream src/pycolmap/scene).
    py::enum_<SensorType> sensor_type(m_scene, "SensorType",
                                      "Type of a rig sensor.");
    sensor_type.value("INVALID", SensorType::INVALID)
            .value("CAMERA", SensorType::CAMERA)
            .value("IMU", SensorType::IMU);

    // Upstream pycolmap parity (src/pycolmap/util/types.cc): the engine's
    // sensor/data identifier structs under their upstream python names.
    py::class_<sensor_t> sensor_id(m_scene, "sensor_t",
                                   "Typed sensor identifier (type + id).");
    sensor_id.def(py::init<>())
            .def(py::init([](SensorType type, uint32_t id) {
                     return sensor_t(type, id);
                 }),
                 "type"_a, "id"_a)
            .def_readwrite("type", &sensor_t::type)
            .def_readwrite("id", &sensor_t::id)
            .def("__repr__", [](const sensor_t& self) {
                return "sensor_t(type=" +
                       std::to_string(static_cast<int>(self.type)) +
                       ", id=" + std::to_string(self.id) + ")";
            });

    py::class_<colmap::data_t> data_id(
            m_scene, "data_t", "Typed data identifier (sensor + data).");
    data_id.def(py::init<>())
            .def(py::init([](SensorType type, uint32_t sensor_id_value,
                             uint64_t data_id_value) {
                     return colmap::data_t(sensor_t(type, sensor_id_value),
                                           data_id_value);
                 }),
                 "type"_a, "sensor_id"_a, "data_id"_a)
            .def_readwrite("sensor_id", &colmap::data_t::sensor_id)
            .def_readwrite("id", &colmap::data_t::id)
            .def("__repr__",
                 [](const colmap::data_t& self) {
                     return "data_t(sensor=(" +
                            std::to_string(
                                    static_cast<int>(self.sensor_id.type)) +
                            ", " + std::to_string(self.sensor_id.id) +
                            "), id=" + std::to_string(self.id) + ")";
                 })
            .def(py::init([](const sensor_t& sensor, uint64_t data_id_value) {
                     return colmap::data_t(sensor, data_id_value);
                 }),
                 "sensor_id"_a, "id"_a,
                 "Upstream-style construction from a sensor id and a data "
                 "id.");

    py::class_<Frame> frame(m_scene, "Frame",
                            "One simultaneous capture of a rig, owning an "
                            "optional rig-from-world pose.");
    // Upstream pycolmap parity (923342c5 #4857): a Frame copy is an editable
    // draft (the finalized flag resets in the copy constructor).
    frame.def(py::init<>())
            .def(py::init<const Frame&>())
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

    // Upstream pycolmap parity (d3ccaf35 scene/image.cc): observation
    // accessors and projection.
    image.def(
                 "get_observation_point2D_idxs",
                 [](const Image& self) {
                     std::vector<point2D_t> point2D_idxs;
                     for (point2D_t idx = 0; idx < self.NumPoints2D(); ++idx) {
                         if (self.Point2D(idx).HasPoint3D()) {
                             point2D_idxs.push_back(idx);
                         }
                     }
                     return point2D_idxs;
                 },
                 "Indices of 2D points that observe a 3D point.")
            .def(
                    "get_observation_points2D",
                    [](const Image& self) {
                        std::vector<Point2D> points2D;
                        for (point2D_t idx = 0; idx < self.NumPoints2D();
                             ++idx) {
                            const Point2D& point2D = self.Point2D(idx);
                            if (point2D.HasPoint3D()) {
                                points2D.push_back(point2D);
                            }
                        }
                        return points2D;
                    },
                    "The 2D points that observe a 3D point.")
            .def(
                    "project_point",
                    [](const Image& self,
                       const Eigen::Vector3d& point3D) -> py::object {
                        THROW_CHECK(self.HasCameraPtr());
                        const Eigen::Vector3d point3D_in_cam =
                                self.CamFromWorld() * point3D;
                        auto uv = self.CameraPtr()->ImgFromCam(point3D_in_cam);
                        if (!uv.has_value()) {
                            return py::none();
                        }
                        return py::cast(*uv);
                    },
                    "point3D"_a,
                    "Project a 3D point onto the image; None if the point "
                    "cannot be projected (behind camera / invalid model).");

    // Upstream pycolmap parity: rig back-pointer and pose seeding on the
    // frame (the fork engine owns both).
    frame.def("reset_rig_ptr", &Frame::ResetRigPtr,
              "Clear the cached rig back-pointer (after rig removal).")
            .def(
                    "set_cam_from_world",
                    [](Frame& self, colmap::camera_t camera_id,
                       const Rigid3d& cam_from_world) {
                        self.SetCamFromWorld(camera_id, cam_from_world);
                    },
                    "camera_id"_a, "cam_from_world"_a,
                    "Set the cam_from_world pose for the given camera sensor.");

    frame.def("finalize_data_ids", &Frame::FinalizeDataIds,
              "Mark the frame data ids as final (upstream parity).")
            .def("has_final_data_ids", &Frame::HasFinalDataIds,
                 "Whether the data ids were finalized.")
            .def(
                    "data_ids_by_sensor",
                    [](const Frame& self, colmap::SensorType type) {
                        std::vector<colmap::data_t> out;
                        for (const auto& data_id : self.DataIds()) {
                            if (data_id.sensor_id.type == type) {
                                out.push_back(data_id);
                            }
                        }
                        return out;
                    },
                    "type"_a, "The associated data for a given sensor type.");

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
    // Upstream top-level name for the nested ConfigurationType enum.
    m_scene.attr("TwoViewGeometryConfiguration") =
            two_view_geometry.attr("ConfigurationType");

    // Upstream top-level name (pycolmap sfm/incremental_mapper.cc): the
    // reprojection-error-type enum used by ObservationManager.
    py::enum_<colmap::ReprojectionErrorType>(m_scene, "ReprojectionErrorType")
            .value("PIXEL", colmap::ReprojectionErrorType::PIXEL)
            .value("NORMALIZED", colmap::ReprojectionErrorType::NORMALIZED)
            .value("ANGULAR", colmap::ReprojectionErrorType::ANGULAR)
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

    // Upstream pycolmap parity: sensor membership queries on the rig.
    rig.def("has_sensor_from_rig", &Rig::HasSensorFromRig, "sensor_id"_a,
            "Whether the rig contains the given non-reference sensor.")
            .def("reset_sensor_from_rig", &Rig::ResetSensorFromRig,
                 "sensor_id"_a,
                 "Clear the known sensor_from_rig extrinsic of the given "
                 "sensor (it becomes unknown again).");

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

    // Upstream pycolmap parity (d3ccaf35 scene/database_cache.cc): the
    // mutable construction surface of the cache.
    database_cache.def(py::init<>())
            .def_static(
                    "create",
                    [](const colmap::Database& database,
                       const DatabaseCache::Options& options) {
                        return colmap::DatabaseCache::Create(database, options);
                    },
                    "database"_a, "options"_a,
                    "Create a cache from the database.")
            .def_static(
                    "create_from_cache",
                    [](const DatabaseCache& cache,
                       const DatabaseCache::Options& options) {
                        return colmap::DatabaseCache::CreateFromCache(cache,
                                                                      options);
                    },
                    "database_cache"_a, "options"_a,
                    "Create a filtered cache from an existing cache.")
            .def(
                    "load",
                    [](DatabaseCache& self, colmap::Database& database,
                       const DatabaseCache::Options& options) {
                        return self.Load(database, options);
                    },
                    "database"_a, "options"_a,
                    "Load the cache from the database (returns True on "
                    "success).")
            .def(
                    "add_rig",
                    [](DatabaseCache& self, colmap::Rig rig) {
                        self.AddRig(std::move(rig));
                    },
                    "rig"_a)
            .def(
                    "add_camera",
                    [](DatabaseCache& self, colmap::Camera camera) {
                        self.AddCamera(std::move(camera));
                    },
                    "camera"_a)
            .def(
                    "add_frame",
                    [](DatabaseCache& self, colmap::Frame frame) {
                        self.AddFrame(std::move(frame));
                    },
                    "frame"_a)
            .def(
                    "add_image",
                    [](DatabaseCache& self, colmap::Image image) {
                        self.AddImage(std::move(image));
                    },
                    "image"_a)
            .def(
                    "add_pose_prior",
                    [](DatabaseCache& self, colmap::PosePrior prior) {
                        self.AddPosePrior(std::move(prior));
                    },
                    "pose_prior"_a)
            .def("exists_rig", &DatabaseCache::ExistsRig, "rig_id"_a)
            .def("exists_camera", &DatabaseCache::ExistsCamera, "camera_id"_a)
            .def("exists_frame", &DatabaseCache::ExistsFrame, "frame_id"_a)
            .def("exists_image", &DatabaseCache::ExistsImage, "image_id"_a)
            .def(
                    "find_image_with_name",
                    [](DatabaseCache& self,
                       const std::string& name) -> py::object {
                        const colmap::Image* image =
                                self.FindImageWithName(name);
                        if (image == nullptr) {
                            return py::none();
                        }
                        return py::cast(image);
                    },
                    py::return_value_policy::reference_internal, "name"_a,
                    "Find a cached image by its name; None if not found.")
            .def(
                    "rig",
                    [](DatabaseCache& self, colmap::rig_t rig_id) -> Rig& {
                        return self.Rig(rig_id);
                    },
                    py::return_value_policy::reference_internal, "rig_id"_a,
                    "Mutable reference to the cached rig.")
            .def(
                    "frame",
                    [](DatabaseCache& self, colmap::frame_t frame_id)
                            -> Frame& { return self.Frame(frame_id); },
                    py::return_value_policy::reference_internal, "frame_id"_a,
                    "Mutable reference to the cached frame.");

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
                    "Apply a similarity transform to the reconstruction.")
            // --- Upstream pycolmap parity (d3ccaf35 scene/
            // reconstruction.cc): the full mutation and IO surface. Member
            // names follow the fork engine API; divergences are documented
            // per binding.
            .def(
                    "add_rig",
                    [](Reconstruction& self, colmap::Rig rig) {
                        self.AddRig(std::move(rig));
                    },
                    "rig"_a, "Add new rig.")
            .def(
                    "add_camera",
                    [](Reconstruction& self, const colmap::Camera& camera) {
                        self.AddCamera(camera);
                    },
                    "camera"_a,
                    "Add new camera. There is only one camera per image, "
                    "while multiple images can have the same camera.")
            .def(
                    "add_camera_with_trivial_rig",
                    [](Reconstruction& self, colmap::Camera camera) {
                        self.AddCameraWithTrivialRig(std::move(camera));
                    },
                    "camera"_a,
                    "Add a new camera and a trivial rig with the camera as "
                    "the only reference sensor.")
            .def(
                    "add_frame",
                    [](Reconstruction& self, colmap::Frame frame) {
                        self.AddFrame(std::move(frame));
                    },
                    "frame"_a, "Add new frame.")
            .def(
                    "add_image",
                    [](Reconstruction& self, colmap::Image image) {
                        self.AddImage(std::move(image));
                    },
                    "image"_a, "Add new image. Its frame must already exist.")
            .def(
                    "add_image_with_trivial_frame",
                    [](Reconstruction& self, colmap::Image image,
                       std::optional<Rigid3d> cam_from_world) {
                        if (cam_from_world.has_value()) {
                            self.AddImageWithTrivialFrame(std::move(image),
                                                          *cam_from_world);
                        } else {
                            self.AddImageWithTrivialFrame(std::move(image));
                        }
                    },
                    "image"_a, "cam_from_world"_a = py::none(),
                    "Add a new image and a trivial frame for it; optionally "
                    "seeding the frame pose.")
            .def(
                    "set_rigs_and_frames",
                    [](Reconstruction& self, std::vector<colmap::Rig> rigs,
                       std::vector<colmap::Frame> frames) {
                        self.SetRigsAndFrames(std::move(rigs),
                                              std::move(frames));
                    },
                    "rigs"_a, "frames"_a,
                    "Set the rigs and frames of the reconstruction (both "
                    "cleared and replaced).")
            .def(
                    "add_point3D",
                    [](Reconstruction& self, const Eigen::Vector3d& xyz,
                       const colmap::Track& track,
                       py::object color) -> point3D_t {
                        Eigen::Vector3ub rgb = Eigen::Vector3ub::Zero();
                        if (!color.is_none()) {
                            // The repo-wide stl caster rejects plain lists
                            // for vector<uint8_t>; go through a forced
                            // ndarray instead.
                            auto rgb_arr = color.cast<py::array_t<
                                    uint8_t, py::array::c_style |
                                                     py::array::forcecast>>();
                            rgb = Eigen::Vector3ub(*rgb_arr.data(0),
                                                   *rgb_arr.data(1),
                                                   *rgb_arr.data(2));
                        }
                        return self.AddPoint3D(xyz, track, rgb);
                    },
                    "xyz"_a, "track"_a, "color"_a = py::none(),
                    "Add a new 3D point with known color and return its ID.")
            .def(
                    "add_point3D_with_id",
                    [](Reconstruction& self, point3D_t point3D_id,
                       colmap::Point3D point3D) {
                        self.AddPoint3D(point3D_id, std::move(point3D));
                    },
                    "point3D_id"_a, "point3D"_a,
                    "Add a 3D point with a specific identifier.")
            .def(
                    "add_observation",
                    [](Reconstruction& self, point3D_t point3D_id,
                       const colmap::TrackElement& track_el) {
                        self.AddObservation(point3D_id, track_el);
                    },
                    "point3D_id"_a, "track_el"_a,
                    "Add an observation to an existing 3D point.")
            .def(
                    "merge_points3D",
                    [](Reconstruction& self, point3D_t point3D_id1,
                       point3D_t point3D_id2) {
                        return self.MergePoints3D(point3D_id1, point3D_id2);
                    },
                    "point3D_id1"_a, "point3D_id2"_a,
                    "Merge two 3D points and return the new identifier; the "
                    "merged tracks and properties are combined.")
            .def(
                    "delete_point3D",
                    [](Reconstruction& self, point3D_t point3D_id) {
                        self.DeletePoint3D(point3D_id);
                    },
                    "point3D_id"_a,
                    "Delete a 3D point and all of its observations.")
            .def(
                    "delete_observation",
                    [](Reconstruction& self, colmap::image_t image_id,
                       colmap::point2D_t point2D_idx) {
                        self.DeleteObservation(image_id, point2D_idx);
                    },
                    "image_id"_a, "point2D_idx"_a,
                    "Delete one observation of an image from its 3D point.")
            .def(
                    "delete_all_points2D_and_points3D",
                    [](Reconstruction& self) {
                        self.DeleteAllPoints2DAndPoints3D();
                    },
                    "Delete all 2D points and 3D points.")
            .def(
                    "normalize",
                    [](Reconstruction& self, double extent, double p0,
                       double p1, bool use_images) {
                        self.Normalize(extent, p0, p1, use_images);
                    },
                    "extent"_a = 10.0, "p0"_a = 0.1, "p1"_a = 0.9,
                    "use_images"_a = true,
                    "Scale the reconstruction so that the bounding box of "
                    "the given percentile range has the requested extent.")
            .def(
                    "compute_centroid",
                    [](const Reconstruction& self, double p0, double p1) {
                        return self.ComputeCentroid(p0, p1);
                    },
                    "p0"_a = 0.1, "p1"_a = 0.9,
                    "Compute the centroid of the reconstruction.")
            .def(
                    "compute_bounding_box",
                    [](const Reconstruction& self, double p0, double p1) {
                        return self.ComputeBoundingBox(p0, p1);
                    },
                    "p0"_a = 0.0, "p1"_a = 1.0,
                    "Compute the bounding box corner points over the given "
                    "percentile range.")
            .def(
                    "transform",
                    [](Reconstruction& self, const Sim3d& new_from_old) {
                        self.Transform(new_from_old);
                    },
                    "tform"_a, "Upstream name of ``sim3d_transform``.")
            // Upstream pycolmap signature: crop(AlignedBox3d), matching the
            // return type of compute_bounding_box().
            .def("crop", &Reconstruction::Crop, "bbox"_a,
                 "Crop the reconstruction to the given bounding box; "
                 "returns a new reconstruction.")
            // Legacy fork overload: accept a (min, max) pair as well.
            .def(
                    "crop",
                    [](const Reconstruction& self,
                       const std::pair<Eigen::Vector3d, Eigen::Vector3d>&
                               bbox) {
                        return self.Crop(
                                Eigen::AlignedBox3d(bbox.first, bbox.second));
                    },
                    "bbox"_a, "Legacy overload taking a ``(min, max)`` pair.")
            .def(
                    "merge",
                    [](Reconstruction& self, const Reconstruction& other,
                       double max_reproj_error) {
                        return self.Merge(other, max_reproj_error);
                    },
                    "reconstruction2"_a, "max_reproj_error"_a,
                    "Merge the given reconstruction into this one; returns "
                    "True on success.")
            .def(
                    "find_image_with_name",
                    [](const Reconstruction& self,
                       const std::string& name) -> py::object {
                        const colmap::Image* image =
                                self.FindImageWithName(name);
                        if (image == nullptr) {
                            return py::none();
                        }
                        return py::cast(image);
                    },
                    py::return_value_policy::reference_internal, "name"_a,
                    "Find an image by its name; None if not found.")
            .def(
                    "find_common_reg_image_ids",
                    [](const Reconstruction& self,
                       const Reconstruction& other) {
                        return self.FindCommonRegImageIds(other);
                    },
                    "other"_a,
                    "Common registered images between two reconstructions.")
            .def(
                    "transcribe_image_ids_to_database",
                    [](Reconstruction& self, colmap::Database& database) {
                        self.TranscribeImageIdsToDatabase(database);
                    },
                    "database"_a,
                    "Write the reconstruction image names to the database, "
                    "transcribing image ids.")
            .def(
                    "point3D_ids",
                    [](const Reconstruction& self) {
                        const auto ids = self.Point3DIds();
                        return std::vector<point3D_t>(ids.begin(), ids.end());
                    },
                    "Identifiers of all 3D points.")
            .def(
                    "update_point_3d_errors",
                    [](Reconstruction& self) { self.UpdatePoint3DErrors(); },
                    "Update the 3D point error statistics.")
            .def("compute_num_observations",
                 &Reconstruction::ComputeNumObservations,
                 "The total number of observations.")
            .def("compute_mean_track_length",
                 &Reconstruction::ComputeMeanTrackLength,
                 "The mean length of all tracks.")
            .def("compute_mean_observations_per_reg_image",
                 &Reconstruction::ComputeMeanObservationsPerRegImage,
                 "The mean observations per registered image.")
            .def("compute_mean_reprojection_error",
                 &Reconstruction::ComputeMeanReprojectionError,
                 "The mean reprojection error in pixels.")
            .def(
                    "load",
                    [](Reconstruction& self, DatabaseCache& database_cache) {
                        self.Load(database_cache);
                    },
                    "database_cache"_a,
                    "Load the reconstruction state from a database cache.")
            .def("tear_down", &Reconstruction::TearDown,
                 "Finalize the reconstruction after reconstruction.")
            .def(
                    "read_text",
                    [](Reconstruction& self, const std::string& path) {
                        self.ReadText(path);
                    },
                    "path"_a, "Read the reconstruction in text format.")
            .def(
                    "read_binary",
                    [](Reconstruction& self, const std::string& path) {
                        self.ReadBinary(path);
                    },
                    "path"_a, "Read the reconstruction in binary format.")
            .def(
                    "write_text",
                    [](Reconstruction& self, const std::string& path) {
                        self.WriteText(path);
                    },
                    "path"_a, "Write the reconstruction in text format.")
            .def(
                    "write_binary",
                    [](Reconstruction& self, const std::string& path) {
                        self.WriteBinary(path);
                    },
                    "path"_a, "Write the reconstruction in binary format.")
            .def(
                    "import_PLY",
                    [](Reconstruction& self, const std::string& path) {
                        self.ImportPLY(path);
                    },
                    "path"_a, "Import 3D points from a PLY file.")
            .def(
                    "export_PLY",
                    [](const Reconstruction& self, const std::string& path) {
                        // Binding-level composition: the engine converts to
                        // PlyPoint records and the shared ply writer emits
                        // them (upstream has a fused ExportPLY helper).
                        colmap::WriteTextPlyPoints(path, self.ConvertToPLY());
                    },
                    "output_path"_a, "Export 3D points to PLY format (.ply).")
            .def(
                    "extract_colors_for_image",
                    [](Reconstruction& self, colmap::image_t image_id,
                       const std::string& path) {
                        return self.ExtractColorsForImage(image_id, path);
                    },
                    "image_id"_a, "path"_a,
                    "Extract colors from an image for the observed 3D "
                    "points; returns True on success.")
            .def(
                    "extract_colors_for_all_images",
                    [](Reconstruction& self, const std::string& path) {
                        self.ExtractColorsForAllImages(path);
                    },
                    "path"_a,
                    "Extract colors for all images from the given image "
                    "folder.")
            .def(
                    "create_image_dirs",
                    [](const Reconstruction& self, const std::string& path) {
                        self.CreateImageDirs(path);
                    },
                    "path"_a,
                    "Create the image sub-directories used by the dense "
                    "workspace layout.")
            .def("is_valid", &Reconstruction::IsValid,
                 "Check whether the reconstruction object is internally "
                 "consistent.")
            .def("__copy__",
                 [](const Reconstruction& self) {
                     return Reconstruction(self);
                 })
            .def("__deepcopy__", [](const Reconstruction& self,
                                    py::dict) { return Reconstruction(self); })
            .def("__repr__", [](const Reconstruction& self) {
                return "Reconstruction(num_images=" +
                       std::to_string(self.NumImages()) +
                       ", num_points3D=" + std::to_string(self.NumPoints3D()) +
                       ")";
            });

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
    // Upstream pycolmap parity: Correspondence and the borrowed range view
    // are top-level names (the fork registers the value type nested under
    // CorrespondenceGraph; both spellings stay available).
    m_scene.attr("Correspondence") = corr_graph.attr("Correspondence");
    py::class_<colmap::CorrespondenceGraph::CorrespondenceRange>
            correspondence_range(
                    m_scene, "CorrespondenceRange",
                    "A borrowed [beg, end) range of correspondences.");
    correspondence_range
            .def_property_readonly(
                    "empty",
                    [](const colmap::CorrespondenceGraph::CorrespondenceRange&
                               self) { return self.beg == self.end; },
                    "Whether the range is empty.")
            .def(
                    "to_list",
                    [](const colmap::CorrespondenceGraph::CorrespondenceRange&
                               self) {
                        return std::vector<
                                colmap::CorrespondenceGraph::Correspondence>(
                                self.beg, self.end);
                    },
                    "Convert the range to a list of correspondences.");
    corr_graph.def(py::init<>())
            .def(
                    "extract_correspondences",
                    [](const colmap::CorrespondenceGraph& self,
                       colmap::image_t image_id,
                       colmap::point2D_t point2D_idx) {
                        std::vector<colmap::CorrespondenceGraph::Correspondence>
                                corrs;
                        self.ExtractCorrespondences(image_id, point2D_idx,
                                                    &corrs);
                        return corrs;
                    },
                    "image_id"_a, "point2D_idx"_a,
                    "All correspondences of the given observation.")
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
                    "relative poses of verified two-view geometries).")
            .def("is_valid_edge", &colmap::PoseGraph::IsValid,
                 "image_pair_id"_a,
                 "Whether the edge is marked valid (used in the largest "
                 "connected component search).")
            .def("set_valid_edge", &colmap::PoseGraph::SetValidEdge,
                 "image_pair_id"_a, "Mark the edge valid.")
            .def("set_invalid_edge", &colmap::PoseGraph::SetInvalidEdge,
                 "image_pair_id"_a, "Mark the edge invalid.")
            .def(
                    "largest_connected_frame_component",
                    [](const colmap::PoseGraph& self,
                       const Reconstruction& reconstruction,
                       bool filter_unregistered) {
                        const auto frames = self.LargestConnectedFrameComponent(
                                reconstruction, filter_unregistered);
                        return std::vector<colmap::frame_t>(frames.begin(),
                                                            frames.end());
                    },
                    "reconstruction"_a, "filter_unregistered"_a = true,
                    "Frame ids of the largest connected component of valid "
                    "edges.")
            .def(
                    "invalidate_pairs_outside_active_image_ids",
                    [](colmap::PoseGraph& self,
                       const std::vector<colmap::image_t>& image_ids) {
                        self.InvalidatePairsOutsideActiveImageIds(
                                colmap::FlatHashSet<colmap::image_t>(
                                        image_ids.begin(), image_ids.end()));
                    },
                    "image_ids"_a,
                    "Mark every edge whose images are not both in the given "
                    "active set as invalid.")
            .def("is_valid", &colmap::PoseGraph::IsValid, "pair_id"_a,
                 "Upstream name of ``is_valid_edge``.")
            .def_property_readonly(
                    "edges",
                    [](const colmap::PoseGraph& self) {
                        // The engine stores edges in a NodeHashMap (custom
                        // hash), which py::bind_map cannot wrap; expose a
                        // copied {pair_id: Edge} dict view instead.
                        py::dict out;
                        for (const auto& [pair_id, edge] : self.Edges()) {
                            out[py::int_(pair_id)] = edge;
                        }
                        return out;
                    },
                    "Copied {pair_id: PoseGraphEdge} view of all edges.");
    // Upstream pycolmap parity: the edge value type and its map are
    // top-level names.
    m_scene.attr("PoseGraphEdge") = pose_graph.attr("Edge");

    // Upstream pycolmap parity (src/pycolmap/scene/database_cache.cc): the
    // cache options are a top-level class name (the fork nests them under
    // DatabaseCache; both spellings stay available).
    m_scene.attr("DatabaseCacheOptions") = database_cache.attr("Options");

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

    // Upstream pycolmap parity (scene/synthetic.cc): the match config is
    // exported at top level under the upstream class name.
    py::enum_<colmap::SyntheticDatasetOptions::MatchConfig>(
            m_scene, "SyntheticDatasetMatchConfig",
            "The match generation strategy of the synthetic dataset.")
            .value("EXHAUSTIVE",
                   colmap::SyntheticDatasetOptions::MatchConfig::EXHAUSTIVE)
            .value("CHAINED",
                   colmap::SyntheticDatasetOptions::MatchConfig::CHAINED)
            .export_values();

    m_scene.attr("__docstring__") = map_shared_argument_docstrings;
}

}  // namespace scene
}  // namespace reconstruction
}  // namespace cloudViewer

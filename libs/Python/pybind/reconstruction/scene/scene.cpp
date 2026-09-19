// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/scene/scene.h"

#include "pybind/docstring.h"
#include "scene/camera.h"
#include "scene/image.h"
#include "scene/point2d.h"
#include "scene/point3d.h"
#include "scene/reconstruction.h"
#include "scene/track.h"

namespace cloudViewer {
namespace reconstruction {
namespace scene {

// The COLMAP fork engine types live in namespace colmap.
using colmap::Camera;
using colmap::camera_t;
using colmap::Image;
using colmap::image_t;
using colmap::Point2D;
using colmap::point2D_t;
using colmap::Point3D;
using colmap::point3D_t;
using colmap::Reconstruction;
using colmap::Track;
using colmap::TrackElement;

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
            .def_property_readonly("camera_id", &Camera::CameraId)
            .def_property_readonly("model_id", &Camera::ModelId)
            .def_property_readonly("model_name", &Camera::ModelName)
            .def_property_readonly("width", &Camera::Width)
            .def_property_readonly("height", &Camera::Height)
            .def_property_readonly("focal_length", &Camera::FocalLength)
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
            .def_property_readonly(
                    "image_id",
                    [](const Image& self) { return self.ImageId(); })
            .def_property_readonly(
                    "name", [](const Image& self) { return self.Name(); })
            .def_property_readonly(
                    "camera_id",
                    [](const Image& self) { return self.CameraId(); })
            .def_property_readonly(
                    "num_points2D",
                    [](const Image& self) { return self.NumPoints2D(); })
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

    py::class_<Reconstruction> reconstruction(
            m_scene, "Reconstruction",
            "Incremental reconstruction container (cameras, images, 3D "
            "points).");
    reconstruction.def(py::init<>())
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
            .def_property_readonly("num_images", &Reconstruction::NumImages)
            .def_property_readonly("num_reg_images",
                                   &Reconstruction::NumRegImages)
            .def_property_readonly("num_cameras", &Reconstruction::NumCameras)
            .def_property_readonly("num_points3D", &Reconstruction::NumPoints3D)
            .def("exists_image", &Reconstruction::ExistsImage, "image_id"_a,
                 "Whether an image with the identifier exists.")
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
                    py::return_value_policy::copy);

    m_scene.attr("__docstring__") = map_shared_argument_docstrings;
}

}  // namespace scene
}  // namespace reconstruction
}  // namespace cloudViewer

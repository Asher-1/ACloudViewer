// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/geometry/geometry.h"

#include <sstream>
#include <unordered_map>

#include "geometry/essential_matrix.h"
#include "geometry/gps.h"
#include "geometry/homography_matrix.h"
#include "geometry/pose.h"
#include "geometry/pose_prior.h"
#include "geometry/triangulation.h"
#include "pybind/docstring.h"

namespace cloudViewer {
namespace reconstruction {
namespace geometry {

// The COLMAP fork engine types live in namespace colmap.
using colmap::GPSTransform;
using colmap::Inverse;
using colmap::PosePrior;
using colmap::Rigid3d;
using colmap::Sim3d;

// Upstream pycolmap parity (src/pycolmap/geometry): rigid and similarity
// transforms as lightweight value types, exposed Open3D style (plain class
// bindings, no dataclass machinery).
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"rotation", "Quaternion as (w, x, y, z)."},
                {"translation", "Translation vector as (x, y, z)."},
};

void pybind_geometry(py::module& m) {
    py::module m_geometry = m.def_submodule("geometry");

    py::class_<Rigid3d> rigid3d(m_geometry, "Rigid3d",
                                "Rigid transform (rotation + translation).");
    rigid3d.def(py::init<>())
            .def(py::init([](const Eigen::Vector4d& wxyz,
                             const Eigen::Vector3d& translation) {
                     return Rigid3d(Eigen::Quaterniond(wxyz(0), wxyz(1),
                                                       wxyz(2), wxyz(3)),
                                    translation);
                 }),
                 "rotation_wxyz"_a, "translation"_a,
                 "Build from a quaternion given as (w, x, y, z).")
            // The fork's W3-2b Rigid3d stores a single params block with
            // accessor methods; quaternions are exposed as (w, x, y, z)
            // arrays to avoid registering Eigen quaternion types.
            .def_property(
                    "rotation",
                    [](const Rigid3d& self) {
                        const Eigen::Quaterniond q(self.rotation());
                        return Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
                    },
                    [](Rigid3d& self, const Eigen::Vector4d& wxyz) {
                        self.rotation() = Eigen::Quaterniond(wxyz(0), wxyz(1),
                                                             wxyz(2), wxyz(3));
                    },
                    // By-value Eigen getters must be copied out: the
                    // default reference_internal policy would wrap the
                    // temporary's buffer and dangle (observed as garbage
                    // pose values in downstream conversions).
                    py::return_value_policy::copy)
            .def_property(
                    "translation",
                    [](const Rigid3d& self) {
                        return Eigen::Vector3d(self.translation());
                    },
                    [](Rigid3d& self, const Eigen::Vector3d& t) {
                        self.translation() = t;
                    },
                    py::return_value_policy::copy)
            .def(
                    "inverse",
                    [](const Rigid3d& self) { return Inverse(self); },
                    "Return the inverse transform.")
            .def(
                    "__mul__",
                    [](const Rigid3d& self, const Rigid3d& other) {
                        return self * other;
                    },
                    py::is_operator())
            .def(
                    "apply",
                    [](const Rigid3d& self, const Eigen::Vector3d& point) {
                        return self * point;
                    },
                    "point"_a,
                    "Transform a world point into the destination frame.")
            .def("__repr__", [](const Rigid3d& self) {
                const Eigen::Quaterniond& q = self.rotation();
                const Eigen::Vector3d& t = self.translation();
                return "Rigid3d(rotation=(" + std::to_string(q.w()) + ", " +
                       std::to_string(q.x()) + ", " + std::to_string(q.y()) +
                       ", " + std::to_string(q.z()) + "), translation=(" +
                       std::to_string(t.x()) + ", " + std::to_string(t.y()) +
                       ", " + std::to_string(t.z()) + "))";
            });

    py::class_<Sim3d> sim3d(m_geometry, "Sim3d",
                            "Similarity transform (scale + rotation + "
                            "translation).");
    sim3d.def(py::init<>())
            .def(py::init([](double scale, const Eigen::Vector4d& wxyz,
                             const Eigen::Vector3d& translation) {
                     return Sim3d(scale,
                                  Eigen::Quaterniond(wxyz(0), wxyz(1), wxyz(2),
                                                     wxyz(3)),
                                  translation);
                 }),
                 "scale"_a, "rotation_wxyz"_a, "translation"_a)
            .def_property(
                    "scale", [](const Sim3d& self) { return self.scale(); },
                    [](Sim3d& self, double s) { self.scale() = s; })
            .def_property(
                    "rotation",
                    [](const Sim3d& self) {
                        const Eigen::Quaterniond q(self.rotation());
                        return Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
                    },
                    [](Sim3d& self, const Eigen::Vector4d& wxyz) {
                        self.rotation() = Eigen::Quaterniond(wxyz(0), wxyz(1),
                                                             wxyz(2), wxyz(3));
                    },
                    // See the Rigid3d note: by-value Eigen getters must be
                    // copied out to avoid dangling temporaries.
                    py::return_value_policy::copy)
            .def_property(
                    "translation",
                    [](const Sim3d& self) {
                        return Eigen::Vector3d(self.translation());
                    },
                    [](Sim3d& self, const Eigen::Vector3d& t) {
                        self.translation() = t;
                    },
                    py::return_value_policy::copy)
            .def(
                    "inverse", [](const Sim3d& self) { return Inverse(self); },
                    "Return the inverse transform.")
            .def("__repr__", [](const Sim3d& self) {
                return "Sim3d(scale=" + std::to_string(self.scale()) + ")";
            });

    m_geometry.def(
            "calculate_triangulation_angle",
            [](const Eigen::Vector3d& proj_center1,
               const Eigen::Vector3d& proj_center2,
               const Eigen::Vector3d& point3D) {
                return colmap::CalculateTriangulationAngle(
                        proj_center1, proj_center2, point3D);
            },
            "proj_center1"_a, "proj_center2"_a, "point3D"_a,
            "Angle in radians between the two rays of a triangulated "
            "point.");

    m_geometry.def(
            "calculate_triangulation_angles",
            [](const Eigen::Vector3d& proj_center1,
               const Eigen::Vector3d& proj_center2,
               const std::vector<Eigen::Vector3d>& points3D) {
                return colmap::CalculateTriangulationAngles(
                        proj_center1, proj_center2, points3D);
            },
            "proj_center1"_a, "proj_center2"_a, "points3D"_a,
            "Vectorized triangulation-angle computation for a list of 3D "
            "points.");

    py::enum_<GPSTransform::ELLIPSOID> ellipsoid(
            m_geometry, "GPSTransformEllipsoid",
            "Reference ellipsoid models for GPS transforms.");
    ellipsoid.value("GRS80", GPSTransform::GRS80)
            .value("WGS84", GPSTransform::WGS84);

    py::class_<GPSTransform> gps_transform(m_geometry, "GPSTransform",
                                           "Geodetic coordinate transforms.");
    gps_transform
            .def(py::init([](GPSTransform::ELLIPSOID ellipsoid) {
                     return GPSTransform(static_cast<int>(ellipsoid));
                 }),
                 "ellipsoid"_a = GPSTransform::GRS80)
            .def("ellipsoid_to_ecef", &GPSTransform::EllipsoidToECEF,
                 "lat_lon_alt"_a,
                 "Convert ellipsoidal (lat/lon/alt) to ECEF coordinates.")
            .def("ecef_to_ellipsoid", &GPSTransform::ECEFToEllipsoid,
                 "xyz_in_ecef"_a,
                 "Convert ECEF to ellipsoidal (lat/lon/alt) coordinates.")
            .def("ellipsoid_to_enu", &GPSTransform::EllipsoidToENU,
                 "lat_lon_alt"_a, "ref_lat"_a, "ref_lon"_a, "ref_alt"_a,
                 "Convert ellipsoidal coordinates to a local ENU frame.")
            .def("ecef_to_enu", &GPSTransform::ECEFToENU, "xyz_in_ecef"_a,
                 "ref_ecef"_a, "Convert ECEF coordinates to a local ENU frame.")
            .def("enu_to_ellipsoid", &GPSTransform::ENUToEllipsoid,
                 "xyz_in_enu"_a, "ref_lat"_a, "ref_lon"_a, "ref_alt"_a,
                 "Convert local ENU coordinates to ellipsoidal coordinates.")
            .def("enu_to_ecef", &GPSTransform::ENUToECEF, "xyz_in_enu"_a,
                 "ref_lat"_a, "ref_lon"_a, "ref_alt"_a,
                 "Convert local ENU coordinates to ECEF coordinates.");

    py::enum_<PosePrior::CoordinateSystem> coordinate_system(
            m_geometry, "PosePriorCoordinateSystem",
            "Coordinate system of a pose prior position.");
    coordinate_system.value("UNDEFINED", PosePrior::CoordinateSystem::UNDEFINED)
            .value("WGS84", PosePrior::CoordinateSystem::WGS84)
            .value("CARTESIAN", PosePrior::CoordinateSystem::CARTESIAN);

    py::class_<PosePrior> pose_prior(
            m_geometry, "PosePrior",
            "Prior information about the pose of a sensor.");
    pose_prior.def(py::init<>())
            .def_readwrite("pose_prior_id", &PosePrior::pose_prior_id)
            .def_readwrite("corr_data_id", &PosePrior::corr_data_id)
            .def_readwrite("position", &PosePrior::position)
            .def_readwrite("position_covariance",
                           &PosePrior::position_covariance)
            .def_readwrite("coordinate_system", &PosePrior::coordinate_system)
            .def_readwrite("gravity", &PosePrior::gravity)
            .def("has_position", &PosePrior::HasPosition,
                 "Whether the position is finite.")
            .def("has_position_cov", &PosePrior::HasPositionCov,
                 "Whether the position covariance is finite.")
            .def("has_gravity", &PosePrior::HasGravity,
                 "Whether the gravity vector is finite.")
            .def("__repr__", [](const PosePrior& self) {
                std::ostringstream ss;
                ss << self;
                return ss.str();
            });

    m_geometry.def(
            "gravity_from_exif_orientation",
            [](int orientation) {
                return colmap::GravityFromExifOrientation(orientation);
            },
            "orientation"_a,
            "Gravity (down) direction from the EXIF orientation tag; None if "
            "the orientation is not upright.");

    m_geometry.def(
            "compute_rot90_from_gravity",
            [](const Eigen::Vector3d& gravity) {
                return colmap::ComputeRot90FromGravity(gravity);
            },
            "gravity"_a,
            "Number of 90 degree counter-clockwise rotations needed to make "
            "the sensor upright.");

    m_geometry.def(
            "essential_matrix_from_pose",
            [](const Rigid3d& cam2_from_cam1) {
                const Eigen::Matrix3d R =
                        cam2_from_cam1.rotation().toRotationMatrix();
                const Eigen::Vector3d& t = cam2_from_cam1.translation();
                return colmap::EssentialMatrixFromPose(R, t);
            },
            "cam2_from_cam1"_a,
            "Construct the essential matrix from a relative pose.");

    m_geometry.def(
            "pose_from_homography_matrix",
            [](const Eigen::Matrix3d& H, const Eigen::Matrix3d& K1,
               const Eigen::Matrix3d& K2,
               const std::vector<Eigen::Vector3d>& cam_rays1,
               const std::vector<Eigen::Vector3d>& cam_rays2) {
                Rigid3d cam2_from_cam1;
                Eigen::Vector3d normal;
                std::vector<Eigen::Vector3d> points3D;
                colmap::PoseFromHomographyMatrix(H, K1, K2, cam_rays1,
                                                 cam_rays2, &cam2_from_cam1,
                                                 &normal, &points3D);
                py::dict out;
                out["cam2_from_cam1"] = cam2_from_cam1;
                out["normal"] = normal;
                out["points3D"] = points3D;
                return out;
            },
            "H"_a, "K1"_a, "K2"_a, "cam_rays1"_a, "cam_rays2"_a,
            "Recover the most probable pose from the given homography matrix "
            "using the cheirality check; returns a dict with "
            "cam2_from_cam1/normal/points3D.");

    m_geometry.attr("__docstring__") = map_shared_argument_docstrings;
}

}  // namespace geometry
}  // namespace reconstruction
}  // namespace cloudViewer

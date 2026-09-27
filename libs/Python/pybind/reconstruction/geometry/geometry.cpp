// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/geometry/geometry.h"

#include <limits>
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

    // Upstream pycolmap parity (src/pycolmap/geometry/eigen.cc): a
    // standalone rotation wrapper. The upstream uses a zero-copy numpy
    // wrapper type; the fork binds Eigen::Quaterniond directly, which gives
    // the same surface with copy semantics.
    py::class_<Eigen::Quaterniond> rotation3d(
            m_geometry, "Rotation3d",
            "A 3D rotation (quaternion, coeffs in [x, y, z, w] order).");
    rotation3d.def(py::init<>())
            .def(py::init([](const Eigen::Vector4d& xyzw) {
                     return Eigen::Quaterniond(xyzw(3), xyzw(0), xyzw(1),
                                               xyzw(2));
                 }),
                 "xyzw"_a, "Quaternion in [x, y, z, w] format.")
            .def(py::init([](const Eigen::Matrix3d& matrix) {
                     return Eigen::Quaterniond(matrix);
                 }),
                 "matrix"_a, "3x3 rotation matrix.")
            .def(py::init([](const Eigen::Vector3d& axis_angle) {
                     const double angle = axis_angle.norm();
                     if (angle < std::numeric_limits<double>::epsilon()) {
                         return Eigen::Quaterniond::Identity();
                     }
                     return Eigen::Quaterniond(
                             Eigen::AngleAxisd(angle, axis_angle / angle));
                 }),
                 "axis_angle"_a, "Axis-angle 3D vector.")
            .def_property(
                    "quat",
                    [](const Eigen::Quaterniond& self) {
                        // Eigen stores coefficients as [x, y, z, w].
                        return Eigen::Vector4d(self.x(), self.y(), self.z(),
                                               self.w());
                    },
                    [](Eigen::Quaterniond& self, const Eigen::Vector4d& xyzw) {
                        self.coeffs() = xyzw;
                    },
                    "Quaternion in [x, y, z, w] format.")
            .def(
                    "__mul__",
                    [](const Eigen::Quaterniond& self,
                       const Eigen::Quaterniond& other) {
                        return self * other;
                    },
                    py::is_operator())
            .def(
                    "__mul__",
                    [](const Eigen::Quaterniond& self,
                       const Eigen::Vector3d& v) { return self * v; },
                    py::is_operator())
            .def(
                    "normalize",
                    [](Eigen::Quaterniond& self) { self.normalize(); },
                    "Normalize the quaternion in place.")
            .def(
                    "matrix",
                    [](const Eigen::Quaterniond& self) {
                        return self.toRotationMatrix();
                    },
                    "The 3x3 rotation matrix.")
            .def("norm",
                 [](const Eigen::Quaterniond& self) { return self.norm(); })
            .def(
                    "angle",
                    [](const Eigen::Quaterniond& self) {
                        return Eigen::AngleAxisd(self).angle();
                    },
                    "The rotation angle in radians.")
            .def(
                    "angle_to",
                    [](const Eigen::Quaterniond& self,
                       const Eigen::Quaterniond& other) {
                        return self.angularDistance(other);
                    },
                    "other"_a,
                    "The rotation angle to another rotation in radians.")
            .def(
                    "inverse",
                    [](const Eigen::Quaterniond& self) {
                        return self.inverse();
                    },
                    "The inverse rotation.")
            .def("__repr__",
                 [](const Eigen::Quaterniond& self) {
                     std::ostringstream ss;
                     ss << "Rotation3d(xyzw=[" << self.x() << ", " << self.y()
                        << ", " << self.z() << ", " << self.w() << "])";
                     return ss.str();
                 })
            .def("__deepcopy__", [](const Eigen::Quaterniond& self,
                                    const py::dict&) { return self; });

    // Upstream pycolmap parity (src/pycolmap/geometry/eigen.cc): the
    // axis-aligned bounding box from Eigen.
    py::class_<Eigen::AlignedBox3d> aligned_box3d(
            m_geometry, "AlignedBox3d", "An axis-aligned 3D bounding box.");
    aligned_box3d.def(py::init<>())
            .def(py::init([](const Eigen::Vector3d& min_pt,
                             const Eigen::Vector3d& max_pt) {
                     return Eigen::AlignedBox3d(min_pt, max_pt);
                 }),
                 "min"_a, "max"_a)
            .def_property(
                    "min",
                    [](const Eigen::AlignedBox3d& self) { return self.min(); },
                    [](Eigen::AlignedBox3d& self,
                       const Eigen::Vector3d& min_pt) { self.min() = min_pt; },
                    "The minimal corner.")
            .def_property(
                    "max",
                    [](const Eigen::AlignedBox3d& self) { return self.max(); },
                    [](Eigen::AlignedBox3d& self,
                       const Eigen::Vector3d& max_pt) { self.max() = max_pt; },
                    "The maximal corner.")
            .def(
                    "contains",
                    [](const Eigen::AlignedBox3d& self,
                       const Eigen::Vector3d& point) {
                        return self.contains(point);
                    },
                    "point"_a, "Whether the box contains the given point.")
            .def(
                    "diagonal",
                    [](const Eigen::AlignedBox3d& self) {
                        return self.diagonal();
                    },
                    "The diagonal vector of the box.")
            .def(
                    "center",
                    [](const Eigen::AlignedBox3d& self) {
                        return self.center();
                    },
                    "The center point of the box.")
            .def("volume",
                 [](const Eigen::AlignedBox3d& self) { return self.volume(); });

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
            .def(
                    "__mul__",
                    [](const Rigid3d& self, const Eigen::Vector3d& point) {
                        return self * point;
                    },
                    py::is_operator())
            .def("matrix", &Rigid3d::ToMatrix,
                 "The 3x4 [R | t] matrix representation.")
            .def("tgt_origin_in_src", &Rigid3d::TgtOriginInSrc,
                 "The origin of the target frame expressed in the source "
                 "frame.")
            .def("adjoint", &Rigid3d::Adjoint,
                 "The 6x6 adjoint matrix for uncertainty propagation.")
            .def("adjoint_inverse", &Rigid3d::AdjointInverse,
                 "The 6x6 inverse adjoint matrix.")
            // Upstream pycolmap parity: the quaternion/covariance helpers
            // are also methods on Rigid3d (the same functionality is
            // available as module-level free functions).
            .def(
                    "average_quaternions",
                    [](const Rigid3d&,
                       const std::vector<Eigen::Vector4d>& quats,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               weights) {
                        return colmap::AverageQuaternions(
                                quats,
                                std::vector<double>(
                                        weights.data(),
                                        weights.data() + weights.size()));
                    },
                    "quats"_a, "weights"_a,
                    "Weighted-average a list of (w, x, y, z) quaternions.")
            .def(
                    "get_covariance_for_inverse",
                    [](const Rigid3d& self, const Eigen::Matrix6d& cov) {
                        return colmap::GetCovarianceForRigid3dInverse(self,
                                                                      cov);
                    },
                    "cov"_a, "Covariance of the inverse rigid transform.")
            .def(
                    "get_covariance_for_composed_rigid3d",
                    [](const Rigid3d& a_from_b,
                       const Eigen::Matrix<double, 12, 12>& covar) {
                        return colmap::GetCovarianceForComposedRigid3d(a_from_b,
                                                                       covar);
                    },
                    "covar"_a,
                    "Propagate a 12x12 covariance of two composed rigid "
                    "transforms to a 6x6 covariance of the composition.")
            .def(
                    "get_covariance_for_relative_rigid3d",
                    [](const Rigid3d& a_from_c, const Rigid3d& b_from_c,
                       const Eigen::Matrix<double, 12, 12>& covar) {
                        return colmap::GetCovarianceForRelativeRigid3d(
                                a_from_c, b_from_c, covar);
                    },
                    "b_from_c"_a, "covar"_a,
                    "Propagate a 12x12 covariance to the 6x6 covariance of the "
                    "relative transform.")
            .def(
                    "interpolate_camera_poses",
                    [](const Rigid3d& cam1_from_world,
                       const Rigid3d& cam2_from_world, double t) {
                        return colmap::InterpolateCameraPoses(
                                cam1_from_world, cam2_from_world, t);
                    },
                    "cam2_from_world"_a, "t"_a,
                    "Interpolate between this pose and the given one (slerp "
                    "rotation, lerp translation); t in [0, 1].")
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
            .def(
                    "transform_camera_world",
                    [](const Sim3d& self, const Rigid3d& cam_from_world) {
                        return colmap::TransformCameraWorld(self,
                                                            cam_from_world);
                    },
                    "cam_from_world"_a,
                    "Compose the camera pose from world after applying this "
                    "similarity transform to the world frame (upstream "
                    "Sim3d::TransformCameraWorld).")
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
            .def(
                    "__mul__",
                    [](const Sim3d& self, const Sim3d& other) {
                        return self * other;
                    },
                    py::is_operator())
            .def(
                    "__mul__",
                    [](const Sim3d& self, const Eigen::Vector3d& point) {
                        return self * point;
                    },
                    py::is_operator())
            .def("matrix", &Sim3d::ToMatrix,
                 "The 3x4 [s*R | t] matrix representation.")
            .def("__repr__", [](const Sim3d& self) {
                return "Sim3d(scale=" + std::to_string(self.scale()) + ")";
            });

    // Upstream pycolmap geometry free functions (src/pycolmap/geometry/
    // rigid3.cc). interpolate_camera_poses is not bound: the fork engine
    // does not implement InterpolateCameraPoses yet (recorded in the
    // alignment manifest).
    m_geometry
            .def(
                    "average_quaternions",
                    [](const std::vector<Eigen::Vector4d>& quats,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>
                               weights_array) {
                        // The repo-wide std::vector<double> caster
                        // (DoubleVector) only accepts numpy arrays; forcecast
                        // here so python lists work like upstream pycolmap.
                        py::buffer_info info = weights_array.request();
                        std::vector<double> weights(
                                static_cast<double*>(info.ptr),
                                static_cast<double*>(info.ptr) + info.size);
                        return colmap::AverageQuaternions(quats, weights);
                    },
                    "quats"_a, "weights"_a,
                    "Weighted-average a list of (w, x, y, z) quaternions.")
            .def(
                    "get_covariance_for_inverse",
                    [](const Rigid3d& rigid3, const Eigen::Matrix6d& covar) {
                        return colmap::GetCovarianceForRigid3dInverse(rigid3,
                                                                      covar);
                    },
                    "rigid3d"_a, "covar"_a,
                    "Propagate a 6x6 pose covariance through an inversion.")
            .def(
                    "get_covariance_for_composed_rigid3d",
                    [](const Rigid3d& a_from_b,
                       const Eigen::Matrix<double, 12, 12>& covar) {
                        return colmap::GetCovarianceForComposedRigid3d(a_from_b,
                                                                       covar);
                    },
                    "a_from_b"_a, "covar"_a,
                    "Propagate a 12x12 covariance of two composed rigid "
                    "transforms to a 6x6 covariance of the composition.")
            .def(
                    "get_covariance_for_relative_rigid3d",
                    [](const Rigid3d& a_from_c, const Rigid3d& b_from_c,
                       const Eigen::Matrix<double, 12, 12>& covar) {
                        return colmap::GetCovarianceForRelativeRigid3d(
                                a_from_c, b_from_c, covar);
                    },
                    "a_from_c"_a, "b_from_c"_a, "covar"_a,
                    "Propagate a 12x12 covariance to the 6x6 covariance of "
                    "the relative transform.");

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

    // Upstream pycolmap parity: pose interpolation on Rigid3d (fork engine
    // geometry/pose.cc InterpolateCameraPoses) and the gravity-based rot90
    // helper on PosePrior.
    m_geometry.def(
            "interpolate_camera_poses",
            [](const Rigid3d& cam1_from_world, const Rigid3d& cam2_from_world,
               double t) {
                return colmap::InterpolateCameraPoses(cam1_from_world,
                                                      cam2_from_world, t);
            },
            "cam1_from_world"_a, "cam2_from_world"_a, "t"_a,
            "Interpolate two camera poses (slerp rotation, lerp "
            "translation); t in [0, 1].");

    // Upstream pycolmap parity (d3ccaf35 geometry/gps.cc): UTM conversions.
    m_geometry
            .def(
                    "ellipsoid_to_utm",
                    [](const GPSTransform& self,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               lat_lon_alt) {
                        THROW_CHECK_EQ(lat_lon_alt.ndim(), 2);
                        THROW_CHECK_EQ(lat_lon_alt.shape(1), 3);
                        std::vector<Eigen::Vector3d> lla(lat_lon_alt.shape(0));
                        auto buf = lat_lon_alt.unchecked<2>();
                        for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
                            lla[i] = Eigen::Vector3d(buf(i, 0), buf(i, 1),
                                                     buf(i, 2));
                        }
                        auto utm = self.EllipsoidToUTM(lla);
                        py::dict out;
                        out["xyz_in_utm"] = py::cast(utm.first);
                        out["zone"] = utm.second;
                        return out;
                    },
                    "gps"_a, "lat_lon_alt"_a,
                    "Convert ellipsoidal coordinates to UTM; returns a dict "
                    "with "
                    "xyz_in_utm and zone.")
            .def(
                    "utm_to_ellipsoid",
                    [](const GPSTransform& self,
                       const py::array_t<double, py::array::c_style |
                                                         py::array::forcecast>&
                               xyz_in_utm,
                       int zone, bool is_north) {
                        THROW_CHECK_EQ(xyz_in_utm.ndim(), 2);
                        THROW_CHECK_EQ(xyz_in_utm.shape(1), 3);
                        std::vector<Eigen::Vector3d> utm(xyz_in_utm.shape(0));
                        auto buf = xyz_in_utm.unchecked<2>();
                        for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
                            utm[i] = Eigen::Vector3d(buf(i, 0), buf(i, 1),
                                                     buf(i, 2));
                        }
                        return self.UTMToEllipsoid(utm, zone, is_north);
                    },
                    "gps"_a, "xyz_in_utm"_a, "zone"_a, "is_north"_a,
                    "Convert UTM coordinates back to ellipsoidal coordinates.");

    py::class_<PosePrior> pose_prior(
            m_geometry, "PosePrior",
            "Prior information about the pose of a sensor.");
    pose_prior.def(py::init<>())
            .def_static("compute_rot90_from_gravity",
                        &colmap::ComputeRot90FromGravity, "gravity"_a,
                        "Compute the number of 90 deg CCW rotations needed "
                        "to make the sensor upright from its gravity "
                        "direction.")
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

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/geometry/geometry.h"

#include "geometry/pose.h"
#include "pybind/docstring.h"

namespace cloudViewer {
namespace reconstruction {
namespace geometry {

// The COLMAP fork engine types live in namespace colmap.
using colmap::Inverse;
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
                    })
            .def_property(
                    "translation",
                    [](const Rigid3d& self) {
                        return Eigen::Vector3d(self.translation());
                    },
                    [](Rigid3d& self, const Eigen::Vector3d& t) {
                        self.translation() = t;
                    })
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
                    })
            .def_property(
                    "translation",
                    [](const Sim3d& self) {
                        return Eigen::Vector3d(self.translation());
                    },
                    [](Sim3d& self, const Eigen::Vector3d& t) {
                        self.translation() = t;
                    })
            .def(
                    "inverse", [](const Sim3d& self) { return Inverse(self); },
                    "Return the inverse transform.")
            .def("__repr__", [](const Sim3d& self) {
                return "Sim3d(scale=" + std::to_string(self.scale()) + ")";
            });

    m_geometry.attr("__docstring__") = map_shared_argument_docstrings;
}

}  // namespace geometry
}  // namespace reconstruction
}  // namespace cloudViewer

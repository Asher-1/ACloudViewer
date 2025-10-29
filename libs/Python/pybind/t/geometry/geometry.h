// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cloudViewer/core/Device.h"
#include "pybind/cloudViewer_pybind.h"
#include "t/geometry/Geometry.h"

namespace cloudViewer {
namespace t {
namespace geometry {

// Geometry trampoline class.
template <class GeometryBase = Geometry>
class PyGeometry : public GeometryBase {
public:
    using GeometryBase::GeometryBase;

    GeometryBase& Clear() override {
        PYBIND11_OVERLOAD_PURE(GeometryBase&, GeometryBase, );
    }

    bool IsEmpty() const override {
        PYBIND11_OVERLOAD_PURE(bool, GeometryBase, );
    }

    core::Device GetDevice() const override {
        PYBIND11_OVERLOAD_PURE(core::Device, GeometryBase, );
    }
};

void pybind_geometry(py::module& m);
void pybind_geometry_class(py::module& m);
void pybind_drawable_geometry(py::module& m);
void pybind_tensormap(py::module& m);
void pybind_pointcloud(py::module& m);
void pybind_lineset(py::module& m);
void pybind_trianglemesh(py::module& m);
void pybind_image(py::module& m);
void pybind_boundingvolume(py::module& m);
void pybind_voxel_block_grid(py::module& m);
void pybind_raycasting_scene(py::module& m);

}  // namespace geometry
}  // namespace t
}  // namespace cloudViewer

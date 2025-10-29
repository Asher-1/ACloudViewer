// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace geometry {

void pybind_geometry(py::module &m);

void pybind_cloudbase(py::module &m);
void pybind_pointcloud(py::module &m);
void pybind_keypoint(py::module &m);
void pybind_voxelgrid(py::module &m);
void pybind_lineset(py::module &m);
void pybind_meshbase(py::module &m);
void pybind_trianglemesh(py::module &m);
void pybind_primitives(py::module &m);
void pybind_facet(py::module &m);
void pybind_polyline(py::module &m);
void pybind_halfedgetrianglemesh(py::module &m);
void pybind_image(py::module &m);
void pybind_tetramesh(py::module &m);
void pybind_kdtreeflann(py::module &m);
void pybind_cloudbase_methods(py::module &m);
void pybind_pointcloud_methods(py::module &m);
void pybind_voxelgrid_methods(py::module &m);
void pybind_meshbase_methods(py::module &m);
void pybind_trianglemesh_methods(py::module &m);
void pybind_primitives_methods(py::module &m);
void pybind_facet_methods(py::module &m);
void pybind_polyline_methods(py::module &m);
void pybind_lineset_methods(py::module &m);
void pybind_image_methods(py::module &m);
void pybind_octree_methods(py::module &m);
void pybind_octree(py::module &m);
void pybind_boundingvolume(py::module &m);

}  // namespace geometry
}  // namespace cloudViewer

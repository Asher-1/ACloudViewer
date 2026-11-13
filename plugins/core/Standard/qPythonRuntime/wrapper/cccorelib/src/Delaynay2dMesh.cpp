// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include <Delaunay2dMesh.h>
#include <GenericIndexedCloud.h>
#include <GenericIndexedCloudPersist.h>
#include <GenericIndexedMesh.h>

#include "wrappers.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_Delaunay2dMesh(py::module &cccorelib)
{
    py::class_<cloudViewer::Delaunay2dMesh, cloudViewer::GenericIndexedMesh>(cccorelib, "Delaunay2dMesh")
        .def_property_readonly_static("USE_ALL_POINTS",
                                      [](const py::object & /* self */)
                                      { return cloudViewer::Delaunay2dMesh::USE_ALL_POINTS; })
        .def(py::init<>())
        .def_static("Available", &cloudViewer::Delaunay2dMesh::Available)
        .def(
            "linkMeshWith", &cloudViewer::Delaunay2dMesh::linkMeshWith, "aCloud"_a, "passOwnership"_a = false)
        .def(
            "buildMesh",
            [](cloudViewer::Delaunay2dMesh &mesh, py::handle points, std::size_t count, std::string &error)
            {
                return PyCC::WithContainer<std::vector<CCVector2>>(
                    points, [&](const auto &vertices) { return mesh.buildMesh(vertices, count, error); });
            },
            "points2D"_a,
            "pointCountToUse"_a,
            "outputErrorStr"_a)
        .def(
            "buildMesh",
            [](cloudViewer::Delaunay2dMesh &mesh,
               py::handle points,
               const std::vector<int> &segments,
               std::string &error)
            {
                return PyCC::WithContainer<std::vector<CCVector2>>(
                    points, [&](const auto &vertices) { return mesh.buildMesh(vertices, segments, error); });
            },
            "points2D"_a,
            "segments2D"_a,
            "outputErrorStr"_a)
        .def(
            "removeOuterTriangles",
            [](cloudViewer::Delaunay2dMesh &mesh, py::handle vertices, py::handle polygon, bool outside)
            {
                return PyCC::WithContainer<std::vector<CCVector2>>(
                    vertices,
                    [&](const auto &points)
                    {
                        return PyCC::WithContainer<std::vector<CCVector2>>(
                            polygon,
                            [&](const auto &contour)
                            { return mesh.removeOuterTriangles(points, contour, outside); });
                    });
            },
            "vertices2D"_a,
            "polygon2D"_a,
            "removeOutside"_a = true)
        .def("getTriangleVertIndexesArray",
             &cloudViewer::Delaunay2dMesh::getTriangleVertIndexesArray,
             py::return_value_policy::reference)
        .def("getTriangleVertIndexesArray",
             &cloudViewer::Delaunay2dMesh::removeTrianglesWithEdgesLongerThan,
             "maxEdgeLength"_a)
        .def("getAssociatedCloud",
             &cloudViewer::Delaunay2dMesh::getAssociatedCloud,
             py::return_value_policy::reference)
        .def_static("TesselateContour",
                    (cloudViewer::Delaunay2dMesh *
                     (*)(const std::vector<CCVector2> &))(&cloudViewer::Delaunay2dMesh::TesselateContour),
                    "contourPoints"_a)
        .def_static("TesselateContour",
                    (cloudViewer::Delaunay2dMesh * (*)(cloudViewer::GenericIndexedCloudPersist *, int))(
                        &cloudViewer::Delaunay2dMesh::TesselateContour),
                    "contourPoints"_a,
                    "flatDimension"_a = -1)
        // Keep the typed cloud overload ahead of the legacy Python sequence adapter.
        .def_static(
            "TesselateContour",
            [](py::handle points)
            {
                return PyCC::WithContainer<std::vector<CCVector2>>(
                    points,
                    [](const auto &contour)
                    { return cloudViewer::Delaunay2dMesh::TesselateContour(contour); });
            },
            "contourPoints"_a);
}

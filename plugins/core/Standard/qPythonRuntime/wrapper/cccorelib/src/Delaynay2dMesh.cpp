// ##########################################################################
// #                                                                        #
// #                ACLOUDVIEWER PLUGIN: PythonRuntime                       #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 of the License.               #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #                   COPYRIGHT: Thomas Montaigu                           #
// #                                                                        #
// ##########################################################################

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include <Delaunay2dMesh.h>
#include <GenericIndexedCloud.h>
#include <GenericIndexedCloudPersist.h>
#include <GenericIndexedMesh.h>

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
        .def("buildMesh",
             (bool(cloudViewer::Delaunay2dMesh::*)(const std::vector<CCVector2> &,
                                                   std::size_t,
                                                   std::string &))(&cloudViewer::Delaunay2dMesh::buildMesh),
             "points2D"_a,
             "pointCountToUse"_a,
             "outputErrorStr"_a)
        .def("buildMesh",
             (bool(cloudViewer::Delaunay2dMesh::*)(const std::vector<CCVector2> &,
                                                   const std::vector<int> &,
                                                   std::string &))(&cloudViewer::Delaunay2dMesh::buildMesh),
             "points2D"_a,
             "segments2D"_a,
             "outputErrorStr"_a)
        .def("removeOuterTriangles",
             (bool(cloudViewer::Delaunay2dMesh::*)(
                 const std::vector<CCVector2> &, const std::vector<CCVector2> &, bool removeOutside))(
                 &cloudViewer::Delaunay2dMesh::removeOuterTriangles),
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
                    "flatDimension"_a = -1);
}

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ecvGenericMesh.h>
#include <ecvMaterialSet.h>
#include <ecvPointCloud.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccGenericMesh(py::module &m)
{
    py::class_<ccGenericMesh, cloudViewer::GenericIndexedMesh, ccHObject>(m, "ccGenericMesh")
        .def("getAssociatedCloud",
             &ccGenericMesh::getAssociatedCloud,
             py::return_value_policy::reference)
        .def("refreshBB", &ccGenericMesh::refreshBB)
        .def("capacity", &ccGenericMesh::capacity)
        .def("hasMaterials", &ccGenericMesh::hasMaterials)
        .def("getMaterialSet", &ccGenericMesh::getMaterialSet, py::return_value_policy::reference)
        .def("getTriangleMtlIndex", &ccGenericMesh::getTriangleMtlIndex, "triangleIndex"_a)
        .def("hasTextures", &ccGenericMesh::hasTextures)
        .def("getTexCoordinatesTable",
             &ccGenericMesh::getTexCoordinatesTable,
             py::return_value_policy::reference)
        .def(
            "getTriangleTexCoordinates",
            [](ccGenericMesh &self, unsigned triIndex)
            {
                TexCoords2D *tx1, *tx2, *tx3;
                self.getTriangleTexCoordinates(triIndex, tx1, tx2, tx3);
                return py::make_tuple(tx1, tx2, tx2);
            },
            "triIndex"_a,
            py::return_value_policy::reference)
        .def("hasPerTriangleTexCoordinates", &ccGenericMesh::hasPerTriangleTexCoordIndexes)
        .def(
            "getTriangleTexCoordinatesIndexes",
            [](ccGenericMesh &self, unsigned triangleIndex)
            {
                int i1, i2, i3;
                self.getTriangleTexCoordinatesIndexes(triangleIndex, i1, i2, i3);
                return py::make_tuple(i1, i2, i3);
            },
            "triangleIndex"_a)
        .def("hasTriNormals", &ccGenericMesh::hasTriNormals)
        .def(
            "getTriangleNormalIndexes",
            [](ccGenericMesh &self, unsigned triangleIndex)
            {
                int i1, i2, i3;
                self.getTriangleNormalIndexes(triangleIndex, i1, i2, i3);
                return py::make_tuple(i1, i2, i3);
            },
            "triangleIndex"_a)
        .def(
            "getTriangleNormals",
            [](ccGenericMesh &self, unsigned triangleIndex)
            {
                CCVector3 Na, Nb, Nc;
                self.getTriangleNormals(triangleIndex, Na, Nb, Nc);
                return py::make_tuple(Na, Nb, Nc);
            },
            "triangleIndex"_a);
    // FIXME blocked by NormsIndexesTableType not being wrappable due to link errors
    //.def("getTriNormsTable", &ccGenericMesh::getTriNormsTable,
    // py::return_value_policy::reference);
    // TODD compute interpolation weights and following
}

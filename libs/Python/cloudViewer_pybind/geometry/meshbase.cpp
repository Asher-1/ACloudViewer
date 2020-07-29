// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <GenericMesh.h>

#include "cloudViewer_pybind/docstring.h"
#include "cloudViewer_pybind/geometry/geometry.h"
#include "cloudViewer_pybind/geometry/geometry_trampoline.h"

using namespace CVLib;

void pybind_meshbase(py::module &m) {
    py::class_<GenericMesh, PyGenericMesh<GenericMesh>, std::shared_ptr<GenericMesh>>
            meshbase(m, "GenericMesh",
                     "GenericMesh class. Triangle mesh contains vertices. "
                     "Optionally, the mesh "
                     "may also contain vertex normals and vertex colors.");
    py::enum_<GenericMesh::SimplificationContraction>(
            m, "SimplificationContraction")
            .value("Average",
                   GenericMesh::SimplificationContraction::Average,
                   "The vertex positions are computed by the averaging.")
            .value("Quadric",
                   GenericMesh::SimplificationContraction::Quadric,
                   "The vertex positions are computed by minimizing the "
                   "distance to the adjacent triangle planes.")
            .export_values();
    py::enum_<GenericMesh::FilterScope>(m, "FilterScope")
            .value("All", GenericMesh::FilterScope::All,
                   "All properties (color, normal, vertex position) are "
                   "filtered.")
            .value("Color", GenericMesh::FilterScope::Color,
                   "Only the color values are filtered.")
            .value("Normal", GenericMesh::FilterScope::Normal,
                   "Only the normal values are filtered.")
            .value("Vertex", GenericMesh::FilterScope::Vertex,
                   "Only the vertex positions are filtered.")
            .export_values();
	py::enum_<GenericMesh::DeformAsRigidAsPossibleEnergy>(
		m, "DeformAsRigidAsPossibleEnergy")
		.value("Spokes",
			GenericMesh::DeformAsRigidAsPossibleEnergy::Spokes,
			"is the original energy as formulated in orkine and Alexa, "
			"\"As-Rigid-As-Possible Surface Modeling\", 2007.")
		.value("Smoothed",
			GenericMesh::DeformAsRigidAsPossibleEnergy::Smoothed,
			"adds a rotation smoothing term to the rotations.")
		.export_values();

	meshbase.def("__repr__",
		[](const GenericMesh &mesh) {
		return std::string("GenericMesh with ") +
			std::to_string(mesh.size()) + " triangles";
	})
	.def("size", &GenericMesh::size,
		"Returns the number of triangles.")
	.def("has_triangles", &GenericMesh::hasTriangles,
		"Returns whether triangles are empty.");

    docstring::ClassMethodDocInject(m, "GenericMesh", "size");
    docstring::ClassMethodDocInject(m, "GenericMesh", "has_triangles");
}

void pybind_meshbase_methods(py::module &m) {}

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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
#include <ecvMeshBase.h>
#include <ecvPointCloud.h>

#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

#ifdef CV_WINDOWS
#pragma warning(disable : 4715)
#endif

using namespace cloudViewer;
namespace cloudViewer {
namespace geometry {

void pybind_meshbase(py::module& m) {
    py::enum_<CC_TRIANGULATION_TYPES>(m, "TriangulationType")
            .value("DELAUNAY_2D_AXIS_ALIGNED",
                   CC_TRIANGULATION_TYPES::DELAUNAY_2D_AXIS_ALIGNED,
                   "Triangulation types.")
            .value("DELAUNAY_2D_BEST_LS_PLANE",
                   CC_TRIANGULATION_TYPES::DELAUNAY_2D_BEST_LS_PLANE,
                   "Triangulation types.")
            .export_values();
    py::class_<GenericMesh, PyGenericMesh<GenericMesh>,
               std::shared_ptr<GenericMesh>>
            meshbase(m, "GenericMesh",
                     "GenericMesh class. Triangle mesh contains vertices. "
                     "Optionally, the mesh "
                     "may also contain vertex normals and vertex colors.");
    py::enum_<GenericMesh::SimplificationContraction>(
            m, "SimplificationContraction")
            .value("Average", GenericMesh::SimplificationContraction::Average,
                   "The vertex positions are computed by the averaging.")
            .value("Quadric", GenericMesh::SimplificationContraction::Quadric,
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
            .value("Spokes", GenericMesh::DeformAsRigidAsPossibleEnergy::Spokes,
                   "is the original energy as formulated in orkine and Alexa, "
                   "\"As-Rigid-As-Possible Surface Modeling\", 2007.")
            .value("Smoothed",
                   GenericMesh::DeformAsRigidAsPossibleEnergy::Smoothed,
                   "adds a rotation smoothing term to the rotations.")
            .export_values();

    meshbase.def("__repr__",
                 [](const GenericMesh& mesh) {
                     return std::string("GenericMesh with ") +
                            std::to_string(mesh.size()) + " triangles";
                 })
            .def("size", &GenericMesh::size, "Returns the number of triangles.")
            .def("has_triangles", &GenericMesh::hasTriangles,
                 "Returns whether triangles are empty.")
            .def(
                    "get_bbox_corner",
                    [](GenericMesh& mesh) {
                        CCVector3 bbMin, bbMax;
                        mesh.getBoundingBox(bbMin, bbMax);
                        return std::make_tuple(CCVector3d::fromArray(bbMin),
                                               CCVector3d::fromArray(bbMax));
                    },
                    "Returns the mesh bounding-box.")
            .def(
                    "get_next_triangle",
                    [](GenericMesh& mesh) {
                        if (mesh._getNextTriangle()) {
                            return std::ref(*mesh._getNextTriangle());
                        } else {
                            cloudViewer::utility::LogWarning(
                                    "[GenericMesh] does not have next "
                                    "triangle!");
                        }
                    },
                    "Returns the next triangle (relatively to the global "
                    "iterator position).")
            .def("place_iterator_at_beginning",
                 &GenericMesh::placeIteratorAtBeginning,
                 "Places the mesh iterator at the beginning.");

    docstring::ClassMethodDocInject(m, "GenericMesh", "size");
    docstring::ClassMethodDocInject(m, "GenericMesh", "has_triangles");
    docstring::ClassMethodDocInject(m, "GenericMesh", "get_bbox_corner");
    docstring::ClassMethodDocInject(m, "GenericMesh", "get_next_triangle");
    docstring::ClassMethodDocInject(m, "GenericMesh",
                                    "place_iterator_at_beginning");

    py::class_<GenericTriangle, std::shared_ptr<GenericTriangle>>
            genericTriangle(m, "GenericTriangle",
                            "GenericTriangle, a generic triangle interface.");
    genericTriangle
            .def(
                    "get_A",
                    [](const GenericTriangle& triangle) {
                        return CCVector3d::fromArray(*triangle._getA());
                    },
                    "Returns the first vertex (A).")
            .def(
                    "get_B",
                    [](const GenericTriangle& triangle) {
                        return CCVector3d::fromArray(*triangle._getB());
                    },
                    "Returns second vertex (B).")
            .def(
                    "get_C",
                    [](const GenericTriangle& triangle) {
                        return CCVector3d::fromArray(*triangle._getC());
                    },
                    "Returns the third vertex (C).");

    docstring::ClassMethodDocInject(m, "GenericTriangle", "get_A");
    docstring::ClassMethodDocInject(m, "GenericTriangle", "get_B");
    docstring::ClassMethodDocInject(m, "GenericTriangle", "get_C");

    py::class_<cloudViewer::VerticesIndexes,
               std::shared_ptr<cloudViewer::VerticesIndexes>>
            verticesindexes(m, "VerticesIndexes",
                            "VerticesIndexes, Triangle described by the "
                            "indexes of its 3 vertices.");
    py::detail::bind_default_constructor<cloudViewer::VerticesIndexes>(
            verticesindexes);
    py::detail::bind_copy_functions<cloudViewer::VerticesIndexes>(
            verticesindexes);
    verticesindexes.def(py::init<>())
            .def(py::init([](unsigned i1, unsigned i2, unsigned i3) {
                     return new cloudViewer::VerticesIndexes(i1, i2, i3);
                 }),
                 "Constructor with specified indexes", "i1"_a, "i2"_a, "i3"_a);

    py::class_<cloudViewer::GenericIndexedMesh,
               PyGenericIndexedMesh<cloudViewer::GenericIndexedMesh>,
               std::shared_ptr<cloudViewer::GenericIndexedMesh>,
               cloudViewer::GenericMesh>
            genericIndexedMesh(
                    m, "GenericIndexedMesh",
                    "GenericIndexedMesh with index-based vertex access.");
    genericIndexedMesh
            .def("__repr__",
                 [](const cloudViewer::GenericIndexedMesh& mesh) {
                     return std::string("GenericIndexedMesh with ") +
                            std::to_string(mesh.size()) + " triangles";
                 })
            .def(
                    "get_triangle",
                    [](cloudViewer::GenericIndexedMesh& mesh,
                       size_t triangle_index) {
                        if (mesh._getTriangle(
                                    static_cast<unsigned>(triangle_index))) {
                            return std::ref(*mesh._getTriangle(
                                    static_cast<unsigned>(triangle_index)));
                        } else {
                            cloudViewer::utility::LogWarning(
                                    "[cloudViewer::GenericIndexedMesh] does "
                                    "not have triangle!");
                        }
                    },
                    "Returns the ith triangle.", "triangle_index"_a)
            .def(
                    "get_vertice_indexes",
                    [](cloudViewer::GenericIndexedMesh& mesh,
                       size_t triangle_index) {
                        if (mesh.getTriangleVertIndexes(
                                    static_cast<unsigned>(triangle_index))) {
                            return std::ref(*mesh.getTriangleVertIndexes(
                                    static_cast<unsigned>(triangle_index)));
                        } else {
                            cloudViewer::utility::LogWarning(
                                    "[cloudViewer::GenericIndexedMesh] does "
                                    "not have vertice indexes!");
                        }
                    },
                    "Returns the indexes of the vertices of a given triangle.",
                    "triangle_index"_a)
            .def(
                    "get_triangle_vertices",
                    [](const cloudViewer::GenericIndexedMesh& mesh,
                       size_t triangle_index) {
                        CCVector3 A, B, C;
                        mesh.getTriangleVertices(
                                static_cast<unsigned>(triangle_index), A, B, C);
                        return std::make_tuple(CCVector3d::fromArray(A),
                                               CCVector3d::fromArray(B),
                                               CCVector3d::fromArray(C));
                    },
                    "Returns the vertices of a given triangle.",
                    "triangle_index"_a)
            .def(
                    "get_next_vertice_indexes",
                    [](cloudViewer::GenericIndexedMesh& mesh) {
                        if (mesh.getNextTriangleVertIndexes()) {
                            return std::ref(*mesh.getNextTriangleVertIndexes());
                        } else {
                            cloudViewer::utility::LogWarning(
                                    "[cloudViewer::GenericIndexedMesh] does "
                                    "not have next vertice indexes!");
                        }
                    },
                    "Returns the indexes of the vertices of the next triangle "
                    "(relatively to the global iterator position).");

    docstring::ClassMethodDocInject(m, "GenericIndexedMesh", "get_triangle");
    docstring::ClassMethodDocInject(m, "GenericIndexedMesh",
                                    "get_triangle_vertices");
    docstring::ClassMethodDocInject(m, "GenericIndexedMesh",
                                    "get_vertice_indexes");
    docstring::ClassMethodDocInject(m, "GenericIndexedMesh",
                                    "get_next_vertice_indexes");

    py::class_<ccGenericMesh, PyGeometry<ccGenericMesh>,
               std::shared_ptr<ccGenericMesh>, cloudViewer::GenericIndexedMesh,
               ccHObject>
            genericMesh(m, "ccGenericMesh", py::multiple_inheritance(),
                        "ccGenericMesh class. Generic mesh interface.");
    genericMesh
            .def("__repr__",
                 [](const ccGenericMesh& mesh) {
                     return std::string("ccGenericMesh with ") +
                            std::to_string(mesh.size()) + " triangles";
                 })
            .def(
                    "get_associated_cloud",
                    [](const ccGenericMesh& mesh) {
                        if (mesh.getAssociatedCloud()) {
                            return std::ref(*mesh.getAssociatedCloud());
                        } else {
                            cloudViewer::utility::LogWarning(
                                    "[ccGenericMesh] does not have associated "
                                    "cloud!");
                        }
                    },
                    "Returns the associated cloud.")
            .def("refresh_bbox", &ccGenericMesh::refreshBB,
                 "Forces bounding-box update.")
            .def("capacity", &ccGenericMesh::capacity, "Returns max capacity.")
            .def("has_materials", &ccGenericMesh::hasMaterials,
                 "Returns whether the mesh has materials/textures.")
            .def("has_textures", &ccGenericMesh::hasTextures,
                 "Returns whether textures are available for this mesh.")
            .def("has_triangle_normals", &ccGenericMesh::hasTriNormals,
                 "Returns whether the mesh has per-triangle normals.")
            .def(
                    "get_triangle_normal_indexes",
                    [](const ccGenericMesh& mesh, unsigned triangle_index) {
                        int i1, i2, i3;
                        mesh.getTriangleNormalIndexes(triangle_index, i1, i2,
                                                      i3);
                        return std::make_tuple(i1, i2, i3);
                    },
                    "Returns a triplet of normal indexes for a given triangle "
                    "(if any).",
                    "triangle_index"_a)
            .def(
                    "get_triangle_normals",
                    [](const ccGenericMesh& mesh, unsigned triangle_index) {
                        Eigen::Vector3d Na, Nb, Nc;
                        mesh.getTriangleNormals(triangle_index, Na, Nb, Nc);
                        return std::make_tuple(Na, Nb, Nc);
                    },
                    "Returns a given triangle normal.", "triangle_index"_a)
            .def(
                    "interpolate_normals",
                    [](ccGenericMesh& mesh, unsigned triangle_index,
                       const Eigen::Vector3d& point) {
                        CCVector3 normal;
                        mesh.interpolateNormals(triangle_index, point, normal);
                        return CCVector3d::fromArray(normal);
                    },
                    "Interpolates normal(s) inside a given triangle.",
                    "triangle_index"_a, "point"_a)
            .def(
                    "compute_interpolation_weights",
                    [](const ccGenericMesh& mesh, unsigned triangle_index,
                       const Eigen::Vector3d& point) {
                        CCVector3d weights;
                        mesh.computeInterpolationWeights(triangle_index, point,
                                                         weights);
                        return CCVector3d::fromArray(weights);
                    },
                    "Returns the (barycentric) interpolation weights for a "
                    "given triangle.",
                    "triangle_index"_a, "point"_a)
            .def(
                    "interpolate_colors",
                    [](ccGenericMesh& mesh, unsigned triangle_index,
                       const Eigen::Vector3d& point) {
                        ecvColor::Rgb color;
                        bool success = mesh.interpolateColors(triangle_index,
                                                              point, color);
                        return std::make_tuple(success,
                                               ecvColor::Rgb::ToEigen(color));
                    },
                    "Interpolates RGB colors inside a given triangle.",
                    "triangle_index"_a, "point"_a)
            .def(
                    "get_color_from_material",
                    [](ccGenericMesh& mesh, unsigned triangle_index,
                       const Eigen::Vector3d& point,
                       bool interpolate_color_if_no_texture) {
                        ecvColor::Rgb color;
                        bool success = mesh.getColorFromMaterial(
                                triangle_index, point, color,
                                interpolate_color_if_no_texture);
                        return std::make_tuple(success,
                                               ecvColor::Rgb::ToEigen(color));
                    },
                    "Returns RGB color from a given triangle material/texture.",
                    "triangle_index"_a, "point"_a,
                    "interpolate_color_if_no_texture"_a)
            .def(
                    "get_vertex_color_from_material",
                    [](ccGenericMesh& mesh, unsigned triangle_index,
                       unsigned char vertex_index,
                       bool return_color_if_no_texture) {
                        ecvColor::Rgb color;
                        bool success = mesh.getVertexColorFromMaterial(
                                triangle_index, vertex_index, color,
                                return_color_if_no_texture);
                        return std::make_tuple(success,
                                               ecvColor::Rgb::ToEigen(color));
                    },
                    "Returns RGB color of a vertex from a given triangle "
                    "material/texture.",
                    "triangle_index"_a, "vertex_index"_a,
                    "return_color_if_no_texture"_a)
            .def("is_shown_as_wire", &ccGenericMesh::isShownAsWire,
                 "Returns whether the mesh is displayed as wired or with plain "
                 "facets.")
            .def("show_wired", &ccGenericMesh::showWired,
                 "Sets whether mesh should be displayed as a wire or with "
                 "plain facets.",
                 "state"_a)
            .def("is_shown_as_points", &ccGenericMesh::isShownAsPoints,
                 "Returns whether the mesh is displayed as wired or with plain "
                 "facets.")
            .def("show_points", &ccGenericMesh::showPoints,
                 "Sets whether mesh should be displayed as a point cloud or "
                 "with plain facets.",
                 "state"_a)
            .def("triangle_norms_shown", &ccGenericMesh::triNormsShown,
                 "Returns whether per-triangle normals are shown or not .")
            .def("show_triangle_norms", &ccGenericMesh::showTriNorms,
                 "Sets whether to show or not per-triangle normals.", "state"_a)
            .def("materials_shown", &ccGenericMesh::materialsShown,
                 "Sets whether textures/material should be displayed or not.")
            .def("show_materials", &ccGenericMesh::showMaterials,
                 "Sets whether textures should be displayed or not.", "state"_a)
            .def("stippling_enabled", &ccGenericMesh::stipplingEnabled,
                 "Returns whether polygon stippling is enabled or not.")
            .def("enable_stippling", &ccGenericMesh::enableStippling,
                 "Enables polygon stippling.", "state"_a)
            .def(
                    "sample_points",
                    [](ccGenericMesh& mesh, bool density_based,
                       double sampling_parameter, bool with_normals,
                       bool with_rgb, bool with_texture) {
                        ccPointCloud* cloud = mesh.samplePoints(
                                density_based, sampling_parameter, with_normals,
                                with_rgb, with_texture, nullptr);
                        return std::shared_ptr<ccPointCloud>(cloud);
                    },
                    "Samples points on a mesh.", "density_based"_a,
                    "sampling_parameter"_a, "with_normals"_a, "with_rgb"_a,
                    "with_texture"_a)
            .def(
                    "import_parameters_from",
                    [](ccGenericMesh& mesh, const ccGenericMesh& source) {
                        mesh.importParametersFrom(&source);
                    },
                    "Imports the parameters from another mesh.", "source"_a);

    docstring::ClassMethodDocInject(m, "ccGenericMesh", "get_associated_cloud");
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "refresh_bbox");
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "capacity");
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "has_materials");
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "has_textures");
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "has_triangle_normals");
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "get_triangle_normals");
    docstring::ClassMethodDocInject(
            m, "ccGenericMesh", "get_triangle_normal_indexes",
            {{"triangle_index", "triIndex triangle index"}});
    docstring::ClassMethodDocInject(
            m, "ccGenericMesh", "interpolate_normals",
            {{"triangle_index", "triIndex triangle index"},
             {"point",
              "point where to interpolate (should be inside the triangle!)"}});
    docstring::ClassMethodDocInject(
            m, "ccGenericMesh", "interpolate_colors",
            {{"triangle_index", "triIndex triangle index"},
             {"point",
              "point where to interpolate (should be inside the triangle!)"}});
    docstring::ClassMethodDocInject(m, "ccGenericMesh",
                                    "compute_interpolation_weights");
    docstring::ClassMethodDocInject(
            m, "ccGenericMesh", "get_color_from_material",
            {{"triangle_index", "triIndex triangle index"},
             {"point",
              "point where to grab color (should be inside the triangle!)"},
             {"interpolate_color_if_no_texture",
              "whether to return the color interpolated from the RGB field if "
              "no texture/material is associated to the given triangles"}});
    docstring::ClassMethodDocInject(
            m, "ccGenericMesh", "get_vertex_color_from_material",
            {{"triangle_index", "triIndex triangle index"},
             {"vertex_index", "vertex index inside triangle (i.e. 0, 1 or 2!)"},
             {"return_color_if_no_texture",
              "whether to return the color from the vertex RGB field if no "
              "texture/material is associated to the given triangle"}});
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "is_shown_as_wire");
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "show_wired");
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "is_shown_as_points");
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "show_points");
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "triangle_norms_shown");
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "show_triangle_norms");
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "materials_shown");
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "show_materials");
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "stippling_enabled");
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "enable_stippling");
    docstring::ClassMethodDocInject(m, "ccGenericMesh", "sample_points");
    docstring::ClassMethodDocInject(m, "ccGenericMesh",
                                    "import_parameters_from");

    py::class_<ecvMeshBase, PyGeometry<ecvMeshBase>,
               std::shared_ptr<ecvMeshBase>, cloudViewer::GenericMesh,
               ccHObject>
            meshbase2(m, "ecvMeshBase",
                      "ecvMeshBase class. Triangle mesh contains vertices. "
                      "Optionally, the mesh "
                      "may also contain vertex normals and vertex colors.");
    py::detail::bind_default_constructor<ecvMeshBase>(meshbase2);
    py::detail::bind_copy_functions<ecvMeshBase>(meshbase2);

    meshbase2
            .def("__repr__",
                 [](const ecvMeshBase& mesh) {
                     return std::string("ecvMeshBase with ") +
                            std::to_string(mesh.vertices_.size()) + " points";
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("has_vertices", &ecvMeshBase::hasVertices,
                 "Returns ``True`` if the mesh contains vertices.")
            .def("has_vertex_normals", &ecvMeshBase::hasVertexNormals,
                 "Returns ``True`` if the mesh contains vertex normals.")
            .def("has_vertex_colors", &ecvMeshBase::hasVertexColors,
                 "Returns ``True`` if the mesh contains vertex colors.")
            .def("normalize_normals", &ecvMeshBase::normalizeNormals,
                 "Normalize vertex normals to length 1.")
            .def("paint_uniform_color", &ecvMeshBase::paintUniformColor,
                 "Assigns each vertex in the ecvMeshBase the same color.",
                 "color"_a)
            .def("compute_convex_hull", &ecvMeshBase::computeConvexHull,
                 "Computes the convex hull of the triangle mesh.")
            .def_readwrite("vertices", &ecvMeshBase::vertices_,
                           "``float64`` array of shape ``(num_vertices, 3)``, "
                           "use ``numpy.asarray()`` to access data: Vertex "
                           "coordinates.")
            .def_readwrite("vertex_normals", &ecvMeshBase::vertex_normals_,
                           "``float64`` array of shape ``(num_vertices, 3)``, "
                           "use ``numpy.asarray()`` to access data: Vertex "
                           "normals.")
            .def_readwrite(
                    "vertex_colors", &ecvMeshBase::vertex_colors_,
                    "``float64`` array of shape ``(num_vertices, 3)``, "
                    "range ``[0, 1]`` , use ``numpy.asarray()`` to access "
                    "data: RGB colors of vertices.");
    docstring::ClassMethodDocInject(m, "ecvMeshBase", "has_vertex_colors");
    docstring::ClassMethodDocInject(
            m, "ecvMeshBase", "has_vertex_normals",
            {{"normalized",
              "Set to ``True`` to normalize the normal to length 1."}});
    docstring::ClassMethodDocInject(m, "ecvMeshBase", "has_vertices");
    docstring::ClassMethodDocInject(m, "ecvMeshBase", "normalize_normals");
    docstring::ClassMethodDocInject(m, "ecvMeshBase", "paint_uniform_color",
                                    {{"color", "RGB colors of vertices."}});
    docstring::ClassMethodDocInject(m, "ecvMeshBase", "compute_convex_hull");
}

void pybind_meshbase_methods(py::module& m) {}

}  // namespace geometry
}  // namespace cloudViewer
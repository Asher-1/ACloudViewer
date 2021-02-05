// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
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

#include <Console.h>
#include <Image.h>
#include <ecvMesh.h>
#include <ecvPolyline.h>
#include <ecvTetraMesh.h>
#include <ecvPointCloud.h>
#include <ecvHObjectCaster.h>

#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

namespace cloudViewer {
namespace geometry {

void pybind_trianglemesh(py::module &m) {
	py::enum_<ccMesh::MESH_SCALAR_FIELD_PROCESS>(m, "MeshScalarFieldProcessType")
		.value("SMOOTH_MESH_SF", ccMesh::MESH_SCALAR_FIELD_PROCESS::SMOOTH_MESH_SF,
			"Smooth Scalar fields.")
		.value("ENHANCE_MESH_SF", ccMesh::MESH_SCALAR_FIELD_PROCESS::ENHANCE_MESH_SF,
			"Enhance Scalar fields.")
		.export_values();

    py::class_<ccMesh, PyGeometry<ccMesh>, 
			std::shared_ptr<ccMesh>, ccGenericMesh, ccHObject>
            trianglemesh(m, "ccMesh", py::multiple_inheritance(),
                         "ccMesh class. Triangle mesh contains vertices "
                         "and triangles represented by the indices to the "
                         "vertices. Optionally, the mesh may also contain "
                         "triangle normals, vertex normals and vertex colors.");
    py::detail::bind_default_constructor<ccMesh>(trianglemesh);
    py::detail::bind_copy_functions<ccMesh>(trianglemesh);
    trianglemesh.def(py::init<const std::vector<Eigen::Vector3d> &,
                          const std::vector<Eigen::Vector3i> &>(),
                 "Create a triangle mesh from vertices and triangle indices",
                 "vertices"_a, "triangles"_a)
		.def(py::init([](std::shared_ptr<ccGenericPointCloud> cloud) {
				if (cloud) {
					if (!cloud->isKindOf(CV_TYPES::POINT_CLOUD))
					{
						return new ccMesh(nullptr);
					}
					ccPointCloud* vertices = ccHObjectCaster::ToPointCloud(cloud.get());
					return new ccMesh(vertices->cloneThis());
				} else {
					return new ccMesh(nullptr);
				}
				
			}), "Create a triangle mesh from vertices", "cloud"_a = nullptr)
		.def(py::init([](std::shared_ptr<cloudViewer::GenericIndexedMesh> index_mesh,
			std::shared_ptr<ccGenericPointCloud> cloud) {
				cloudViewer::GenericIndexedMesh* indexMesh = nullptr;
				if (index_mesh)
				{
					indexMesh = index_mesh.get();
				}

				if (cloud) {
					if (!cloud->isKindOf(CV_TYPES::POINT_CLOUD))
					{
						return new ccMesh(indexMesh, nullptr);
					}
					ccPointCloud* vertices = ccHObjectCaster::ToPointCloud(cloud.get());
					return new ccMesh(indexMesh, vertices->cloneThis());
				} else {
					return new ccMesh(indexMesh, nullptr);
				}
				
			}), "Create a triangle mesh from index_mesh and cloud", "index_mesh"_a, "cloud"_a )
		.def("__repr__",
			[](const ccMesh &mesh) {
				std::string info = fmt::format(
				"ccMesh with {} points and {} triangles",
				mesh.getVerticeSize(), mesh.size());

				if (mesh.hasEigenTextures()) {
					info += fmt::format(", and textures of size ");
					for (auto &tex : mesh.textures_) {
						info += fmt::format("({}, {}) ", tex.width_,
							tex.height_);
					}
				} else {
					info += ".";
				}
				return info;
			})
		.def(py::self + py::self)
		.def(py::self += py::self)

		.def("clone_mesh", [](ccMesh& mesh, std::shared_ptr<ccGenericPointCloud> cloud) {
				if (!cloud)
				{
					return std::shared_ptr<ccMesh>(mesh.cloneMesh());
				}
				return std::shared_ptr<ccMesh>(mesh.cloneMesh(cloud.get()));
			}, "Returns vertices number.", "cloud"_a = nullptr)
		.def("merge", [](ccMesh& mesh, const ccMesh& input_mesh, bool create_submesh) {
				return mesh.merge(&input_mesh, create_submesh);
			}, "Returns vertices number.", "input_mesh"_a, "create_submesh"_a)
		.def("clear", &ccMesh::clear, "Release internal memory.")
		.def("shift_triangle_indexes", &ccMesh::shiftTriangleIndexes,
			"Shifts all triangles indexes.", "shift"_a)
		.def("reserve", &ccMesh::reserve, 
			"Reserves the memory to store the vertex indexes (3 per triangle).", 
			"triangles_number"_a)
		.def("resize", &ccMesh::resize, 
			"If the new number of elements is smaller than the actual size,"
			"the overflooding elements will be deleted.", "triangles_number"_a)
		.def("flip_triangles", &ccMesh::flipTriangles, "Flips the triangle.")
		.def("shrink_triangles", &ccMesh::shrinkToFit, "Removes unused triangle capacity.")
		.def("shrink_vertexes", &ccMesh::shrinkVertexToFit, "Removes unused vertex capacity.")
		.def("clear_triangle_normals", &ccMesh::clearTriNormals, "Removes per-triangle normals.")
		.def("are_triangle_normals_enabled", &ccMesh::arePerTriangleNormalsEnabled,
			"Returns whether per triangle normals are enabled.")
		.def("reserve_triangle_normal_indexes", &ccMesh::reservePerTriangleNormalIndexes,
			"Reserves memory to store per-triangle triplets of normal indexes.")
		.def("add_triangle_normal_indexes", &ccMesh::addTriangleNormalIndexes,
			"Adds a triplet of normal indexes for next triangle.")
		.def("set_triangle_normal_indexes", 
			py::overload_cast<std::size_t, CompressedNormType>(&ccMesh::setTriangleNormalIndexes),
			"Adds a triplet of normal indexes for next triangle.", 
			"triangle_index"_a, "value"_a)
		.def("set_triangle_normal_indexes", 
			py::overload_cast<unsigned, int, int, int>(&ccMesh::setTriangleNormalIndexes),
			"Adds a triplet of normal indexes for next triangle.", 
			"triangle_index"_a, "i1"_a, "i2"_a, "i3"_a)
		.def("get_triangle_normal_indexes", 
			py::overload_cast<std::size_t>(&ccMesh::getTriangleNormalIndexes),
			"Returns a compressed normal indexes for a given triangle index (if any).",
			"triangle_index"_a)
		.def("remove_triangle_normal_indexes", &ccMesh::removePerTriangleNormalIndexes,
			"Removes any per-triangle triplets of normal indexes.")
		.def("convert_materials_to_vertex_colors", &ccMesh::convertMaterialsToVertexColors,
			"Converts materials to vertex colors.")
		.def("has_material_indexes", &ccMesh::hasPerTriangleMtlIndexes,
			"Returns whether this mesh as per-triangle material index.")
		.def("reserve_material_indexes", &ccMesh::reservePerTriangleMtlIndexes,
			"Reserves memory to store per-triangle material index.")
		.def("remove_material_indexes", &ccMesh::removePerTriangleMtlIndexes,
			"Removes any per-triangle material indexes.")
		.def("add_material_index", &ccMesh::addTriangleMtlIndex,
			"Adds triangle material index for next triangle.", "material_index"_a)
		.def("set_material_index", &ccMesh::setTriangleMtlIndex,
			"Adds triangle material index for next triangle.", "triangle_index"_a, "material_index"_a)
		.def("reserve_texture_coord_indexes", &ccMesh::reservePerTriangleTexCoordIndexes,
			"Reserves memory to store per-triangle triplets of tex coords indexes.")
		.def("remove_texture_coord_indexes", &ccMesh::removePerTriangleTexCoordIndexes,
			"Remove per-triangle tex coords indexes.")
		.def("add_texture_coord_indexes", &ccMesh::addTriangleTexCoordIndexes,
			"Adds a triplet of tex coords indexes for next triangle.",
			"i1"_a, "i2"_a, "i3"_a)
		.def("set_texture_coord_indexes", &ccMesh::setTriangleTexCoordIndexes,
			"Sets a triplet of tex coords indexes for a given triangle.", 
			"triangle_index"_a, "i1"_a, "i2"_a, "i3"_a)
		.def("compute_normals", &ccMesh::computeNormals, "Computes normals.", "per_vertex"_a)
		.def("laplacian_smooth", [](ccMesh& mesh, unsigned iterations, PointCoordinateType factor) {
				return mesh.laplacianSmooth(iterations, factor, nullptr);
			}, "Laplacian smoothing.", "iterations"_a = 100, "factor"_a = 0.01f)
		.def("process_scalar_field", &ccMesh::processScalarField, 
			"Applies process to the mesh scalar field (the one associated to its vertices in fact).",
			"process_type"_a)
		.def("subdivide", [](const ccMesh& mesh, PointCoordinateType max_area) {
				return std::shared_ptr<ccMesh>(mesh.subdivide(max_area));
			}, "Subdivides mesh (so as to ensure that all triangles are falls below 'max_area').",
			"max_area"_a)
		.def("create_mesh_from_selection", [](ccMesh& mesh, bool remove_selected_faces) {
				return std::shared_ptr<ccMesh>(mesh.createNewMeshFromSelection(remove_selected_faces));
			}, "Creates a new mesh with the selected vertices only.",
			"remove_selected_faces"_a)
		.def("swap_triangles", &ccMesh::swapTriangles,
			"Swaps two triangles.", "first_index"_a, "second_index"_a)
		.def("remove_triangles", &ccMesh::removeTriangles,
			"Removes triangles.", "index"_a)
		.def("transform_triangle_normals", [](ccMesh& mesh, const Eigen::Matrix4d& transformation) {
				mesh.transformTriNormals(ccGLMatrix::FromEigenMatrix(transformation));
			}, "Transforms the mesh per-triangle normals.", "transformation"_a)
		.def("get_triangle_area", &ccMesh::getTriangleArea, 
			"Function that computes the area of a mesh triangle identified by the triangle index.",
			"triangle_index"_a)
		.def("is_bbox_intersecting", &ccMesh::isBoundingBoxIntersecting,
			"Function that tests if the bounding boxes of the triangle meshes are intersecting.",
			"other"_a)
		.def("get_edge_to_triangles_map", &ccMesh::getEdgeToTrianglesMap,
			"Function that returns a map from edges (vertex0, vertex1) to the"
			" triangle indices the given edge belongs to.")
		.def("get_edge_to_vertices_map", &ccMesh::getEdgeToVerticesMap,
			"Function that returns a map from edges (vertex0, vertex1) to the"
			" vertex (vertex2) indices the given edge belongs to.")
		.def("get_triangle_plane", &ccMesh::getTrianglePlane,
			"Function that computes the plane equation of a mesh triangle identified by the triangle index."
			"triangle_index"_a)

		.def("vertice_size", &ccMesh::getVerticeSize, "Returns vertices number.")
		.def("set_associated_cloud", [](ccMesh& mesh, ccGenericPointCloud& cloud) {
				if (cloud.isKindOf(CV_TYPES::POINT_CLOUD))
				{
					ccPointCloud* vertices = ccHObjectCaster::ToPointCloud(&cloud);
					mesh.setAssociatedCloud(vertices->cloneThis());
				}
			}, "Sets the associated vertices cloud (warning)", "cloud"_a)
		.def("create_internal_cloud",
			&ccMesh::createInternalCloud,
			"Sets the associated vertices cloud (warning)")
		.def("compute_triangle_normals",
			&ccMesh::computeTriangleNormals,
			"Function to compute triangle normals, usually called before "
			"rendering",
			"normalized"_a = true)
		.def("compute_vertex_normals",
			&ccMesh::computeVertexNormals,
			"Function to compute vertex normals, usually called before "
			"rendering",
			"normalized"_a = true)
		.def("compute_adjacency_list",
			&ccMesh::computeAdjacencyList,
			"Function to compute adjacency list, call before adjacency "
			"list is needed")
		.def("remove_duplicated_vertices",
			&ccMesh::removeDuplicatedVertices,
			"Function that removes duplicated verties, i.e., vertices "
			"that have identical coordinates.")
		.def("remove_duplicated_triangles",
			&ccMesh::removeDuplicatedTriangles,
			"Function that removes duplicated triangles, i.e., removes "
			"triangles that reference the same three vertices, "
			"independent of their order.")
        .def("remove_unreferenced_vertices",
                &ccMesh::removeUnreferencedVertices,
                "This function removes vertices from the triangle mesh that "
                "are not referenced in any triangle of the mesh.")
        .def("remove_degenerate_triangles",
                &ccMesh::removeDegenerateTriangles,
                "Function that removes degenerate triangles, i.e., triangles "
                "that references a single vertex multiple times in a single "
                "triangle. They are usually the product of removing "
                "duplicated vertices.")
        .def("remove_non_manifold_edges",
                &ccMesh::removeNonManifoldEdges,
                "Function that removes all non-manifold edges, by "
                "successively deleting  triangles with the smallest surface "
                "area adjacent to the non-manifold edge until the number of "
                "adjacent triangles to the edge is `<= 2`.")
        .def("merge_close_vertices",
                &ccMesh::mergeCloseVertices,
                "Function that will merge close by vertices to a single one. "
                "The vertex position, "
                "normal and color will be the average of the vertices. The "
                "parameter eps "
                "defines the maximum distance of close by vertices.  This "
                "function might help to "
                "close triangle soups.",
                "eps"_a)
        .def("filter_sharpen", &ccMesh::filterSharpen,
                "Function to sharpen triangle mesh. The output value "
                "(:math:`v_o`) is the input value (:math:`v_i`) plus strength "
                "times the input value minus he sum of he adjacent values. "
                ":math:`v_o = v_i x strength (v_i * |N| - \\sum_{n \\in N} "
                "v_n)`",
                "number_of_iterations"_a = 1, "strength"_a = 1,
                "filter_scope"_a = ccMesh::FilterScope::All)
        .def("filter_smooth_simple",
                &ccMesh::filterSmoothSimple,
                "Function to smooth triangle mesh with simple neighbor "
                "average. :math:`v_o = \\frac{v_i + \\sum_{n \\in N} "
                "v_n)}{|N| + 1}`, with :math:`v_i` being the input value, "
                ":math:`v_o` the output value, and :math:`N` is the set of "
                "adjacent neighbours.",
                "number_of_iterations"_a = 1,
                "filter_scope"_a = ccMesh::FilterScope::All)
        .def("filter_smooth_laplacian",
                &ccMesh::filterSmoothLaplacian,
                "Function to smooth triangle mesh using Laplacian. :math:`v_o "
                "= v_i \\cdot \\lambda (sum_{n \\in N} w_n v_n - v_i)`, with "
                ":math:`v_i` being the input value, :math:`v_o` the output "
                "value, :math:`N` is the  set of adjacent neighbours, "
                ":math:`w_n` is the weighting of the neighbour based on the "
                "inverse distance (closer neighbours have higher weight), and "
                "lambda is the smoothing parameter.",
                "number_of_iterations"_a = 1, "lambda"_a = 0.5,
                "filter_scope"_a = ccMesh::FilterScope::All)
        .def("filter_smooth_taubin",
                &ccMesh::filterSmoothTaubin,
                "Function to smooth triangle mesh using method of Taubin, "
                "\"Curve and Surface Smoothing Without Shrinkage\", 1995. "
                "Applies in each iteration two times filter_smooth_laplacian, "
                "first with filter parameter lambda and second with filter "
                "parameter mu as smoothing parameter. This method avoids "
                "shrinkage of the triangle mesh.",
                "number_of_iterations"_a = 1, "lambda"_a = 0.5, "mu"_a = -0.53,
                "filter_scope"_a = ccMesh::FilterScope::All)
        .def("has_vertices", &ccMesh::hasVertices,
                "Returns ``True`` if the mesh contains vertices.")
        .def("has_triangles", &ccMesh::hasTriangles,
                "Returns ``True`` if the mesh contains triangles.")
        .def("has_vertex_normals",
                &ccMesh::hasNormals,
                "Returns ``True`` if the mesh contains vertex normals.")
        .def("has_vertex_colors", &ccMesh::hasColors,
			"Returns ``True`` if the mesh contains vertex colors.")
		.def("set_triangle", &ccMesh::setTriangle,
			"set triangle indices by index",
			"index"_a, "triangle"_a)
		.def("get_triangle", &ccMesh::getTriangle,
			"get triangle indices by index",
			"index"_a)
		.def("add_triangles", &ccMesh::addTriangles,
			"``int`` array of shape ``(num_triangles, 3)``, use "
			"``numpy.asarray()`` to access data: List of "
			"triangles denoted by the index of points forming "
			"the triangle.", "triangles"_a)
		.def("set_triangles", &ccMesh::setTriangles,
			"``int`` array of shape ``(num_triangles, 3)``, use "
			"``numpy.asarray()`` to access data: List of "
			"triangles denoted by the index of points forming "
			"the triangle.", "triangles"_a)
		.def("get_triangles", &ccMesh::getTriangles,
			"``int`` array of shape ``(num_triangles, 3)``, use "
			"``numpy.asarray()`` to access data: List of "
			"triangles denoted by the index of points forming "
			"the triangle.")
		.def("set_triangle_normal", &ccMesh::setTriangleNorm,
			"set triangle normal by index",
			"index"_a, "triangle_normal"_a)
		.def("get_triangle_normal", &ccMesh::getTriangleNorm,
			"get triangle indices by index",
			"index"_a)
		.def("set_triangle_normals", &ccMesh::setTriangleNorms,
			"``int`` array of shape ``(num_triangles, 3)``, use "
			"``numpy.asarray()`` to access data: List of "
			"triangles denoted by the index of points forming "
			"the triangle.", "triangle_normals"_a)
		.def("get_triangle_normals", &ccMesh::getTriangleNorms,
			"``int`` array of shape ``(num_triangles, 3)``, use "
			"``numpy.asarray()`` to access data: List of "
			"triangles denoted by the index of points forming "
			"the triangle.")
		.def("set_vertice", &ccMesh::setVertice,
			"set vertex coordinate by given index.",
			"index"_a, "vertice"_a)
		.def("get_vertice", &ccMesh::getVertice,
			"get vertex coordinate by given index.",
			"index"_a)
		.def("set_vertices", &ccMesh::addEigenVertices,
			"``float64`` array of shape ``(num_vertices, 3)``, "
			"use ``numpy.asarray()`` to access data: Vertex "
			"coordinates.", "vertices"_a)
		.def("get_vertices", &ccMesh::getEigenVertices,
			"``float64`` array of shape ``(num_vertices, 3)``, "
			"use ``numpy.asarray()`` to access data: Vertex "
			"coordinates.")
		.def("set_vertex_normal", &ccMesh::setVertexNormal,
			"set vertex normal by given index.",
			"index"_a, "normal"_a)
		.def("get_vertex_normal", &ccMesh::getVertexNormal,
			"get vertex normal by given index.",
			"index"_a)
		.def("set_vertex_normals", &ccMesh::addVertexNormals,
			"``float64`` array of shape ``(num_vertices, 3)``, "
			"use ``numpy.asarray()`` to access data: Vertex "
			"normals.", "normals"_a)
		.def("get_vertex_normals", &ccMesh::getVertexNormals,
			"``float64`` array of shape ``(num_vertices, 3)``, "
			"use ``numpy.asarray()`` to access data: Vertex "
			"normals.")
		.def("set_vertex_color", &ccMesh::setVertexColor,
			"set vertex color by given index.",
			"index"_a, "color"_a)
		.def("get_vertex_color", &ccMesh::getVertexColor,
			"get vertex color by given index.",
			"index"_a)
		.def("set_vertex_colors", &ccMesh::addVertexColors,
			"``float64`` array of shape ``(num_vertices, 3)``, "
			"range ``[0, 1]`` , use ``numpy.asarray()`` to access "
			"data: RGB colors of vertices.", "colors"_a)
		.def("get_vertex_colors", &ccMesh::getVertexColors,
			"``float64`` array of shape ``(num_vertices, 3)``, "
			"range ``[0, 1]`` , use ``numpy.asarray()`` to access "
			"data: RGB colors of vertices.")
        .def("has_adjacency_list", &ccMesh::hasAdjacencyList,
                "Returns ``True`` if the mesh contains adjacency normals.")
        .def("has_triangle_uvs", &ccMesh::hasTriangleUvs,
                "Returns ``True`` if the mesh contains uv coordinates.")
        .def("has_triangle_material_ids",
                &ccMesh::hasTriangleMaterialIds,
                "Returns ``True`` if the mesh contains material ids.")
        .def("has_textures", &ccMesh::hasEigenTextures,
                "Returns ``True`` if the mesh contains a texture image.")
        .def("normalize_normals", &ccMesh::normalizeNormals,
                "Normalize both triangle normals and vertex normals to length "
                "1.")
        .def("paint_uniform_color",
                &ccMesh::paintUniformColor,
                "Assigns each vertex in the TriangleMesh the same color.")
        .def("euler_poincare_characteristic",
                &ccMesh::eulerPoincareCharacteristic,
                "Function that computes the Euler-Poincaré characteristic, "
                "i.e., V + F - E, where V is the number of vertices, F is the "
                "number of triangles, and E is the number of edges.")
        .def("get_non_manifold_edges",
                &ccMesh::getNonManifoldEdges,
                "Get list of non-manifold edges.",
                "allow_boundary_edges"_a = true)
        .def("is_edge_manifold", &ccMesh::isEdgeManifold,
                "Tests if the triangle mesh is edge manifold.",
                "allow_boundary_edges"_a = true)
        .def("get_non_manifold_vertices",
                &ccMesh::getNonManifoldVertices,
                "Returns a list of indices to non-manifold vertices.")
        .def("is_vertex_manifold",
                &ccMesh::isVertexManifold,
                "Tests if all vertices of the triangle mesh are manifold.")
        .def("is_self_intersecting",
                &ccMesh::isSelfIntersecting,
                "Tests if the triangle mesh is self-intersecting.")
        .def("get_self_intersecting_triangles",
                &ccMesh::getSelfIntersectingTriangles,
                "Returns a list of indices to triangles that intersect the "
                "mesh.")
        .def("is_intersecting", &ccMesh::isIntersecting,
                "Tests if the triangle mesh is intersecting the other "
                "triangle mesh.")
        .def("is_orientable", &ccMesh::isOrientable,
                "Tests if the triangle mesh is orientable.")
        .def("is_watertight", &ccMesh::isWatertight,
                "Tests if the triangle mesh is watertight.")
        .def("orient_triangles", &ccMesh::orientTriangles,
                "If the mesh is orientable this function orients all "
                "triangles such that all normals point towards the same "
                "direction.")
		.def("select_by_index", &ccMesh::selectByIndex,
			"Function to select mesh from input triangle mesh into output "
			"triangle mesh. ``input``: The input triangle mesh. "
			"``indices``: "
			"Indices of vertices to be selected.",
			"indices"_a, "cleanup"_a = true)
        .def("crop",
                (std::shared_ptr<ccMesh>(ccMesh::*)(const ccBBox &) const)
				&ccMesh::crop,
                "Function to crop input TriangleMesh into output TriangleMesh",
                "bounding_box"_a)
        .def("crop",
                (std::shared_ptr<ccMesh>(ccMesh::*)(
                        const ecvOrientedBBox &) const) &
                        ccMesh::crop,
                "Function to crop input TriangleMesh into output TriangleMesh",
                "bounding_box"_a)
        .def("get_surface_area",
			py::overload_cast<>(&ccMesh::getSurfaceArea, py::const_),
                "Function that computes the surface area of the mesh, i.e. "
                "the sum of the individual triangle surfaces.")
        .def("get_surface_area",
                py::overload_cast<std::vector<double> &>(&ccMesh::getSurfaceArea, py::const_),
                "Function that computes the surface area of the mesh, i.e. "
                "the sum of the individual triangle surfaces.", "triangle_areas"_a)
        .def("get_volume",
                (double (ccMesh::*)() const) & ccMesh::getVolume,
                "Function that computes the volume of the mesh, under the "
                "condition that it is watertight and orientable.")
        .def("sample_points_uniformly",
                &ccMesh::samplePointsUniformly,
                "Function to uniformly sample points from the mesh.",
                "number_of_points"_a = 100, "use_triangle_normal"_a = false,
                 "seed"_a = -1)
        .def("sample_points_poisson_disk",
                &ccMesh::samplePointsPoissonDisk,
                "Function to sample points from the mesh, where each point "
                "has "
                "approximately the same distance to the neighbouring points "
                "(blue "
                "noise). Method is based on Yuksel, \"Sample Elimination for "
                "Generating Poisson Disk Sample Sets\", EUROGRAPHICS, 2015.",
                "number_of_points"_a, "init_factor"_a = 5, "pcl"_a = nullptr,
                 "use_triangle_normal"_a = false, "seed"_a = -1)
        .def("subdivide_midpoint",
                &ccMesh::subdivideMidpoint,
                "Function subdivide mesh using midpoint algorithm.",
                "number_of_iterations"_a = 1)
        .def("subdivide_loop", &ccMesh::subdivideLoop,
                "Function subdivide mesh using Loop's algorithm. Loop, "
                "\"Smooth "
                "subdivision surfaces based on triangles\", 1987.",
                "number_of_iterations"_a = 1)
        .def("simplify_vertex_clustering",
                &ccMesh::simplifyVertexClustering,
                "Function to simplify mesh using vertex clustering.",
                "voxel_size"_a,
                "contraction"_a =
                ccMesh::SimplificationContraction::Average)
        .def("simplify_quadric_decimation",
                &ccMesh::simplifyQuadricDecimation,
                "Function to simplify mesh using Quadric Error Metric "
                "Decimation by "
                "Garland and Heckbert",
                 "target_number_of_triangles"_a,
                 "maximum_error"_a = std::numeric_limits<double>::infinity(),
                 "boundary_weight"_a = 1.0)
        .def("compute_convex_hull",
                &ccMesh::computeConvexHull,
                "Computes the convex hull of the triangle mesh.")
        .def("cluster_connected_triangles",
                &ccMesh::clusterConnectedTriangles,
                "Function that clusters connected triangles, i.e., triangles "
                "that are connected via edges are assigned the same cluster "
                "index.  This function retuns an array that contains the "
                "cluster index per triangle, a second array contains the "
                "number of triangles per cluster, and a third vector contains "
                "the surface area per cluster.")
        .def("remove_triangles_by_index",
                &ccMesh::removeTrianglesByIndex,
                "This function removes the triangles with index in "
                "triangle_indices.  Call remove_unreferenced_vertices to "
                "clean up vertices afterwards.",
                "triangle_indices"_a)
        .def("remove_triangles_by_mask",
                &ccMesh::removeTrianglesByMask,
                "This function removes the triangles where triangle_mask is "
                "set to true.  Call remove_unreferenced_vertices to clean up "
                "vertices afterwards.",
                "triangle_mask"_a)
        .def("remove_vertices_by_index",
                &ccMesh::removeVerticesByIndex,
                "This function removes the vertices with index in "
                "vertex_indices. Note that also all triangles associated with "
                "the vertices are removed.",
                "vertex_indices"_a)
        .def("remove_vertices_by_mask",
            &ccMesh::removeVerticesByMask,
            "This function removes the vertices that are masked in "
            "vertex_mask. Note that also all triangles associated with "
            "the vertices are removed.",
            "vertex_mask"_a)
        .def("deform_as_rigid_as_possible",
            &ccMesh::deformAsRigidAsPossible,
            "This function deforms the mesh using the method by Sorkine "
            "and Alexa, "
            "'As-Rigid-As-Possible Surface Modeling', 2007",
            "constraint_vertex_indices"_a, "constraint_vertex_positions"_a,
            "max_iter"_a,
            "energy"_a = cloudViewer::GenericMesh::DeformAsRigidAsPossibleEnergy::Spokes,
            "smoothed_alpha"_a = 0.01)
		.def_static("compute_triangle_area", &ccMesh::ComputeTriangleArea,
					"Function that computes the area of a mesh triangle.", "p0"_a, "p1"_a, "p2"_a)
		.def_static("compute_triangle_plane", &ccMesh::ComputeTrianglePlane,
					"Function that computes the plane equation from the three points.", "p0"_a, "p1"_a, "p2"_a)
		.def_static("get_ordered_edge", &ccMesh::GetOrderedEdge,
					"Helper function to get an edge with ordered vertex indices.",
					"vidx0"_a, "vidx1"_a)
		.def_static("get_eigne_ordered_triangle", &ccMesh::GetEigneOrderedTriangle,
					"Returns eigne ordered triangle.", "vidx0"_a, "vidx1"_a, "vidx2"_a)
        .def_static("triangulate",
                    [](ccGenericPointCloud &cloud, CC_TRIANGULATION_TYPES type, 
						bool update_normals, PointCoordinateType max_edge_length,
						unsigned char dim) {
                        return std::shared_ptr<ccMesh>(
							ccMesh::Triangulate(&cloud, type, update_normals, max_edge_length, dim));
                    },
                    "Creates a Delaunay 2.5D mesh from a point cloud \n"
					"See cloudViewer::PointProjectionTools::computeTriangulation.",
					"cloud"_a, "type"_a, "update_normals"_a = false, 
					"max_edge_length"_a = 0, "dim"_a = 2)
        .def_static("triangulate_two_polylines",
                    [](ccPolyline &poly1, ccPolyline &poly2) {
                        return std::shared_ptr<ccMesh>(
							ccMesh::TriangulateTwoPolylines(&poly1, &poly2));
                    },
                    "Creates a Delaunay 2.5D mesh from two polylines.",
                    "poly1"_a, "poly2"_a)
        .def_static("create_from_point_cloud_alpha_shape",
                    [](const ccPointCloud &pcd, double alpha) {
                        return ccMesh::CreateFromPointCloudAlphaShape(pcd, alpha);
                    },
                    "Alpha shapes are a generalization of the convex hull. "
                    "With decreasing alpha value the shape schrinks and "
                    "creates cavities. See Edelsbrunner and Muecke, "
                    "\"Three-Dimensional Alpha Shapes\", 1994.",
                    "pcd"_a, "alpha"_a)
        .def_static("create_from_point_cloud_alpha_shape",
                    &ccMesh::CreateFromPointCloudAlphaShape,
                    "Alpha shapes are a generalization of the convex hull. "
                    "With decreasing alpha value the shape schrinks and "
                    "creates cavities. See Edelsbrunner and Muecke, "
                    "\"Three-Dimensional Alpha Shapes\", 1994.",
                    "pcd"_a, "alpha"_a, "tetra_mesh"_a, "pt_map"_a)
        .def_static(
                "create_from_point_cloud_ball_pivoting",
                &ccMesh::CreateFromPointCloudBallPivoting,
                "Function that computes a triangle mesh from a oriented "
                "PointCloud. This implements the Ball Pivoting algorithm "
                "proposed in F. Bernardini et al., \"The ball-pivoting "
                "algorithm for surface reconstruction\", 1999. The "
                "implementation is also based on the algorithms outlined "
                "in Digne, \"An Analysis and Implementation of a Parallel "
                "Ball Pivoting Algorithm\", 2014. The surface "
                "reconstruction is done by rolling a ball with a given "
                "radius over the point cloud, whenever the ball touches "
                "three points a triangle is created.",
                "pcd"_a, "radii"_a)
        .def_static("create_from_point_cloud_poisson",
                    &ccMesh::CreateFromPointCloudPoisson,
                    "Function that computes a triangle mesh from a "
                    "oriented PointCloud pcd. This implements the Screened "
                    "Poisson Reconstruction proposed in Kazhdan and Hoppe, "
                    "\"Screened Poisson Surface Reconstruction\", 2013. "
                    "This function uses the original implementation by "
                    "Kazhdan. See https://github.com/mkazhdan/PoissonRecon",
                    "pcd"_a, "depth"_a = 8, 
					"width"_a = 0, "scale"_a = 1.1,
                    "linear_fit"_a = false, 
					"point_weight"_a = 2.0,
					"samples_per_node"_a = 1.5, 
                    "boundary_type"_a = 2,
                    "n_threads"_a = -1)
        .def_static("create_plane", &ccMesh::CreatePlane,
                    "Factory function to create a plane. The center of  "
                    "the plane will be placed at (0, 0, 0).",
                    "width"_a = 1.0, "height"_a = 1.0)
        .def_static("create_box", &ccMesh::CreateBox,
                    "Factory function to create a box. The left bottom "
                    "corner on the "
                    "front will be placed at (0, 0, 0).",
                    "width"_a = 1.0, "height"_a = 1.0, "depth"_a = 1.0)
        .def_static("create_tetrahedron",
                    &ccMesh::CreateTetrahedron,
                    "Factory function to create a tetrahedron. The "
                    "centroid of the mesh "
                    "will be placed at (0, 0, 0) and the vertices have a "
                    "distance of "
                    "radius to the center.",
                    "radius"_a = 1.0)
        .def_static("create_octahedron",
                    &ccMesh::CreateOctahedron,
                    "Factory function to create a octahedron. The centroid "
                    "of the mesh "
                    "will be placed at (0, 0, 0) and the vertices have a "
                    "distance of "
                    "radius to the center.",
                    "radius"_a = 1.0)
        .def_static("create_icosahedron",
                    &ccMesh::CreateIcosahedron,
                    "Factory function to create a icosahedron. The "
                    "centroid of the mesh "
                    "will be placed at (0, 0, 0) and the vertices have a "
                    "distance of "
                    "radius to the center.",
                    "radius"_a = 1.0)
        .def_static("create_sphere", &ccMesh::CreateSphere,
                    "Factory function to create a sphere mesh centered at "
                    "(0, 0, 0).",
                    "radius"_a = 1.0, "resolution"_a = 20)
        .def_static("create_cylinder",
                    &ccMesh::CreateCylinder,
                    "Factory function to create a cylinder mesh.",
                    "radius"_a = 1.0, "height"_a = 2.0, "resolution"_a = 20,
                    "split"_a = 4)
        .def_static("create_cone", &ccMesh::CreateCone,
                    "Factory function to create a cone mesh.",
                    "radius"_a = 1.0, "height"_a = 2.0, "resolution"_a = 20,
                    "split"_a = 1)
        .def_static("create_torus", &ccMesh::CreateTorus,
                    "Factory function to create a torus mesh.",
                    "torus_radius"_a = 1.0, "tube_radius"_a = 0.5,
                    "radial_resolution"_a = 30, "tubular_resolution"_a = 20)
        .def_static("create_arrow", &ccMesh::CreateArrow,
                    "Factory function to create an arrow mesh",
                    "cylinder_radius"_a = 1.0, "cone_radius"_a = 1.5,
                    "cylinder_height"_a = 5.0, "cone_height"_a = 4.0,
                    "resolution"_a = 20, "cylinder_split"_a = 4,
                    "cone_split"_a = 1)
        .def_static("create_coordinate_frame",
                    &ccMesh::CreateCoordinateFrame,
                    "Factory function to create a coordinate frame mesh. "
                    "The coordinate "
                    "frame will be centered at ``origin``. The x, y, z "
                    "axis will be "
                    "rendered as red, green, and blue arrows respectively.",
                    "size"_a = 1.0,
                    "origin"_a = Eigen::Vector3d(0.0, 0.0, 0.0))
        .def_static("create_moebius",
					&ccMesh::CreateMoebius,
					"Factory function to create a Moebius strip.",
					"length_split"_a = 70, "width_split"_a = 15,
					"twists"_a = 1, "radius"_a = 1, "flatness"_a = 1,
					"width"_a = 1, "scale"_a = 1)
        .def_readwrite(
                "adjacency_list", &ccMesh::adjacency_list_,
                "List of Sets: The set ``adjacency_list[i]`` contains the "
                "indices of adjacent vertices of vertex i.")
        .def_readwrite("triangle_uvs",
                        &ccMesh::triangle_uvs_,
                        "``float64`` array of shape ``(3 * num_triangles, "
                        "2)``, use "
                        "``numpy.asarray()`` to access data: List of "
                        "uvs denoted by the index of points forming "
                        "the triangle.")
        .def_readwrite("triangle_material_ids",
                   &ccMesh::triangle_material_ids_,
                   "`int` array of shape ``(num_trianges, 1)``, use "
                   "``numpy.asarray()`` to access data: material index "
                   "associated with each triangle")
        .def_readwrite("textures", &ccMesh::textures_,
                   "cloudViewer.geometry.Image: The texture images.");

    docstring::ClassMethodDocInject(m, "ccMesh", "clone_mesh");
    docstring::ClassMethodDocInject(m, "ccMesh", "merge");
    docstring::ClassMethodDocInject(m, "ccMesh", "clear");
    docstring::ClassMethodDocInject(m, "ccMesh", "shift_triangle_indexes");
    docstring::ClassMethodDocInject(m, "ccMesh", "reserve");
    docstring::ClassMethodDocInject(m, "ccMesh", "resize");
    docstring::ClassMethodDocInject(m, "ccMesh", "flip_triangles");
    docstring::ClassMethodDocInject(m, "ccMesh", "shrink_triangles");
    docstring::ClassMethodDocInject(m, "ccMesh", "shrink_vertexes");
    docstring::ClassMethodDocInject(m, "ccMesh", "clear_triangle_normals");
    docstring::ClassMethodDocInject(m, "ccMesh", "are_triangle_normals_enabled");
    docstring::ClassMethodDocInject(m, "ccMesh", "reserve_triangle_normal_indexes");
    docstring::ClassMethodDocInject(m, "ccMesh", "add_triangle_normal_indexes");
    docstring::ClassMethodDocInject(m, "ccMesh", "set_triangle_normal_indexes");
    docstring::ClassMethodDocInject(m, "ccMesh", "get_triangle_normal_indexes");
    docstring::ClassMethodDocInject(m, "ccMesh", "remove_triangle_normal_indexes");
    docstring::ClassMethodDocInject(m, "ccMesh", "convert_materials_to_vertex_colors");
    docstring::ClassMethodDocInject(m, "ccMesh", "has_material_indexes");
    docstring::ClassMethodDocInject(m, "ccMesh", "reserve_material_indexes");
    docstring::ClassMethodDocInject(m, "ccMesh", "remove_material_indexes");
    docstring::ClassMethodDocInject(m, "ccMesh", "add_material_index");
    docstring::ClassMethodDocInject(m, "ccMesh", "set_material_index");
    docstring::ClassMethodDocInject(m, "ccMesh", "reserve_texture_coord_indexes");
    docstring::ClassMethodDocInject(m, "ccMesh", "remove_texture_coord_indexes");
    docstring::ClassMethodDocInject(m, "ccMesh", "add_texture_coord_indexes");
    docstring::ClassMethodDocInject(m, "ccMesh", "set_texture_coord_indexes");
    docstring::ClassMethodDocInject(m, "ccMesh", "compute_normals");
    docstring::ClassMethodDocInject(m, "ccMesh", "laplacian_smooth");
    docstring::ClassMethodDocInject(m, "ccMesh", "process_scalar_field");
    docstring::ClassMethodDocInject(m, "ccMesh", "subdivide");
    docstring::ClassMethodDocInject(m, "ccMesh", "create_mesh_from_selection");
    docstring::ClassMethodDocInject(m, "ccMesh", "swap_triangles");
    docstring::ClassMethodDocInject(m, "ccMesh", "remove_triangles");
    docstring::ClassMethodDocInject(m, "ccMesh", "transform_triangle_normals");
    docstring::ClassMethodDocInject(m, "ccMesh", "get_triangle_area");
    docstring::ClassMethodDocInject(m, "ccMesh", "is_bbox_intersecting");
    docstring::ClassMethodDocInject(m, "ccMesh", "get_edge_to_triangles_map");
    docstring::ClassMethodDocInject(m, "ccMesh", "get_edge_to_vertices_map");
    docstring::ClassMethodDocInject(m, "ccMesh", "get_triangle_plane");
    docstring::ClassMethodDocInject(m, "ccMesh", "vertice_size");
    docstring::ClassMethodDocInject(m, "ccMesh", "create_internal_cloud");
    docstring::ClassMethodDocInject(m, "ccMesh", "set_associated_cloud");
    docstring::ClassMethodDocInject(m, "ccMesh", "compute_adjacency_list");
    docstring::ClassMethodDocInject(m, "ccMesh", "compute_triangle_normals");
    docstring::ClassMethodDocInject(m, "ccMesh", "compute_vertex_normals");
    docstring::ClassMethodDocInject(m, "ccMesh", "has_adjacency_list");
    docstring::ClassMethodDocInject(m, "ccMesh", "set_vertices");
    docstring::ClassMethodDocInject(m, "ccMesh", "set_vertice");
    docstring::ClassMethodDocInject(m, "ccMesh", "get_vertice");
    docstring::ClassMethodDocInject(m, "ccMesh", "get_vertices");
    docstring::ClassMethodDocInject(m, "ccMesh", "set_vertex_normal");
    docstring::ClassMethodDocInject(m, "ccMesh", "set_vertex_normals");
    docstring::ClassMethodDocInject(m, "ccMesh", "get_vertex_normal");
    docstring::ClassMethodDocInject(m, "ccMesh", "get_vertex_normals");
    docstring::ClassMethodDocInject(m, "ccMesh", "set_vertex_colors");
    docstring::ClassMethodDocInject(m, "ccMesh", "set_vertex_color");
    docstring::ClassMethodDocInject(m, "ccMesh", "get_vertex_color");
    docstring::ClassMethodDocInject(m, "ccMesh", "get_vertex_colors");
    docstring::ClassMethodDocInject(m, "ccMesh", "set_triangle");
    docstring::ClassMethodDocInject(m, "ccMesh", "set_triangles");
    docstring::ClassMethodDocInject(m, "ccMesh", "get_triangle");
    docstring::ClassMethodDocInject(m, "ccMesh", "get_triangles");
    docstring::ClassMethodDocInject(m, "ccMesh", "add_triangles");
    docstring::ClassMethodDocInject(m, "ccMesh", "set_triangle_normal");
    docstring::ClassMethodDocInject(m, "ccMesh", "set_triangle_normals");
    docstring::ClassMethodDocInject(m, "ccMesh", "get_triangle_normal");
    docstring::ClassMethodDocInject(m, "ccMesh", "get_triangle_normals");
    docstring::ClassMethodDocInject(m, "ccMesh", "has_triangles");
    docstring::ClassMethodDocInject(m, "ccMesh", "has_triangle_uvs");
    docstring::ClassMethodDocInject(m, "ccMesh",
                                    "has_triangle_material_ids");
    docstring::ClassMethodDocInject(m, "ccMesh", "has_textures");
    docstring::ClassMethodDocInject(m, "ccMesh", "has_vertex_colors");
    docstring::ClassMethodDocInject(m, "ccMesh", "normalize_normals");
    docstring::ClassMethodDocInject(
            m, "ccMesh", "has_vertex_normals",
            {{"normalized",
              "Set to ``True`` to normalize the normal to length 1."}});
    docstring::ClassMethodDocInject(m, "ccMesh", "has_vertices");
    docstring::ClassMethodDocInject(
            m, "ccMesh", "paint_uniform_color",
            {{"color", "RGB color for the PointCloud."}});
    docstring::ClassMethodDocInject(m, "ccMesh",
                                    "euler_poincare_characteristic");
    docstring::ClassMethodDocInject(
            m, "ccMesh", "get_non_manifold_edges",
            {{"allow_boundary_edges",
              "If true, than non-manifold edges are defined as edges with more "
              "than two adjacent triangles, otherwise each edge that is not "
              "adjacent to two triangles is defined as non-manifold."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "is_edge_manifold",
            {{"allow_boundary_edges",
              "If true, than non-manifold edges are defined as edges with more "
              "than two adjacent triangles, otherwise each edge that is not "
              "adjacent to two triangles is defined as non-manifold."}});
    docstring::ClassMethodDocInject(m, "ccMesh", "is_vertex_manifold");
    docstring::ClassMethodDocInject(m, "ccMesh",
                                    "get_non_manifold_vertices");
    docstring::ClassMethodDocInject(m, "ccMesh", "is_self_intersecting");
    docstring::ClassMethodDocInject(m, "ccMesh",
                                    "get_self_intersecting_triangles");
    docstring::ClassMethodDocInject(
            m, "ccMesh", "is_intersecting",
            {{"other", "Other triangle mesh to test intersection with."}});
    docstring::ClassMethodDocInject(m, "ccMesh", "is_orientable");
    docstring::ClassMethodDocInject(m, "ccMesh", "is_watertight");
    docstring::ClassMethodDocInject(m, "ccMesh", "orient_triangles");
    docstring::ClassMethodDocInject(m, "ccMesh",
                                    "remove_duplicated_vertices");
    docstring::ClassMethodDocInject(m, "ccMesh",
                                    "remove_duplicated_triangles");
    docstring::ClassMethodDocInject(m, "ccMesh",
                                    "remove_unreferenced_vertices");
    docstring::ClassMethodDocInject(m, "ccMesh",
                                    "remove_degenerate_triangles");
    docstring::ClassMethodDocInject(m, "ccMesh",
                                    "remove_non_manifold_edges");
    docstring::ClassMethodDocInject(
            m, "ccMesh", "merge_close_vertices",
            {{"eps",
              "Parameter that defines the distance between close vertices."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "filter_sharpen",
            {{"number_of_iterations",
              " Number of repetitions of this operation"},
             {"strength", "Filter parameter."},
             {"scope", "Mesh property that should be filtered."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "filter_smooth_simple",
            {{"number_of_iterations",
              " Number of repetitions of this operation"},
             {"scope", "Mesh property that should be filtered."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "filter_smooth_laplacian",
            {{"number_of_iterations",
              " Number of repetitions of this operation"},
             {"lambda", "Filter parameter."},
             {"scope", "Mesh property that should be filtered."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "filter_smooth_taubin",
            {{"number_of_iterations",
              " Number of repetitions of this operation"},
             {"lambda", "Filter parameter."},
             {"mu", "Filter parameter."},
             {"scope", "Mesh property that should be filtered."}});
	docstring::ClassMethodDocInject(
			m, "ccMesh", "select_by_index",
			{ {"indices", "Indices of vertices to be selected."},
			{"cleanup",
			"If true calls number of mesh cleanup functions to remove "
			"unreferenced vertices and degenerate triangles"} });
    docstring::ClassMethodDocInject(
            m, "ccMesh", "crop",
            {{"bounding_box", "AxisAlignedBoundingBox to crop points"}});
    docstring::ClassMethodDocInject(m, "ccMesh", "get_volume");
    docstring::ClassMethodDocInject(
            m, "ccMesh", "sample_points_uniformly",
            {{"number_of_points",
              "Number of points that should be uniformly sampled."},
             {"use_triangle_normal",
              "If True assigns the triangle normals instead of the "
              "interpolated vertex normals to the returned points. The "
              "triangle normals will be computed and added to the mesh if "
              "necessary."},
             {"seed",
              "Seed value used in the random generator, set to -1 to use a "
              "random seed value with each function call."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "sample_points_poisson_disk",
            {{"number_of_points", "Number of points that should be sampled."},
             {"init_factor",
              "Factor for the initial uniformly sampled PointCloud. This init "
              "PointCloud is used for sample elimination."},
             {"pcl",
              "Initial PointCloud that is used for sample elimination. If this "
              "parameter is provided the init_factor is ignored."},
             {"use_triangle_normal",
              "If True assigns the triangle normals instead of the "
              "interpolated vertex normals to the returned points. The "
              "triangle normals will be computed and added to the mesh if "
              "necessary."},
             {"seed",
              "Seed value used in the random generator, set to -1 to use a "
              "random seed value with each function call."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "subdivide_midpoint",
            {{"number_of_iterations",
              "Number of iterations. A single iteration splits each triangle "
              "into four triangles that cover the same surface."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "subdivide_loop",
            {{"number_of_iterations",
              "Number of iterations. A single iteration splits each triangle "
              "into four triangles."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "simplify_vertex_clustering",
            {{"voxel_size",
              "The size of the voxel within vertices are pooled."},
             {"contraction",
              "Method to aggregate vertex information. Average computes a "
              "simple average, Quadric minimizes the distance to the adjacent "
              "planes."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "simplify_quadric_decimation",
            {{"target_number_of_triangles",
              "The number of triangles that the simplified mesh should have. "
              "It is not guranteed that this number will be reached."},
             {"maximum_error",
              "The maximum error where a vertex is allowed to be merged"},
             {"boundary_weight",
              "A weight applied to edge vertices used to preserve "
              "boundaries"}});
    docstring::ClassMethodDocInject(m, "ccMesh", "compute_convex_hull");
    docstring::ClassMethodDocInject(m, "ccMesh",
                                    "cluster_connected_triangles");
    docstring::ClassMethodDocInject(
            m, "ccMesh", "remove_triangles_by_index",
            {{"triangle_indices",
              "1D array of triangle indices that should be removed from the "
              "TriangleMesh."}});
    docstring::ClassMethodDocInject(m, "ccMesh",
                                    "remove_triangles_by_mask",
                                    {{"triangle_mask",
                                      "1D bool array, True values indicate "
                                      "triangles that should be removed."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "remove_vertices_by_index",
            {{"vertex_indices",
              "1D array of vertex indices that should be removed from the "
              "TriangleMesh."}});
    docstring::ClassMethodDocInject(m, "ccMesh",
                                    "remove_vertices_by_mask",
                                    {{"vertex_mask",
                                      "1D bool array, True values indicate "
                                      "vertices that should be removed."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "deform_as_rigid_as_possible",
            {{"constraint_vertex_indices",
              "Indices of the triangle vertices that should be constrained by "
              "the vertex positions "
              "in constraint_vertex_positions."},
             {"constraint_vertex_positions",
              "Vertex positions used for the constraints."},
             {"max_iter",
              "Maximum number of iterations to minimize energy functional."},
             {"energy",
              "Energy model that is minimized in the deformation process"},
             {"smoothed_alpha",
              "trade-off parameter for the smoothed energy functional for the "
              "regularization term."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "triangulate",
            {{"cloud", "a point cloud."},
             {"type", "the triangulation strategy."},
             {"update_normals", "compute per-vertex normals if true."},
             {"max_edge_length", "max edge length for output triangles (0 = ignored)."},
             {"dim", "projection dimension (for axis-aligned meshes)."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "create_from_point_cloud_alpha_shape",
            {{"pcd",
              "PointCloud from whicht the TriangleMesh surface is "
              "reconstructed."},
             {"alpha",
              "Parameter to controll the shape. A very big value will give a "
              "shape close to the convex hull."},
             {"tetra_mesh",
              "If not None, than uses this to construct the alpha shape. "
              "Otherwise, TetraMesh is computed from pcd."},
             {"pt_map",
              "Optional map from tetra_mesh vertex indices to pcd points."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "create_from_point_cloud_ball_pivoting",
            {{"pcd",
              "PointCloud from which the TriangleMesh surface is "
              "reconstructed. Has to contain normals."},
             {"radii",
              "The radii of the ball that are used for the surface "
              "reconstruction."}});
	docstring::ClassMethodDocInject(
            m, "ccMesh", "create_from_point_cloud_poisson",
            {{"pcd",
              "PointCloud from which the TriangleMesh surface is "
              "reconstructed. Has to contain normals."},
             {"depth",
              "Maximum depth of the tree that will be used for surface "
              "reconstruction. Running at depth d corresponds to solving on a "
              "grid whose resolution is no larger than 2^d x 2^d x 2^d. Note "
              "that since the reconstructor adapts the octree to the sampling "
              "density, the specified reconstruction depth is only an upper "
              "bound."},
             {"width",
              "Specifies the target width of the finest level octree cells. "
              "This parameter is ignored if depth is specified"},
             {"scale",
              "Specifies the ratio between the diameter of the cube used for "
              "reconstruction and the diameter of the samples' bounding cube."},
             {"linear_fit",
              "If true, the reconstructor use linear interpolation to estimate "
              "the positions of iso-vertices."},
             {"point_weight",
			  "The importance that interpolation of the point samples "
			  "is given in the formulation of the screened Poisson equation."
			  "The results of the original (unscreened) Poisson Reconstruction" 
			  "can be obtained by setting this value to 0"},
             {"samples_per_node",
			  "The minimum number of sample points that should fall within" 
			  "an octree node as the octree construction is adapted to sampling density."
			  "This parameter specifies the minimum number of points that should fall" 
			  "within an octree node. For noise-free samples, small values in the range [1.0 - 5.0]" 
			  "can be used. For more noisy samples, larger values in the range [15.0 - 20.0]"
			  "may be needed to provide a smoother, noise-reduced, reconstruction."},
			 {"boundary_type", "Boundary type for the finite elements"},
             {"n_threads",
              "Number of threads used for reconstruction. Set to -1 to "
              "automatically determine it."}});

    docstring::ClassMethodDocInject(m, "ccMesh", "create_box",
                                    {{"width", "x-directional length."},
                                     {"height", "y-directional length."},
                                     {"depth", "z-directional length."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "create_tetrahedron",
            {{"radius", "Distance from centroid to mesh vetices."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "create_octahedron",
            {{"radius", "Distance from centroid to mesh vetices."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "create_icosahedron",
            {{"radius", "Distance from centroid to mesh vetices."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "create_sphere",
            {{"radius", "The radius of the sphere."},
             {"resolution",
              "The resolution of the sphere. The longitues will be split into "
              "``resolution`` segments (i.e. there are ``resolution + 1`` "
              "latitude lines including the north and south pole). The "
              "latitudes will be split into ```2 * resolution`` segments (i.e. "
              "there are ``2 * resolution`` longitude lines.)"}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "create_cylinder",
            {{"radius", "The radius of the cylinder."},
             {"height",
              "The height of the cylinder. The axis of the cylinder will be "
              "from (0, 0, -height/2) to (0, 0, height/2)."},
             {"resolution",
              " The circle will be split into ``resolution`` segments"},
             {"split",
              "The ``height`` will be split into ``split`` segments."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "create_cone",
            {{"radius", "The radius of the cone."},
             {"height",
              "The height of the cone. The axis of the cone will be from (0, "
              "0, 0) to (0, 0, height)."},
             {"resolution",
              "The circle will be split into ``resolution`` segments"},
             {"split",
              "The ``height`` will be split into ``split`` segments."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "create_torus",
            {{"torus_radius",
              "The radius from the center of the torus to the center of the "
              "tube."},
             {"tube_radius", "The radius of the torus tube."},
             {"radial_resolution",
              "The number of segments along the radial direction."},
             {"tubular_resolution",
              "The number of segments along the tubular direction."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "create_arrow",
            {{"cylinder_radius", "The radius of the cylinder."},
             {"cone_radius", "The radius of the cone."},
             {"cylinder_height",
              "The height of the cylinder. The cylinder is from (0, 0, 0) to "
              "(0, 0, cylinder_height)"},
             {"cone_height",
              "The height of the cone. The axis of the cone will be from (0, "
              "0, cylinder_height) to (0, 0, cylinder_height + cone_height)"},
             {"resolution",
              "The cone will be split into ``resolution`` segments."},
             {"cylinder_split",
              "The ``cylinder_height`` will be split into ``cylinder_split`` "
              "segments."},
             {"cone_split",
              "The ``cone_height`` will be split into ``cone_split`` "
              "segments."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "create_coordinate_frame",
            {{"size", "The size of the coordinate frame."},
             {"origin", "The origin of the cooridnate frame."}});
    docstring::ClassMethodDocInject(
            m, "ccMesh", "create_moebius",
            {{"length_split",
              "The number of segments along the Moebius strip."},
             {"width_split",
              "The number of segments along the width of the Moebius strip."},
             {"twists", "Number of twists of the Moebius strip."},
             {"radius", "The radius of the Moebius strip."},
             {"flatness", "Controls the flatness/height of the Moebius strip."},
             {"width", "Width of the Moebius strip."},
             {"scale", "Scale the complete Moebius strip."}});
}

void pybind_trianglemesh_methods(py::module &m) {}

}  // namespace geometry
}  // namespace cloudViewer

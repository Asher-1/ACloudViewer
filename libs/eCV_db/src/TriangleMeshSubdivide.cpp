// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 asher-1.github.io
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

#include "ecvMesh.h"
#include "ecvPointCloud.h"
#include "ecvHObjectCaster.h"

#include <Eigen/Dense>
#include <queue>
#include <tuple>

#include <Logging.h>

#include <unordered_map>
#include <unordered_set>

using namespace cloudViewer;
std::shared_ptr<ccMesh> ccMesh::subdivideMidpoint(
	int number_of_iterations) const {
	if (hasTriangleUvs()) {
		utility::LogWarning(
			"[SubdivideMidpoint] This mesh contains triangle uvs that are "
			"not handled in this function");
	}

	ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
	if (!cloud)
	{
		utility::LogError(
			"[ccMesh::simplifyVertexClustering] mesh"
			"should set associated cloud before using!");
	}

	ccPointCloud* baseVertices = new ccPointCloud("vertices");
	assert(baseVertices);
	baseVertices->setEnabled(false);
	// DGM: no need to lock it as it is only used by one mesh!
	baseVertices->setLocked(false);
	auto mesh = cloudViewer::make_shared<ccMesh>(baseVertices);
	mesh->addChild(baseVertices);
	mesh->merge(this, false);

	bool has_vert_normal = cloud->hasNormals();
	bool has_vert_color = cloud->hasColors();

	// Compute and return midpoint.
	// Also adds edge - new vertex refrence to new_verts map.
	auto SubdivideEdge =
		[&](std::unordered_map<Eigen::Vector2i, int,
			utility::hash_eigen<Eigen::Vector2i>>&
			new_verts,
			int vidx0, int vidx1) {
		size_t min = static_cast<size_t>(std::min(vidx0, vidx1));
		size_t max = static_cast<size_t>(std::max(vidx0, vidx1));
		Eigen::Vector2i edge(min, max);
		if (new_verts.count(edge) == 0) {
			baseVertices->addEigenPoint(0.5 * 
				(baseVertices->getEigenPoint(min) +
				baseVertices->getEigenPoint(max)));

			if (has_vert_normal) {
				baseVertices->addEigenNorm(0.5 * 
					(baseVertices->getEigenNormal(min) +
						baseVertices->getEigenNormal(max)));
			}
			if (has_vert_color) {
				baseVertices->addEigenColor(0.5 *
					(baseVertices->getEigenColor(min) +
						baseVertices->getEigenColor(max)));
			}
			int vidx01 = int(baseVertices->size()) - 1;
			new_verts[edge] = vidx01;
			return vidx01;
		}
		else {
			return new_verts[edge];
		}
	};

	for (int iter = 0; iter < number_of_iterations; ++iter) {
		std::unordered_map<Eigen::Vector2i, int,
			utility::hash_eigen<Eigen::Vector2i>>
			new_verts;
		std::vector<Eigen::Vector3i> new_triangles(4 * mesh->size());
		for (size_t tidx = 0; tidx < mesh->size(); ++tidx) {
			const auto& triangle = mesh->getTriangle(tidx);
			int vidx0 = triangle(0);
			int vidx1 = triangle(1);
			int vidx2 = triangle(2);
			int vidx01 = SubdivideEdge(new_verts, vidx0, vidx1);
			int vidx12 = SubdivideEdge(new_verts, vidx1, vidx2);
			int vidx20 = SubdivideEdge(new_verts, vidx2, vidx0);
			new_triangles[tidx * 4 + 0] =
				Eigen::Vector3i(vidx0, vidx01, vidx20);
			new_triangles[tidx * 4 + 1] =
				Eigen::Vector3i(vidx01, vidx1, vidx12);
			new_triangles[tidx * 4 + 2] =
				Eigen::Vector3i(vidx12, vidx2, vidx20);
			new_triangles[tidx * 4 + 3] =
				Eigen::Vector3i(vidx01, vidx12, vidx20);
		}
		mesh->setTriangles(new_triangles);
	}

	if (hasTriNormals()) {
		mesh->computeTriangleNormals();
	}

	return mesh;
}

std::shared_ptr<ccMesh> ccMesh::subdivideLoop(
	int number_of_iterations) const {
	if (hasTriangleUvs()) {
		utility::LogWarning(
			"[SubdivideLoop] This mesh contains triangle uvs that are not "
			"handled in this function");
	}
	typedef std::unordered_map<Eigen::Vector2i, int,
		utility::hash_eigen<Eigen::Vector2i>>
		EdgeNewVertMap;
	typedef std::unordered_map<Eigen::Vector2i, std::unordered_set<int>,
		utility::hash_eigen<Eigen::Vector2i>>
		EdgeTrianglesMap;
	typedef std::vector<std::unordered_set<int>> VertexNeighbours;

	ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
	if (!cloud)
	{
		utility::LogError(
			"[ccMesh::simplifyVertexClustering] mesh"
			"should set associated cloud before using!");
	}

	bool has_vert_normal = cloud->hasNormals();
	bool has_vert_color = cloud->hasColors();

	auto UpdateVertex = [&](int vidx,
		const std::shared_ptr<ccMesh>& old_mesh,
		std::shared_ptr<ccMesh>& new_mesh,
		const std::unordered_set<int>& nbs,
		const EdgeTrianglesMap& edge_to_triangles) {
		// check if boundary edge and get nb vertices in that case
		std::unordered_set<int> boundary_nbs;
		for (int nb : nbs) {
			const Eigen::Vector2i edge = GetOrderedEdge(vidx, nb);
			if (edge_to_triangles.at(edge).size() == 1) {
				boundary_nbs.insert(nb);
			}
		}

		// in manifold meshes this should not happen
		if (boundary_nbs.size() > 2) {
			utility::LogWarning(
				"[SubdivideLoop] boundary edge with > 2 neighbours, maybe "
				"mesh is not manifold.");
		}

		double beta, alpha;
		if (boundary_nbs.size() >= 2) {
			beta = 1. / 8.;
			alpha = 1. - boundary_nbs.size() * beta;
		}
		else if (nbs.size() == 3) {
			beta = 3. / 16.;
			alpha = 1. - nbs.size() * beta;
		}
		else {
			beta = 3. / (8. * nbs.size());
			alpha = 1. - nbs.size() * beta;
		}

		ccPointCloud* newVertice = 
			ccHObjectCaster::ToPointCloud(new_mesh->getAssociatedCloud());
		if (!newVertice)
		{
			utility::LogError(
				"[ccMesh::simplifyVertexClustering] mesh"
				"should set associated cloud before using!");
		}
		ccPointCloud* oldVertice =
			ccHObjectCaster::ToPointCloud(old_mesh->getAssociatedCloud());
		if (!oldVertice)
		{
			utility::LogError(
				"[ccMesh::simplifyVertexClustering] mesh"
				"should set associated cloud before using!");
		}

		newVertice->setPoint(static_cast<unsigned>(vidx), 
			*oldVertice->getPoint(static_cast<unsigned>(vidx))* alpha);
		if (has_vert_normal) {
			newVertice->setPointNormal(static_cast<unsigned>(vidx),
				static_cast<PointCoordinateType>(alpha) * oldVertice->getPointNormal(static_cast<unsigned>(vidx)));
		}
		if (has_vert_color) {
			newVertice->setEigenColor(static_cast<unsigned>(vidx),
				alpha * oldVertice->getEigenColor(static_cast<unsigned>(vidx)));
		}

		auto Update = [&](int nb) {
			CCVector3* p = newVertice->getPointPtr(static_cast<unsigned>(vidx));
			*p += *oldVertice->getPoint(static_cast<unsigned>(nb)) * beta;
			if (has_vert_normal) {
				newVertice->setPointNormal(static_cast<unsigned>(vidx),
					newVertice->getPointNormal(static_cast<unsigned>(vidx)) +
					static_cast<PointCoordinateType>(beta) * oldVertice->getPointNormal(static_cast<unsigned>(nb)));
			}
			if (has_vert_color) {
				newVertice->setEigenColor(static_cast<unsigned>(vidx),
					newVertice->getEigenColor(static_cast<unsigned>(vidx)) +
					beta * oldVertice->getEigenColor(static_cast<unsigned>(nb)));
			}
		};

		if (boundary_nbs.size() >= 2) {
			for (int nb : boundary_nbs) {
				Update(nb);
			}
		}
		else {
			for (int nb : nbs) {
				Update(nb);
			}
		}
	};

	auto SubdivideEdge = [&](int vidx0, int vidx1,
		const std::shared_ptr<ccMesh>& old_mesh,
		std::shared_ptr<ccMesh>& new_mesh,
		EdgeNewVertMap& new_verts,
		const EdgeTrianglesMap& edge_to_triangles) {
		Eigen::Vector2i edge = GetOrderedEdge(vidx0, vidx1);

		ccPointCloud* newVertice =
			ccHObjectCaster::ToPointCloud(new_mesh->getAssociatedCloud());
		if (!newVertice)
		{
			utility::LogError(
				"[ccMesh::simplifyVertexClustering] mesh"
				"should set associated cloud before using!");
		}
		const ccPointCloud* oldVertice =
			ccHObjectCaster::ToPointCloud(old_mesh->getAssociatedCloud());
		if (!oldVertice)
		{
			utility::LogError(
				"[ccMesh::simplifyVertexClustering] mesh"
				"should set associated cloud before using!");
		}

		if (new_verts.count(edge) == 0) {
			Eigen::Vector3d new_vert =
				oldVertice->getEigenPoint(static_cast<size_t>(vidx0)) + 
				oldVertice->getEigenPoint(static_cast<size_t>(vidx1));
			Eigen::Vector3d new_normal;
			if (has_vert_normal) {
				new_normal = oldVertice->getEigenNormal(static_cast<size_t>(vidx0)) +
					oldVertice->getEigenNormal(static_cast<size_t>(vidx1));
			}
			Eigen::Vector3d new_color;
			if (has_vert_color) {
				new_color = oldVertice->getEigenColor(static_cast<size_t>(vidx0)) +
					oldVertice->getEigenColor(static_cast<size_t>(vidx1));
			}

			const auto& edge_triangles = edge_to_triangles.at(edge);
			if (edge_triangles.size() < 2) {
				new_vert *= 0.5;
				if (has_vert_normal) {
					new_normal *= 0.5;
				}
				if (has_vert_color) {
					new_color *= 0.5;
				}
			}
			else {
				new_vert *= 3. / 8.;
				if (has_vert_normal) {
					new_normal *= 3. / 8.;
				}
				if (has_vert_color) {
					new_color *= 3. / 8.;
				}
				size_t n_adjacent_trias = edge_triangles.size();
				double scale = 1. / (4. * n_adjacent_trias);
				for (int tidx : edge_triangles) {
					const auto& tria = 
						old_mesh->getTriangle(static_cast<size_t>(tidx));
					int vidx2 =
						(tria(0) != vidx0 && tria(0) != vidx1)
						? tria(0)
						: ((tria(1) != vidx0 && tria(1) != vidx1)
							? tria(1)
							: tria(2));
					new_vert += scale * oldVertice->getEigenPoint(static_cast<size_t>(vidx2));
					if (has_vert_normal) {
						new_normal += scale * oldVertice->getEigenNormal(static_cast<size_t>(vidx2));
					}
					if (has_vert_color) {
						new_color += scale * oldVertice->getEigenColor(static_cast<size_t>(vidx2));
					}
				}
			}

			int vidx01 = int(oldVertice->size() + new_verts.size());

			newVertice->setPoint(static_cast<size_t>(vidx01), new_vert);
			if (has_vert_normal) {
				newVertice->setEigenNormal(static_cast<unsigned>(vidx01), new_normal);
			}
			if (has_vert_color) {
				newVertice->setEigenColor(static_cast<unsigned>(vidx01), new_color);
			}

			new_verts[edge] = vidx01;
			return vidx01;
		}
		else {
			int vidx01 = new_verts[edge];
			return vidx01;
		}
	};

	auto InsertTriangle = [&](int tidx, int vidx0, int vidx1, int vidx2,
							  std::shared_ptr<ccMesh>& mesh,
							  EdgeTrianglesMap& edge_to_triangles,
							  VertexNeighbours& vertex_neighbours) {
		mesh->setTriangle(static_cast<size_t>(tidx), Eigen::Vector3i(vidx0, vidx1, vidx2));
		edge_to_triangles[GetOrderedEdge(vidx0, vidx1)].insert(tidx);
		edge_to_triangles[GetOrderedEdge(vidx1, vidx2)].insert(tidx);
		edge_to_triangles[GetOrderedEdge(vidx2, vidx0)].insert(tidx);
		vertex_neighbours[vidx0].insert(vidx1);
		vertex_neighbours[vidx0].insert(vidx2);
		vertex_neighbours[vidx1].insert(vidx0);
		vertex_neighbours[vidx1].insert(vidx2);
		vertex_neighbours[vidx2].insert(vidx0);
		vertex_neighbours[vidx2].insert(vidx1);
	};

	EdgeTrianglesMap edge_to_triangles;
	VertexNeighbours vertex_neighbours(cloud->size());
	for (size_t tidx = 0; tidx < size(); ++tidx) {
		const auto& tria = getTriangle(tidx);
		Eigen::Vector2i e0 = GetOrderedEdge(tria(0), tria(1));
		edge_to_triangles[e0].insert(int(tidx));
		Eigen::Vector2i e1 = GetOrderedEdge(tria(1), tria(2));
		edge_to_triangles[e1].insert(int(tidx));
		Eigen::Vector2i e2 = GetOrderedEdge(tria(2), tria(0));
		edge_to_triangles[e2].insert(int(tidx));

		if (edge_to_triangles[e0].size() > 2 ||
			edge_to_triangles[e1].size() > 2 ||
			edge_to_triangles[e2].size() > 2) {
			utility::LogWarning("[SubdivideLoop] non-manifold edge.");
		}

		vertex_neighbours[tria(0)].insert(tria(1));
		vertex_neighbours[tria(0)].insert(tria(2));
		vertex_neighbours[tria(1)].insert(tria(0));
		vertex_neighbours[tria(1)].insert(tria(2));
		vertex_neighbours[tria(2)].insert(tria(0));
		vertex_neighbours[tria(2)].insert(tria(1));
	}

	ccPointCloud* oldVertices = new ccPointCloud("vertices");
	assert(oldVertices);
	oldVertices->setEnabled(false);
	// DGM: no need to lock it as it is only used by one mesh!
	oldVertices->setLocked(false);
	auto old_mesh = cloudViewer::make_shared<ccMesh>(oldVertices);
	old_mesh->addChild(oldVertices);
	old_mesh->merge(this, false);

	for (int iter = 0; iter < number_of_iterations; ++iter) {
		size_t n_new_vertices =
			old_mesh->getVerticeSize() + edge_to_triangles.size();
		size_t n_new_triangles = 4 * old_mesh->size();

		ccPointCloud* newVertices = new ccPointCloud("vertices");
		assert(newVertices);
		newVertices->setEnabled(false);
		// DGM: no need to lock it as it is only used by one mesh!
		newVertices->setLocked(false);
		auto new_mesh = cloudViewer::make_shared<ccMesh>(newVertices);
		new_mesh->addChild(newVertices);

		newVertices->resize(static_cast<unsigned>(n_new_vertices));
		if (has_vert_normal) {
			newVertices->resizeTheNormsTable();
		}
		if (has_vert_color) {
			newVertices->resizeTheRGBTable();
		}
		new_mesh->resize(n_new_triangles);

		EdgeNewVertMap new_verts;
		EdgeTrianglesMap new_edge_to_triangles;
		VertexNeighbours new_vertex_neighbours(n_new_vertices);

		for (size_t vidx = 0; vidx < old_mesh->getVerticeSize(); ++vidx) {
			UpdateVertex(int(vidx), old_mesh, new_mesh, vertex_neighbours[vidx],
				edge_to_triangles);
		}

		for (size_t tidx = 0; tidx < old_mesh->size(); ++tidx) {
			const auto& triangle = old_mesh->getTriangle(tidx);
			int vidx0 = triangle(0);
			int vidx1 = triangle(1);
			int vidx2 = triangle(2);

			int vidx01 = SubdivideEdge(vidx0, vidx1, old_mesh, new_mesh,
				new_verts, edge_to_triangles);
			int vidx12 = SubdivideEdge(vidx1, vidx2, old_mesh, new_mesh,
				new_verts, edge_to_triangles);
			int vidx20 = SubdivideEdge(vidx2, vidx0, old_mesh, new_mesh,
				new_verts, edge_to_triangles);

			InsertTriangle(int(tidx) * 4 + 0, vidx0, vidx01, vidx20, new_mesh,
				new_edge_to_triangles, new_vertex_neighbours);
			InsertTriangle(int(tidx) * 4 + 1, vidx01, vidx1, vidx12, new_mesh,
				new_edge_to_triangles, new_vertex_neighbours);
			InsertTriangle(int(tidx) * 4 + 2, vidx12, vidx2, vidx20, new_mesh,
				new_edge_to_triangles, new_vertex_neighbours);
			InsertTriangle(int(tidx) * 4 + 3, vidx01, vidx12, vidx20, new_mesh,
				new_edge_to_triangles, new_vertex_neighbours);
		}

		old_mesh = std::move(new_mesh);
		edge_to_triangles = std::move(new_edge_to_triangles);
		vertex_neighbours = std::move(new_vertex_neighbours);
	}

	if (hasTriNormals()) {
		old_mesh->computeTriangleNormals();
	}

	return old_mesh;
}

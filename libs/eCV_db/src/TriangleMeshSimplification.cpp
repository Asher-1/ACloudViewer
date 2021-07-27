// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.erow.cn
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

#include <unordered_set>
#include <unordered_map>

using namespace cloudViewer;
/// Error quadric that is used to minimize the squared distance of a point to
/// its neigbhouring triangle planes.
/// Cf. "Simplifying Surfaces with Color and Texture using Quadric Error
/// Metrics" by Garland and Heckbert.
class Quadric {
public:
	Quadric() {
		A_.fill(0);
		b_.fill(0);
		c_ = 0;
	}

	Quadric(const Eigen::Vector4d& plane, double weight = 1) {
		Eigen::Vector3d n = plane.head<3>();
		A_ = weight * n * n.transpose();
		b_ = weight * plane(3) * n;
		c_ = weight * plane(3) * plane(3);
	}

	Quadric& operator+=(const Quadric& other) {
		A_ += other.A_;
		b_ += other.b_;
		c_ += other.c_;
		return *this;
	}

	Quadric operator+(const Quadric& other) const {
		Quadric res;
		res.A_ = A_ + other.A_;
		res.b_ = b_ + other.b_;
		res.c_ = c_ + other.c_;
		return res;
	}

	double Eval(const Eigen::Vector3d& v) const {
		Eigen::Vector3d Av = A_ * v;
		double q = v.dot(Av) + 2 * b_.dot(v) + c_;
		return q;
	}

	bool IsInvertible() const { return std::fabs(A_.determinant()) > 1e-4; }

	Eigen::Vector3d Minimum() const { return -A_.ldlt().solve(b_); }

public:
	/// A_ = n . n^T, where n is the plane normal
	Eigen::Matrix3d A_;
	/// b_ = d . n, where n is the plane normal and d the non-normal component
	/// of the plane parameters
	Eigen::Vector3d b_;
	/// c_ = d . d, where d the non-normal component pf the plane parameters
	double c_;
};

std::shared_ptr<ccMesh> ccMesh::simplifyVertexClustering(
	double voxel_size,
	SimplificationContraction
	contraction /* = SimplificationContraction::Average */) const {
	if (hasTriangleUvs()) {
		utility::LogWarning(
			"[SimplifyVertexClustering] This mesh contains triangle uvs "
			"that are not handled in this function");
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

	if (voxel_size <= 0.0) {
		utility::LogError("[VoxelGridFromPointCloud] voxel_size <= 0.0");
	}

	Eigen::Vector3d voxel_size3 =
		Eigen::Vector3d(voxel_size, voxel_size, voxel_size);
	Eigen::Vector3d voxel_min_bound = getMinBound() - voxel_size3 * 0.5;
	Eigen::Vector3d voxel_max_bound = getMaxBound() + voxel_size3 * 0.5;
	if (voxel_size * std::numeric_limits<int>::max() <
		(voxel_max_bound - voxel_min_bound).maxCoeff()) {
		utility::LogError("[VoxelGridFromPointCloud] voxel_size is too small.");
	}

	auto GetVoxelIdx = [&](const Eigen::Vector3d& vert) {
		Eigen::Vector3d ref_coord = (vert - voxel_min_bound) / voxel_size;
		Eigen::Vector3i idx(int(floor(ref_coord(0))), int(floor(ref_coord(1))),
			int(floor(ref_coord(2))));
		return idx;
	};

	std::unordered_map<Eigen::Vector3i, std::unordered_set<int>,
		utility::hash_eigen::hash<Eigen::Vector3i>>
		voxel_vertices;
	std::unordered_map<Eigen::Vector3i, int,
		utility::hash_eigen::hash<Eigen::Vector3i>>
		voxel_vert_ind;
	int new_vidx = 0;
	for (size_t vidx = 0; vidx < cloud->size(); ++vidx) {
		const Eigen::Vector3i vox_idx = GetVoxelIdx(cloud->getEigenPoint(vidx));
		voxel_vertices[vox_idx].insert(int(vidx));

		if (voxel_vert_ind.count(vox_idx) == 0) {
			voxel_vert_ind[vox_idx] = new_vidx;
			new_vidx++;
		}
	}

	// aggregate vertex info
	bool has_vert_normal = cloud->hasNormals();
	bool has_vert_color = cloud->hasColors();
	baseVertices->resize(static_cast<unsigned int>(voxel_vertices.size()));
	if (has_vert_normal) {
		baseVertices->resizeTheNormsTable();
		baseVertices->showNormals(true);
	}
	if (has_vert_color) {
		baseVertices->resizeTheRGBTable();
		baseVertices->showColors(true);
	}

	auto AvgVertex = [&](const std::unordered_set<int> ind) {
		Eigen::Vector3d aggr(0, 0, 0);
		for (int vidx : ind) {
			aggr += cloud->getEigenPoint(vidx);
		}
		aggr /= double(ind.size());
		return aggr;
	};
	auto AvgNormal = [&](const std::unordered_set<int> ind) {
		Eigen::Vector3d aggr(0, 0, 0);
		for (int vidx : ind) {
			aggr += cloud->getEigenNormal(static_cast<size_t>(vidx));
		}
		aggr /= double(ind.size());
		return aggr;
	};
	auto AvgColor = [&](const std::unordered_set<int> ind) {
		Eigen::Vector3d aggr(0, 0, 0);
		for (int vidx : ind) {
			aggr += cloud->getEigenColor(static_cast<size_t>(vidx));
		}
		aggr /= double(ind.size());
		return aggr;
	};

	if (contraction == SimplificationContraction::Average) {
		for (const auto& voxel : voxel_vertices) {
			int vox_vidx = voxel_vert_ind[voxel.first];
			baseVertices->setPoint(static_cast<size_t>(vox_vidx), AvgVertex(voxel.second));
			if (has_vert_normal) {
				baseVertices->setPointNormal(
					static_cast<size_t>(vox_vidx), AvgNormal(voxel.second));
			}
			if (has_vert_color) {
				baseVertices->setPointColor(
					static_cast<size_t>(vox_vidx), AvgColor(voxel.second));
			}
		}
	}
	else if (contraction == SimplificationContraction::Quadric) {
		// Map triangles
		std::unordered_map<int, std::unordered_set<int>> vert_to_triangles;
		for (size_t tidx = 0; tidx < size(); ++tidx) {
			const cloudViewer::VerticesIndexes* tri = 
				getTriangleVertIndexes(static_cast<unsigned>(tidx));
			vert_to_triangles[tri->i1].emplace(int(tidx));
			vert_to_triangles[tri->i2].emplace(int(tidx));
			vert_to_triangles[tri->i3].emplace(int(tidx));
		}

		for (const auto& voxel : voxel_vertices) {
			size_t vox_vidx = static_cast<size_t>(voxel_vert_ind[voxel.first]);
			Quadric q;
			for (int vidx : voxel.second) {
				for (int tidx : vert_to_triangles[vidx]) {
					Eigen::Vector4d p = getTrianglePlane(tidx);
					double area = getTriangleArea(tidx);
					q += Quadric(p, area);
				}
			}
			if (q.IsInvertible()) {
				Eigen::Vector3d v = q.Minimum();
				baseVertices->setPoint(vox_vidx, v);
			}
			else {
				baseVertices->setPoint(vox_vidx, AvgVertex(voxel.second));
			}

			if (has_vert_normal) {
				baseVertices->setPointNormal(vox_vidx, AvgNormal(voxel.second));
			}
			if (has_vert_color) {
				baseVertices->setPointColor(vox_vidx, AvgColor(voxel.second));
			}
		}
	}

	//  connect vertices
	std::unordered_set<Eigen::Vector3i,
		utility::hash_eigen::hash<Eigen::Vector3i>>
		triangles;
	for (unsigned triIndx = 0; triIndx < size(); ++triIndx) {
		Eigen::Vector3d v0, v1, v2;
		getTriangleVertices(triIndx, v0.data(), v1.data(), v2.data());
		int vidx0 = voxel_vert_ind[GetVoxelIdx(v0)];
		int vidx1 = voxel_vert_ind[GetVoxelIdx(v1)];
		int vidx2 = voxel_vert_ind[GetVoxelIdx(v2)];

		// only connect if in different voxels
		if (vidx0 == vidx1 || vidx0 == vidx2 || vidx1 == vidx2) {
			continue;
		}

		// Note: there can be still double faces with different orientation
		// The user has to clean up manually
		if (vidx1 < vidx0 && vidx1 < vidx2) {
			int tmp = vidx0;
			vidx0 = vidx1;
			vidx1 = vidx2;
			vidx2 = tmp;
		}
		else if (vidx2 < vidx0 && vidx2 < vidx1) {
			int tmp = vidx1;
			vidx1 = vidx0;
			vidx0 = vidx2;
			vidx2 = tmp;
		}

		triangles.emplace(Eigen::Vector3i(vidx0, vidx1, vidx2));
	}

	mesh->resize(triangles.size());
	unsigned int tidx = 0;
	for (const Eigen::Vector3i& triangle : triangles) {
		mesh->setTriangle(tidx, triangle);
		tidx++;
	}

	if (hasTriNormals()) {
		mesh->computeTriangleNormals();
	}

	return mesh;
}

std::shared_ptr<ccMesh> ccMesh::simplifyQuadricDecimation(
        int target_number_of_triangles,
        double maximum_error/* = std::numeric_limits<double>::infinity()*/,
        double boundary_weight/* = 1.0*/) const {
	if (hasTriangleUvs()) {
		utility::LogWarning(
			"[SimplifyQuadricDecimation] This mesh contains triangle uvs "
			"that are not handled in this function");
	}
	typedef std::tuple<double, int, int> CostEdge;

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

	std::vector<bool> vertices_deleted(cloud->size(), false);
	std::vector<bool> triangles_deleted(this->size(), false);

	// Map vertices to triangles and compute triangle planes and areas
	std::vector<std::unordered_set<int>> vert_to_triangles(cloud->size());
	std::vector<Eigen::Vector4d> triangle_planes(this->size());
	std::vector<double> triangle_areas(this->size());
	for (size_t tidx = 0; tidx < this->size(); ++tidx) {
		const cloudViewer::VerticesIndexes* tri =
			getTriangleVertIndexes(static_cast<unsigned>(tidx));
		vert_to_triangles[tri->i1].emplace(static_cast<int>(tidx));
		vert_to_triangles[tri->i2].emplace(static_cast<int>(tidx));
		vert_to_triangles[tri->i3].emplace(static_cast<int>(tidx));

		triangle_planes[tidx] = getTrianglePlane(tidx);
		triangle_areas[tidx] = getTriangleArea(tidx);
	}

	// Compute the error metric per vertex
	std::vector<Quadric> Qs(cloud->size());
	for (size_t vidx = 0; vidx < cloud->size(); ++vidx) {
		for (int tidx : vert_to_triangles[vidx]) {
			Qs[vidx] += Quadric(triangle_planes[tidx], triangle_areas[tidx]);
		}
	}

	// For boundary edges add perpendicular plane quadric
	auto edge_triangle_count = getEdgeToTrianglesMap();
	auto AddPerpPlaneQuadric = [&](int vidx0, int vidx1, int vidx2,
		double area) {
		int min = std::min(vidx0, vidx1);
		int max = std::max(vidx0, vidx1);
		Eigen::Vector2i edge(min, max);
		if (edge_triangle_count[edge].size() != 1) {
			return;
		}
		const auto& vert0 = baseVertices->getEigenPoint(static_cast<size_t>(vidx0));
		const auto& vert1 = baseVertices->getEigenPoint(static_cast<size_t>(vidx1));
		const auto& vert2 = baseVertices->getEigenPoint(static_cast<size_t>(vidx2));
		Eigen::Vector3d vert2p = (vert2 - vert0).cross(vert2 - vert1);
		Eigen::Vector4d plane = ComputeTrianglePlane(vert0, vert1, vert2p);
        Quadric quad(plane, area * boundary_weight);
		Qs[vidx0] += quad;
		Qs[vidx1] += quad;
	};
	for (size_t tidx = 0; tidx < this->size(); ++tidx) {
		const auto& tria = getTriangle(tidx);
		double area = triangle_areas[tidx];
		AddPerpPlaneQuadric(tria(0), tria(1), tria(2), area);
		AddPerpPlaneQuadric(tria(1), tria(2), tria(0), area);
		AddPerpPlaneQuadric(tria(2), tria(0), tria(1), area);
	}

	// Get valid edges and compute cost
	// Note: We could also select all vertex pairs as edges with dist < eps
	std::unordered_map<Eigen::Vector2i, Eigen::Vector3d,
		utility::hash_eigen::hash<Eigen::Vector2i>> vbars;
	std::unordered_map<Eigen::Vector2i, double,
		utility::hash_eigen::hash<Eigen::Vector2i>> costs;
	auto CostEdgeComp = [](const CostEdge& a, const CostEdge& b) {
		return std::get<0>(a) > std::get<0>(b);
	};
	std::priority_queue<CostEdge, std::vector<CostEdge>, decltype(CostEdgeComp)>
		queue(CostEdgeComp);

	auto AddEdge = [&](int vidx0, int vidx1, bool update) {
		int min = std::min(vidx0, vidx1);
		int max = std::max(vidx0, vidx1);
		Eigen::Vector2i edge(min, max);
		if (update || vbars.count(edge) == 0) {
			const Quadric& Q0 = Qs[min];
			const Quadric& Q1 = Qs[max];
			Quadric Qbar = Q0 + Q1;
			double cost;
			Eigen::Vector3d vbar;
			if (Qbar.IsInvertible()) {
				vbar = Qbar.Minimum();
				cost = Qbar.Eval(vbar);
			}
			else {
				const Eigen::Vector3d& v0 = 
					baseVertices->getEigenPoint(static_cast<size_t>(vidx0));
				const Eigen::Vector3d& v1 = 
					baseVertices->getEigenPoint(static_cast<size_t>(vidx1));
				Eigen::Vector3d vmid = (v0 + v1) / 2;
				double cost0 = Qbar.Eval(v0);
				double cost1 = Qbar.Eval(v1);
				double costmid = Qbar.Eval(vbar);
				cost = std::min(cost0, std::min(cost1, costmid));
				if (cost == costmid) {
					vbar = vmid;
				}
				else if (cost == cost0) {
					vbar = v0;
				}
				else {
					vbar = v1;
				}
			}
			vbars[edge] = vbar;
			costs[edge] = cost;
			queue.push(CostEdge(cost, min, max));
		}
	};

	// add all edges to priority queue
	for (size_t trindx = 0; trindx < this->size(); ++trindx) {
		const auto& triangle = getTriangle(trindx);
		AddEdge(triangle(0), triangle(1), false);
		AddEdge(triangle(1), triangle(2), false);
		AddEdge(triangle(2), triangle(0), false);
	}

	// perform incremental edge collapse
	bool has_vert_normal = cloud->hasNormals();
	bool has_vert_color = cloud->hasColors();
	int n_triangles = int(this->size());
	while (n_triangles > target_number_of_triangles && !queue.empty()) {
		// retrieve edge from queue
		double cost;
		int vidx0, vidx1;
		std::tie(cost, vidx0, vidx1) = queue.top();
		queue.pop();

		if (cost > maximum_error) {
            break;
        }

		// test if the edge has been updated (reinserted into queue)
		Eigen::Vector2i edge(vidx0, vidx1);
		bool valid = !vertices_deleted[vidx0] && !vertices_deleted[vidx1] &&
			cost == costs[edge];
		if (!valid) {
			continue;
		}

		// avoid flip of triangle normal
		bool flipped = false;
		for (int tidx : vert_to_triangles[vidx1]) {
			if (triangles_deleted[tidx]) {
				continue;
			}

			const Eigen::Vector3i& tria = 
				mesh->getTriangle(static_cast<size_t>(tidx));
			bool has_vidx0 =
				vidx0 == tria(0) || vidx0 == tria(1) || vidx0 == tria(2);
			bool has_vidx1 =
				vidx1 == tria(0) || vidx1 == tria(1) || vidx1 == tria(2);
			if (has_vidx0 && has_vidx1) {
				continue;
			}

			Eigen::Vector3d vert0 = baseVertices->getEigenPoint(static_cast<size_t>(tria(0)));
			Eigen::Vector3d vert1 = baseVertices->getEigenPoint(static_cast<size_t>(tria(1)));
			Eigen::Vector3d vert2 = baseVertices->getEigenPoint(static_cast<size_t>(tria(2)));
			Eigen::Vector3d norm_before = (vert1 - vert0).cross(vert2 - vert0);
			norm_before /= norm_before.norm();

			if (vidx1 == tria(0)) {
				vert0 = vbars[edge];
			} else if (vidx1 == tria(1)) {
				vert1 = vbars[edge];
			} else if (vidx1 == tria(2)) {
				vert2 = vbars[edge];
			}

			Eigen::Vector3d norm_after = (vert1 - vert0).cross(vert2 - vert0);
			norm_after /= norm_after.norm();
			if (norm_before.dot(norm_after) < 0) {
				flipped = true;
				break;
			}
		}
		if (flipped) {
			continue;
		}

		// Connect triangles from vidx1 to vidx0, or mark deleted
		for (int tidx : vert_to_triangles[vidx1]) {
			if (triangles_deleted[tidx]) {
				continue;
			}

			cloudViewer::VerticesIndexes* tria = mesh->getTriangleVertIndexes(
				static_cast<unsigned>(tidx));
			bool has_vidx0 = vidx0 == static_cast<int>(tria->i[0]) || 
							 vidx0 == static_cast<int>(tria->i[1]) ||
							 vidx0 == static_cast<int>(tria->i[2]);
			bool has_vidx1 = vidx1 == static_cast<int>(tria->i[0]) ||
							 vidx1 == static_cast<int>(tria->i[1]) ||
							 vidx1 == static_cast<int>(tria->i[2]);
			if (has_vidx0 && has_vidx1) {
				triangles_deleted[tidx] = true;
				n_triangles--;
				continue;
			}

			if (vidx1 == static_cast<int>(tria->i[0])) {
				tria->i[0] = static_cast<unsigned>(vidx0);
			} else if (vidx1 == static_cast<int>(tria->i[1])) {
				tria->i[1] = static_cast<unsigned>(vidx0);
			} else if (vidx1 == static_cast<int>(tria->i[2])) {
				tria->i[2] = static_cast<unsigned>(vidx0);
			}
			vert_to_triangles[vidx0].insert(tidx);
		}

		// update vertex vidx0 to vbar
		baseVertices->setPoint(static_cast<size_t>(vidx0), vbars[edge]);
		Qs[vidx0] += Qs[vidx1];
		if (has_vert_normal) {
			baseVertices->setEigenNormal(static_cast<size_t>(vidx0), 0.5 * (
				baseVertices->getEigenNormal(static_cast<size_t>(vidx0)) +
				baseVertices->getEigenNormal(static_cast<size_t>(vidx1))));
		}
		if (has_vert_color) {
			baseVertices->setEigenColor(static_cast<size_t>(vidx0), 0.5 * (
					baseVertices->getEigenColor(static_cast<size_t>(vidx0)) +
					baseVertices->getEigenColor(static_cast<size_t>(vidx1))));
		}
		vertices_deleted[vidx1] = true;

		// Update edge costs for all triangles connecting to vidx0
		for (const auto& tidx : vert_to_triangles[vidx0]) {
			if (triangles_deleted[tidx]) {
				continue;
			}
			const Eigen::Vector3i& tria = mesh->getTriangle(static_cast<size_t>(tidx));
			if (tria(0) == vidx0 || tria(1) == vidx0) {
				AddEdge(tria(0), tria(1), true);
			}
			if (tria(1) == vidx0 || tria(2) == vidx0) {
				AddEdge(tria(1), tria(2), true);
			}
			if (tria(2) == vidx0 || tria(0) == vidx0) {
				AddEdge(tria(2), tria(0), true);
			}
		}
	}

	// Apply changes to the triangle mesh
	unsigned int next_free = 0;
	std::unordered_map<int, int> vert_remapping;
	for (unsigned int idx = 0; idx < baseVertices->size(); ++idx) {
		if (!vertices_deleted[idx]) {
			vert_remapping[int(idx)] = next_free;
			baseVertices->setPoint(next_free, *baseVertices->getPoint(idx));
			if (has_vert_normal) {
				baseVertices->setPointNormal(next_free, baseVertices->getPointNormal(idx));
			}
			if (has_vert_color) {
				baseVertices->setPointColor(next_free, baseVertices->getPointColor(idx));
			}
			next_free++;
		}
	}

	baseVertices->resize(next_free);
	if (has_vert_normal) {
		baseVertices->resizeTheNormsTable();
	}
	if (has_vert_color) {
		baseVertices->resizeTheRGBTable();
	}

	next_free = 0;
	for (size_t idx = 0; idx < mesh->size(); ++idx) {
		if (!triangles_deleted[idx]) {
			Eigen::Vector3i tria = mesh->getTriangle(idx);
			mesh->setTriangle(next_free, 
				Eigen::Vector3i(
					vert_remapping[tria(0)], 
					vert_remapping[tria(1)],
					vert_remapping[tria(2)]));
			next_free++;
		}
	}
	mesh->resize(next_free);

	if (hasTriNormals()) {
		mesh->computeTriangleNormals();
	}

	return mesh;
}

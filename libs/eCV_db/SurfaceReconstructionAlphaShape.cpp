// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include "ecvPointCloud.h"
#include "ecvQhull.h"
#include "ecvMesh.h"
#include "ecvTetraMesh.h"
#include "ecvHObjectCaster.h"
#include <Console.h>

#include <Eigen/Dense>

#include <iostream>
#include <list>

using namespace CVLib;
std::shared_ptr<ccMesh> ccMesh::CreateFromPointCloudAlphaShape(
	const ccPointCloud& pcd,
	double alpha,
	std::shared_ptr<cloudViewer::geometry::TetraMesh> tetra_mesh,
	std::vector<size_t>* pt_map) {
	std::vector<size_t> pt_map_computed;
	if (tetra_mesh == nullptr) {
		utility::LogDebug(
			"[CreateFromPointCloudAlphaShape] "
			"ComputeDelaunayTetrahedralization");
		std::tie(tetra_mesh, pt_map_computed) =
			cloudViewer::utility::Qhull::ComputeDelaunayTetrahedralization(pcd.getPoints());
		pt_map = &pt_map_computed;
		utility::LogDebug(
			"[CreateFromPointCloudAlphaShape] done "
			"ComputeDelaunayTetrahedralization");
	}

	utility::LogDebug("[CreateFromPointCloudAlphaShape] init triangle mesh");
	
	ccPointCloud* baseVertices = new ccPointCloud("vertices");
	assert(baseVertices);
	baseVertices->setEnabled(false);
	// DGM: no need to lock it as it is only used by one mesh!
	baseVertices->setLocked(false);
	auto mesh = std::make_shared<ccMesh>(baseVertices);
	mesh->addChild(baseVertices);

	baseVertices->addPoints(tetra_mesh->vertices_);
	if (pcd.hasNormals()) {
		baseVertices->resizeTheNormsTable();
		for (size_t idx = 0; idx < (*pt_map).size(); ++idx) {
			baseVertices->setPointNormal(
				static_cast<unsigned>(idx), 
				pcd.getPointNormal(static_cast<unsigned>((*pt_map)[idx])));
		}
	}
	if (pcd.hasColors()) {
		baseVertices->resizeTheRGBTable();
		for (size_t idx = 0; idx < (*pt_map).size(); ++idx) {
			baseVertices->setPointColor(
				static_cast<unsigned>(idx),
				pcd.getPointColor(static_cast<unsigned>((*pt_map)[idx])));
		}
	}
	utility::LogDebug(
		"[CreateFromPointCloudAlphaShape] done init triangle mesh");

	std::vector<double> vsqn(tetra_mesh->vertices_.size());
	for (size_t vidx = 0; vidx < vsqn.size(); ++vidx) {
		vsqn[vidx] = tetra_mesh->vertices_[vidx].squaredNorm();
	}

	utility::LogDebug(
		"[CreateFromPointCloudAlphaShape] add triangles from tetras that "
		"satisfy constraint");
	const auto& verts = tetra_mesh->vertices_;
	for (size_t tidx = 0; tidx < tetra_mesh->tetras_.size(); ++tidx) {
		const auto& tetra = tetra_mesh->tetras_[tidx];
		// clang-format off
		Eigen::Matrix4d tmp;
		tmp << verts[tetra(0)](0), verts[tetra(0)](1), verts[tetra(0)](2), 1,
			verts[tetra(1)](0), verts[tetra(1)](1), verts[tetra(1)](2), 1,
			verts[tetra(2)](0), verts[tetra(2)](1), verts[tetra(2)](2), 1,
			verts[tetra(3)](0), verts[tetra(3)](1), verts[tetra(3)](2), 1;
		double a = tmp.determinant();
		tmp << vsqn[tetra(0)], verts[tetra(0)](0), verts[tetra(0)](1), verts[tetra(0)](2),
			vsqn[tetra(1)], verts[tetra(1)](0), verts[tetra(1)](1), verts[tetra(1)](2),
			vsqn[tetra(2)], verts[tetra(2)](0), verts[tetra(2)](1), verts[tetra(2)](2),
			vsqn[tetra(3)], verts[tetra(3)](0), verts[tetra(3)](1), verts[tetra(3)](2);
		double c = tmp.determinant();
		tmp << vsqn[tetra(0)], verts[tetra(0)](1), verts[tetra(0)](2), 1,
			vsqn[tetra(1)], verts[tetra(1)](1), verts[tetra(1)](2), 1,
			vsqn[tetra(2)], verts[tetra(2)](1), verts[tetra(2)](2), 1,
			vsqn[tetra(3)], verts[tetra(3)](1), verts[tetra(3)](2), 1;
		double dx = tmp.determinant();
		tmp << vsqn[tetra(0)], verts[tetra(0)](0), verts[tetra(0)](2), 1,
			vsqn[tetra(1)], verts[tetra(1)](0), verts[tetra(1)](2), 1,
			vsqn[tetra(2)], verts[tetra(2)](0), verts[tetra(2)](2), 1,
			vsqn[tetra(3)], verts[tetra(3)](0), verts[tetra(3)](2), 1;
		double dy = tmp.determinant();
		tmp << vsqn[tetra(0)], verts[tetra(0)](0), verts[tetra(0)](1), 1,
			vsqn[tetra(1)], verts[tetra(1)](0), verts[tetra(1)](1), 1,
			vsqn[tetra(2)], verts[tetra(2)](0), verts[tetra(2)](1), 1,
			vsqn[tetra(3)], verts[tetra(3)](0), verts[tetra(3)](1), 1;
		double dz = tmp.determinant();
		// clang-format on
		if (a == 0) {
			utility::LogError(
				"[CreateFromPointCloudAlphaShape] invalid tetra in "
				"TetraMesh");
		}
		double r = std::sqrt(dx * dx + dy * dy + dz * dz - 4 * a * c) /
			(2 * std::abs(a));

		if (r <= alpha) {
			mesh->addTriangle(ccMesh::GetOrderedTriangle(
				tetra(0), tetra(1), tetra(2)));
			mesh->addTriangle(ccMesh::GetOrderedTriangle(
				tetra(0), tetra(1), tetra(3)));
			mesh->addTriangle(ccMesh::GetOrderedTriangle(
				tetra(0), tetra(2), tetra(3)));
			mesh->addTriangle(ccMesh::GetOrderedTriangle(
				tetra(1), tetra(2), tetra(3)));
		}
	}
	utility::LogDebug(
		"[CreateFromPointCloudAlphaShape] done add triangles from tetras "
		"that satisfy constraint");

	utility::LogDebug(
		"[CreateFromPointCloudAlphaShape] remove triangles within "
		"the mesh");
	std::unordered_map<Eigen::Vector3i, int,
		utility::hash_eigen::hash<Eigen::Vector3i>>
		triangle_count;
	for (size_t tidx = 0; tidx < mesh->size(); ++tidx) {
		Eigen::Vector3i triangle = mesh->getTriangle(tidx);
		if (triangle_count.count(triangle) == 0) {
			triangle_count[triangle] = 1;
		}
		else {
			triangle_count[triangle] += 1;
		}
	}

	size_t to_idx = 0;
	for (size_t tidx = 0; tidx < mesh->size(); ++tidx) {
		Eigen::Vector3i triangle = mesh->getTriangle(tidx);
		if (triangle_count[triangle] == 1) {
			mesh->setTriangle(to_idx, triangle);
			to_idx++;
		}
	}
	mesh->resize(to_idx);
	utility::LogDebug(
		"[CreateFromPointCloudAlphaShape] done remove triangles within "
		"the mesh");

	utility::LogDebug(
		"[CreateFromPointCloudAlphaShape] remove duplicate triangles and "
		"unreferenced vertices");

	//do some cleaning
	{
		baseVertices->shrinkToFit();
		mesh->shrinkToFit();
		NormsIndexesTableType* normals = mesh->getTriNormsTable();
		if (normals)
		{
			normals->shrink_to_fit();
		}
	}

	utility::LogDebug(
		"[CreateFromPointCloudAlphaShape] done remove duplicate triangles "
		"and unreferenced vertices");

	return mesh;
}
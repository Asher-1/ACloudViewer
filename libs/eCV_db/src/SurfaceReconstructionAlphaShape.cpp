// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Logging.h>

#include <Eigen/Dense>
#include <iostream>
#include <list>

#include "ecvHObjectCaster.h"
#include "ecvMesh.h"
#include "ecvPointCloud.h"
#include "ecvQhull.h"
#include "ecvTetraMesh.h"

std::shared_ptr<ccMesh> ccMesh::CreateFromPointCloudAlphaShape(
        const ccPointCloud& pcd,
        double alpha,
        std::shared_ptr<cloudViewer::geometry::TetraMesh> tetra_mesh,
        std::vector<size_t>* pt_map) {
    std::vector<size_t> pt_map_computed;
    if (tetra_mesh == nullptr) {
        cloudViewer::utility::LogDebug(
                "[CreateFromPointCloudAlphaShape] "
                "ComputeDelaunayTetrahedralization");
        std::tie(tetra_mesh, pt_map_computed) =
                cloudViewer::geometry::Qhull::ComputeDelaunayTetrahedralization(
                        pcd.getPoints());
        pt_map = &pt_map_computed;
        cloudViewer::utility::LogDebug(
                "[CreateFromPointCloudAlphaShape] done "
                "ComputeDelaunayTetrahedralization");
    }

    cloudViewer::utility::LogDebug(
            "[CreateFromPointCloudAlphaShape] init triangle mesh");

    ccPointCloud* baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);
    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    auto mesh = cloudViewer::make_shared<ccMesh>(baseVertices);
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
    cloudViewer::utility::LogDebug(
            "[CreateFromPointCloudAlphaShape] done init triangle mesh");

    std::vector<double> vsqn(tetra_mesh->vertices_.size());
    for (size_t vidx = 0; vidx < vsqn.size(); ++vidx) {
        vsqn[vidx] = tetra_mesh->vertices_[vidx].squaredNorm();
    }

    cloudViewer::utility::LogDebug(
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
            cloudViewer::utility::LogError(
                    "[CreateFromPointCloudAlphaShape] invalid tetra in "
                    "TetraMesh");
        }
        double r = std::sqrt(dx * dx + dy * dy + dz * dz - 4 * a * c) /
                   (2 * std::abs(a));

        if (r <= alpha) {
            mesh->addTriangle(
                    ccMesh::GetOrderedTriangle(tetra(0), tetra(1), tetra(2)));
            mesh->addTriangle(
                    ccMesh::GetOrderedTriangle(tetra(0), tetra(1), tetra(3)));
            mesh->addTriangle(
                    ccMesh::GetOrderedTriangle(tetra(0), tetra(2), tetra(3)));
            mesh->addTriangle(
                    ccMesh::GetOrderedTriangle(tetra(1), tetra(2), tetra(3)));
        }
    }
    cloudViewer::utility::LogDebug(
            "[CreateFromPointCloudAlphaShape] done add triangles from tetras "
            "that satisfy constraint");

    cloudViewer::utility::LogDebug(
            "[CreateFromPointCloudAlphaShape] remove triangles within "
            "the mesh");
    std::unordered_map<Eigen::Vector3i, int,
                       cloudViewer::utility::hash_eigen<Eigen::Vector3i>>
            triangle_count;
    for (size_t tidx = 0; tidx < mesh->size(); ++tidx) {
        Eigen::Vector3i triangle = mesh->getTriangle(tidx);
        if (triangle_count.count(triangle) == 0) {
            triangle_count[triangle] = 1;
        } else {
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
    cloudViewer::utility::LogDebug(
            "[CreateFromPointCloudAlphaShape] done remove triangles within "
            "the mesh");

    cloudViewer::utility::LogDebug(
            "[CreateFromPointCloudAlphaShape] remove duplicate triangles and "
            "unreferenced vertices");

    // do some cleaning
    {
        baseVertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType* normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
        mesh->RemoveDuplicatedTriangles();
        mesh->RemoveUnreferencedVertices();
    }

    cloudViewer::utility::LogDebug(
            "[CreateFromPointCloudAlphaShape] done remove duplicate triangles "
            "and unreferenced vertices");

    return mesh;
}
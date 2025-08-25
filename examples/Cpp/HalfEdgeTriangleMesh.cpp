// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <iostream>

#include "CloudViewer.h"

using namespace cloudViewer;

void PrintHelp() {
    cloudViewer::utility::LogInfo("Usage :");
    cloudViewer::utility::LogInfo("    > ecvHalfEdgeMesh <file>");
}

void ColorizeBoundaryVertices(geometry::ecvHalfEdgeMesh &halfMesh,
                              const Eigen::Vector3d &color) {
    std::vector<Eigen::Vector3d> vertextColors(halfMesh.vertices_.size(),
                                               {0.75, 0.75, 0.75});
    std::vector<std::vector<int>> boundaries = halfMesh.getBoundaries();
    for (std::size_t i = 0; i < boundaries.size(); ++i) {
        for (std::size_t j = 0; j < boundaries[i].size(); ++j) {
            vertextColors[static_cast<std::size_t>(boundaries[i][j])] = color;
        }
    }
    halfMesh.vertex_colors_ = vertextColors;
}

void DrawGeometriesWithBackFace(
        const std::vector<std::shared_ptr<const ccHObject>> &geometry_ptrs) {
    visualization::DrawGeometries(geometry_ptrs, "CloudViewer", 640, 480, 50,
                                  50, false, false, true);
}

int main(int argc, char *argv[]) {
    cloudViewer::utility::SetVerbosityLevel(
            cloudViewer::utility::VerbosityLevel::Debug);

    if (argc < 2) {
        PrintHelp();
        return 1;
    }

    auto mesh = io::CreateMeshFromFile(argv[1]);
    ccBBox bbox =
            ccBBox(CCVector3(-1.f, -1.f, -1.f), CCVector3(1.f, 0.6f, 1.f));
    auto croppedMesh = mesh->crop(bbox);
    auto halfEdgeMesh =
            geometry::ecvHalfEdgeMesh::CreateFromTriangleMesh(*croppedMesh);
    DrawGeometriesWithBackFace({halfEdgeMesh});

    ColorizeBoundaryVertices(*halfEdgeMesh, {1, 0, 0});
    DrawGeometriesWithBackFace({halfEdgeMesh});
    return 0;
}

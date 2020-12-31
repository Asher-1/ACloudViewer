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

#include <iostream>

#include "CloudViewer.h"

#include <ecvHObjectCaster.h>

using namespace cloudViewer;

void PrintHelp() {
    CVLib::utility::LogInfo("Usage :");
    CVLib::utility::LogInfo("    > ecvHalfEdgeMesh <file>");
}

void ColorizeBoundaryVertices(geometry::ecvHalfEdgeMesh &halfMesh, const Eigen::Vector3d &color) {
    std::vector<Eigen::Vector3d> vertextColors(halfMesh.vertices_.size(), {0.75, 0.75, 0.75});
    std::vector<std::vector<int>> boundaries = halfMesh.getBoundaries();
    for (std::size_t i = 0; i < boundaries.size(); ++i) {
        for (std::size_t j = 0; j < boundaries[i].size(); ++j) {
            vertextColors[static_cast<std::size_t>(boundaries[i][j])] = color;
        }
    }
    halfMesh.vertex_colors_ = vertextColors;
}

void DrawGeometriesWithBackFace(const std::vector<std::shared_ptr<const ccHObject>> &geometry_ptrs)
{
    visualization::DrawGeometries(geometry_ptrs, "CloudViewer", 640, 480, 50, 50, false, false, true);
}

int main(int argc, char *argv[]) {

    CVLib::utility::SetVerbosityLevel(CVLib::utility::VerbosityLevel::Debug);

    if (argc < 2) {
        PrintHelp();
        return 1;
    }

    auto mesh = io::CreateMeshFromFile(argv[1]);
    ccBBox bbox = ccBBox(CCVector3(-1.f, -1.f, -1.f), CCVector3(1.f, 0.6f, 1.f));
    auto croppedMesh = mesh->crop(bbox);
    auto halfEdgeMesh = geometry::ecvHalfEdgeMesh::CreateFromTriangleMesh(*croppedMesh);
    DrawGeometriesWithBackFace({halfEdgeMesh});

    ColorizeBoundaryVertices(*halfEdgeMesh, {1, 0, 0});
    DrawGeometriesWithBackFace({halfEdgeMesh});
    return 0;
}

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <ecvMesh.h>

#include "cloudViewer/io/TriangleMeshIO.h"
#include "tests/Tests.h"

namespace cloudViewer {
namespace tests {

TEST(FileSTL, WriteReadTriangleMeshFromSTL) {
    geometry::TriangleMesh tm_gt;
    tm_gt.vertices_ = {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    tm_gt.triangles_ = {{0, 1, 2}};
    tm_gt.ComputeVertexNormals();

    io::WriteTriangleMesh("tmp.stl", tm_gt);

    geometry::TriangleMesh tm_test;
    io::ReadTriangleMesh("tmp.stl", tm_test, false);

    ExpectEQ(tm_gt.vertices_, tm_test.vertices_);
    ExpectEQ(tm_gt.triangles_, tm_test.triangles_);
}

}  // namespace tests
}  // namespace cloudViewer

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <ecvMesh.h>

#include "io/TriangleMeshIO.h"
#include "tests/Tests.h"

namespace cloudViewer {
namespace tests {

TEST(FileGLTF, WriteReadTriangleMeshFromGLTF) {
    geometry::TriangleMesh tm_gt;
    tm_gt.vertices_ = {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    tm_gt.triangles_ = {{0, 1, 2}};
    tm_gt.ComputeVertexNormals();

    io::WriteTriangleMesh("tmp.gltf", tm_gt);

    geometry::TriangleMesh tm_test;
    io::ReadTriangleMesh("tmp.gltf", tm_test, false);

    ExpectEQ(tm_gt.vertices_, tm_test.vertices_);
    ExpectEQ(tm_gt.triangles_, tm_test.triangles_);
    ExpectEQ(tm_gt.vertex_normals_, tm_test.vertex_normals_);
}

// NOTE: Temporarily disabled because of a mismatch between GLB export
// (TinyGLTF) and GLB import (ASSIMP)
// TEST(FileGLTF, WriteReadTriangleMeshFromGLB) {
//     geometry::TriangleMesh tm_gt;
//     tm_gt.vertices_ = {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}};
//     tm_gt.triangles_ = {{0, 1, 2}};
//     tm_gt.ComputeVertexNormals();

//     io::WriteTriangleMesh("tmp.glb", tm_gt);

//     geometry::TriangleMesh tm_test;
//     io::ReadTriangleMesh("tmp.glb", tm_test, false);

//     ExpectEQ(tm_gt.vertices_, tm_test.vertices_);
//     ExpectEQ(tm_gt.triangles_, tm_test.triangles_);
//     ExpectEQ(tm_gt.vertex_normals_, tm_test.vertex_normals_);
// }

}  // namespace tests
}  // namespace cloudViewer

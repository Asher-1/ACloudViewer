// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "t/geometry/TriangleMesh.h"

#include "core/CoreTest.h"
#include "core/TensorList.h"
#include "tests/UnitTest.h"

namespace cloudViewer {
namespace tests {

class TriangleMeshPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TriangleMesh,
                         TriangleMeshPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class TriangleMeshPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        TriangleMesh,
        TriangleMeshPermuteDevicePairs,
        testing::ValuesIn(TriangleMeshPermuteDevicePairs::TestCases()));

TEST_P(TriangleMeshPermuteDevices, DefaultConstructor) {
    t::geometry::TriangleMesh mesh;

    // Inherited from Geometry3D.
    EXPECT_EQ(mesh.GetGeometryType(),
              t::geometry::Geometry::GeometryType::TriangleMesh);
    EXPECT_EQ(mesh.Dimension(), 3);

    // Public members.
    EXPECT_TRUE(mesh.IsEmpty());
    EXPECT_FALSE(mesh.HasVertices());
    EXPECT_FALSE(mesh.HasVertexColors());
    EXPECT_FALSE(mesh.HasVertexNormals());
    EXPECT_FALSE(mesh.HasTriangles());
    EXPECT_FALSE(mesh.HasTriangleNormals());

    // Default dtypes.
    EXPECT_EQ(mesh.GetVertexPositions().GetDevice(), core::Device("CPU:0"));
    EXPECT_EQ(mesh.GetVertexPositions().GetDtype(), core::Dtype::Float32);
    EXPECT_EQ(mesh.GetTriangleIndices().GetDevice(), core::Device("CPU:0"));
    EXPECT_EQ(mesh.GetTriangleIndices().GetDtype(), core::Dtype::Int64);
}

TEST_P(TriangleMeshPermuteDevices, ConstructFromVertices) {
    core::Device device = GetParam();

    // Prepare data.
    core::Tensor vertices =
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device);
    core::Tensor single_vertex =
            core::Tensor::Ones({3}, core::Dtype::Float32, device);

    core::Tensor triangles =
            core::Tensor::Ones({10, 3}, core::Dtype::Int64, device);
    core::Tensor single_triangle =
            core::Tensor::Ones({3}, core::Dtype::Int64, device);

    // Copied, push back okay.
    t::geometry::TriangleMesh mesh(
            core::TensorList::FromTensor(vertices, /*inplace=*/false),
            core::TensorList::FromTensor(triangles, /*inplace=*/false));

    EXPECT_TRUE(mesh.HasVertices());
    EXPECT_EQ(mesh.GetVertexPositions().GetSize(), 10);
    mesh.GetVertexPositions().PushBack(single_vertex);
    EXPECT_EQ(mesh.GetVertexPositions().GetSize(), 11);

    EXPECT_TRUE(mesh.HasTriangles());
    EXPECT_EQ(mesh.GetTriangleIndices().GetSize(), 10);
    mesh.GetTriangleIndices().PushBack(single_triangle);
    EXPECT_EQ(mesh.GetTriangleIndices().GetSize(), 11);

    // Inplace tensorlist: cannot push_back.
    mesh = t::geometry::TriangleMesh(
            core::TensorList::FromTensor(vertices, /*inplace=*/true),
            core::TensorList::FromTensor(triangles, /*inplace=*/true));

    EXPECT_TRUE(mesh.HasVertices());
    EXPECT_EQ(mesh.GetVertexPositions().GetSize(), 10);
    EXPECT_ANY_THROW(mesh.GetVertexPositions().PushBack(single_vertex));

    EXPECT_TRUE(mesh.HasTriangles());
    EXPECT_EQ(mesh.GetTriangleIndices().GetSize(), 10);
    EXPECT_ANY_THROW(mesh.GetTriangleIndices().PushBack(single_triangle));
}

TEST_P(TriangleMeshPermuteDevices, SynchronizedPushBack) {
    core::Device device = GetParam();

    // Create trianglemesh.
    core::TensorList vertices = core::TensorList::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device));
    core::TensorList vertex_colors = core::TensorList::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device) * 0.5);
    core::TensorList triangles = core::TensorList::FromTensor(
            core::Tensor::Ones({5, 3}, core::Dtype::Int64, device));
    core::TensorList triangle_normals = core::TensorList::FromTensor(
            core::Tensor::Ones({5, 3}, core::Dtype::Float32, device) * 0.5);

    t::geometry::TriangleMesh mesh(vertices, triangles);
    mesh.SetVertexColors(vertex_colors);
    mesh.SetTriangleNormals(triangle_normals);

    // Exception cases are tested in TensorListMap::SynchronizedPushBack().
    std::unordered_map<std::string, core::Tensor> vertex_struct;
    EXPECT_EQ(mesh.GetVertexPositions().GetSize(), 10);
    EXPECT_EQ(mesh.GetVertexColors().GetSize(), 10);
    mesh.VertexSynchronizedPushBack({
            {"vertices", core::Tensor::Ones({3}, core::Dtype::Float32, device)},
            {"colors", core::Tensor::Ones({3}, core::Dtype::Float32, device)},
    });
    EXPECT_EQ(mesh.GetVertexPositions().GetSize(), 11);
    EXPECT_EQ(mesh.GetVertexColors().GetSize(), 11);

    std::unordered_map<std::string, core::Tensor> triangle_struct;
    EXPECT_EQ(mesh.GetTriangleIndices().GetSize(), 5);
    EXPECT_EQ(mesh.GetTriangleNormals().GetSize(), 5);
    mesh.TriangleSynchronizedPushBack({
            {"triangles", core::Tensor::Ones({3}, core::Dtype::Int64, device)},
            {"normals", core::Tensor::Ones({3}, core::Dtype::Float32, device)},
    });
    EXPECT_EQ(mesh.GetTriangleIndices().GetSize(), 6);
    EXPECT_EQ(mesh.GetTriangleNormals().GetSize(), 6);
}

TEST_P(TriangleMeshPermuteDevices, Getters) {
    core::Device device = GetParam();

    auto vertices = core::TensorList::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device));
    auto vertex_colors = core::TensorList::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 2);
    auto vertex_labels = core::TensorList::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3);

    auto triangles = core::TensorList::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Int64, device));
    auto triangle_normals = core::TensorList::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 2);
    auto triangle_labels = core::TensorList::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3);

    t::geometry::TriangleMesh mesh(vertices, triangles);
    mesh.SetVertexColors(vertex_colors);
    mesh.SetVertexAttr("labels", vertex_labels);
    mesh.SetTriangleNormals(triangle_normals);
    mesh.SetTriangleAttr("labels", triangle_labels);

    EXPECT_TRUE(mesh.GetVertexPositions().AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device)));
    EXPECT_TRUE(mesh.GetVertexColors().AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 2));
    EXPECT_TRUE(mesh.GetVertexAttr("labels").AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3));
    EXPECT_ANY_THROW(mesh.GetVertexNormals());

    EXPECT_TRUE(mesh.GetTriangleIndices().AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Int64, device)));
    EXPECT_TRUE(mesh.GetTriangleNormals().AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 2));
    EXPECT_TRUE(mesh.GetTriangleAttr("labels").AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3));

    // Const getters. (void)tl gets rid of the unused variables warning.
    EXPECT_NO_THROW(const core::TensorList& tl = mesh.GetVertexPositions();
                    (void)tl);
    EXPECT_NO_THROW(const core::TensorList& tl = mesh.GetVertexColors();
                    (void)tl);
    EXPECT_NO_THROW(const core::TensorList& tl = mesh.GetVertexAttr("labels");
                    (void)tl);
    EXPECT_ANY_THROW(const core::TensorList& tl = mesh.GetVertexNormals();
                     (void)tl);

    EXPECT_NO_THROW(const core::TensorList& tl = mesh.GetTriangleIndices();
                    (void)tl);
    EXPECT_NO_THROW(const core::TensorList& tl = mesh.GetTriangleNormals();
                    (void)tl);
    EXPECT_NO_THROW(const core::TensorList& tl = mesh.GetTriangleAttr("labels");
                    (void)tl);
}

TEST_P(TriangleMeshPermuteDevices, Setters) {
    core::Device device = GetParam();

    // Setters are already tested in Getters' unit tests. Here we test that
    // mismatched device should throw an exception. This test is only effective
    // is device is a CUDA device.
    t::geometry::TriangleMesh mesh(core::Dtype::Float32, core::Dtype::Float64,
                                   device);
    core::Device cpu_device = core::Device("CPU:0");
    if (cpu_device != device) {
        core::TensorList cpu_vertices = core::TensorList::FromTensor(
                core::Tensor::Ones({2, 3}, core::Dtype::Float32, cpu_device));
        core::TensorList cpu_colors = core::TensorList::FromTensor(
                core::Tensor::Ones({2, 3}, core::Dtype::Float32, cpu_device) *
                2);
        core::TensorList cpu_labels = core::TensorList::FromTensor(
                core::Tensor::Ones({2, 3}, core::Dtype::Float32, cpu_device) *
                3);

        EXPECT_ANY_THROW(mesh.SetVertices(cpu_vertices));
        EXPECT_ANY_THROW(mesh.SetVertexColors(cpu_colors));
        EXPECT_ANY_THROW(mesh.SetVertexAttr("labels", cpu_labels));
    }
}

TEST_P(TriangleMeshPermuteDevices, Has) {
    core::Device device = GetParam();

    t::geometry::TriangleMesh mesh(core::Dtype::Float32, core::Dtype::Float64,
                                   device);
    EXPECT_FALSE(mesh.HasVertices());
    EXPECT_FALSE(mesh.HasVertexColors());
    EXPECT_FALSE(mesh.HasVertexNormals());
    EXPECT_FALSE(mesh.HasVertexAttr("labels"));
    EXPECT_FALSE(mesh.HasTriangles());
    EXPECT_FALSE(mesh.HasTriangleNormals());
    EXPECT_FALSE(mesh.HasTriangleAttr("labels"));

    mesh.SetVertices(core::TensorList::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device)));
    EXPECT_TRUE(mesh.HasVertices());
    mesh.SetTriangles(core::TensorList::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Int64, device)));
    EXPECT_TRUE(mesh.HasTriangles());

    // Different size.
    mesh.SetVertexColors(core::TensorList::FromTensor(
            core::Tensor::Ones({5, 3}, core::Dtype::Float32, device)));
    EXPECT_FALSE(mesh.HasVertexColors());
    mesh.SetTriangleNormals(core::TensorList::FromTensor(
            core::Tensor::Ones({5, 3}, core::Dtype::Float32, device)));
    EXPECT_FALSE(mesh.HasTriangleNormals());

    // Same size.
    mesh.SetVertexColors(core::TensorList::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device)));
    EXPECT_TRUE(mesh.HasVertexColors());
    mesh.SetTriangleNormals(core::TensorList::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device)));
    EXPECT_TRUE(mesh.HasTriangleNormals());
}

}  // namespace tests
}  // namespace cloudViewer

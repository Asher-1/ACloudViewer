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

#include "t/geometry/TriangleMesh.h"

#include <Eigen/Core>
#include <string>
#include <unordered_map>

#include "core/EigenConverter.h"
#include "core/ShapeUtil.h"
#include "core/Tensor.h"
#include "core/TensorList.h"

namespace cloudViewer {
namespace t {
namespace geometry {

TriangleMesh::TriangleMesh(const core::Device &device)
    : Geometry(Geometry::GeometryType::TriangleMesh, 3),
      device_(device),
      vertex_attr_(TensorMap("vertices")),
      triangle_attr_(TensorMap("triangles")) {
}

TriangleMesh::TriangleMesh(const core::Tensor &vertices,
                           const core::Tensor &triangles)
    : TriangleMesh([&]() {
          if (vertices.GetDevice() != triangles.GetDevice()) {
              CVLib::utility::LogError(
                      "vertices' device {} does not match triangles' device "
                      "{}.",
                      vertices.GetDevice().ToString(),
                      triangles.GetDevice().ToString());
          }
          return vertices.GetDevice();
      }()) {
    SetVertices(vertices);
    SetTriangles(triangles);
}

geometry::TriangleMesh TriangleMesh::FromLegacyTriangleMesh(
        const ccMesh &mesh_legacy,
        core::Dtype float_dtype,
        core::Dtype int_dtype,
        const core::Device &device) {
    if (float_dtype != core::Dtype::Float32 &&
        float_dtype != core::Dtype::Float64) {
        CVLib::utility::LogError("float_dtype must be Float32 or Float64, but got {}.",
                          float_dtype.ToString());
    }
    if (int_dtype != core::Dtype::Int32 && int_dtype != core::Dtype::Int64) {
        CVLib::utility::LogError("int_dtype must be Int32 or Int64, but got {}.",
                          int_dtype.ToString());
    }

    TriangleMesh mesh(device);
    if (mesh_legacy.hasVertices()) {
        mesh.SetVertices(core::eigen_converter::EigenVector3dVectorToTensor(
                mesh_legacy.getEigenVertices(), float_dtype, device));
    } else {
        CVLib::utility::LogWarning("Creating from empty legacy TriangleMesh.");
    }
    if (mesh_legacy.hasColors()) {
        mesh.SetVertexColors(core::eigen_converter::EigenVector3dVectorToTensor(
                mesh_legacy.getVertexColors(), float_dtype, device));
    }
    if (mesh_legacy.hasNormals()) {
        mesh.SetVertexNormals(
                core::eigen_converter::EigenVector3dVectorToTensor(
                        mesh_legacy.getVertexNormals(), float_dtype, device));
    }
    if (mesh_legacy.hasTriangles()) {
        mesh.SetTriangles(core::eigen_converter::EigenVector3iVectorToTensor(
                mesh_legacy.getTriangles(), int_dtype, device));
    }
    if (mesh_legacy.hasTriNormals()) {
        mesh.SetTriangleNormals(
                core::eigen_converter::EigenVector3dVectorToTensor(
                        mesh_legacy.getTriangleNorms(), float_dtype, device));
    }
    return mesh;
}

ccMesh TriangleMesh::ToLegacyTriangleMesh() const {
    ccMesh mesh_legacy;
    mesh_legacy.createInternalCloud();
    if (mesh_legacy.reserveAssociatedCloud(1, HasVertexColors(), HasVertexNormals()))
    {
        CVLib::utility::LogError("[TriangleMesh::ToLegacyTriangleMesh] not enough memory!");
    }

    if (HasVertices()) {
        mesh_legacy.addEigenVertices(
                    core::eigen_converter::TensorToEigenVector3dVector(GetVertices()));
    }
    if (HasVertexColors()) {
        mesh_legacy.addVertexColors(
                    core::eigen_converter::TensorToEigenVector3dVector(GetVertexColors()));
    }
    if (HasVertexNormals()) {
        mesh_legacy.addVertexNormals(
                    core::eigen_converter::TensorToEigenVector3dVector(GetVertexNormals()));
    }
    if (HasTriangles()) {
        mesh_legacy.addTriangles(
                    core::eigen_converter::TensorToEigenVector3iVector(GetTriangles()));
    }
    if (HasTriangleNormals()) {
        mesh_legacy.addTriangleNorms(
                    core::eigen_converter::TensorToEigenVector3dVector(GetTriangleNormals()));
    }

    return mesh_legacy;
}

TriangleMesh TriangleMesh::To(const core::Device &device, bool copy) const {
    if (!copy && GetDevice() == device) {
        return *this;
    }
    TriangleMesh mesh(device);
    for (const auto &kv : triangle_attr_) {
        mesh.SetTriangleAttr(kv.first, kv.second.To(device, /*copy=*/true));
    }
    for (const auto &kv : vertex_attr_) {
        mesh.SetVertexAttr(kv.first, kv.second.To(device, /*copy=*/true));
    }
    return mesh;
}

}  // namespace geometry
}  // namespace t
}  // namespace cloudViewer

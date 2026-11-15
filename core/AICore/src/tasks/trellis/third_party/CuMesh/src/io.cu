// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cumesh.h"

namespace cumesh {

void CuMesh::init(const float* host_vertices,
                  size_t num_vertices,
                  const int* host_faces,
                  size_t num_faces) {
    this->vertices.resize(num_vertices);
    this->faces.resize(num_faces);
    // float3 / int3 are three tightly packed scalars (12 bytes, no padding),
    // so the packed host arrays upload with one contiguous copy each.
    static_assert(sizeof(float3) == 3 * sizeof(float), "float3 must be packed");
    static_assert(sizeof(int3) == 3 * sizeof(int), "int3 must be packed");
    if (num_vertices > 0) {
        CUDA_CHECK(cudaMemcpy(this->vertices.ptr, host_vertices,
                              num_vertices * sizeof(float3),
                              cudaMemcpyHostToDevice));
    }
    if (num_faces > 0) {
        CUDA_CHECK(cudaMemcpy(this->faces.ptr, host_faces,
                              num_faces * sizeof(int3),
                              cudaMemcpyHostToDevice));
    }
}

HostMesh CuMesh::read() {
    HostMesh mesh;
    mesh.vertices.resize(this->vertices.size * 3);
    mesh.faces.resize(this->faces.size * 3);
    if (!mesh.vertices.empty()) {
        CUDA_CHECK(cudaMemcpy(mesh.vertices.data(), this->vertices.ptr,
                              mesh.vertices.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    if (!mesh.faces.empty()) {
        CUDA_CHECK(cudaMemcpy(mesh.faces.data(), this->faces.ptr,
                              mesh.faces.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }
    return mesh;
}

}  // namespace cumesh

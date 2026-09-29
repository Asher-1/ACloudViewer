// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "mesh_uv.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#include "xatlas.h"

namespace sam3d {
namespace {

bool valid_mesh(const NativeMesh& mesh) {
    const size_t vertex_count = mesh.positions.size() / 3;
    return !mesh.positions.empty() && mesh.positions.size() % 3 == 0 &&
           !mesh.indices.empty() && mesh.indices.size() % 3 == 0 &&
           (mesh.normals.empty() ||
            mesh.normals.size() == mesh.positions.size()) &&
           (mesh.colors.empty() ||
            mesh.colors.size() == mesh.positions.size()) &&
           (mesh.vertex_attributes.empty() ||
            mesh.vertex_attributes.size() == vertex_count * 6);
}

void remap_vec3(const std::vector<float>& source,
                const xatlas::Mesh& atlas_mesh,
                std::vector<float>& destination) {
    if (source.empty()) return;
    destination.resize(static_cast<size_t>(atlas_mesh.vertexCount) * 3);
    for (uint32_t output = 0; output < atlas_mesh.vertexCount; ++output) {
        const size_t input =
                static_cast<size_t>(atlas_mesh.vertexArray[output].xref) * 3;
        std::memcpy(destination.data() + static_cast<size_t>(output) * 3,
                    source.data() + input, 3 * sizeof(float));
    }
}

void remap_vertex_attributes(const std::vector<float>& source,
                             const xatlas::Mesh& atlas_mesh,
                             std::vector<float>& destination) {
    if (source.empty()) return;
    constexpr size_t kChannels = 6;
    destination.resize(static_cast<size_t>(atlas_mesh.vertexCount) * kChannels);
    for (uint32_t output = 0; output < atlas_mesh.vertexCount; ++output) {
        const size_t input =
                static_cast<size_t>(atlas_mesh.vertexArray[output].xref) *
                kChannels;
        std::memcpy(
                destination.data() + static_cast<size_t>(output) * kChannels,
                source.data() + input, kChannels * sizeof(float));
    }
}

}  // namespace

bool parameterize_mesh_xatlas(NativeMesh& mesh, std::string& error) {
    if (!valid_mesh(mesh)) {
        error = "xatlas requires non-empty triangle positions and valid "
                "optional vertex attributes";
        return false;
    }
    const size_t vertex_count = mesh.positions.size() / 3;
    for (uint32_t index : mesh.indices) {
        if (index >= vertex_count) {
            error = "xatlas input contains an out-of-range triangle index";
            return false;
        }
    }
    xatlas::Atlas* atlas = xatlas::Create();
    if (atlas == nullptr) {
        error = "xatlas failed to create an atlas";
        return false;
    }
    xatlas::MeshDecl declaration{};
    declaration.vertexCount = static_cast<uint32_t>(vertex_count);
    declaration.vertexPositionData = mesh.positions.data();
    declaration.vertexPositionStride = 3 * sizeof(float);
    declaration.indexCount = static_cast<uint32_t>(mesh.indices.size());
    declaration.indexData = mesh.indices.data();
    declaration.indexFormat = xatlas::IndexFormat::UInt32;
    if (xatlas::AddMesh(atlas, declaration) != xatlas::AddMeshError::Success) {
        xatlas::Destroy(atlas);
        error = "xatlas AddMesh failed";
        return false;
    }
    // The Python binding's parametrize() uses xatlas's default chart and pack
    // settings. Do the same rather than importing unrelated TRELLIS tuning.
    xatlas::Generate(atlas);
    if (atlas->meshCount != 1 || atlas->atlasCount != 1 || atlas->width == 0 ||
        atlas->height == 0) {
        xatlas::Destroy(atlas);
        error = "xatlas did not produce one non-empty UV atlas";
        return false;
    }
    const xatlas::Mesh& atlas_mesh = atlas->meshes[0];
    if (atlas_mesh.vertexCount == 0 || atlas_mesh.indexCount == 0) {
        xatlas::Destroy(atlas);
        error = "xatlas produced an empty mesh";
        return false;
    }
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> colors;
    std::vector<float> vertex_attributes;
    remap_vec3(mesh.positions, atlas_mesh, positions);
    remap_vec3(mesh.normals, atlas_mesh, normals);
    remap_vec3(mesh.colors, atlas_mesh, colors);
    remap_vertex_attributes(mesh.vertex_attributes, atlas_mesh,
                            vertex_attributes);
    std::vector<float> texcoords(static_cast<size_t>(atlas_mesh.vertexCount) *
                                 2);
    for (uint32_t output = 0; output < atlas_mesh.vertexCount; ++output) {
        // The xatlas Python binding normalizes with direct division. Do not
        // precompute a reciprocal: multiplication changes thousands of F32
        // UVs by one ULP and those values seed the texture optimizer.
        texcoords[output * 2 + 0] = atlas_mesh.vertexArray[output].uv[0] /
                                    static_cast<float>(atlas->width);
        texcoords[output * 2 + 1] = atlas_mesh.vertexArray[output].uv[1] /
                                    static_cast<float>(atlas->height);
    }
    std::vector<uint32_t> indices(
            atlas_mesh.indexArray,
            atlas_mesh.indexArray + atlas_mesh.indexCount);
    xatlas::Destroy(atlas);
    mesh.positions = std::move(positions);
    mesh.normals = std::move(normals);
    mesh.colors = std::move(colors);
    mesh.vertex_attributes = std::move(vertex_attributes);
    mesh.texcoords = std::move(texcoords);
    mesh.indices = std::move(indices);
    return true;
}

}  // namespace sam3d

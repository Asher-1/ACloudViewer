// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "types.h"

namespace colmap {

struct PlyPoint {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float nx = 0.0f;
    float ny = 0.0f;
    float nz = 0.0f;
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
};

struct PlyMeshVertex {
    PlyMeshVertex() : x(0), y(0), z(0) {}
    PlyMeshVertex(const float x, const float y, const float z)
        : x(x), y(y), z(z) {}
    PlyMeshVertex(const float x,
                  const float y,
                  const float z,
                  const uint8_t r,
                  const uint8_t g,
                  const uint8_t b)
        : x(x), y(y), z(z), r(r), g(g), b(b) {}

    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    uint8_t r = 200;
    uint8_t g = 200;
    uint8_t b = 200;
    // PoissonRecon sampling density of the vertex. Meaningful only when the
    // owning PlyMesh sets has_vertex_value (the file declared the property).
    float value = 0.0f;
};

struct PlyMeshFace {
    PlyMeshFace() : vertex_idx1(0), vertex_idx2(0), vertex_idx3(0) {}
    PlyMeshFace(const size_t vertex_idx1,
                const size_t vertex_idx2,
                const size_t vertex_idx3)
        : vertex_idx1(vertex_idx1),
          vertex_idx2(vertex_idx2),
          vertex_idx3(vertex_idx3) {}

    size_t vertex_idx1 = 0;
    size_t vertex_idx2 = 0;
    size_t vertex_idx3 = 0;
};

struct PlyMesh {
    std::vector<PlyMeshVertex> vertices;
    std::vector<PlyMeshFace> faces;
    // Presence flags for optional vertex properties, decided by the PLY
    // header on read and honored by the writers on write. Mirrors the
    // upstream meshed-poisson.ply contract where the solver emits
    // x/y/z/value/r/g/b when run with --density and color interpolation.
    bool has_vertex_colors = false;
    bool has_vertex_value = false;
};

struct PlyTexturedMesh {
    PlyMesh mesh;
    // Six values per face: u1, v1, u2, v2, u3, v3.
    std::vector<float> face_uvs;
    // Referenced by the standard PLY "comment TextureFile" header field.
    std::string texture_file;
};

// Read PLY point cloud from text or binary file.
std::vector<PlyPoint> ReadPly(const std::filesystem::path& path);

// Write PLY point cloud to text or binary file.
void WriteTextPlyPoints(const std::filesystem::path& path,
                        const std::vector<PlyPoint>& points,
                        const bool write_normal = true,
                        const bool write_rgb = true);
void WriteBinaryPlyPoints(const std::filesystem::path& path,
                          const std::vector<PlyPoint>& points,
                          const bool write_normal = true,
                          const bool write_rgb = true);

// Read PLY mesh (plain or textured) from text or binary file. The presence
// flags of optional vertex properties (colors, value) follow the file header.
PlyTexturedMesh ReadPlyMesh(const std::filesystem::path& path);

// Returns true if the PLY file contains face elements (i.e., is a mesh).
bool HasPlyMeshFaces(const std::filesystem::path& path);

// Write PLY mesh to text or binary file. Optional vertex properties are
// emitted iff the corresponding PlyMesh presence flag is set: x/y/z, then
// value, then red/green/blue - the upstream PoissonRecon property order.
void WriteTextPlyMesh(const std::filesystem::path& path, const PlyMesh& mesh);
void WriteBinaryPlyMesh(const std::filesystem::path& path, const PlyMesh& mesh);
void WriteTextPlyMesh(const std::filesystem::path& path,
                      const PlyTexturedMesh& mesh);
void WriteBinaryPlyMesh(const std::filesystem::path& path,
                        const PlyTexturedMesh& mesh);

}  // namespace colmap

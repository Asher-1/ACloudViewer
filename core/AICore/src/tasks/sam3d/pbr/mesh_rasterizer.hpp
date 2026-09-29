// Native triangle rasterization used by mesh cleanup and texture baking.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "asset_io.hpp"

namespace sam3d {

// OpenCV world-to-camera extrinsics and normalized OpenCV intrinsics are used
// by the texture-baking path. The visibility cleanup path uses the official
// OpenGL `view_look_at` and `perspective_from_fov_xy` matrices directly, which
// avoids changing its near/far or coordinate convention during conversion.
struct MeshRasterCamera {
    std::array<float, 16> extrinsics{};
    std::array<float, 9> intrinsics{};
    std::array<float, 16> view{};
    std::array<float, 16> projection{};
    bool has_view_projection = false;
};

struct MeshRasterConfig {
    int width = 1024;
    int height = 1024;
    float near_plane = 0.1f;
    float far_plane = 10.0f;
};

// Pixels use a bottom-left origin to match the UV tensors supplied to
// nvdiffrast by the official optimized texture bake. `face_ids` are zero
// based; `-1` denotes no covered face.
struct MeshRasterResult {
    std::vector<float> uv;          // width * height * 2
    std::vector<uint32_t> face_ids; // width * height, UINT32_MAX if empty
    std::vector<uint8_t> coverage;  // width * height, 0 or 1
};

// Exact sphere-Hammersley camera construction used by the official mesh
// visibility pass. The default `view_count=1000` is intentionally explicit at
// its callers because this is a meaningful quality/cost contract.
std::vector<MeshRasterCamera> make_mesh_hammersley_cameras(
    int view_count, const MeshRasterConfig& config, std::string& error);

// Rasterize UVs with OpenGL-compatible view/projection transforms, a depth
// buffer, and perspective-correct interpolation. The mesh must contain one UV
// pair per vertex.
bool rasterize_mesh_uv(const NativeMesh& mesh, const MeshRasterCamera& camera,
                       const MeshRasterConfig& config, MeshRasterResult& result,
                       std::string& error);

// Rasterize the official Hammersley views and count each face once per view.
// The return values are frequencies in [0, 1], in mesh-index order.
bool mesh_visibility_frequency(const NativeMesh& mesh, int view_count,
                               const MeshRasterConfig& config,
                               std::vector<float>& frequency, std::string& error);

// Same visibility contract for explicitly supplied OpenCV cameras. This is
// used by the parity harness to separate camera construction from rasterizer
// semantics, and by callers that persist a deterministic camera trajectory.
bool mesh_visibility_frequency(const NativeMesh& mesh,
                               const std::vector<MeshRasterCamera>& cameras,
                               const MeshRasterConfig& config,
                               std::vector<float>& frequency, std::string& error);

}  // namespace sam3d

// mesh_to_dual_grid.h — C++ wrapper for o-voxel flexible dual grid (QEF).
// Vendored core: third_party/o-voxel-fdg/flexible_dual_grid.cpp (MIT, Microsoft TRELLIS.2)
//
// Output matches shape-encoder input:
//   enc_vert  = dual_vertex * grid_size - voxel_index  (per-axis offset in voxel units)
//   enc_inter = intersected flags (3 bools per voxel)
#pragma once

#include <cstdint>
#include <vector>

namespace mtdg {

struct Result {
    std::vector<int32_t> coords;       // N*3 voxel indices in [0, grid_size)
    std::vector<float>   dual_verts;   // N*3 world-space dual vertices (aabb-relative)
    std::vector<uint8_t> intersected;  // N*3 edge intersection flags
    int grid_size = 0;
};

/// Mesh (verts N*3, tris M*3) -> sparse dual grid at resolution R inside [-0.5,0.5]^3.
/// Weights match dump_texture_reference.py defaults.
Result mesh_to_flexible_dual_grid(
    const float * verts, int n_verts,
    const int32_t * tris, int n_tris,
    int grid_size,
    float face_weight = 1.f,
    float boundary_weight = 0.2f,
    float regularization_weight = 1e-2f);

} // namespace mtdg

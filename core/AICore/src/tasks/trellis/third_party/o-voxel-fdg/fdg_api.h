#pragma once
#include <cstdint>
#include <tuple>
#include <vector>

struct FdgVoxelGrid {
    std::vector<int32_t> coords;       // N*3
    std::vector<float>   dual_verts;   // N*3
    std::vector<uint8_t> intersected;  // N*3 (0/1)
};

/// Pure C++ mesh -> flexible dual grid (no PyTorch).
FdgVoxelGrid mesh_to_flexible_dual_grid_native(
    const float * verts, int n_verts,
    const int32_t * faces, int n_faces,
    int grid_size,
    float face_weight,
    float boundary_weight,
    float regularization_weight,
    bool timing = false);

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "mesh_to_dual_grid.h"

#include "third_party/o-voxel-fdg/fdg_api.h"

namespace mtdg {

Result mesh_to_flexible_dual_grid(const float* verts,
                                  int n_verts,
                                  const int32_t* tris,
                                  int n_tris,
                                  int grid_size,
                                  float face_weight,
                                  float boundary_weight,
                                  float regularization_weight) {
    Result r;
    r.grid_size = grid_size;
    if (!verts || !tris || n_verts <= 0 || n_tris <= 0 || grid_size <= 0)
        return r;

    // o-voxel expects mesh in [0,1]^3; Trellis meshes live in [-0.5,0.5]^3.
    std::vector<float> qef_verts((size_t)n_verts * 3);
    for (int i = 0; i < n_verts * 3; ++i)
        qef_verts[(size_t)i] = verts[i] + 0.5f;

    FdgVoxelGrid g = mesh_to_flexible_dual_grid_native(
            qef_verts.data(), n_verts, tris, n_tris, grid_size, face_weight,
            boundary_weight, regularization_weight, false);
    r.coords = std::move(g.coords);
    r.dual_verts = std::move(g.dual_verts);
    r.intersected = std::move(g.intersected);
    return r;
}

}  // namespace mtdg

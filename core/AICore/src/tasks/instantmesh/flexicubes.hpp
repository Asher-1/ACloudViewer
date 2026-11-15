// FlexiCubes isosurface mesh extraction (host C++, no weights).
#pragma once

#include <cstdint>
#include <vector>

namespace instantmesh {

// Result of extracting a triangle mesh from an SDF field via FlexiCubes.
struct FlexiMesh {
    std::vector<float>   vertices; // [Nv, 3] dual-vertex positions
    std::vector<int32_t> faces;    // [Nf, 3] triangle vertex indices

    bool empty() const { return faces.empty(); }
};

// Extracts a triangle mesh using the FlexiCubes (Dual Marching Cubes) scheme.
// This is the inference path only: grad_func = nullptr, training = false,
// output_tetmesh = false (matches InstantMesh's `get_mesh`).
//
//   x_nx3        [N,   3]  (deformed) voxel-grid vertex positions
//   s_n          [N]       signed distance field (negative = inside)
//   cube_fx8     [C,   8]  corner vertex index of each cube
//   beta_fx12    [C,  12]  per-cube edge weights     (raw network output)
//   alpha_fx8    [C,   8]  per-cube corner weights   (raw network output)
//   gamma_f      [C]       per-cube quad-split weight (raw network output)
//   res                  grid resolution (cubes per axis)
void flexicubes_extract(const float * x_nx3, const float * s_n,
                        const int32_t * cube_fx8, int C,
                        const float * beta_fx12, const float * alpha_fx8,
                        const float * gamma_f, int res,
                        FlexiMesh & out);

} // namespace instantmesh
// Native implementation of SAM 3D Objects' mesh-to-textured-GLB postprocess.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "asset_io.hpp"
#include "gaussian_renderer.hpp"

namespace sam3d {

struct NativePbrPipelineConfig {
    // The official process_mesh() sequence: VTK decimation, 1,000-view
    // visibility/mincut, then pymeshfix.TMesh boundary repair.
    bool input_mesh_already_clean = false;
    int visibility_views = 1000;
    int visibility_resolution = 1024;
    int render_views = 100;
    int render_resolution = 1024;
    int texture_size = 1024;
    int texture_steps = 2500;
    uint32_t random_seed = 0;
};

struct NativePbrPipelineResult {
    NativeMesh mesh;  // Final Y-up mesh with glTF UVs (V flipped after baking).
    size_t rendered_views = 0;
    size_t pre_meshfix_faces = 0;
    size_t final_faces = 0;
};

// Mirrors trimesh's angle-weighted vertex normal calculation in decoder-space
// Z-up coordinates. Invalid or degenerate faces and unreferenced vertices
// contribute a zero normal. Kept separate for direct numeric regression tests.
void compute_official_angle_weighted_vertex_normals(NativeMesh& mesh);

// Runs the official native CUDA postprocess. `source_mesh` must use the
// decoder's Z-up coordinate system. The result is ready for write_pbr_glb().
// Complete cleanup deliberately requires SAM3D_GGML_MESHFIX_GPL=ON because
// the same pymeshfix TMesh code used by the official implementation is
// GPL-3.0-or-commercial. No approximate repair is substituted.
bool assemble_official_pbr_cuda(const NativeMesh& source_mesh,
                                const GaussianSplatSet& splats,
                                const NativePbrPipelineConfig& config,
                                NativePbrPipelineResult& result,
                                std::string& error);

}  // namespace sam3d

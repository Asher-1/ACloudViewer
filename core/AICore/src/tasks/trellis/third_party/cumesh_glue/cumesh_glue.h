#pragma once
//
// cumesh_glue — thin native C wrapper around JeffreyXiang/CuMesh (MIT).
//
// Exposes GPU-accelerated chart clustering (compute_charts) and supporting
// mesh-cleanup routines without requiring PyTorch types in the caller. The
// cu file in this directory compiles the CuMesh CUDA sources and links
// libtorch internally; the header stays torch-free.
//
// Allocated output buffers must be freed via cumesh_free_buffer().

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ── Memory ──────────────────────────────────────────────────────────
void cumesh_free_buffer(void * ptr);

// ── Chart clustering (CuMesh compute_charts) ────────────────────────
// Runs the upstream pipeline:
//   1. CuMesh.init(verts, faces) on GPU
//   2. CuMesh.remove_degenerate_faces()
//   3. CuMesh.get_manifold_face_adjacency()
//   4. CuMesh.compute_charts(threshold_cone_half_angle_rad, ...)
//   5. Copies chart data back to CPU; writes face_chart_ids (nt ints)
//
// threshold_cone_half_angle_rad — cone merging threshold (upstream default ~1.57 rad / 90°)
// refine_iterations / global_iterations / smooth_strength — clustering params (0,1,1 typical)
//
// Returns 1 on success, 0 on failure. On success the caller owns the
// output buffers and must free them via cumesh_free_buffer().
int cumesh_compute_charts(
    const float * verts, int nv,
    const int *   tris,  int nt,
    float threshold_cone_half_angle_rad,
    int   refine_iterations,
    int   global_iterations,
    float smooth_strength,
    float area_penalty_weight,
    float perimeter_area_ratio_weight,
    // Outputs (caller frees via cumesh_free_buffer):
    int ** out_face_chart_ids,     // [nt] chart id per face
    int ** out_chart_vertex_map,   // [nv_in] maps chart vertex → input vertex
    int ** out_chart_faces,        // [nt_out * 3] chart-local face indices
    int ** out_chart_faces_offset, // [n_charts + 1]
    int ** out_chart_vertex_offset, // [n_charts + 1]
    int *  out_n_charts,
    int *  out_nt_out
);

// ── Simplified mesh cleanup (CuMesh remove_degenerate_faces) ────────
// Removes degenerate faces and unreferenced vertices on the GPU.
// Returns updated verts/tris through output pointers.
int cumesh_clean_mesh(
    const float * verts, int nv,
    const int *   tris,  int nt,
    float ** out_verts, int * out_nv,
    int **   out_tris,  int * out_nt
);

#ifdef __cplusplus
}
#endif
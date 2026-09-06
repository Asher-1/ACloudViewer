// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
//
// mesh_export — CUDA-free port of o_voxel.postprocess.to_glb: take the demo's
// dense per-vertex-PBR mesh and produce a portable glTF 2.0 binary (GLB). The
// reference does this on the GPU (CuMesh simplify/unwrap/BVH, nvdiffrast
// raster, flex_gemm grid_sample); here every stage is pure C++:
//
//   preserve topology -> optional decimation -> xatlas UV unwrap -> PBR atlas
//   bake
//   -> standard glTF baseColorTexture + metallicRoughnessTexture GLB.
//   Set T2GLB_VERTEX to opt into the legacy per-vertex COLOR_0 export instead.
//
// A practical fixed-size atlas cannot give millions of preserved triangles
// enough texels, and the dual-grid geometry is heavily non-manifold. glTF
// vertex colour is therefore both more faithful and substantially smaller. Set
// T2GLB_XATLAS to opt into a conventional image atlas on clean meshes. No ggml
// / CUDA dependency: plain float/int arrays.

#include <cstdint>
#include <string>
#include <vector>

namespace t2glb {

enum class ComponentFilter {
    RemoveTiny = 0,   // preserve meaningful disconnected parts
    KeepLargest = 1,  // retain only the component with the most triangles
    KeepAll = 2       // input is already prepared; do not filter again
};

// UV-unwrap backend. Was the T2GLB_XATLAS / T2GLB_NOCUMESH environment
// variable pair upstream:
//   Auto    — CuMesh GPU chart clustering when compiled in (not built
//             in-tree), else the chartless simple_unwrap fallback
//   XAtlas  — legacy pure-xatlas charting (debug / regression path)
//   Simple  — chartless 6-bin projection unwrap (robust on non-manifold
//             dual-grid geometry, seconds-fast)
enum class UnwrapMode { Auto = 0, XAtlas = 1, Simple = 2 };

/** Bake progress sink: fired at each bake stage boundary with the stage
 *  description and the seconds elapsed since that stage started. */
typedef void (*t2glb_bake_progress_fn)(const char* stage, double elapsed_s,
                                       void* user);

struct MeshExportOptions {
    int texture_size = 2048;  // atlas width/height (T2GLB_TEXTURE_SIZE env)
    int padding = 2;          // xatlas chart padding (texels)
    int dilate = 6;           // gutter dilation passes (kills UV seams)
    // Upstream default (500k; same as mesh2glb). NOTE: the sloppy fallback
    // this target triggers scrambles vertex normals, so cone/cumesh chart
    // clustering degrades towards per-triangle fragments on dense dual-grid
    // meshes — raise the target (above the ~1.1-1.4M plain-simplify cap) to
    // keep sloppy decimation out of the chain when atlas quality matters
    // more than file size.
    int decimation_target = 500000;  // max triangles before atlas export
    ComponentFilter components = ComponentFilter::RemoveTiny;
    UnwrapMode unwrap = UnwrapMode::Auto;
    /** Stage-boundary progress sink + cooperative cancel flag
     *  (polled with relaxed semantics at the same boundaries;
     *  both null by default = silent, uncancellable bake). */
    t2glb_bake_progress_fn progress = nullptr;
    void* progress_user = nullptr;
    const volatile int* cancel = nullptr;
    // meshopt_simplify error limit; 2e-1 lets a 3.7M-tri mesh reach a 281K
    // target without stalling into the sloppy path (was T2GLB_DECIMATION_ERROR
    // upstream).
    float decimation_error = 2e-1f;
    // Sliver filter for sloppy decimation: drop triangles whose max edge is
    // more than `sliver_aspect` times the altitude (was T2GLB_SLIVER_ASPECT).
    float sliver_aspect = 100.0f;
    // Snap the decoder's near-transparent activation noise (alpha 0.03-0.5
    // on solid surface) to fully opaque (was T2GLB_ALPHA_CLEAN, default on).
    bool alpha_clean = true;
    // Regularise the geometry with a Taubin smooth before export (opt-in;
    // off by default). Was the T2GLB_XATLAS/T2GLB_NOSMOOTH env pair upstream.
    bool taubin_smooth = false;
    // Write the legacy vertex-colour GLB instead of the UV atlas (debug /
    // regression path). Was T2GLB_VERTEX upstream.
    bool vertex_pbr = false;
};

// Geometry/material streams after the same component filtering used by
// mesh_to_glb. Valid source triangles retain their original polygon density.
struct PreparedMesh {
    std::vector<float> verts;
    std::vector<float> normals;
    std::vector<int32_t> tris;
    std::vector<float> pbr;
};

bool prepare_mesh(const float* verts,
                  int nv,
                  const int32_t* tris,
                  int nt,
                  const float* pbr,
                  const MeshExportOptions& opt,
                  PreparedMesh& out,
                  std::string& err);

// Optional CPU print-remesh path backed by CGAL Alpha Wrap 3. The ratios are
// fractions of the component-filtered mesh's bounding-box diagonal. The result
// is guaranteed by Alpha Wrap to be closed, oriented, intersection-free and
// 2-manifold. Wrapping creates a new enclosing surface, so a textured source is
// sampled onto the wrap vertices (approximate per-vertex preview); the sharper
// per-texel rebake stays in mesh_to_projected_glb for the GLB download.
bool print_remesh_available();
bool prepare_print_mesh(const float* verts,
                        int nv,
                        const int32_t* tris,
                        int nt,
                        const float* pbr,
                        const MeshExportOptions& opt,
                        float alpha_ratio,
                        float offset_ratio,
                        PreparedMesh& out,
                        std::string& err);

// Export a dense per-vertex-PBR mesh as a standard vertex-coloured GLB.
//
//   verts   3*nv  vertex positions (mesh/world space, as fdg::extract emits)
//   tris    3*nt  triangle vertex indices
//   pbr     6*nv  base_color rgb, metallic, roughness, alpha
//                 (null -> untextured grey)
//
// On success fills `out` with the GLB bytes and returns true. On failure
// returns false with a message in `err`. Not reentrant (Simplify.h uses global
// state): serialized internally by a mutex.
bool mesh_to_glb(const float* verts,
                 int nv,
                 const int32_t* tris,
                 int nt,
                 const float* pbr,
                 const MeshExportOptions& opt,
                 std::vector<uint8_t>& out,
                 std::string& err);

// UV-unwrap `target` and bake its atlas by projecting each covered texel onto
// the closest triangle of the dense PBR `source`.  Intended for assigning the
// generated material to Alpha Wrap geometry; always uses xatlas and the CGAL
// CPU closest-surface backend regardless of T2GLB_XATLAS.
bool mesh_to_projected_glb(const float* target_verts,
                           int target_nv,
                           const int32_t* target_tris,
                           int target_nt,
                           const float* source_verts,
                           int source_nv,
                           const int32_t* source_tris,
                           int source_nt,
                           const float* source_pbr,
                           const MeshExportOptions& opt,
                           std::vector<uint8_t>& out,
                           std::string& err);

}  // namespace t2glb

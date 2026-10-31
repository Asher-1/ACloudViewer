// UV unwrapping + texture-map rasterization for the ggml InstantMesh port.
//
// Reference: src/models/geometry/rep_3d/extract_texture_map.py (xatlas_uvmap)
// and src/models/lrm_mesh.py::extract_mesh(use_texture_map=True). The official
// pipeline uses the xatlas library for UV parametrization and nvdiffrast for
// rasterization; this C++ module provides self-contained equivalents so the
// whole textured export runs without Python:
//
//   uv_unwrap(...)       per-vertex UV atlas in [0,1]^2
//                        (chart-cluster + per-chart planar projection + packing)
//   rasterize_texture()  per-pixel world-space position + coverage mask
//                        (barycentric software rasterizer over UV space)
//
// The baked texture is then obtained by pushing each covered pixel's world
// position through net_rgb (see synthesizer_texture_forward) — identical color
// source as the official get_texture_prediction, so the texture *content*
// matches; only the UV layout (xatlas vs. planar charts) may differ.
//
// Coordinate convention: UV v=0 is the bottom row, v=1 the top; texture rows
// with py=0 map to v=0. PNG output flips rows so row 0 (top) = v=1, matching
// the OBJ+MTL loader expectation and the reference save_obj_with_mtl.
#ifndef INSTANTMESH_MODELS_TEXTURE_MAP_HPP
#define INSTANTMESH_MODELS_TEXTURE_MAP_HPP

#include <cstdint>
#include <vector>

namespace instantmesh {

// Unwraps a triangle mesh into a 2D UV atlas in [0,1]^2.
//
//   verts: [nv*3] float32 world positions
//   faces: [nf*3] int32 triangle vertex indices
//   atlas_res: atlas resolution in pixels (packing grid; UVs stay [0,1])
//   angle_threshold_deg: max dihedral angle (deg) between adjacent faces that
//                        are kept in the same chart (default ~75)
//
//   out_uvs:   [3*nf*2] float32, per-corner UV in [0,1]^2 (corner 3f+k is the
//              k-th vertex of face f)
//   out_charts: (optional) number of generated charts
//
// UVs are written per face-corner: a mesh vertex shared by multiple charts is
// duplicated, each copy keeping the UV of its own chart. This corner splitting
// is required for correct texture mapping across chart seams. Use split_mesh()
// to build a per-corner (duplicated-vertex) mesh from these UVs.
// Returns 0 on success.
int uv_unwrap(const float * verts, const int32_t * faces, int nv, int nf,
              int atlas_res, std::vector<float> & out_uvs,
              int * out_charts, float angle_threshold_deg = 75.0f);

// Splits a mesh so each face corner becomes its own vertex carrying its own UV
// and face normal. Vertices on chart seams are duplicated. The result has
// 3*nf vertices and nf faces (face f uses corners 3f, 3f+1, 3f+2).
//
//   verts: [nv*3]; faces: [nf*3]; corner_uvs: [3*nf*2] (from uv_unwrap)
//   out_v: [3*nf*3] per-corner positions
//   out_uv: [3*nf*2] per-corner UVs
//   out_norm: [3*nf*3] per-corner face normals
//   out_f: [3*nf] face index buffer (identity corner ids)
void split_mesh(const float * verts, const int32_t * faces,
                const float * corner_uvs, int nf,
                std::vector<float> & out_v, std::vector<float> & out_uv,
                std::vector<float> & out_norm, std::vector<int32_t> & out_f);

// Rasterizes the UV-mapped mesh into an atlas_res x atlas_res grid, producing
// a per-pixel world-space position (barycentrically interpolated) and a
// coverage mask. Mirrors xatlas_uvmap + nvdiffrast in the reference.
//
//   verts: [nv*3]; faces: [nf*3]; uvs: [nv*2] (from uv_unwrap)
//   out_gb_pos: [atlas_res*atlas_res*3] float32 world position at the pixel
//               center; pixels with mask==0 are left as-is (background).
//   out_gb4:    [atlas_res*atlas_res*3*4] float32 2x2 supersampled world
//               positions (4 subsamples per covered texel) for anti-aliased
//               net_rgb baking.
//   out_mask:   [atlas_res*atlas_res] uint8 (1 = covered by geometry)
// Pass the SPLIT mesh (split_mesh output) so that faces crossing chart seams
// use their own corner UVs; otherwise the rasterized world positions are wrong.
void rasterize_texture(const float * verts, const int32_t * faces,
                       const float * uvs, int nv, int nf, int atlas_res,
                       std::vector<float> & out_gb_pos,
                       std::vector<float> & out_gb4,
                       std::vector<uint8_t> & out_mask);

} // namespace instantmesh

#endif // INSTANTMESH_MODELS_TEXTURE_MAP_HPP

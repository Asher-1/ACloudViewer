// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// UV unwrapping + texture-map rasterization (ggml InstantMesh port).
#include "texture_map.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>
#include <utility>

#include "../trellis/third_party/xatlas/xatlas.h"

namespace instantmesh {

namespace {

inline void cross3(const float* a, const float* b, float* c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}
inline float dot3(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline void sub3(const float* a, const float* b, float* c) {
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];
}
inline void norm3(float* a) {
    float n = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    if (n > 1e-12f) {
        a[0] /= n;
        a[1] /= n;
        a[2] /= n;
    }
}

}  // namespace

// UV unwrapping implemented with xatlas — the same parametrize/chart/pack
// library the reference InstantMesh pipeline uses (see
// src/models/geometry/rep_3d/extract_texture_map.py::xatlas_uvmap). This
// produces a proper non-overlapping UV atlas (per-corner UVs, vertices split
// across chart seams), which the naive planar-projection + shelf-packing
// approach failed to guarantee (it produced heavily overlapping charts).
//
// Output per-corner UVs: corner (f, k) maps to out_uvs[2*(3f+k)..+1], the
// k-th vertex of face f. v=0 is the bottom row, v=1 the top.
int uv_unwrap(const float* verts,
              const int32_t* faces,
              int nv,
              int nf,
              int atlas_res,
              std::vector<float>& out_uvs,
              int* out_charts,
              float angle_threshold_deg) {
    (void)angle_threshold_deg;
    out_uvs.assign((size_t)3 * nf * 2, 0.0f);

    xatlas::Atlas* atlas = xatlas::Create();
    if (!atlas) return -1;

    xatlas::MeshDecl md;
    md.vertexPositionData = verts;
    md.vertexPositionStride = sizeof(float) * 3;
    md.vertexCount = (uint32_t)nv;
    md.indexData = faces;
    md.indexFormat = xatlas::IndexFormat::UInt32;
    md.indexCount = (uint32_t)nf * 3;
    md.faceCount = (uint32_t)nf;
    if (xatlas::AddMesh(atlas, md) != xatlas::AddMeshError::Success) {
        xatlas::Destroy(atlas);
        return -1;
    }

    xatlas::ChartOptions copt;
    xatlas::PackOptions popt;
    popt.resolution = (uint32_t)atlas_res;
    popt.padding = 2;      // pixel gutter between charts
    popt.bilinear = true;  // leave a margin for bilinear filtering
    xatlas::Generate(atlas, copt, popt);

    const xatlas::Mesh& mesh = atlas->meshes[0];
    const float iw = (atlas->width > 0) ? 1.0f / (float)atlas->width : 0.0f;
    const float ih = (atlas->height > 0) ? 1.0f / (float)atlas->height : 0.0f;
    for (size_t c = 0; c < (size_t)mesh.indexCount; ++c) {
        const xatlas::Vertex& vtx = mesh.vertexArray[mesh.indexArray[c]];
        float u = vtx.uv[0] * iw;
        float v = vtx.uv[1] * ih;
        // xatlas uv v increases downward (image space); flip so v=0 is bottom.
        v = 1.0f - v;
        out_uvs[2 * c + 0] = u;
        out_uvs[2 * c + 1] = v;
    }
    if (out_charts) *out_charts = (int)mesh.chartCount;

    xatlas::Destroy(atlas);
    return 0;
}

// Split the mesh so each face corner becomes its own vertex carrying its own
// UV (and a face normal). Vertices on chart seams are duplicated, giving each
// chart a clean copy of the shared vertices. The returned mesh has 3*nf
// vertices and nf faces (face f uses corners 3f, 3f+1, 3f+2).
//
//   verts: [nv*3]; faces: [nf*3]; corner_uvs: [3*nf*2] (from uv_unwrap)
//   out_v: [3*nf*3] per-corner positions
//   out_uv: [3*nf*2] per-corner UVs
//   out_norm: [3*nf*3] per-corner face normals
//   out_f: [3*nf] face index buffer (identity: corner ids)
void split_mesh(const float* verts,
                const int32_t* faces,
                const float* corner_uvs,
                int nf,
                std::vector<float>& out_v,
                std::vector<float>& out_uv,
                std::vector<float>& out_norm,
                std::vector<int32_t>& out_f) {
    const size_t nc = (size_t)3 * nf;
    out_v.assign(nc * 3, 0.0f);
    out_uv.assign(nc * 2, 0.0f);
    out_norm.assign(nc * 3, 0.0f);
    out_f.assign(nc, 0);
    for (int f = 0; f < nf; ++f) {
        const float* v0 = verts + 3 * faces[3 * f + 0];
        const float* v1 = verts + 3 * faces[3 * f + 1];
        const float* v2 = verts + 3 * faces[3 * f + 2];
        float e1[3], e2[3], n[3];
        sub3(v1, v0, e1);
        sub3(v2, v0, e2);
        cross3(e1, e2, n);
        norm3(n);
        for (int k = 0; k < 3; ++k) {
            size_t c = (size_t)3 * f + k;
            int v = faces[3 * f + k];
            out_v[3 * c + 0] = verts[3 * v + 0];
            out_v[3 * c + 1] = verts[3 * v + 1];
            out_v[3 * c + 2] = verts[3 * v + 2];
            out_uv[2 * c + 0] = corner_uvs[2 * c + 0];
            out_uv[2 * c + 1] = corner_uvs[2 * c + 1];
            out_norm[3 * c + 0] = n[0];
            out_norm[3 * c + 1] = n[1];
            out_norm[3 * c + 2] = n[2];
            out_f[3 * f + k] = (int32_t)c;
        }
    }
}

void rasterize_texture(const float* verts,
                       const int32_t* faces,
                       const float* uvs,
                       int nv,
                       int nf,
                       int atlas_res,
                       std::vector<float>& out_gb_pos,
                       std::vector<float>& out_gb4,
                       std::vector<uint8_t>& out_mask) {
    out_gb_pos.assign((size_t)atlas_res * atlas_res * 3, 0.0f);
    out_gb4.assign((size_t)atlas_res * atlas_res * 3 * 4, 0.0f);
    out_mask.assign((size_t)atlas_res * atlas_res, 0);

    // 2x2 subsample offsets within a texel (pixel-space), used to anti-alias
    // the per-texel point-sample of net_rgb (single point samples alias the
    // high-frequency appearance into the "snowflake" speckle).
    static const float SO[4][2] = {
            {0.25f, 0.25f}, {0.75f, 0.25f}, {0.25f, 0.75f}, {0.75f, 0.75f}};

    for (int f = 0; f < nf; ++f) {
        int i0 = faces[3 * f + 0], i1 = faces[3 * f + 1], i2 = faces[3 * f + 2];
        // Pixel-space triangle (u*res, v*res); py increases upward (v up).
        float x0 = uvs[2 * i0 + 0] * atlas_res,
              y0 = uvs[2 * i0 + 1] * atlas_res;
        float x1 = uvs[2 * i1 + 0] * atlas_res,
              y1 = uvs[2 * i1 + 1] * atlas_res;
        float x2 = uvs[2 * i2 + 0] * atlas_res,
              y2 = uvs[2 * i2 + 1] * atlas_res;

        // Bounding box in pixel space.
        float minx = std::max(0.0f, std::floor(std::min({x0, x1, x2})));
        float maxx = std::min((float)atlas_res - 1,
                              std::ceil(std::max({x0, x1, x2})));
        float miny = std::max(0.0f, std::floor(std::min({y0, y1, y2})));
        float maxy = std::min((float)atlas_res - 1,
                              std::ceil(std::max({y0, y1, y2})));

        const float *p0 = verts + 3 * i0, *p1 = verts + 3 * i1,
                    *p2 = verts + 3 * i2;
        const float denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
        if (std::fabs(denom) < 1e-12f) continue;

        for (int py = (int)miny; py <= (int)maxy; ++py) {
            for (int px = (int)minx; px <= (int)maxx; ++px) {
                size_t pix = (size_t)py * atlas_res + px;
                if (out_mask[pix]) continue;
                float cx = px + 0.5f, cy = py + 0.5f;

                // Barycentric coordinates of the pixel center (cx, cy).
                float l0 =
                        ((y1 - y2) * (cx - x2) + (x2 - x1) * (cy - y2)) / denom;
                float l1 =
                        ((y2 - y0) * (cx - x2) + (x0 - x2) * (cy - y2)) / denom;
                float l2 = 1.0f - l0 - l1;
                if (l0 < -1e-4f || l1 < -1e-4f || l2 < -1e-4f) continue;

                // Interpolate world position at the pixel center.
                float* gb = &out_gb_pos[pix * 3];
                gb[0] = l0 * p0[0] + l1 * p1[0] + l2 * p2[0];
                gb[1] = l0 * p0[1] + l1 * p1[1] + l2 * p2[1];
                gb[2] = l0 * p0[2] + l1 * p1[2] + l2 * p2[2];
                out_mask[pix] = 1;

                // 2x2 supersamples: world position at each subsample; fall back
                // to the center position when a subsample leaves the triangle
                // (edge texels), so anti-aliasing never introduces gaps.
                float* gb4 = &out_gb4[pix * 12];
                for (int s = 0; s < 4; ++s) {
                    float sx = (float)px + SO[s][0];
                    float sy = (float)py + SO[s][1];
                    float a0 = ((y1 - y2) * (sx - x2) + (x2 - x1) * (sy - y2)) /
                               denom;
                    float a1 = ((y2 - y0) * (sx - x2) + (x0 - x2) * (sy - y2)) /
                               denom;
                    float a2 = 1.0f - a0 - a1;
                    if (a0 < -1e-4f || a1 < -1e-4f || a2 < -1e-4f) {
                        gb4[3 * s + 0] = gb[0];
                        gb4[3 * s + 1] = gb[1];
                        gb4[3 * s + 2] = gb[2];
                    } else {
                        gb4[3 * s + 0] = a0 * p0[0] + a1 * p1[0] + a2 * p2[0];
                        gb4[3 * s + 1] = a0 * p0[1] + a1 * p1[1] + a2 * p2[1];
                        gb4[3 * s + 2] = a0 * p0[2] + a1 * p1[2] + a2 * p2[2];
                    }
                }
            }
        }
    }
}

}  // namespace instantmesh

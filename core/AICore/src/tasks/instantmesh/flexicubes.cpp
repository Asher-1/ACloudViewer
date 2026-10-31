// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// FlexiCubes isosurface mesh extraction (host C++, inference path).
// Mirrors the no-grad / non-training branch of
// src/models/geometry/rep_3d/flexicubes.py:
//   _identify_surf_cubes -> _normalize_weights -> _get_case_id ->
//   _identify_surf_edges -> _compute_vd -> _triangulate
#include "flexicubes.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "flexicubes_tables.hpp"

namespace {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Linear interpolation of the zero crossing: given per-endpoint weights w (<0
// inside) and positions x, returns the point where the interpolated field is 0.
// Matches FlexiCubes._linear_interp: ue = (x1*w1 - x0*w0) / (w1 - w0).
inline float zero_interp(const float* x0, const float* x1, float w0, float w1) {
    float denom = w1 - w0;
    return (denom == 0.0f) ? 0.5f * (x0[0] + x1[0])
                           : (x1[0] * w1 - x0[0] * w0) / denom;
}

// A sorted-dedup unique over a list of integer pairs, replicating
// torch.unique(return_inverse=True) ordering (lexicographic by (a,b)).
struct UniquePairs {
    std::vector<int32_t> unique;   // flattened [n,2]
    std::vector<int32_t> inverse;  // [in_size] -> index into unique
    std::vector<int32_t> counts;   // [n] occurrence count
};

UniquePairs unique_pairs(const std::vector<int32_t>& ab, size_t n) {
    std::vector<int32_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = (int32_t)i;
    std::sort(order.begin(), order.end(), [&](int32_t i, int32_t j) {
        int32_t ai = ab[2 * i], bi = ab[2 * i + 1], aj = ab[2 * j],
                bj = ab[2 * j + 1];
        return ai != aj ? ai < aj : bi < bj;
    });
    UniquePairs r;
    std::vector<int32_t> idx_of(n);
    for (size_t k = 0; k < n; ++k) {
        int32_t i = order[k], a = ab[2 * i], b = ab[2 * i + 1];
        if (r.unique.size() < 2 || r.unique[r.unique.size() - 2] != a ||
            r.unique[r.unique.size() - 1] != b) {
            r.unique.push_back(a);
            r.unique.push_back(b);
            r.counts.push_back(0);
        }
        idx_of[i] = (int32_t)(r.counts.size() - 1);
        r.counts.back()++;
    }
    r.inverse.resize(n);
    for (size_t i = 0; i < n; ++i) r.inverse[i] = idx_of[i];
    return r;
}

}  // namespace

namespace instantmesh {

void flexicubes_extract(const float* x_nx3,
                        const float* s_n,
                        const int32_t* cube_fx8,
                        int C,
                        const float* beta_fx12,
                        const float* alpha_fx8,
                        const float* gamma_f,
                        int res,
                        FlexiMesh& out) {
    out.vertices.clear();
    out.faces.clear();

    // ---- 1. identify surface cubes ----
    std::vector<int32_t> surf_cube_idx;  // original cube index
    std::vector<int> occ_fx8;            // [C,8]
    occ_fx8.reserve((size_t)C * 8);
    for (int c = 0; c < C; ++c) {
        int occ_sum = 0;
        for (int j = 0; j < 8; ++j) {
            int inside = s_n[cube_fx8[c * 8 + j]] < 0.0f ? 1 : 0;
            occ_fx8.push_back(inside);
            occ_sum += inside;
        }
        if (occ_sum > 0 && occ_sum < 8) surf_cube_idx.push_back(c);
    }
    const int NS = (int)surf_cube_idx.size();
    if (NS == 0) return;  // empty mesh

    // ---- 2. normalize weights (tanh/sigmoid + scale) for surf cubes ----
    // InstantMesh's FlexiCubesGeometry uses weight_scale=0.5 (not the default
    // 0.99), see src/models/geometry/rep_3d/flexicubes_geometry.py.
    std::vector<float> beta(NS * 12), alpha(NS * 8), gamma(NS);
    const float wscale = 0.5f;
    for (int s = 0; s < NS; ++s) {
        int c = surf_cube_idx[s];
        for (int k = 0; k < 12; ++k)
            beta[s * 12 + k] = std::tanh(beta_fx12[c * 12 + k]) * wscale + 1.0f;
        for (int k = 0; k < 8; ++k)
            alpha[s * 8 + k] = std::tanh(alpha_fx8[c * 8 + k]) * wscale + 1.0f;
        gamma[s] = sigmoid(gamma_f[c]) * wscale + (1.0f - wscale) * 0.5f;
    }

    // ---- 3. case ids with ambiguity resolution ----
    std::vector<int32_t> case_ids(NS);
    for (int s = 0; s < NS; ++s)
        for (int j = 0; j < 8; ++j)
            case_ids[s] |= occ_fx8[surf_cube_idx[s] * 8 + j] << j;

    {
        // problem_full[pos] = new case id if this grid cell is a "problem"
        // config, else -1.
        const int64_t V = (int64_t)res * res * res;
        std::vector<int32_t> problem_full(V, -1);
        std::vector<int32_t> problem_cube;  // surf index of problem cubes
        std::vector<int32_t> problem_pos;   // 3D flattened pos
        std::vector<int32_t> problem_new;   // destination case id
        for (int s = 0; s < NS; ++s) {
            const int32_t* pc = CHECK_TABLE[case_ids[s]];
            if (pc[0] == 1) {
                int c = surf_cube_idx[s];
                int x = c % res, y = (c / res) % res, z = c / (res * res);
                problem_cube.push_back(s);
                problem_pos.push_back(x + res * (y + res * z));
                problem_new.push_back(pc[4]);
            }
        }
        const int M = (int)problem_cube.size();
        for (int i = 0; i < M; ++i)
            problem_full[problem_pos[i]] = problem_new[i];
        for (int i = 0; i < M; ++i) {
            int c = surf_cube_idx[problem_cube[i]];
            int x = c % res, y = (c / res) % res, z = c / (res * res);
            int dx = CHECK_TABLE[case_ids[problem_cube[i]]][1];
            int dy = CHECK_TABLE[case_ids[problem_cube[i]]][2];
            int dz = CHECK_TABLE[case_ids[problem_cube[i]]][3];
            int ax = x + dx, ay = y + dy, az = z + dz;
            if (ax >= 0 && ax < res && ay >= 0 && ay < res && az >= 0 &&
                az < res) {
                int adj = ax + res * (ay + res * az);
                if (problem_full[adj] != -1)
                    case_ids[problem_cube[i]] = problem_new[i];
            }
        }
    }

    // ---- 4. identify surface edges ----
    // all_edges[c][e] = (cube_fx8[surf][cube_edges[2e]],
    // cube_fx8[surf][cube_edges[2e+1]])
    std::vector<int32_t> all_edges((size_t)NS * 12 * 2);
    for (int s = 0; s < NS; ++s) {
        int c = surf_cube_idx[s];
        const int32_t* corners = &cube_fx8[c * 8];
        for (int e = 0; e < 12; ++e) {
            all_edges[(s * 12 + e) * 2] = corners[CUBE_EDGES[2 * e]];
            all_edges[(s * 12 + e) * 2 + 1] = corners[CUBE_EDGES[2 * e + 1]];
        }
    }
    UniquePairs up = unique_pairs(all_edges, (size_t)NS * 12);
    const int NE = (int)up.unique.size() / 2;
    // surf edge mask per unique edge: exactly one endpoint inside
    std::vector<int32_t> surf_edge_id(NE, -1);
    int n_surf_edges = 0;
    for (int u = 0; u < NE; ++u) {
        int in = (s_n[up.unique[2 * u]] < 0) + (s_n[up.unique[2 * u + 1]] < 0);
        if (in == 1) surf_edge_id[u] = n_surf_edges++;
    }
    // idx_map: [NS*12], -1 or surf-edge index
    std::vector<int32_t> idx_map(NS * 12);
    std::vector<int32_t> surf_edges_mask(NS * 12);
    std::vector<int32_t> edge_counts(NS * 12);
    for (size_t i = 0; i < up.inverse.size(); ++i) {
        int u = up.inverse[i];
        idx_map[i] = surf_edge_id[u];
        surf_edges_mask[i] = (surf_edge_id[u] != -1) ? 1 : 0;
        edge_counts[i] = up.counts[u];
    }
    // surf_edges: [n_surf_edges, 2]
    std::vector<int32_t> surf_edges(n_surf_edges * 2);
    for (int u = 0; u < NE; ++u)
        if (surf_edge_id[u] != -1) {
            surf_edges[surf_edge_id[u] * 2] = up.unique[2 * u];
            surf_edges[surf_edge_id[u] * 2 + 1] = up.unique[2 * u + 1];
        }
    if (n_surf_edges == 0) return;

    // ---- 5. compute dual vertices ----
    // zero_crossing for each surf edge: [n_surf_edges, 3]
    std::vector<float> zero_crossing((size_t)n_surf_edges * 3);
    for (int i = 0; i < n_surf_edges; ++i) {
        int v0 = surf_edges[2 * i], v1 = surf_edges[2 * i + 1];
        float w0 = s_n[v0], w1 = s_n[v1];
        for (int d = 0; d < 3; ++d)
            zero_crossing[i * 3 + d] =
                    zero_interp(&x_nx3[v0 * 3], &x_nx3[v1 * 3], w0, w1);
    }
    // alpha_nx12x2 = alpha[:, cube_edges].reshape(C,12,2)
    // num_vd per cube
    std::vector<int32_t> num_vd(NS);
    for (int s = 0; s < NS; ++s) num_vd[s] = NUM_VD_TABLE[case_ids[s]];

    // iterate over distinct num_vd (ascending, like torch.unique)
    std::vector<int> distinct;
    for (int s = 0; s < NS; ++s)
        if (std::find(distinct.begin(), distinct.end(), num_vd[s]) ==
            distinct.end())
            distinct.push_back(num_vd[s]);
    std::sort(distinct.begin(), distinct.end());

    // accumulated scatter buffers: edge_group(cube-edge), edge_group_to_vd,
    // edge_group_to_cube
    std::vector<int32_t> eg_cube_edge, eg_vd, eg_cube;
    std::vector<float> vd_gamma;

    int total_num_vd = 0;
    for (int num : distinct) {
        // cubes with this num_vd
        std::vector<int> cubes_with;
        for (int s = 0; s < NS; ++s)
            if (num_vd[s] == num) cubes_with.push_back(s);
        int n_cw = (int)cubes_with.size();
        int curr_num_vd = n_cw * num;
        // curr_edge_group: [n_cw * num, 7]
        std::vector<int32_t> edge_group(n_cw * num * 7, -1);
        for (int i = 0; i < n_cw; ++i) {
            int s = cubes_with[i];
            for (int v = 0; v < num; ++v)
                for (int e = 0; e < 7; ++e)
                    edge_group[(i * num + v) * 7 + e] =
                            DMC_TABLE[case_ids[s]][v][e];
        }
        // curr_edge_group_to_vd: [n_cw*num, 7] =
        // arange(curr_num_vd).repeat(1,7) + total curr_edge_group_to_cube: [..]
        // = cube index within surf cubes
        for (int i = 0; i < n_cw; ++i) {
            int s = cubes_with[i];
            for (int v = 0; v < num; ++v) {
                int vd_base = (i * num + v) + total_num_vd;
                for (int e = 0; e < 7; ++e) {
                    int g = (i * num + v) * 7 + e;
                    if (edge_group[g] != -1) {
                        eg_cube_edge.push_back(s * 12 + edge_group[g]);
                        eg_vd.push_back(vd_base);
                        eg_cube.push_back(s);
                    }
                }
            }
        }
        // vd_num_edges per dual vert = count of valid edges
        // vd_gamma per dual vert = gamma[cube]
        for (int i = 0; i < n_cw; ++i) {
            int s = cubes_with[i];
            for (int v = 0; v < num; ++v) vd_gamma.push_back(gamma[s]);
        }
        total_num_vd += curr_num_vd;
    }
    const int TOT = total_num_vd;
    if (TOT == 0) return;

    // vd = sum(ue*beta)/sum(beta) ; ue = zero_interp(s*alpha, x)
    std::vector<float> vd(TOT * 3, 0.0f), beta_sum(TOT, 0.0f);
    const int K = (int)eg_cube_edge.size();
    for (int k = 0; k < K; ++k) {
        int ce = eg_cube_edge[k];  // surf cube index *12 + edge
        int s = ce / 12, e = ce % 12;
        int edge_idx = idx_map[s * 12 + e];  // surf-edge index
        int v0 = surf_edges[2 * edge_idx], v1 = surf_edges[2 * edge_idx + 1];
        // alpha at the two corners of this edge
        int a0 = CUBE_EDGES[2 * e], a1 = CUBE_EDGES[2 * e + 1];
        float w0 = s_n[v0] * alpha[s * 8 + a0];
        float w1 = s_n[v1] * alpha[s * 8 + a1];
        float denom = w1 - w0;
        float uex, uey, uez;
        if (denom == 0.0f) {
            uex = 0.5f * (x_nx3[v0 * 3] + x_nx3[v1 * 3]);
            uey = 0.5f * (x_nx3[v0 * 3 + 1] + x_nx3[v1 * 3 + 1]);
            uez = 0.5f * (x_nx3[v0 * 3 + 2] + x_nx3[v1 * 3 + 2]);
        }
        // FlexiCubes._linear_interp: ue = (x0*w1 - x1*w0)/(w1-w0)  (w0~v0,
        // w1~v1)
        else {
            uex = (x_nx3[v0 * 3] * w1 - x_nx3[v1 * 3] * w0) / denom;
            uey = (x_nx3[v0 * 3 + 1] * w1 - x_nx3[v1 * 3 + 1] * w0) / denom;
            uez = (x_nx3[v0 * 3 + 2] * w1 - x_nx3[v1 * 3 + 2] * w0) / denom;
        }
        float b = beta[s * 12 + e];
        int j = eg_vd[k];
        vd[j * 3] += uex * b;
        vd[j * 3 + 1] += uey * b;
        vd[j * 3 + 2] += uez * b;
        beta_sum[j] += b;
    }
    for (int j = 0; j < TOT; ++j) {
        vd[j * 3] /= beta_sum[j];
        vd[j * 3 + 1] /= beta_sum[j];
        vd[j * 3 + 2] /= beta_sum[j];
    }

    // vd_idx_map: [NS,12], scatter vd index by cube-edge
    std::vector<int32_t> vd_idx_map(NS * 12, 0);
    for (int k = 0; k < K; ++k) vd_idx_map[eg_cube_edge[k]] = eg_vd[k];

    // ---- 6. triangulate ----
    // group edges shared by exactly 4 cubes and that are surface edges
    std::vector<int32_t> group, vd_idx;
    for (int s = 0; s < NS; ++s) {
        for (int e = 0; e < 12; ++e) {
            int i = s * 12 + e;
            if (edge_counts[i] == 4 && surf_edges_mask[i]) {
                group.push_back(idx_map[i]);
                vd_idx.push_back(vd_idx_map[i]);
            }
        }
    }
    const int n_quad_entries = (int)group.size();
    if (n_quad_entries == 0) {
        out.vertices.swap(vd);
        return;
    }
    // stable sort by group value
    std::vector<int32_t> order(n_quad_entries);
    for (int i = 0; i < n_quad_entries; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](int32_t i, int32_t j) { return group[i] < group[j]; });
    std::vector<int32_t> edge_indices(n_quad_entries), quad_vd(n_quad_entries);
    for (int i = 0; i < n_quad_entries; ++i) {
        edge_indices[i] = group[order[i]];
        quad_vd[i] = vd_idx[order[i]];
    }
    const int n_quads = n_quad_entries / 4;
    std::vector<int32_t> quad_vd_idx((size_t)n_quads * 4);
    for (int q = 0; q < n_quads; ++q)
        for (int k = 0; k < 4; ++k) quad_vd_idx[q * 4 + k] = quad_vd[q * 4 + k];

    // winding: flip so face normals point toward positive SDF
    std::vector<int32_t> flip(n_quads, 0);
    for (int q = 0; q < n_quads; ++q) {
        int edge_idx = edge_indices[q * 4];  // surf-edge index (same for all 4)
        int v0 = surf_edges[2 * edge_idx];
        if (s_n[v0] > 0.0f) flip[q] = 1;
    }
    // build two lists (flipped / non-flipped) then concat, matching pytorch cat
    std::vector<int32_t> flip_list, nflip_list;
    for (int q = 0; q < n_quads; ++q) {
        if (flip[q]) {
            flip_list.push_back(quad_vd_idx[q * 4 + 0]);
            flip_list.push_back(quad_vd_idx[q * 4 + 1]);
            flip_list.push_back(quad_vd_idx[q * 4 + 3]);
            flip_list.push_back(quad_vd_idx[q * 4 + 2]);
        } else {
            nflip_list.push_back(quad_vd_idx[q * 4 + 2]);
            nflip_list.push_back(quad_vd_idx[q * 4 + 3]);
            nflip_list.push_back(quad_vd_idx[q * 4 + 1]);
            nflip_list.push_back(quad_vd_idx[q * 4 + 0]);
        }
    }
    flip_list.insert(flip_list.end(), nflip_list.begin(), nflip_list.end());
    std::copy(flip_list.begin(), flip_list.end(), quad_vd_idx.begin());

    // split quads into triangles by gamma
    std::vector<float> quad_gamma((size_t)n_quads * 4);
    for (int q = 0; q < n_quads; ++q)
        for (int k = 0; k < 4; ++k)
            quad_gamma[q * 4 + k] = vd_gamma[quad_vd_idx[q * 4 + k]];
    std::vector<int32_t> faces((size_t)n_quads * 6);
    for (int q = 0; q < n_quads; ++q) {
        float g02 = quad_gamma[q * 4 + 0] * quad_gamma[q * 4 + 2];
        float g13 = quad_gamma[q * 4 + 1] * quad_gamma[q * 4 + 3];
        const int32_t* split = (g02 > g13) ? QUAD_SPLIT_1 : QUAD_SPLIT_2;
        for (int k = 0; k < 6; ++k)
            faces[q * 6 + k] = quad_vd_idx[q * 4 + split[k]];
    }

    out.vertices.swap(vd);
    out.faces.swap(faces);
}

}  // namespace instantmesh
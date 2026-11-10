// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>

#include "cumesh_glue.h"

// ── CuMesh header only (no .cu includes — those are compiled separately
// as CUDA sources listed in CMakeLists.txt and linked at build time).
#include "third_party/CuMesh/src/cumesh.h"

// ── Local helpers ──────────────────────────────────────────────────
namespace {

// Copy a host vector into a malloc'd array the caller releases through
// cumesh_free_buffer(). Returns nullptr for an empty source or on failure.
template <typename T>
T* to_malloc_array(const std::vector<T>& src) {
    if (src.empty()) return nullptr;
    T* dst = static_cast<T*>(std::malloc(src.size() * sizeof(T)));
    if (dst) std::memcpy(dst, src.data(), src.size() * sizeof(T));
    return dst;
}

}  // namespace

// ── Public API ─────────────────────────────────────────────────────
void cumesh_free_buffer(void* ptr) { std::free(ptr); }

int cumesh_compute_charts(const float* verts,
                          int nv,
                          const int* tris,
                          int nt,
                          float threshold_cone_half_angle_rad,
                          int refine_iterations,
                          int global_iterations,
                          float smooth_strength,
                          float area_penalty_weight,
                          float perimeter_area_ratio_weight,
                          int** out_face_chart_ids,
                          int** out_chart_vertex_map,
                          int** out_chart_faces,
                          int** out_chart_faces_offset,
                          int** out_chart_vertex_offset,
                          int* out_n_charts,
                          int* out_nt_out) {
    if (!out_face_chart_ids || !out_chart_vertex_map || !out_chart_faces ||
        !out_chart_faces_offset || !out_chart_vertex_offset || !out_n_charts ||
        !out_nt_out) {
        return 0;
    }

    // Default outputs
    *out_face_chart_ids = nullptr;
    *out_chart_vertex_map = nullptr;
    *out_chart_faces = nullptr;
    *out_chart_faces_offset = nullptr;
    *out_chart_vertex_offset = nullptr;
    *out_n_charts = 0;
    *out_nt_out = 0;
    if (!verts || !tris || nv <= 0 || nt <= 0) return 0;

    try {
        // Every CuMesh kernel and copy runs on the legacy default stream
        // (cumesh::current_stream()), so no stream state is shared with the
        // caller and nothing needs restoring.
        // 1. Upload the host arrays and init CuMesh on the GPU
        cumesh::CuMesh mesh;
        mesh.init(verts, (size_t)nv, tris, (size_t)nt);

        // 2. Remove degenerate faces (default thresholds)
        mesh.remove_degenerate_faces(1e-4f, 1e-4f);

        // 3. Build manifold face adjacency (required before compute_charts)
        mesh.get_manifold_face_adjacency();

        // 4. Run chart clustering with upstream-compatible parameters
        mesh.compute_charts(threshold_cone_half_angle_rad, refine_iterations,
                            global_iterations, smooth_strength,
                            area_penalty_weight, perimeter_area_ratio_weight);

        // 5. Read chart data from the GPU (synchronous host copies)
        const cumesh::HostAtlasCharts charts = mesh.read_atlas_charts();
        const int num_charts = charts.num_charts;

        // Chart IDs are indexed by the cleaned, compacted faces; chart
        // vertices map to cleaned vertices and may repeat across boundaries;
        // chart faces index chart_vertex_map as a flat [nt_out * 3] array.
        const int nf = (int)charts.chart_ids.size();
        const int nf_chart = (int)(charts.chart_faces.size() / 3);
        if (num_charts <= 0 || nf <= 0) return 0;
        if (charts.chart_vertex_map.empty() || nf_chart <= 0) return 0;
        // Both offset arrays hold [num_charts + 1] entries.
        const size_t n_offsets = (size_t)num_charts + 1;
        if (charts.chart_vertex_offset.size() < n_offsets ||
            charts.chart_face_offset.size() < n_offsets) {
            return 0;
        }

        std::unique_ptr<int, decltype(&std::free)> face_chart_ids(
                to_malloc_array(charts.chart_ids), &std::free);
        std::unique_ptr<int, decltype(&std::free)> chart_vertex_map(
                to_malloc_array(charts.chart_vertex_map), &std::free);
        std::unique_ptr<int, decltype(&std::free)> chart_faces(
                to_malloc_array(charts.chart_faces), &std::free);
        std::unique_ptr<int, decltype(&std::free)> chart_vertex_offset(
                (int*)std::malloc(n_offsets * sizeof(int)), &std::free);
        std::unique_ptr<int, decltype(&std::free)> chart_faces_offset(
                (int*)std::malloc(n_offsets * sizeof(int)), &std::free);
        if (!face_chart_ids || !chart_vertex_map || !chart_faces ||
            !chart_vertex_offset || !chart_faces_offset) {
            return 0;
        }
        std::memcpy(chart_vertex_offset.get(),
                    charts.chart_vertex_offset.data(), n_offsets * sizeof(int));
        std::memcpy(chart_faces_offset.get(), charts.chart_face_offset.data(),
                    n_offsets * sizeof(int));

        *out_face_chart_ids = face_chart_ids.release();
        *out_chart_vertex_map = chart_vertex_map.release();
        *out_chart_faces = chart_faces.release();
        *out_chart_vertex_offset = chart_vertex_offset.release();
        *out_chart_faces_offset = chart_faces_offset.release();
        *out_n_charts = num_charts;
        *out_nt_out = nf_chart;
        return 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "cumesh_compute_charts failed: %s\n", e.what());
        return 0;
    } catch (...) {
        std::fprintf(stderr, "cumesh_compute_charts failed (unknown)\n");
        return 0;
    }
}

int cumesh_clean_mesh(const float* verts,
                      int nv,
                      const int* tris,
                      int nt,
                      float** out_verts,
                      int* out_nv,
                      int** out_tris,
                      int* out_nt) {
    if (!out_verts || !out_nv || !out_tris || !out_nt) return 0;

    *out_verts = nullptr;
    *out_nv = 0;
    *out_tris = nullptr;
    *out_nt = 0;
    if (!verts || !tris || nv <= 0 || nt <= 0) return 0;

    try {
        cumesh::CuMesh mesh;
        mesh.init(verts, (size_t)nv, tris, (size_t)nt);
        mesh.remove_degenerate_faces(1e-4f, 1e-4f);

        const cumesh::HostMesh cleaned = mesh.read();
        const int nvo = (int)(cleaned.vertices.size() / 3);
        const int nto = (int)(cleaned.faces.size() / 3);

        std::unique_ptr<float, decltype(&std::free)> host_verts(
                to_malloc_array(cleaned.vertices), &std::free);
        std::unique_ptr<int, decltype(&std::free)> host_tris(
                to_malloc_array(cleaned.faces), &std::free);
        if ((nvo > 0 && !host_verts) || (nto > 0 && !host_tris)) return 0;
        *out_verts = host_verts.release();
        *out_tris = host_tris.release();
        *out_nv = nvo;
        *out_nt = nto;
        return 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "cumesh_clean_mesh failed: %s\n", e.what());
        return 0;
    } catch (...) {
        std::fprintf(stderr, "cumesh_clean_mesh failed (unknown)\n");
        return 0;
    }
}

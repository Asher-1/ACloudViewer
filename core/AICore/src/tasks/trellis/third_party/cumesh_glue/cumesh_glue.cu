// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "cumesh_glue.h"

// ── CuMesh header only (no .cu includes — those are compiled separately
// as CUDA sources listed in CMakeLists.txt and linked at build time).
#include "third_party/CuMesh/src/cumesh.h"

// ── Local helpers ──────────────────────────────────────────────────
namespace {

// The C API receives host pointers, while CuMesh::init copies device memory.
// Upload before entering CuMesh; the temporary tensors own the GPU storage.
torch::Tensor wrap_positions(const float* v, int nv) {
    return torch::from_blob(const_cast<float*>(v), {(int64_t)nv, 3},
                            torch::TensorOptions().dtype(torch::kFloat32))
            .to(torch::kCUDA);
}
torch::Tensor wrap_faces(const int* f, int nt) {
    return torch::from_blob(const_cast<int*>(f), {(int64_t)nt, 3},
                            torch::TensorOptions().dtype(torch::kInt32))
            .to(torch::kCUDA);
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
        // CuMesh mixes default-stream kernels with current-stream work.
        // Keep both on the default stream and restore the caller's stream.
        const c10::cuda::CUDAStreamGuard stream_guard(
                c10::cuda::getDefaultCUDAStream());
        // 1. Wrap raw pointers as torch tensors and init CuMesh on GPU
        cumesh::CuMesh mesh;
        torch::Tensor tvert = wrap_positions(verts, nv);
        torch::Tensor tface = wrap_faces(tris, nt);
        mesh.init(tvert, tface);

        // 2. Remove degenerate faces (default thresholds)
        mesh.remove_degenerate_faces(1e-4f, 1e-4f);

        // 3. Build manifold face adjacency (required before compute_charts)
        mesh.get_manifold_face_adjacency();

        // 4. Run chart clustering with upstream-compatible parameters
        mesh.compute_charts(threshold_cone_half_angle_rad, refine_iterations,
                            global_iterations, smooth_strength,
                            area_penalty_weight, perimeter_area_ratio_weight);

        // 5. Read chart data from GPU
        auto [num_charts, chart_ids_t, chart_vmap_t, chart_faces_t,
              chart_vertex_off_t, chart_face_off_t] = mesh.read_atlas_charts();

        // Chart IDs are indexed by the cleaned, compacted faces.
        const int nf = (int)chart_ids_t.size(0);
        if (num_charts <= 0 || nf <= 0) return 0;
        std::unique_ptr<int, decltype(&std::free)> face_chart_ids(
                (int*)std::malloc((size_t)nf * sizeof(int)), &std::free);
        if (!face_chart_ids ||
            cudaMemcpy(face_chart_ids.get(), chart_ids_t.data_ptr<int>(),
                       (size_t)nf * sizeof(int),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            return 0;
        }

        // Chart vertices map to cleaned mesh vertices and may be duplicated
        // across chart boundaries.
        const int nv_in = (int)chart_vmap_t.size(0);
        if (nv_in <= 0) return 0;
        std::unique_ptr<int, decltype(&std::free)> chart_vertex_map(
                (int*)std::malloc((size_t)nv_in * sizeof(int)), &std::free);
        if (!chart_vertex_map ||
            cudaMemcpy(chart_vertex_map.get(), chart_vmap_t.data_ptr<int>(),
                       (size_t)nv_in * sizeof(int),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            return 0;
        }

        // Chart faces (indices into chart_vertex_map, flat [nt_out * 3])
        const int nf_chart = (int)chart_faces_t.size(0);
        if (nf_chart <= 0) return 0;
        const size_t chart_faces_bytes = (size_t)nf_chart * 3 * sizeof(int);
        std::unique_ptr<int, decltype(&std::free)> chart_faces(
                (int*)std::malloc(chart_faces_bytes), &std::free);
        if (!chart_faces ||
            cudaMemcpy(chart_faces.get(), chart_faces_t.data_ptr<int>(),
                       chart_faces_bytes,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            return 0;
        }

        // Chart vertex offset (per chart, [num_charts + 1])
        const size_t offsets_bytes = (size_t)(num_charts + 1) * sizeof(int);
        std::unique_ptr<int, decltype(&std::free)> chart_vertex_offset(
                (int*)std::malloc(offsets_bytes), &std::free);
        if (!chart_vertex_offset ||
            cudaMemcpy(chart_vertex_offset.get(),
                       chart_vertex_off_t.data_ptr<int>(), offsets_bytes,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            return 0;
        }

        // Chart face offset (per chart, [num_charts + 1])
        std::unique_ptr<int, decltype(&std::free)> chart_faces_offset(
                (int*)std::malloc(offsets_bytes), &std::free);
        if (!chart_faces_offset ||
            cudaMemcpy(chart_faces_offset.get(),
                       chart_face_off_t.data_ptr<int>(), offsets_bytes,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            return 0;
        }

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
        const c10::cuda::CUDAStreamGuard stream_guard(
                c10::cuda::getDefaultCUDAStream());
        cumesh::CuMesh mesh;
        mesh.init(wrap_positions(verts, nv), wrap_faces(tris, nt));
        mesh.remove_degenerate_faces(1e-4f, 1e-4f);

        auto [new_verts, new_faces] = mesh.read();

        const int nvo = (int)new_verts.size(0);
        const int nto = (int)new_faces.size(0);

        if (nvo < 0 || nto < 0) return 0;
        const size_t verts_bytes = (size_t)nvo * 3 * sizeof(float);
        const size_t tris_bytes = (size_t)nto * 3 * sizeof(int);
        std::unique_ptr<float, decltype(&std::free)> host_verts(
                nvo > 0 ? (float*)std::malloc(verts_bytes) : nullptr,
                &std::free);
        std::unique_ptr<int, decltype(&std::free)> host_tris(
                nto > 0 ? (int*)std::malloc(tris_bytes) : nullptr, &std::free);
        if ((nvo > 0 && !host_verts) || (nto > 0 && !host_tris)) return 0;
        if ((verts_bytes > 0 &&
             cudaMemcpy(host_verts.get(), new_verts.data_ptr<float>(),
                        verts_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) ||
            (tris_bytes > 0 &&
             cudaMemcpy(host_tris.get(), new_faces.data_ptr<int>(), tris_bytes,
                        cudaMemcpyDeviceToHost) != cudaSuccess)) {
            return 0;
        }
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

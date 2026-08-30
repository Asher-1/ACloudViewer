// cumesh_glue.cu — wraps JeffreyXiang/CuMesh (MIT) behind a torch-free C API.
//
// This single translation unit is the only file in the project that
// includes <torch/extension.h>.  It compiles the CuMesh CUDA sources as
// part of itself (via #include) so every CuMesh method is available
// without linking a separate Python extension .so.

#include "cumesh_glue.h"

// ── CuMesh header only (no .cu includes — those are compiled separately
// as CUDA sources listed in CMakeLists.txt and linked at build time).
#include "third_party/CuMesh/src/cumesh.h"

// ── Local helpers ──────────────────────────────────────────────────
namespace {

// Wrap a CPU raw pointer as a torch tensor (zero-copy). The caller must
// keep the source data alive for the tensor's lifetime.
template <typename T>
torch::Tensor wrap(int n, const T * data) {
    auto opts = torch::TensorOptions()
        .dtype(torch::CppTypeToScalarType<T>())
        .device(torch::kCPU);
    return torch::from_blob(const_cast<T*>(data), {(int64_t) n}, opts);
}
torch::Tensor wrap_positions(const float * v, int nv) {
    return torch::from_blob(const_cast<float*>(v), {(int64_t) nv, 3},
                            torch::TensorOptions().dtype(torch::kFloat32));
}
torch::Tensor wrap_faces(const int * f, int nt) {
    return torch::from_blob(const_cast<int*>(f), {(int64_t) nt, 3},
                            torch::TensorOptions().dtype(torch::kInt32));
}

} // namespace

// ── Public API ─────────────────────────────────────────────────────
void cumesh_free_buffer(void * ptr) { std::free(ptr); }

int cumesh_compute_charts(
    const float * verts, int nv,
    const int *   tris,  int nt,
    float threshold_cone_half_angle_rad,
    int   refine_iterations,
    int   global_iterations,
    float smooth_strength,
    float area_penalty_weight,
    float perimeter_area_ratio_weight,
    int ** out_face_chart_ids,
    int ** out_chart_vertex_map,
    int ** out_chart_faces,
    int ** out_chart_faces_offset,
    int ** out_chart_vertex_offset,
    int *  out_n_charts,
    int *  out_nt_out)
{
    // Default outputs
    *out_face_chart_ids = nullptr;
    *out_chart_vertex_map = nullptr;
    *out_chart_faces = nullptr;
    *out_chart_faces_offset = nullptr;
    *out_chart_vertex_offset = nullptr;
    *out_n_charts = 0;
    *out_nt_out = 0;

    try {
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
        mesh.compute_charts(threshold_cone_half_angle_rad,
                            refine_iterations,
                            global_iterations,
                            smooth_strength,
                            area_penalty_weight,
                            perimeter_area_ratio_weight);

        // 5. Read chart data from GPU
        auto [num_charts, chart_ids_t, chart_vmap_t,
              chart_faces_t, chart_vertex_off_t, chart_face_off_t]
            = mesh.read_atlas_charts();

        *out_n_charts = num_charts;

        // Chart face IDs per input face (from read_atlas_charts: chart_ids_t is [nt])
        // chart_ids_t is a 1D tensor of int, one per input face
        const int nf = (int) chart_ids_t.size(0);
        *out_face_chart_ids = (int*) std::malloc((size_t) nf * sizeof(int));
        if (!*out_face_chart_ids) return 0;
        cudaMemcpy(*out_face_chart_ids, chart_ids_t.data_ptr<int>(),
                   (size_t) nf * sizeof(int), cudaMemcpyDeviceToHost);

        // Chart vertex map (per input vertex)
        const int nv_in = (int) chart_vmap_t.size(0);
        *out_chart_vertex_map = (int*) std::malloc((size_t) nv_in * sizeof(int));
        if (!*out_chart_vertex_map) return 0;
        cudaMemcpy(*out_chart_vertex_map, chart_vmap_t.data_ptr<int>(),
                   (size_t) nv_in * sizeof(int), cudaMemcpyDeviceToHost);

        // Chart faces (local indices per chart, flat [nt_out * 3])
        const int nf_chart = (int) chart_faces_t.size(0);
        *out_chart_faces = (int*) std::malloc((size_t) nf_chart * 3 * sizeof(int));
        if (!*out_chart_faces) return 0;
        cudaMemcpy(*out_chart_faces, chart_faces_t.data_ptr<int>(),
                   (size_t) nf_chart * 3 * sizeof(int), cudaMemcpyDeviceToHost);
        *out_nt_out = nf_chart;

        // Chart vertex offset (per chart, [num_charts + 1])
        *out_chart_vertex_offset = (int*) std::malloc((size_t)(num_charts + 1) * sizeof(int));
        if (!*out_chart_vertex_offset) return 0;
        cudaMemcpy(*out_chart_vertex_offset, chart_vertex_off_t.data_ptr<int>(),
                   (size_t)(num_charts + 1) * sizeof(int), cudaMemcpyDeviceToHost);

        // Chart face offset (per chart, [num_charts + 1])
        *out_chart_faces_offset = (int*) std::malloc((size_t)(num_charts + 1) * sizeof(int));
        if (!*out_chart_faces_offset) return 0;
        cudaMemcpy(*out_chart_faces_offset, chart_face_off_t.data_ptr<int>(),
                   (size_t)(num_charts + 1) * sizeof(int), cudaMemcpyDeviceToHost);

        return 1;

    } catch (const std::exception & e) {
        std::fprintf(stderr, "cumesh_compute_charts failed: %s\n", e.what());
        return 0;
    } catch (...) {
        std::fprintf(stderr, "cumesh_compute_charts failed (unknown)\n");
        return 0;
    }
}

int cumesh_clean_mesh(
    const float * verts, int nv,
    const int *   tris,  int nt,
    float ** out_verts, int * out_nv,
    int **   out_tris,  int * out_nt)
{
    *out_verts = nullptr; *out_nv = 0;
    *out_tris = nullptr;  *out_nt = 0;

    try {
        cumesh::CuMesh mesh;
        mesh.init(wrap_positions(verts, nv), wrap_faces(tris, nt));
        mesh.remove_degenerate_faces(1e-4f, 1e-4f);

        auto [new_verts, new_faces] = mesh.read();

        const int nvo = (int) new_verts.size(0);
        const int nto = (int) new_faces.size(0);

        *out_verts = (float*) std::malloc((size_t) nvo * 3 * sizeof(float));
        *out_tris  = (int*)   std::malloc((size_t) nto * 3 * sizeof(int));
        if (!*out_verts || !*out_tris) return 0;
        cudaMemcpy(*out_verts, new_verts.data_ptr<float>(),
                   (size_t) nvo * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(*out_tris, new_faces.data_ptr<int>(),
                   (size_t) nto * 3 * sizeof(int), cudaMemcpyDeviceToHost);
        *out_nv = nvo; *out_nt = nto;
        return 1;

    } catch (const std::exception & e) {
        std::fprintf(stderr, "cumesh_clean_mesh failed: %s\n", e.what());
        return 0;
    } catch (...) {
        std::fprintf(stderr, "cumesh_clean_mesh failed (unknown)\n");
        return 0;
    }
}
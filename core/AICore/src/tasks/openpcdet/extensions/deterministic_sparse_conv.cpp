#include "deterministic_sparse_conv.h"

#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace openpcdet::det {

void det_sparse_conv_cpu(const float * feat, int C_in,
                         const int32_t * idx, const float * W,
                         const float * bias, float * out,
                         int n_out, int C_out, int K) {
    // Parallelize across disjoint (v, o) cells only: each cell's k-outer /
    // c-inner serial det_fma reduction never changes with scheduling, so the
    // output is bit-identical to the serial run and to the CUDA kernel.
#ifdef _OPENMP
#pragma omp parallel for collapse(2) num_threads(std::min(8, omp_get_max_threads()))
#endif
    for (int v = 0; v < n_out; ++v)
        for (int o = 0; o < C_out; ++o)
            out[static_cast<size_t>(v) * C_out + o] =
                det_sparse_conv_cell(feat, C_in, idx + static_cast<size_t>(v) * K,
                                     W, bias[o], o, C_out, K, det_fma);
}

bool det_expand_neighbor_rows(const ext::sparse_indice_map & map,
                              std::vector<int32_t> & neighbor_rows,
                              std::string & error) {
    const int64_t rows = map.output_active_count();
    if (map.kernel_volume <= 0 || rows <= 0 ||
        map.indices.size() != static_cast<size_t>(rows) * map.kernel_volume) {
        error = "deterministic sparse conv: rulebook shape mismatch";
        return false;
    }
    // map.indices is stored row-major [output_row, kernel] with -1 for missing
    // neighbors — exactly the [n_out, K] layout the deterministic kernels want.
    neighbor_rows = map.indices;
    return true;
}

}  // namespace openpcdet::det

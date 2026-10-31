// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Deterministic focal sparse convolution — the single shared numeric core of
// Option C (see ../DETERMINISTIC_FOCAL_DESIGN.md).
//
// Why: the focal conv_imp sigmoid feeds a hard 0.5 threshold, and ~2945 sites
// sit within 3e-5 of it, so the ~2e-7 ggml mul_mat reduction-order noise flips
// voxel selections and the topology drift compounds through the chain. The fix
// is a fixed reduction order (k outer, c inner) with a fixed rounding semantic
// (single-rounding fma on BOTH sides) and one shared exp/sigmoid polynomial
// (no libm expf vs CUDA expf divergence, no __expf). IEEE-754 then guarantees
// bit-identical CPU/GPU outputs for identical inputs.
//
// Scope: only the FocalSparseConv layers of the focal chain (conv_imp + main
// conv each), gated by hparams::focal_deterministic in voxelrcnn.cpp. Every
// other model keeps the stock ggml mul_mat path untouched.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "sparse_ops.h"

#ifndef __CUDACC__
#include <cmath>
#endif

namespace openpcdet::det {

// Single-rounding multiply-add: CPU std::fma, GPU fmaf. Both are correctly
// rounded (one rounding step), so given identical inputs and identical order
// the results are bit-identical. A plain `a*b + c` expression is NOT usable
// here: with -march=x86-64-v3 + default -ffp-contract=fast the host compiler
// may or may not fuse it independently of nvcc's own contraction decisions.
#ifdef __CUDACC__
__device__ __host__ inline float det_fma(float a, float b, float c) {
    return fmaf(a, b, c);
}
#else
inline float det_fma(float a, float b, float c) { return std::fma(a, b, c); }
#endif

// Deterministic expf shared by host and device. Explicit det_fma everywhere
// (plain products/adds cannot contract; every product+add pair is spelled out),
// no libm, no __expf. Range reduction k = round(x * log2(e)) with a two-part
// ln2, then a degree-6 Taylor core for r in [-ln2/2, ln2/2]; max error ~2 ULP
// over the whole float range the sigmoid can reach.
inline float det_expf(float x) {
    // Sigmoid saturation clamps (keep 2^k * exp(r) inside the normal range;
    // k = round(x*log2e) <= 126 must hold so 2^k * (1 + exp(r)) < FLT_MAX).
    if (x > 87.0f) return 0x1.fffffep+127f;  // ~FLT_MAX: 1/(1+this) rounds to 1
    if (x < -87.0f) return 0.0f;
    constexpr float kInvLn2 = 0x1.715476p+0f;  // 1.4426950216293335
    constexpr float kLn2Hi = 0x1.62e400p-1f;   // 6.93145751953125e-1
    constexpr float kLn2Lo = 0x1.7f7d1cp-20f;  // 1.4286067653e-6
    const float biased = det_fma(x, kInvLn2, x >= 0.0f ? 0.5f : -0.5f);
    const int k = static_cast<int>(biased);
    const float kf = static_cast<float>(k);
    const float hi = det_fma(kf, -kLn2Hi, x);
    const float r = det_fma(kf, -kLn2Lo, hi);
    // exp(r) = 1 + r + r^2/2 + r^3 * q(r), q = 1/6 + r/24 + r^2/120 + r^3/720.
    constexpr float kC720 = 1.38888889e-3f;
    constexpr float kC120 = 8.33333333e-3f;
    constexpr float kC24 = 4.16666667e-2f;
    constexpr float kC6 = 1.66666667e-1f;
    constexpr float kHalf = 0.5f;
    float q = det_fma(r, kC720, kC120);
    q = det_fma(r, q, kC24);
    q = det_fma(r, q, kC6);
    const float r2 = r * r;
    const float p = det_fma(q, r, kHalf);  // 1/2 + r/6 + r^2/24 + r^3/720
    const float e = det_fma(p, r2, r);     // r + r^2/2 + r^3 * q
    // Scale by 2^k exactly, then one final rounding. memcpy works in CUDA
    // device code too (nvcc lowers it to register moves), so both sides build
    // the scale bits identically.
    const int bits = (k + 127) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    return det_fma(scale, e, scale);
}

// det_sigmoid(x) = 1 / (1 + det_expf(-x)) — the shared split_voxels importance
// sigmoid. Host and device evaluate the identical instruction sequence.
inline float det_sigmoid(float x) { return 1.0f / (1.0f + det_expf(-x)); }

// Deterministic ReLU (exact on both sides; -0 normalizes to +0).
inline float det_relu(float v) { return v > 0.0f ? v : 0.0f; }

// One output cell of the deterministic sparse conv:
//   out[v,o] = bias[o] + sum_k sum_c feat[idx[v,k], c] * W[k*C_in + c, o]
// Fixed order: k outer (skip -1 empty neighbors), c inner, fma accumulation.
// W is the GGUF/ggml-native [C_out, K*C_in] o-major layout (element (o, r) at
// o*(K*C_in) + r, exactly what ggml_backend_tensor_get yields for the fused
// sparse weight tensor), which also makes the c-inner walk contiguous.
template <typename FMA>
inline float det_sparse_conv_cell(const float* feat,
                                  int C_in,
                                  const int32_t* idx_row,
                                  const float* W,
                                  float bias_o,
                                  int o,
                                  int C_out,
                                  int K,
                                  FMA fma) {
    float acc = bias_o;
    for (int k = 0; k < K; ++k) {
        const int r = idx_row[k];
        if (r < 0)
            continue;  // zero padding: skip (never touches the accumulator)
        const float* row = feat + static_cast<size_t>(r) * C_in;
        const float* w = W + static_cast<size_t>(o) * (K * C_in) +
                         static_cast<size_t>(k) * C_in;
        for (int c = 0; c < C_in; ++c) acc = fma(row[c], w[c], acc);
    }
    return acc;
}

// Deterministic sparse conv over full row sets (design pseudocode): v outer,
// o middle, k outer / c inner reduction inside each cell. CPU instantiates it
// with std::fma; the CUDA kernel in openpcdet_cuda_ext.cu implements the same
// order per thread with fmaf.
template <typename FMA>
inline void det_sparse_conv_rows(const float* feat,
                                 int C_in,
                                 const int32_t* idx,  // [n_out, K], -1 = empty
                                 const float* W,      // [K*C_in, C_out]
                                 const float* bias,   // [C_out]
                                 float* out,          // [n_out, C_out]
                                 int n_out,
                                 int C_out,
                                 int K,
                                 FMA fma) {
    for (int v = 0; v < n_out; ++v)
        for (int o = 0; o < C_out; ++o)
            out[static_cast<size_t>(v) * C_out + o] = det_sparse_conv_cell(
                    feat, C_in, idx + static_cast<size_t>(v) * K, W, bias[o], o,
                    C_out, K, fma);
}

// CPU deterministic sparse conv (std::fma instantiation). Parallelizes ONLY
// across disjoint output cells, so the per-cell reduction order is independent
// of thread scheduling — results are bit-identical to a serial run and to the
// CUDA kernel.
void det_sparse_conv_cpu(const float* feat,
                         int C_in,
                         const int32_t* idx,
                         const float* W,
                         const float* bias,
                         float* out,
                         int n_out,
                         int C_out,
                         int K);

// Expand the rulebook into the [n_out, K] row-major neighbor table the
// deterministic kernels consume (-1 = no neighbor). map.indices is already
// stored [output_row, kernel] (see sparse_ops.cpp build_sparse_indice_map),
// so this is a pure data move with a shape sanity check.
bool det_expand_neighbor_rows(const ext::sparse_indice_map& map,
                              std::vector<int32_t>& neighbor_rows,
                              std::string& error);

#ifdef AICORE_OPENPCDET_CUDA_EXT
// CUDA deterministic sparse conv: stages the host inputs onto the device,
// launches det_sparse_conv_kernel (same order/rounding as det_sparse_conv_cpu)
// and copies the result back. Synchronous w.r.t. the host on return.
//   features       [in_rows, C_in]
//   neighbor_rows  [n_out, K], -1 = empty
//   weights        [C_out, K*C_in] o-major (ggml native)   bias [C_out] or
//   nullptr out            [n_out, C_out]
bool det_sparse_conv_cuda(const float* features_host,
                          int in_rows,
                          int in_channels,
                          const int32_t* neighbor_rows_host,
                          const float* weights_host,
                          const float* bias_host,
                          float* out_host,
                          int n_out,
                          int out_channels,
                          int kernel_volume,
                          std::string& error);
#endif

}  // namespace openpcdet::det

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// ggml helpers shared by the vision graphs (radio / mmproj / simplefpn /
// wedetect).
#pragma once

#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "vlx_gguf_loader.h"

namespace vlx {

// GGUF tensor dtype -> ggml type. Unsupported dtypes abort: silently mapping
// them to F32 would reinterpret the bytes.
inline ggml_type to_ggml_type(GgmlDType t) {
    switch (t) {
        case GgmlDType::F32:
            return GGML_TYPE_F32;
        case GgmlDType::F16:
            return GGML_TYPE_F16;
        case GgmlDType::Q8_0:
            return GGML_TYPE_Q8_0;
        default:
            GGML_ABORT("unsupported GGUF tensor dtype %u", (unsigned)t);
    }
}

// Weight GEMM w^T x with fp32 accumulation for float weights.
//  * F32 weight: IEEE fp32. CUDA's cuBLAS handle defaults to
//    CUBLAS_TF32_TENSOR_OP_MATH and Vulkan fp16 devices stage F32 tiles in
//    f16 shared memory — ~1e-2 rel on deep conv stacks, enough to flip
//    WeDetect top-100 boxes. The GGML_HINT_STRICT_F32 hint selects the
//    repository's strict-IEEE-fp32 GEMM routes (CUDA CUBLAS_DEFAULT_MATH
//    guard in 0002-acv-cuda-operators, Vulkan matmul_f32_strict scalar
//    pipeline in 0005-acv-vulkan-operators); CPU computes fp32 natively.
//  * F16 weight: the default GPU path is f16 x f16 with f16 accumulation
//    (cls rel 2e-2, box flips); PREC_F32 keeps the CPU semantics -- operands
//    at f16 mantissa, fp32 accumulate.
//  * Q8_0: integer-dot kernels, precision is fixed by the format.
inline ggml_tensor* mul_mat_w(ggml_context* ctx,
                              ggml_tensor* w,
                              ggml_tensor* x) {
    ggml_tensor* r = ggml_mul_mat(ctx, w, x);
    if (w->type == GGML_TYPE_F32) {
        ggml_prec_set_acc(r, GGML_PREC_F32);
        // F32 x F32: request the strict IEEE fp32 GEMM route on GPU backends
        // (see the note above; the backend routes gate on F32 x F32, so the
        // hint is a no-op elsewhere).
        ggml_mul_mat_set_hint(r, GGML_HINT_STRICT_F32);
    } else if (w->type == GGML_TYPE_F16) {
        ggml_prec_set_acc(r, GGML_PREC_F32);
    }
    return r;
}

// mm_projector_aux (mlp2x_gelu): Linear -> erf-GELU -> Linear over N row-major
// rows of D_in, weights `<prefix>.{0,2}.{weight,bias}` in their GGUF dtype.
// Output (N, D_out) row-major. One-shot graph: weights live only for the call
// (N <= 100 objects; the whole projector is ~0.1 GFLOP per row).
inline bool run_mlp2x_gelu(ggml_backend_t be,
                           const GgufFile& g,
                           const std::string& prefix,
                           const std::vector<float>& rows,
                           int64_t n_rows,
                           std::vector<float>* out) {
    const char* names[4] = {".0.weight", ".0.bias", ".2.weight", ".2.bias"};
    const TensorInfo* ti[4];
    for (int i = 0; i < 4; ++i)
        if (!(ti[i] = g.tensor(prefix + names[i]))) return false;
    const int64_t d_in = int64_t(ti[0]->ne[0]);
    if (n_rows <= 0 || int64_t(rows.size()) != d_in * n_rows) return false;
    ggml_init_params ip = {ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
                           nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* t[4];
    for (int i = 0; i < 4; ++i)  // 1-D biases: ne[1] = 1 (loader default)
        t[i] = ggml_new_tensor_2d(ctx, to_ggml_type(ti[i]->dtype),
                                  int64_t(ti[i]->ne[0]), int64_t(ti[i]->ne[1]));
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx, be);
    for (int i = 0; i < 4; ++i)
        ggml_backend_tensor_set(t[i], ti[i]->bytes.data(), 0,
                                ggml_nbytes(t[i]));
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_in, n_rows);
    ggml_set_input(x);
    ggml_tensor* y =
            ggml_gelu_erf(ctx, ggml_add(ctx, mul_mat_w(ctx, t[0], x), t[1]));
    y = ggml_add(ctx, mul_mat_w(ctx, t[2], y),
                 t[3]);  // (D_out, N): addr d + n*D_out
    ggml_set_output(y);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t al =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    const bool ok = ggml_gallocr_alloc_graph(al, gf);
    if (ok) {
        ggml_backend_tensor_set(x, rows.data(), 0, ggml_nbytes(x));
        ggml_backend_graph_compute(be, gf);
        out->resize(ggml_nelements(
                y));  // token-major (D_out, N) == row-major (N, D_out)
        ggml_backend_tensor_get(y, out->data(), 0, ggml_nbytes(y));
    }
    ggml_gallocr_free(al);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx);
    return ok;
}

}  // namespace vlx

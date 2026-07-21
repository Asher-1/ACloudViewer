// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Shared ggml graph-level utility ops for all ggml-based inference modules
// (DA3, FreeSplatter, etc.). Extracted from core/DA3/src/ggml_extend.hpp.
#pragma once
#include "ggml.h"

namespace ggml_common {

// y = W x (+ bias). W is [in, out] as stored by ggml (row-major torch Linear weight).
inline ggml_tensor* linear(ggml_context* ctx, ggml_tensor* W, ggml_tensor* x, ggml_tensor* bias = nullptr) {
    ggml_tensor* y = ggml_mul_mat(ctx, W, x);
    if (bias) y = ggml_add(ctx, y, bias);
    return y;
}

// LayerNorm over dim-0 with affine (w,b); eps from config.
inline ggml_tensor* layernorm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, float eps) {
    ggml_tensor* n = ggml_norm(ctx, x, eps);
    n = ggml_mul(ctx, n, w);
    if (b) n = ggml_add(ctx, n, b);
    return n;
}

// LayerNorm without bias (gamma-only, used by free-splatter pre-norm blocks).
inline ggml_tensor* layernorm_gamma_only(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, float eps) {
    return ggml_mul(ctx, ggml_norm(ctx, x, eps), w);
}

// LayerScale: elementwise multiply by gamma vector (broadcast over tokens).
inline ggml_tensor* layerscale(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gamma) {
    return ggml_mul(ctx, x, gamma);
}

// Exact (erf) GELU - matches torch nn.GELU() default.
inline ggml_tensor* gelu_erf(ggml_context* ctx, ggml_tensor* x) {
    return ggml_gelu_erf(ctx, x);
}

// SiLU / swish - matches torch F.silu (SwiGLU FFN gate activation).
inline ggml_tensor* silu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_silu(ctx, x);
}

// Sigmoid activation (used for opacity and scale activation in free-splatter).
inline ggml_tensor* sigmoid(ggml_context* ctx, ggml_tensor* x) {
    return ggml_sigmoid(ctx, x);
}

}  // namespace ggml_common

// Backward-compatible alias: DA3 code uses `aicore::depth::` namespace.
// New code should use `ggml_common::` directly.
namespace aicore {
namespace depth {
using ggml_common::linear;
using ggml_common::layernorm;
using ggml_common::layerscale;
using ggml_common::gelu_erf;
using ggml_common::silu;
} // namespace depth
} // namespace aicore

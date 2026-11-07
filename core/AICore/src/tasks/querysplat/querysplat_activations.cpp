// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/querysplat/querysplat_activations.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace qs {

namespace {

constexpr float kShDcScale = 0.28209479177387814f;

inline float expm1_abs(float x) {
    // position = torch.sign(position) * torch.expm1(torch.abs(position));
    // expm1 in f32: for |x| >= 1e-4 the exp(x)-1 rewrite is exact to f32
    // ulps; below that the absolute error (~x^2/2) is ~1e-8 world units.
    return std::expm1(x);
}

}  // namespace

GaussianOutput activate_gaussians(const float* geom_raw,
                                  const float* color_raw,
                                  int64_t n_tokens,
                                  int gpt,
                                  int post_output_dim,
                                  float scale_cap,
                                  float z_offset,
                                  int sh_degree,
                                  float sh_rest_scale) {
    const int color_dim = 3 * (sh_degree + 1) * (sh_degree + 1);
    const int stride = 11 + color_dim;
    const int64_t N = n_tokens * gpt;
    GaussianOutput out;
    out.N = N;
    out.color_dim = color_dim;
    out.data.assign((size_t)N * stride, 0.0f);
    out.opacity.resize((size_t)N);

    const float scale_shift = 1.0f - std::log(scale_cap);
    const float scale_log_cap = std::log(scale_cap);

    // geometry branch: ggml mul_mat output ne = {640, n_tokens} means the
    // out-channel (j = p*10+c) is the FASTEST axis: flat = n*640 + j — the
    // torch (n_tokens, 640) C-order layout. (An earlier j*n_tokens+n read
    // was transposed and scrambled the gaussians.)
    const int geom_stride = 10 * gpt;  // 640
    for (int64_t n = 0; n < n_tokens; n++) {
        const float* grow = geom_raw + (size_t)n * geom_stride;
        for (int p = 0; p < gpt; p++) {
            const int64_t g = n * gpt + p;
            float* dst = &out.data[(size_t)g * stride];
            // position (c 0..2): sign*expm1(|x|); +z_offset on component 2
            for (int c = 0; c < 3; c++) {
                const float x = grow[p * 10 + c];
                float v = (x >= 0.0f ? 1.0f : -1.0f) * expm1_abs(std::fabs(x));
                if (c == 2) v += z_offset;
                dst[c] = v;
            }
            // scaling (c 3..5): exp(clamp(x - shift, max = ln(cap)))
            for (int c = 0; c < 3; c++) {
                const float x = grow[p * 10 + 3 + c];
                dst[4 + c] = std::exp(std::min(x - scale_shift, scale_log_cap));
            }
            // rotation (c 6..9): F.normalize(dim=-1) (eps 1e-12)
            float r[4];
            float n2 = 0.0f;
            for (int c = 0; c < 4; c++) {
                r[c] = grow[p * 10 + 6 + c];
                n2 += r[c] * r[c];
            }
            const float inv = 1.0f / std::sqrt(std::max(n2, 1e-12f));
            for (int c = 0; c < 4; c++) dst[7 + c] = r[c] * inv;
        }
    }

    // appearance branch: rows j = p*post_output_dim + c, same n-major layout
    const int color_stride = post_output_dim * gpt;
    for (int64_t n = 0; n < n_tokens; n++) {
        const float* crow = color_raw + (size_t)n * color_stride;
        for (int p = 0; p < gpt; p++) {
            const int64_t g = n * gpt + p;
            float* dst = &out.data[(size_t)g * stride];
            if (sh_degree == 0) {
                // color = 0.5 * tanh(raw_color) + 0.5
                for (int c = 0; c < 3; c++) {
                    const float x = crow[p * post_output_dim + c];
                    dst[11 + c] = 0.5f * std::tanh(x) + 0.5f;
                }
            } else {
                // sh degree 1: dc = (0.5*tanh(sh0)+0.5 - 0.5) / 0.28209479...,
                // rest = sh_rest_scale * tanh(sh1..3)
                for (int c = 0; c < 3; c++) {
                    const float x = crow[p * post_output_dim + c];
                    const float rgb = 0.5f * std::tanh(x) + 0.5f;
                    dst[11 + c] = (rgb - 0.5f) / kShDcScale;
                }
                const int sh_basis = (sh_degree + 1) * (sh_degree + 1);
                for (int b = 1; b < sh_basis; b++) {
                    for (int c = 0; c < 3; c++) {
                        const float x = crow[p * post_output_dim + b * 3 + c];
                        dst[11 + b * 3 + c] = sh_rest_scale * std::tanh(x);
                    }
                }
            }
            // opacity = sigmoid(raw[..., color_dim] - 2)
            const float xo = crow[p * post_output_dim + color_dim];
            const float op = 1.0f / (1.0f + std::exp(-(xo - 2.0f)));
            dst[3] = op;
            out.opacity[(size_t)g] = op;
        }
    }
    return out;
}

// Exact VJP of activate_gaussians (TTO t2). Torch corner cases preserved:
// sign'==0 and |x|'==0 at x==0 make the position gradient 0 there; the
// scale min() saturates to 0 once x-shift >= log_cap (inclusive); the
// rotation F.normalize gradient is the projection (dq - q(q.dq))/|r|.
void activate_gaussians_backward(const float* geom_raw,
                                 const float* color_raw,
                                 const float* d_gaussians,
                                 int64_t n_tokens,
                                 int gpt,
                                 int post_output_dim,
                                 float scale_cap,
                                 float z_offset,
                                 int sh_degree,
                                 float sh_rest_scale,
                                 float* d_geom_raw,
                                 float* d_color_raw) {
    const int color_dim = 3 * (sh_degree + 1) * (sh_degree + 1);
    const int stride = 11 + color_dim;
    const int geom_stride = 10 * gpt;
    const int color_stride = post_output_dim * gpt;
    const float scale_shift = 1.0f - std::log(scale_cap);
    const float scale_log_cap = std::log(scale_cap);
    std::memset(d_geom_raw, 0, (size_t)n_tokens * geom_stride * sizeof(float));
    std::memset(d_color_raw, 0,
                (size_t)n_tokens * color_stride * sizeof(float));
    // geometry branch: rows j = p*10 + c (see the forward comment)
    for (int64_t n = 0; n < n_tokens; n++) {
        const float* grow = geom_raw + (size_t)n * geom_stride;
        float* dgrad = d_geom_raw + (size_t)n * geom_stride;
        for (int p = 0; p < gpt; p++) {
            const int64_t g = n * gpt + p;
            const float* dg = &d_gaussians[(size_t)g * stride];
            for (int c = 0; c < 3; c++) {
                const float x = grow[p * 10 + c];
                dgrad[p * 10 + c] =
                        x == 0.0f ? 0.0f : dg[c] * std::exp(std::fabs(x));
            }
            for (int c = 0; c < 3; c++) {
                const float x = grow[p * 10 + 3 + c];
                const float t = x - scale_shift;
                dgrad[p * 10 + 3 + c] =
                        t <= scale_log_cap ? dg[4 + c] * std::exp(t) : 0.0f;
            }
            float q[4], r2 = 0.f;
            for (int c = 0; c < 4; c++) {
                q[c] = grow[p * 10 + 6 + c];
                r2 += q[c] * q[c];
            }
            const float rn = std::sqrt(std::max(r2, 1e-12f));
            if (rn > 1e-6f) {  // matches the forward's max(n2, 1e-12) sqrt
                // F.normalize backward: (dg - y (y.dg)) / |x| with y = x/|x|
                // — the projection uses the NORMALIZED quaternion, not the
                // raw one (an easy slip that inflates the gradient by |x|^2)
                float dot = 0.f;
                for (int c = 0; c < 4; c++) {
                    q[c] /= rn;
                    dot += dg[7 + c] * q[c];
                }
                for (int c = 0; c < 4; c++)
                    dgrad[p * 10 + 6 + c] = (dg[7 + c] - q[c] * dot) / rn;
            }
        }
    }
    // appearance branch: rows j = p*post_output_dim + c
    for (int64_t n = 0; n < n_tokens; n++) {
        const float* crow = color_raw + (size_t)n * color_stride;
        float* dcrow = d_color_raw + (size_t)n * color_stride;
        for (int p = 0; p < gpt; p++) {
            const int64_t g = n * gpt + p;
            const float* dg = &d_gaussians[(size_t)g * stride];
            if (sh_degree == 0) {
                for (int c = 0; c < 3; c++) {
                    const float x = crow[p * post_output_dim + c];
                    const float t = std::tanh(x);
                    dcrow[p * post_output_dim + c] =
                            dg[11 + c] * 0.5f * (1.0f - t * t);
                }
            } else {
                for (int c = 0; c < 3; c++) {
                    const float x = crow[p * post_output_dim + c];
                    const float t = std::tanh(x);
                    dcrow[p * post_output_dim + c] =
                            dg[11 + c] * 0.5f * (1.0f - t * t) / kShDcScale;
                }
                const int sh_basis = (sh_degree + 1) * (sh_degree + 1);
                for (int b = 1; b < sh_basis; b++)
                    for (int c = 0; c < 3; c++) {
                        const float x = crow[p * post_output_dim + b * 3 + c];
                        const float t = std::tanh(x);
                        dcrow[p * post_output_dim + b * 3 + c] =
                                dg[11 + b * 3 + c] * sh_rest_scale *
                                (1.0f - t * t);
                    }
            }
            const float xo = crow[p * post_output_dim + color_dim];
            const float op = 1.0f / (1.0f + std::exp(-(xo - 2.0f)));
            dcrow[p * post_output_dim + color_dim] = dg[3] * op * (1.0f - op);
        }
    }
}

}  // namespace qs

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Asset-free golden-op tier: pin the ggml ops pieces 1-3 depend on against
// hand-computed f64 references, so a wrong op (or a ggml version that renames /
// changes one) fails instantly with no fixtures. New graph ops get a pin here
// BEFORE any fixture-based parity claim.
#include <ggml.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;
#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, \
                         #cond);                                         \
            failures++;                                                  \
        }                                                                \
    } while (0)

static double erf_gelu(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// GELU must be the exact erf variant (nn.GELU()), NOT the tanh approximation:
// the error compounds across 24 transformer blocks. This proves ggml_gelu_erf
// is available and is the erf form.
static void test_gelu_erf() {
    const float xs[] = {-3.f, -1.f, -0.5f, 0.f, 0.5f, 1.f, 2.f, 3.f};
    const int n = (int)(sizeof(xs) / sizeof(xs[0]));

    ggml_init_params p = {(size_t)16 * 1024 * 1024, nullptr,
                          /*no_alloc =*/false};
    ggml_context *ctx = ggml_init(p);
    ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    std::memcpy(x->data, xs, sizeof(xs));
    ggml_tensor *y = ggml_gelu_erf(ctx, x);
    ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    for (int i = 0; i < n; i++) {
        const double ref = erf_gelu(xs[i]);
        const double got = ((float *)y->data)[i];
        CHECK(std::fabs(got - ref) < 1e-5);
    }
    ggml_free(ctx);
}

// Pin ggml_mul_mat orientation: this is the contract the converter relies on
// (PyTorch Linear weight (out,in) stored as-is loads as ggml ne=[in,out], and
// mul_mat(W, x) then yields the correct (out) projection). result[i,j] over
// A ne=[k,m], B ne=[k,n] is sum_l A[l,i] * B[l,j], giving C ne=[m,n].
static void test_mul_mat_orientation() {
    const int k = 2, m = 3, n = 1;
    ggml_init_params p = {(size_t)16 * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(p);

    ggml_tensor *A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k,
                                        m);  // "weight" [in=k, out=m]
    ggml_tensor *B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k,
                                        n);  // "input"  [in=k, tok=n]
    // A rows (out): [1,2], [3,4], [5,6]; column-major ne=[k,m] => A[l + i*k]
    const float a[] = {1, 2, 3, 4, 5, 6};
    const float b[] = {10, 20};  // one token, k=2
    std::memcpy(A->data, a, sizeof(a));
    std::memcpy(B->data, b, sizeof(b));

    ggml_tensor *C = ggml_mul_mat(ctx, A, B);  // ne=[m,n]
    ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    CHECK(C->ne[0] == m && C->ne[1] == n);
    const float *c = (float *)C->data;
    // out_i = a[i,0]*10 + a[i,1]*20
    CHECK(std::fabs(c[0] - (1 * 10 + 2 * 20)) < 1e-5);  // 50
    CHECK(std::fabs(c[1] - (3 * 10 + 4 * 20)) < 1e-5);  // 110
    CHECK(std::fabs(c[2] - (5 * 10 + 6 * 20)) < 1e-5);  // 170
    ggml_free(ctx);
}

// Pin the patchify path (im2col F32 + kernel-reshape + mul_mat) against a
// direct nested-loop reference conv. This isolates the #1 silent-bug hotspot —
// that the im2col patch-element order agrees with reshape(kernel, [IC*KH*KW,
// OC]) — with no model and no fixtures. stride == kernel (non-overlapping
// patches).
static void test_patchify_order() {
    const int KW = 2, KH = 2, IC = 2, OC = 3, W = 4, H = 4, N = 1,
              S = 2;  // OW=OH=2
    const int OW = (W - KW) / S + 1, OH = (H - KH) / S + 1;

    ggml_init_params p = {(size_t)16 * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(p);
    ggml_tensor *ker = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, KW, KH, IC,
                                          OC);  // [KW,KH,IC,OC]
    ggml_tensor *img =
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, IC, N);  // [W,H,C,N]
    float *kd = (float *)ker->data;
    float *id = (float *)img->data;
    for (int i = 0; i < KW * KH * IC * OC; i++)
        kd[i] = 0.1f * ((i * 7) % 11) - 0.3f;
    for (int i = 0; i < W * H * IC * N; i++)
        id[i] = 0.1f * ((i * 5) % 13) - 0.5f;

    // engine path
    ggml_tensor *cols =
            ggml_im2col(ctx, ker, img, S, S, 0, 0, 1, 1, true, GGML_TYPE_F32);
    ggml_tensor *wflat = ggml_reshape_2d(ctx, ker, IC * KH * KW, OC);
    ggml_tensor *conv = ggml_mul_mat(ctx, wflat, cols);  // [OC, OW, OH, N]
    ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, conv);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    CHECK(conv->ne[0] == OC && conv->ne[1] == OW && conv->ne[2] == OH);
    auto kat = [&](int kw, int kh, int ic, int oc) {
        return kd[kw + KW * (kh + KH * (ic + IC * oc))];
    };
    auto iat = [&](int w, int h, int ic) { return id[w + W * (h + H * ic)]; };
    const float *c = (float *)conv->data;  // ne=[OC,OW,OH,N]
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++)
            for (int oc = 0; oc < OC; oc++) {
                double ref = 0.0;
                for (int ic = 0; ic < IC; ic++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++)
                            ref += (double)kat(kw, kh, ic, oc) *
                                   iat(ow * S + kw, oh * S + kh, ic);
                const float got = c[oc + OC * (ow + OW * oh)];
                CHECK(std::fabs(got - ref) < 1e-5);
            }
    ggml_free(ctx);
}

// Pin the SH-residual DC constant used by the scene head (RGB2SH dc =
// (rgb-0.5)/C0).
static void test_sh_c0() {
    const double C0 = 0.28209479177387814;
    CHECK(std::fabs(C0 - 1.0 / (2.0 * std::sqrt(M_PI))) < 1e-15);
}

int main() {
    test_gelu_erf();
    test_mul_mat_orientation();
    test_patchify_order();
    test_sh_c0();
    std::printf(failures ? "test_graph_blocks: %d FAILURES\n"
                         : "test_graph_blocks: ok\n",
                failures);
    return failures ? 1 : 0;
}

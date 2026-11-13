// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/depth/winograd.hpp"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__FMA__)
#include <immintrin.h>
#endif

namespace aicore {
namespace depth {
namespace {

// ========================================================================
// Winograd F(2x2,3x3) conv3x3: BLOCKED GEMM over a block of tiles
// (parity-identical to the direct conv, pure-speed: reuses each U row across
// TB tiles; ~193ms vs direct ~242ms @504 BASE/16t; ~490ms vs ~625ms on
// GIANT). The historical DA_WINO variants (f2 per-tile GEMV / f4 F(4x4))
// were A/B scaffolding and are removed; f2b is the variant that won. See
// benchmarks/BENCHMARK.md. Parity is exact (halves+ints, max|d|~1e-5).
// ========================================================================

// Tile-block width for the blocked GEMM microkernel (number of tiles batched
// per winograd-domain multiply). 8 keeps all accumulators in zmm registers
// while amortizing each U-row load across 8 tiles.
constexpr int TB = 8;

// ------------------------------------------------------------------------
// F(2x2,3x3) transforms (exact: halves + integers).
//   B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
//   G   = [[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]]
//   A^T = [[1,1,1,0],[0,1,-1,-1]]
// ------------------------------------------------------------------------
struct F2Policy {
    static constexpr int IT = 4, OT = 2,
                         P = 16;  // input tile / output tile / positions

    // U = G g G^T, 3x3 -> 4x4 (u[16]).
    static void filt(const float g[9], float u[16]) {
        float Gg[4][3];
        for (int j = 0; j < 3; ++j) {
            float c0 = g[0 * 3 + j], c1 = g[1 * 3 + j], c2 = g[2 * 3 + j];
            Gg[0][j] = c0;
            Gg[1][j] = 0.5f * (c0 + c1 + c2);
            Gg[2][j] = 0.5f * (c0 - c1 + c2);
            Gg[3][j] = c2;
        }
        for (int i = 0; i < 4; ++i) {
            float c0 = Gg[i][0], c1 = Gg[i][1], c2 = Gg[i][2];
            u[i * 4 + 0] = c0;
            u[i * 4 + 1] = 0.5f * (c0 + c1 + c2);
            u[i * 4 + 2] = 0.5f * (c0 - c1 + c2);
            u[i * 4 + 3] = c2;
        }
    }
    // V = B^T d B, 4x4 -> 4x4 (v[16]).
    static void inp(const float d[16], float v[16]) {
        float m[16];
        for (int j = 0; j < 4; ++j) {
            float r0 = d[0 * 4 + j], r1 = d[1 * 4 + j], r2 = d[2 * 4 + j],
                  r3 = d[3 * 4 + j];
            m[0 * 4 + j] = r0 - r2;
            m[1 * 4 + j] = r1 + r2;
            m[2 * 4 + j] = r2 - r1;
            m[3 * 4 + j] = r1 - r3;
        }
        for (int i = 0; i < 4; ++i) {
            float c0 = m[i * 4 + 0], c1 = m[i * 4 + 1], c2 = m[i * 4 + 2],
                  c3 = m[i * 4 + 3];
            v[i * 4 + 0] = c0 - c2;
            v[i * 4 + 1] = c1 + c2;
            v[i * 4 + 2] = c2 - c1;
            v[i * 4 + 3] = c1 - c3;
        }
    }
    // Y = A^T m A, 4x4 -> 2x2 (y[4]).
    static void outp(const float m[16], float y[4]) {
        float p[8];
        for (int j = 0; j < 4; ++j) {
            float r0 = m[0 * 4 + j], r1 = m[1 * 4 + j], r2 = m[2 * 4 + j],
                  r3 = m[3 * 4 + j];
            p[0 * 4 + j] = r0 + r1 + r2;
            p[1 * 4 + j] = r1 - r2 - r3;
        }
        for (int i = 0; i < 2; ++i) {
            float c0 = p[i * 4 + 0], c1 = p[i * 4 + 1], c2 = p[i * 4 + 2],
                  c3 = p[i * 4 + 3];
            y[i * 2 + 0] = c0 + c1 + c2;
            y[i * 2 + 1] = c1 - c2 - c3;
        }
    }
};

// ------------------------------------------------------------------------
// Persistent per-op state: caches the filter transform U (computed once from
// w->data; reused across forwards). Scratch (V,M) is per-thread on the stack
// / in small per-thread vectors.
// ------------------------------------------------------------------------
struct WinogradState {
    int W = 0, H = 0, IC = 0, OC = 0, N = 0, pad = 0;
    int Wout = 0, Hout = 0, tilesX = 0, tilesY = 0;
    const void* wdata = nullptr;
    // U layout: U[pos*IC*OC + ic*OC + oc], pos in 0..P-1. OC innermost so the
    // winograd-domain multiply vectorizes over OC.
    std::vector<float> U;
    std::once_flag once;
};

template <class Pol>
static void build_U(WinogradState* st, const float* w) {
    const int IC = st->IC, OC = st->OC;
    st->U.assign((size_t)Pol::P * IC * OC, 0.0f);
    float u[Pol::P];
    for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
            const float* g = w + ((size_t)oc * IC + ic) * 9;
            Pol::filt(g, u);
            for (int pos = 0; pos < Pol::P; ++pos)
                st->U[(size_t)pos * IC * OC + (size_t)ic * OC + oc] = u[pos];
        }
    }
}

// ------------------------------------------------------------------------
// Blocked GEMM microkernel for one winograd position:
//   M[t][oc] = sum_ic U[ic][oc] * V[ic][t],  t in [0,TBcur), oc in [0,OC).
// U: [IC][OC] row-major (OC innermost). V: [IC][TB] row-major. M: [TB][OC].
// Each loaded U-row (16 OC lanes) is reused across all TBcur tiles -> far
// better arithmetic intensity than the per-tile GEMV.
// ------------------------------------------------------------------------
static inline void wino_gemm_block(
        const float* U, const float* V, float* M, int IC, int OC, int TBcur) {
#if defined(__AVX512F__)
    int oc = 0;
    for (; oc + 16 <= OC; oc += 16) {
        __m512 acc[TB];
        for (int t = 0; t < TB; ++t) acc[t] = _mm512_setzero_ps();
        const float* up = U + oc;
        for (int ic = 0; ic < IC; ++ic) {
            const __m512 u = _mm512_loadu_ps(up + (size_t)ic * OC);
            const float* vp = V + (size_t)ic * TB;
            for (int t = 0; t < TBcur; ++t)
                acc[t] = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[t]), acc[t]);
        }
        for (int t = 0; t < TBcur; ++t)
            _mm512_storeu_ps(M + (size_t)t * OC + oc, acc[t]);
    }
    if (oc < OC) {
        const int rem = OC - oc;
        const __mmask16 mask = (__mmask16)((1u << rem) - 1u);
        __m512 acc[TB];
        for (int t = 0; t < TB; ++t) acc[t] = _mm512_setzero_ps();
        const float* up = U + oc;
        for (int ic = 0; ic < IC; ++ic) {
            const __m512 u = _mm512_maskz_loadu_ps(mask, up + (size_t)ic * OC);
            const float* vp = V + (size_t)ic * TB;
            for (int t = 0; t < TBcur; ++t)
                acc[t] = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[t]), acc[t]);
        }
        for (int t = 0; t < TBcur; ++t)
            _mm512_mask_storeu_ps(M + (size_t)t * OC + oc, mask, acc[t]);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    int oc = 0;
    for (; oc + 8 <= OC; oc += 8) {
        __m256 acc[TB];
        for (int t = 0; t < TB; ++t) acc[t] = _mm256_setzero_ps();
        const float* up = U + oc;
        for (int ic = 0; ic < IC; ++ic) {
            const __m256 u = _mm256_loadu_ps(up + (size_t)ic * OC);
            const float* vp = V + (size_t)ic * TB;
            for (int t = 0; t < TBcur; ++t)
                acc[t] = _mm256_fmadd_ps(u, _mm256_set1_ps(vp[t]), acc[t]);
        }
        for (int t = 0; t < TBcur; ++t)
            _mm256_storeu_ps(M + (size_t)t * OC + oc, acc[t]);
    }
    for (; oc < OC; ++oc) {
        for (int t = 0; t < TBcur; ++t) {
            float s = 0.0f;
            for (int ic = 0; ic < IC; ++ic)
                s += U[(size_t)ic * OC + oc] * V[(size_t)ic * TB + t];
            M[(size_t)t * OC + oc] = s;
        }
    }
#else
    for (int t = 0; t < TBcur; ++t)
        for (int oc = 0; oc < OC; ++oc) M[(size_t)t * OC + oc] = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        const float* up = U + (size_t)ic * OC;
        const float* vp = V + (size_t)ic * TB;
        for (int t = 0; t < TBcur; ++t) {
            const float vv = vp[t];
            float* mt = M + (size_t)t * OC;
            for (int oc = 0; oc < OC; ++oc) mt[oc] += up[oc] * vv;
        }
    }
#endif
}

// ------------------------------------------------------------------------
// Blocked path: batch TB tiles per winograd-domain GEMM.
// ------------------------------------------------------------------------
template <class Pol>
static void compute_blocked(WinogradState* st,
                            ggml_tensor* dst,
                            int ith,
                            int nth) {
    constexpr int IT = Pol::IT, OT = Pol::OT, P = Pol::P;
    const ggml_tensor* xt = dst->src[0];
    const float* x = (const float*)xt->data;
    float* y = (float*)dst->data;

    const int W = st->W, H = st->H, IC = st->IC, OC = st->OC, pad = st->pad;
    const int Wout = st->Wout, Hout = st->Hout;
    const int tilesX = st->tilesX, tilesY = st->tilesY;
    const float* U = st->U.data();

    const int ntiles = tilesX * tilesY;
    const int64_t total = (int64_t)st->N * ntiles;
    // Split the work by tile-BLOCKS so each thread's GEMMs stay full-width.
    const int64_t nblocks = (total + TB - 1) / TB;
    const int64_t bbeg = nblocks * ith / nth;
    const int64_t bend = nblocks * (ith + 1) / nth;

    // Per-thread scratch.
    std::vector<float> Vblk((size_t)P * IC * TB);  // V[pos][ic][t]
    std::vector<float> Mblk((size_t)P * TB * OC);  // M[pos][t][oc]
    float dpatch[IT * IT], vpatch[P], mpatch[P], ypatch[OT * OT];

    for (int64_t b = bbeg; b < bend; ++b) {
        const int64_t t0 = b * TB;
        const int TBcur = (int)std::min<int64_t>(TB, total - t0);

        // 1. Input transform for each tile in the block -> Vblk[pos][ic][t].
        for (int tb = 0; tb < TBcur; ++tb) {
            const int64_t idx = t0 + tb;
            const int n = (int)(idx / ntiles);
            const int t = (int)(idx % ntiles);
            const int ty = t / tilesX, tx = t % tilesX;
            const int iy0 = ty * OT - pad, ix0 = tx * OT - pad;
            const float* xn = x + (size_t)n * IC * H * W;
            for (int ic = 0; ic < IC; ++ic) {
                const float* xc = xn + (size_t)ic * H * W;
                for (int i = 0; i < IT; ++i) {
                    const int yy = iy0 + i;
                    const bool yok = (yy >= 0 && yy < H);
                    const float* row = yok ? (xc + (size_t)yy * W) : nullptr;
                    for (int j = 0; j < IT; ++j) {
                        const int xx = ix0 + j;
                        dpatch[i * IT + j] =
                                (yok && xx >= 0 && xx < W) ? row[xx] : 0.0f;
                    }
                }
                Pol::inp(dpatch, vpatch);
                float* vbase = Vblk.data() + (size_t)ic * TB + tb;
                for (int pos = 0; pos < P; ++pos)
                    vbase[(size_t)pos * IC * TB] = vpatch[pos];
            }
        }

        // 2. Winograd-domain blocked GEMM per position.
        for (int pos = 0; pos < P; ++pos)
            wino_gemm_block(U + (size_t)pos * IC * OC,
                            Vblk.data() + (size_t)pos * IC * TB,
                            Mblk.data() + (size_t)pos * TB * OC, IC, OC, TBcur);

        // 3. Output transform per tile per oc -> scatter OTxOT into dst.
        for (int tb = 0; tb < TBcur; ++tb) {
            const int64_t idx = t0 + tb;
            const int n = (int)(idx / ntiles);
            const int t = (int)(idx % ntiles);
            const int ty = t / tilesX, tx = t % tilesX;
            const int oy0 = ty * OT, ox0 = tx * OT;
            float* yn = y + (size_t)n * OC * Hout * Wout;
            for (int oc = 0; oc < OC; ++oc) {
                const float* mbase = Mblk.data() + (size_t)tb * OC + oc;
                for (int pos = 0; pos < P; ++pos)
                    mpatch[pos] = mbase[(size_t)pos * TB * OC];
                Pol::outp(mpatch, ypatch);
                float* yc = yn + (size_t)oc * Hout * Wout;
                for (int i = 0; i < OT; ++i) {
                    const int oy = oy0 + i;
                    if (oy >= Hout) continue;
                    for (int j = 0; j < OT; ++j) {
                        const int ox = ox0 + j;
                        if (ox >= Wout) continue;
                        yc[(size_t)oy * Wout + ox] = ypatch[i * OT + j];
                    }
                }
            }
        }
    }
}

static void winograd_compute(ggml_tensor* dst,
                             int ith,
                             int nth,
                             void* userdata) {
    WinogradState* st = (WinogradState*)userdata;
    const ggml_tensor* wt = dst->src[1];
    const float* w = (const float*)wt->data;

    std::call_once(st->once, [&] { build_U<F2Policy>(st, w); });
    compute_blocked<F2Policy>(st, dst, ith, nth);
}

// ------------------------------------------------------------------------
// Keyed cache of op states (U transformed once per (filter,shape)).
// ------------------------------------------------------------------------
struct StateKey {
    const void* wdata;
    int W, H, IC, OC, N, pad;
    bool operator==(const StateKey& o) const {
        return wdata == o.wdata && W == o.W && H == o.H && IC == o.IC &&
               OC == o.OC && N == o.N && pad == o.pad;
    }
};
struct StateKeyHash {
    size_t operator()(const StateKey& k) const {
        size_t h = (size_t)k.wdata;
        auto mix = [&h](int v) { h = h * 1000003u + (size_t)(uint32_t)v; };
        mix(k.W);
        mix(k.H);
        mix(k.IC);
        mix(k.OC);
        mix(k.N);
        mix(k.pad);
        return h;
    }
};

static std::mutex g_states_mtx;
static std::unordered_map<StateKey, WinogradState*, StateKeyHash> g_states;

static WinogradState* get_state(ggml_tensor* w, ggml_tensor* x, int pad) {
    const int W = (int)x->ne[0], H = (int)x->ne[1], IC = (int)x->ne[2],
              N = (int)x->ne[3];
    const int OC = (int)w->ne[3];
    StateKey key{w->data, W, H, IC, OC, N, pad};
    std::lock_guard<std::mutex> lk(g_states_mtx);
    auto it = g_states.find(key);
    if (it != g_states.end()) return it->second;
    WinogradState* st = new WinogradState();
    st->W = W;
    st->H = H;
    st->IC = IC;
    st->OC = OC;
    st->N = N;
    st->pad = pad;
    st->Wout = W + 2 * pad - 2;
    st->Hout = H + 2 * pad - 2;
    constexpr int OT = 2;  // F(2x2) output tile
    st->tilesX = (st->Wout + OT - 1) / OT;
    st->tilesY = (st->Hout + OT - 1) / OT;
    st->wdata = w->data;
    g_states[key] = st;
    return st;
}

}  // namespace

ggml_tensor* winograd_conv3x3(ggml_context* ctx,
                              ggml_tensor* w,
                              ggml_tensor* x,
                              int pad) {
    const int OC = (int)w->ne[3];
    const int N = (int)x->ne[3];
    const int Wout = (int)x->ne[0] + 2 * pad - 2;
    const int Hout = (int)x->ne[1] + 2 * pad - 2;
    WinogradState* st = get_state(w, x, pad);
    ggml_tensor* args[2] = {x, w};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, Wout, Hout, OC, N, args, 2,
                          winograd_compute, GGML_N_TASKS_MAX, st);
}

}  // namespace depth
}  // namespace aicore

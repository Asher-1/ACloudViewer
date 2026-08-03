#include "winograd.hpp"
#include <algorithm>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

// AVX-512 winograd GEMM microkernel availability + runtime dispatch.
//
// The zmm register-blocked microkernel ships in the default portable build as a
// RUNTIME-CPUID-DISPATCHED path. Its functions carry
// __attribute__((target("avx512f,avx512bw,avx512vl"))) (GCC/Clang function
// multiversioning) so the compiler emits the zmm code even though this TU is NOT
// given a global -mavx512f (FACEDETECT_WINO_AVX512 is defined by a CMake probe).
// wino_gemm_block calls it ONLY when __builtin_cpu_supports reports AVX-512 at
// run time, so AVX2-only CPUs never execute a zmm instruction -- no SIGILL, one
// portable binary. A global -mavx512f build (e.g. GGML_NATIVE=ON on an AVX-512
// host) defines __AVX512F__ instead; then the ISA is already on and the same
// kernel is compiled without the per-function attribute and still runtime-gated.
#if defined(__AVX512F__)
#  define FD_WINO_HAVE_AVX512 1
#  define FD_WINO_AVX512_TARGET   /* ISA already global; no per-fn attribute */
#elif defined(FACEDETECT_WINO_AVX512) && defined(__AVX2__)
#  define FD_WINO_HAVE_AVX512 1
#  define FD_WINO_AVX512_TARGET __attribute__((target("avx512f,avx512bw,avx512vl")))
#endif

namespace fd {
namespace {

#if defined(FD_WINO_HAVE_AVX512)
// Cached decision: run the AVX-512 GEMM microkernel? True only when the running
// CPU advertises avx512f+bw+vl (so the zmm code never executes on an AVX2-only
// host) AND the FACEDETECT_DISABLE_AVX512 test hook is unset (forcing it lets us
// exercise + parity-check the AVX2 fallback on an AVX-512 box). __builtin_cpu_*
// reads the CPUID feature bits the C runtime fills in before main; checked once.
// FACEDETECT_WINO_VERBOSE prints the one-time selection for ship-safety proofs.
static bool wino_use_avx512() {
    static const bool use512 = [] {
        const char* off = std::getenv("FACEDETECT_DISABLE_AVX512");
        const bool disabled = off && off[0] != '\0' && off[0] != '0';
        const bool supported = __builtin_cpu_supports("avx512f")
                            && __builtin_cpu_supports("avx512bw")
                            && __builtin_cpu_supports("avx512vl");
        const bool sel = supported && !disabled;
        if (std::getenv("FACEDETECT_WINO_VERBOSE"))
            std::fprintf(stderr,
                         "[winograd] GEMM microkernel: %s (avx512 supported=%d, "
                         "disabled=%d)\n", sel ? "AVX-512" : "AVX2",
                         (int)supported, (int)disabled);
        return sel;
    }();
    return use512;
}
#endif

// ========================================================================
// Mode selection (FACEDETECT_WINO env): which Winograd algorithm + kernel.
//   "f2"  : F(2x2,3x3), per-tile GEMV
//   "f2b" : F(2x2,3x3), blocked GEMM over a block of tiles  <-- auto default
//           (parity-identical to f2; reuses each U row across TB tiles)
//   "f4"  : F(4x4,3x3), blocked GEMM (4x fewer mults vs direct; less accurate)
// Parity: f2/f2b are exact (halves+ints, max|d|~1e-5). f4 uses 1/6,1/24
// fractions so it is less accurate -- gated by the suite.
// Ported from depth-anything.cpp/src/winograd.cpp; the SIMD inner kernels are
// re-targeted to AVX2 (GGML_NATIVE=OFF here -> no AVX-512).
// ========================================================================
enum class Mode { F2, F2B, F4 };

// `prefer_f4` is the per-conv hint from the caller (set only for SCRFD's large
// early maps). The FACEDETECT_WINO env, if set, forces ONE mode globally and
// wins over the hint (A/B harness). With the env unset the hint decides: F4 for
// the large maps, F2B (parity-exact) everywhere else.
static Mode parse_mode(bool prefer_f4) {
    const char* m = std::getenv("FACEDETECT_WINO");
    if (m) {
        if (!std::strcmp(m, "f2"))  return Mode::F2;
        if (!std::strcmp(m, "f2b")) return Mode::F2B;
        if (!std::strcmp(m, "f4"))  return Mode::F4;
    }
    return prefer_f4 ? Mode::F4 : Mode::F2B;
}

// Tile-block width for the blocked GEMM microkernel (number of tiles batched
// per winograd-domain multiply). 8 keeps the AVX2 accumulators in ymm registers
// while amortizing each U-row load across 8 tiles.
constexpr int TB = 8;

// ------------------------------------------------------------------------
// F(2x2,3x3) transforms (exact: halves + integers).
//   B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
//   G   = [[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]]
//   A^T = [[1,1,1,0],[0,1,-1,-1]]
// ------------------------------------------------------------------------
struct F2Policy {
    static constexpr int IT = 4, OT = 2, P = 16;   // input tile / output tile / positions

    // U = G g G^T, 3x3 -> 4x4 (u[16]).
    static void filt(const float g[9], float u[16]) {
        float Gg[4][3];
        for (int j = 0; j < 3; ++j) {
            float c0 = g[0*3 + j], c1 = g[1*3 + j], c2 = g[2*3 + j];
            Gg[0][j] = c0;
            Gg[1][j] = 0.5f * (c0 + c1 + c2);
            Gg[2][j] = 0.5f * (c0 - c1 + c2);
            Gg[3][j] = c2;
        }
        for (int i = 0; i < 4; ++i) {
            float c0 = Gg[i][0], c1 = Gg[i][1], c2 = Gg[i][2];
            u[i*4 + 0] = c0;
            u[i*4 + 1] = 0.5f * (c0 + c1 + c2);
            u[i*4 + 2] = 0.5f * (c0 - c1 + c2);
            u[i*4 + 3] = c2;
        }
    }
    // V = B^T d B, 4x4 -> 4x4 (v[16]).
    static void inp(const float d[16], float v[16]) {
        float m[16];
        for (int j = 0; j < 4; ++j) {
            float r0 = d[0*4 + j], r1 = d[1*4 + j], r2 = d[2*4 + j], r3 = d[3*4 + j];
            m[0*4 + j] = r0 - r2;
            m[1*4 + j] = r1 + r2;
            m[2*4 + j] = r2 - r1;
            m[3*4 + j] = r1 - r3;
        }
        for (int i = 0; i < 4; ++i) {
            float c0 = m[i*4 + 0], c1 = m[i*4 + 1], c2 = m[i*4 + 2], c3 = m[i*4 + 3];
            v[i*4 + 0] = c0 - c2;
            v[i*4 + 1] = c1 + c2;
            v[i*4 + 2] = c2 - c1;
            v[i*4 + 3] = c1 - c3;
        }
    }
    // Y = A^T m A, 4x4 -> 2x2 (y[4]).
    static void outp(const float m[16], float y[4]) {
        float p[8];
        for (int j = 0; j < 4; ++j) {
            float r0 = m[0*4 + j], r1 = m[1*4 + j], r2 = m[2*4 + j], r3 = m[3*4 + j];
            p[0*4 + j] = r0 + r1 + r2;
            p[1*4 + j] = r1 - r2 - r3;
        }
        for (int i = 0; i < 2; ++i) {
            float c0 = p[i*4 + 0], c1 = p[i*4 + 1], c2 = p[i*4 + 2], c3 = p[i*4 + 3];
            y[i*2 + 0] = c0 + c1 + c2;
            y[i*2 + 1] = c1 - c2 - c3;
        }
    }

#if defined(__AVX2__)
    // AVX2 input/output transforms across 8 lanes (8 channels or 8 ocs) at once.
    // Each ymm lane runs the EXACT same add/sub sequence (and association) as the
    // scalar transform above, so the per-lane result is bit-identical to scalar:
    // the only vectorization is lane-parallelism over the channel/oc dimension,
    // never a cross-lane reduction, so no f32 reassociation is introduced here.
    static inline void inp_avx2(const __m256 d[16], __m256 v[16]) {
        __m256 m[16];
        for (int j = 0; j < 4; ++j) {
            __m256 r0 = d[0*4 + j], r1 = d[1*4 + j], r2 = d[2*4 + j], r3 = d[3*4 + j];
            m[0*4 + j] = _mm256_sub_ps(r0, r2);
            m[1*4 + j] = _mm256_add_ps(r1, r2);
            m[2*4 + j] = _mm256_sub_ps(r2, r1);
            m[3*4 + j] = _mm256_sub_ps(r1, r3);
        }
        for (int i = 0; i < 4; ++i) {
            __m256 c0 = m[i*4 + 0], c1 = m[i*4 + 1], c2 = m[i*4 + 2], c3 = m[i*4 + 3];
            v[i*4 + 0] = _mm256_sub_ps(c0, c2);
            v[i*4 + 1] = _mm256_add_ps(c1, c2);
            v[i*4 + 2] = _mm256_sub_ps(c2, c1);
            v[i*4 + 3] = _mm256_sub_ps(c1, c3);
        }
    }
    static inline void outp_avx2(const __m256 m[16], __m256 y[4]) {
        __m256 p[8];
        for (int j = 0; j < 4; ++j) {
            __m256 r0 = m[0*4 + j], r1 = m[1*4 + j], r2 = m[2*4 + j], r3 = m[3*4 + j];
            p[0*4 + j] = _mm256_add_ps(_mm256_add_ps(r0, r1), r2);
            p[1*4 + j] = _mm256_sub_ps(_mm256_sub_ps(r1, r2), r3);
        }
        for (int i = 0; i < 2; ++i) {
            __m256 c0 = p[i*4 + 0], c1 = p[i*4 + 1], c2 = p[i*4 + 2], c3 = p[i*4 + 3];
            y[i*2 + 0] = _mm256_add_ps(_mm256_add_ps(c0, c1), c2);
            y[i*2 + 1] = _mm256_sub_ps(_mm256_sub_ps(c1, c2), c3);
        }
    }
#endif
};

// ------------------------------------------------------------------------
// F(4x4,3x3) transforms (Lavin & Gray). 6x6 input tile -> 4x4 output, 36
// winograd positions. Uses 1/6 and 1/24, so float32 accuracy is lower than F2.
// ------------------------------------------------------------------------
struct F4Policy {
    static constexpr int IT = 6, OT = 4, P = 36;

    static inline void Brow(const float x[6], float r[6]) {
        r[0] = 4.0f*x[0] - 5.0f*x[2] + x[4];
        r[1] = -4.0f*x[1] - 4.0f*x[2] + x[3] + x[4];
        r[2] = 4.0f*x[1] - 4.0f*x[2] - x[3] + x[4];
        r[3] = -2.0f*x[1] - x[2] + 2.0f*x[3] + x[4];
        r[4] = 2.0f*x[1] - x[2] - 2.0f*x[3] + x[4];
        r[5] = 4.0f*x[1] - 5.0f*x[3] + x[5];
    }
    static inline void Grow(const float y[3], float u[6]) {
        const float a = y[0], b = y[1], c = y[2];
        u[0] = 0.25f * a;
        u[1] = -(a + b + c) * (1.0f/6.0f);
        u[2] = (-a + b - c) * (1.0f/6.0f);
        u[3] = a*(1.0f/24.0f) + b*(1.0f/12.0f) + c*(1.0f/6.0f);
        u[4] = a*(1.0f/24.0f) - b*(1.0f/12.0f) + c*(1.0f/6.0f);
        u[5] = c;
    }
    static inline void Arow(const float m[6], float o[4]) {
        o[0] = m[0] + m[1] + m[2] + m[3] + m[4];
        o[1] = m[1] - m[2] + 2.0f*m[3] - 2.0f*m[4];
        o[2] = m[1] + m[2] + 4.0f*m[3] + 4.0f*m[4];
        o[3] = m[1] - m[2] + 8.0f*m[3] - 8.0f*m[4] + m[5];
    }

    // U = G g G^T, 3x3 -> 6x6 (u[36]).
    static void filt(const float g[9], float u[36]) {
        float Gg[6][3];
        for (int j = 0; j < 3; ++j) {
            float col[3] = { g[0*3 + j], g[1*3 + j], g[2*3 + j] };
            float out[6]; Grow(col, out);
            for (int i = 0; i < 6; ++i) Gg[i][j] = out[i];
        }
        for (int i = 0; i < 6; ++i) {
            float out[6]; Grow(Gg[i], out);
            for (int k = 0; k < 6; ++k) u[i*6 + k] = out[k];
        }
    }
    // V = B^T d B, 6x6 -> 6x6 (v[36]).
    static void inp(const float d[36], float v[36]) {
        float m[36];
        for (int j = 0; j < 6; ++j) {
            float col[6] = { d[0*6+j], d[1*6+j], d[2*6+j], d[3*6+j], d[4*6+j], d[5*6+j] };
            float out[6]; Brow(col, out);
            for (int i = 0; i < 6; ++i) m[i*6 + j] = out[i];
        }
        for (int i = 0; i < 6; ++i) {
            float out[6]; Brow(m + i*6, out);
            for (int k = 0; k < 6; ++k) v[i*6 + k] = out[k];
        }
    }
    // Y = A^T m A, 6x6 -> 4x4 (y[16]).
    static void outp(const float m[36], float y[16]) {
        float p[24];
        for (int j = 0; j < 6; ++j) {
            float col[6] = { m[0*6+j], m[1*6+j], m[2*6+j], m[3*6+j], m[4*6+j], m[5*6+j] };
            float out[4]; Arow(col, out);
            for (int i = 0; i < 4; ++i) p[i*6 + j] = out[i];
        }
        for (int i = 0; i < 4; ++i) {
            float out[4]; Arow(p + i*6, out);
            for (int k = 0; k < 4; ++k) y[i*4 + k] = out[k];
        }
    }

#if defined(__AVX2__)
    // AVX2 row transforms across 8 lanes (8 channels for inp, 8 ocs for outp).
    // Each lane runs the same B/A row transform as the scalar Brow/Arow; the
    // constant multiplies are folded into FMAs (fnmadd = c - a*b), so per-lane
    // values can differ from the scalar non-fused form by a few f32 ULPs -- F4 is
    // already the less-accurate (1/6,1/24) path, and this is gated downstream.
    // No cross-lane reduction is performed; the lanes are fully independent.
    static inline void Brow_avx2(const __m256 x[6], __m256 r[6]) {
        const __m256 c2 = _mm256_set1_ps(2.0f);
        const __m256 c4 = _mm256_set1_ps(4.0f);
        const __m256 c5 = _mm256_set1_ps(5.0f);
        r[0] = _mm256_fnmadd_ps(c5, x[2], _mm256_fmadd_ps(c4, x[0], x[4]));
        r[1] = _mm256_fnmadd_ps(c4, x[2], _mm256_fnmadd_ps(c4, x[1], _mm256_add_ps(x[3], x[4])));
        r[2] = _mm256_fnmadd_ps(c4, x[2], _mm256_fmadd_ps(c4, x[1], _mm256_sub_ps(x[4], x[3])));
        r[3] = _mm256_fnmadd_ps(c2, x[1], _mm256_fmadd_ps(c2, x[3], _mm256_sub_ps(x[4], x[2])));
        r[4] = _mm256_fnmadd_ps(c2, x[3], _mm256_fmadd_ps(c2, x[1], _mm256_sub_ps(x[4], x[2])));
        r[5] = _mm256_fnmadd_ps(c5, x[3], _mm256_fmadd_ps(c4, x[1], x[5]));
    }
    static inline void Arow_avx2(const __m256 m[6], __m256 o[4]) {
        const __m256 c2 = _mm256_set1_ps(2.0f);
        const __m256 c4 = _mm256_set1_ps(4.0f);
        const __m256 c8 = _mm256_set1_ps(8.0f);
        o[0] = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(m[0], m[1]), m[2]), m[3]), m[4]);
        o[1] = _mm256_fnmadd_ps(c2, m[4], _mm256_fmadd_ps(c2, m[3], _mm256_sub_ps(m[1], m[2])));
        o[2] = _mm256_fmadd_ps(c4, m[4], _mm256_fmadd_ps(c4, m[3], _mm256_add_ps(m[1], m[2])));
        o[3] = _mm256_add_ps(_mm256_fnmadd_ps(c8, m[4], _mm256_fmadd_ps(c8, m[3], _mm256_sub_ps(m[1], m[2]))), m[5]);
    }
    // V = B^T d B, 6x6 -> 6x6 (v[36]), 8 channels per ymm lane.
    static inline void inp_avx2(const __m256 d[36], __m256 v[36]) {
        __m256 m[36];
        for (int j = 0; j < 6; ++j) {
            __m256 col[6] = { d[0*6+j], d[1*6+j], d[2*6+j], d[3*6+j], d[4*6+j], d[5*6+j] };
            __m256 out[6]; Brow_avx2(col, out);
            for (int i = 0; i < 6; ++i) m[i*6 + j] = out[i];
        }
        for (int i = 0; i < 6; ++i) {
            __m256 out[6]; Brow_avx2(m + i*6, out);
            for (int k = 0; k < 6; ++k) v[i*6 + k] = out[k];
        }
    }
    // Y = A^T m A, 6x6 -> 4x4 (y[16]), 8 ocs per ymm lane.
    static inline void outp_avx2(const __m256 m[36], __m256 y[16]) {
        __m256 p[24];
        for (int j = 0; j < 6; ++j) {
            __m256 col[6] = { m[0*6+j], m[1*6+j], m[2*6+j], m[3*6+j], m[4*6+j], m[5*6+j] };
            __m256 out[4]; Arow_avx2(col, out);
            for (int i = 0; i < 4; ++i) p[i*6 + j] = out[i];
        }
        for (int i = 0; i < 4; ++i) {
            __m256 out[4]; Arow_avx2(p + i*6, out);
            for (int k = 0; k < 4; ++k) y[i*4 + k] = out[k];
        }
    }
#endif
};

// ------------------------------------------------------------------------
// Persistent per-op state: caches the filter transform U (computed once from
// w->data; reused across forwards). Scratch (V,M) is per-thread.
// ------------------------------------------------------------------------
struct WinogradState {
    Mode mode = Mode::F2B;
    int W = 0, H = 0, IC = 0, OC = 0, N = 0, pad = 0;
    int Wout = 0, Hout = 0, tilesX = 0, tilesY = 0;
    // Fused activation: when set, the output (inverse) transform store folds in
    // bias[oc] (from dst->src[2], may be null) + relu, removing a separate
    // ggml_add + ggml_relu pass over the [Wout,Hout,OC,N] output.
    bool fuse_relu = false;
    const void* wdata = nullptr;
    // U layout: U[pos*IC*OC + ic*OC + oc], pos in 0..P-1. OC innermost so the
    // winograd-domain multiply vectorizes over OC.
    std::vector<float> U;
    std::once_flag once;
};

template<class Pol>
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
// Per-tile GEMV (mode "f2"): M[oc] = sum_ic U[ic,oc]*V[ic].
// ------------------------------------------------------------------------
static inline void wino_gemv(const float* Upos, const float* Vpos, float* out, int IC, int OC) {
#if defined(__AVX512F__)
    int oc = 0;
    for (; oc + 16 <= OC; oc += 16) {
        __m512 acc = _mm512_setzero_ps();
        const float* up = Upos + oc;
        for (int ic = 0; ic < IC; ++ic)
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(up + (size_t)ic * OC),
                                  _mm512_set1_ps(Vpos[ic]), acc);
        _mm512_storeu_ps(out + oc, acc);
    }
    if (oc < OC) {
        const int rem = OC - oc;
        const __mmask16 mask = (__mmask16)((1u << rem) - 1u);
        __m512 acc = _mm512_setzero_ps();
        const float* up = Upos + oc;
        for (int ic = 0; ic < IC; ++ic)
            acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, up + (size_t)ic * OC),
                                  _mm512_set1_ps(Vpos[ic]), acc);
        _mm512_mask_storeu_ps(out + oc, mask, acc);
    }
#elif defined(__AVX2__)
    int oc = 0;
    for (; oc + 8 <= OC; oc += 8) {
        __m256 acc = _mm256_setzero_ps();
        const float* up = Upos + oc;
        for (int ic = 0; ic < IC; ++ic)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(up + (size_t)ic * OC),
                                  _mm256_set1_ps(Vpos[ic]), acc);
        _mm256_storeu_ps(out + oc, acc);
    }
    for (; oc < OC; ++oc) {           // scalar tail for OC % 8
        float acc = 0.0f;
        const float* up = Upos + oc;
        for (int ic = 0; ic < IC; ++ic) acc += up[(size_t)ic * OC] * Vpos[ic];
        out[oc] = acc;
    }
#else
    for (int oc = 0; oc < OC; ++oc) out[oc] = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        const float vv = Vpos[ic];
        const float* up = Upos + (size_t)ic * OC;
        for (int oc = 0; oc < OC; ++oc) out[oc] += up[oc] * vv;
    }
#endif
}

// ------------------------------------------------------------------------
// Blocked GEMM microkernel for one winograd position:
//   M[t][oc] = sum_ic U[ic][oc] * V[ic][t],  t in [0,TBcur), oc in [0,OC).
// U: [IC][OC] row-major (OC innermost). V: [IC][TB] row-major. M: [TB][OC].
// Each loaded U-row is reused across all TBcur tiles -> far better arithmetic
// intensity than the per-tile GEMV.
// ------------------------------------------------------------------------
// The AVX2 microkernel is always compiled on an AVX2 base build: it is both the
// non-AVX512 path and the runtime fallback the dispatcher picks on AVX2-only CPUs
// (and under the FACEDETECT_DISABLE_AVX512 hook), so it must exist even when the
// AVX-512 microkernel is also compiled in.
#if defined(__AVX2__)
// Full-block (TBcur == TB == 8) AVX2 winograd-domain GEMM microkernel, register-
// blocked vendor-style. For each winograd position this computes
//   M[t][oc] = sum_ic U[ic][oc] * V[ic][t],   t in [0,8), oc in [0,OC).
// U: [IC][OC] row-major (OC innermost), V: [IC][8] row-major, M: [8][OC].
//
// Why this shape. The reduction is over IC; the two output axes are OC (the
// 8-wide SIMD lane, contiguous in U and M) and the 8 tiles (broadcast from V).
// The kernel keeps the accumulators resident in ymm registers across the whole
// IC reduction (the previous code bounded the tile loop on the runtime TBcur, so
// the compiler spilled the accumulator array to the stack -> ~3x slower). It is
// tiled NR x MR over (OC-vectors) x (tiles): the widest tile, k24x8, holds a
// 3(OC-vec) x 4(tile) = 12-accumulator register block and runs each tile group
// (tiles 0-3, 4-7) so all 8 tiles are covered. Three OC-vectors share ONE V
// broadcast per tile (1 broadcast feeds 3 FMAs) and twelve independent FMA
// accumulators hide the FMA latency, which lifts this to ~140 GFLOP/s single-core
// (~0.9x the AVX2 FMA roofline) vs ~120 for the 1-OC-vector form and ~42 for the
// spilling generic path. OC tails fall to 16- then 8-wide blocks then a scalar
// 8-tile remainder. Every block sums in ascending ic, identical to the generic
// path, so the result is bit-for-bit parity-exact. SCRFD/ArcFace tile counts are
// all multiples of 8, so this full-block path takes essentially every block.

// 3 OC-vectors (24 ocs) x 4 tiles (group g -> tiles 4g..4g+3): 12-acc block.
static inline void gemm_k24x4(const float* U, const float* V, float* M,
                              int IC, int OC, int oc, int g) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
    __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
    const float* up = U + oc;
    const int base = g * 4;
    for (int ic = 0; ic < IC; ++ic) {
        const float* ur = up + (size_t)ic * OC;
        const __m256 u0 = _mm256_loadu_ps(ur);
        const __m256 u1 = _mm256_loadu_ps(ur + 8);
        const __m256 u2 = _mm256_loadu_ps(ur + 16);
        const float* vp = V + (size_t)ic * TB + base;
        __m256 bc;
        bc = _mm256_broadcast_ss(vp + 0); a0 = _mm256_fmadd_ps(u0, bc, a0); b0 = _mm256_fmadd_ps(u1, bc, b0); c0 = _mm256_fmadd_ps(u2, bc, c0);
        bc = _mm256_broadcast_ss(vp + 1); a1 = _mm256_fmadd_ps(u0, bc, a1); b1 = _mm256_fmadd_ps(u1, bc, b1); c1 = _mm256_fmadd_ps(u2, bc, c1);
        bc = _mm256_broadcast_ss(vp + 2); a2 = _mm256_fmadd_ps(u0, bc, a2); b2 = _mm256_fmadd_ps(u1, bc, b2); c2 = _mm256_fmadd_ps(u2, bc, c2);
        bc = _mm256_broadcast_ss(vp + 3); a3 = _mm256_fmadd_ps(u0, bc, a3); b3 = _mm256_fmadd_ps(u1, bc, b3); c3 = _mm256_fmadd_ps(u2, bc, c3);
    }
    float* m;
    m = M + (size_t)(base + 0) * OC + oc; _mm256_storeu_ps(m, a0); _mm256_storeu_ps(m + 8, b0); _mm256_storeu_ps(m + 16, c0);
    m = M + (size_t)(base + 1) * OC + oc; _mm256_storeu_ps(m, a1); _mm256_storeu_ps(m + 8, b1); _mm256_storeu_ps(m + 16, c1);
    m = M + (size_t)(base + 2) * OC + oc; _mm256_storeu_ps(m, a2); _mm256_storeu_ps(m + 8, b2); _mm256_storeu_ps(m + 16, c2);
    m = M + (size_t)(base + 3) * OC + oc; _mm256_storeu_ps(m, a3); _mm256_storeu_ps(m + 8, b3); _mm256_storeu_ps(m + 16, c3);
}

// 2 OC-vectors (16 ocs) x 4 tiles (group g): 8-acc block, for the 16-wide tail.
static inline void gemm_k16x4(const float* U, const float* V, float* M,
                              int IC, int OC, int oc, int g) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
    __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
    const float* up = U + oc;
    const int base = g * 4;
    for (int ic = 0; ic < IC; ++ic) {
        const float* ur = up + (size_t)ic * OC;
        const __m256 u0 = _mm256_loadu_ps(ur);
        const __m256 u1 = _mm256_loadu_ps(ur + 8);
        const float* vp = V + (size_t)ic * TB + base;
        __m256 bc;
        bc = _mm256_broadcast_ss(vp + 0); a0 = _mm256_fmadd_ps(u0, bc, a0); b0 = _mm256_fmadd_ps(u1, bc, b0);
        bc = _mm256_broadcast_ss(vp + 1); a1 = _mm256_fmadd_ps(u0, bc, a1); b1 = _mm256_fmadd_ps(u1, bc, b1);
        bc = _mm256_broadcast_ss(vp + 2); a2 = _mm256_fmadd_ps(u0, bc, a2); b2 = _mm256_fmadd_ps(u1, bc, b2);
        bc = _mm256_broadcast_ss(vp + 3); a3 = _mm256_fmadd_ps(u0, bc, a3); b3 = _mm256_fmadd_ps(u1, bc, b3);
    }
    float* m;
    m = M + (size_t)(base + 0) * OC + oc; _mm256_storeu_ps(m, a0); _mm256_storeu_ps(m + 8, b0);
    m = M + (size_t)(base + 1) * OC + oc; _mm256_storeu_ps(m, a1); _mm256_storeu_ps(m + 8, b1);
    m = M + (size_t)(base + 2) * OC + oc; _mm256_storeu_ps(m, a2); _mm256_storeu_ps(m + 8, b2);
    m = M + (size_t)(base + 3) * OC + oc; _mm256_storeu_ps(m, a3); _mm256_storeu_ps(m + 8, b3);
}

// 1 OC-vector (8 ocs) x 8 tiles: 8-acc block, for the 8-wide tail.
static inline void gemm_k8x8(const float* U, const float* V, float* M,
                             int IC, int OC, int oc) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
    __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();
    const float* up = U + oc;
    for (int ic = 0; ic < IC; ++ic) {
        const __m256 u = _mm256_loadu_ps(up + (size_t)ic * OC);
        const float* vp = V + (size_t)ic * TB;
        a0 = _mm256_fmadd_ps(u, _mm256_broadcast_ss(vp + 0), a0);
        a1 = _mm256_fmadd_ps(u, _mm256_broadcast_ss(vp + 1), a1);
        a2 = _mm256_fmadd_ps(u, _mm256_broadcast_ss(vp + 2), a2);
        a3 = _mm256_fmadd_ps(u, _mm256_broadcast_ss(vp + 3), a3);
        a4 = _mm256_fmadd_ps(u, _mm256_broadcast_ss(vp + 4), a4);
        a5 = _mm256_fmadd_ps(u, _mm256_broadcast_ss(vp + 5), a5);
        a6 = _mm256_fmadd_ps(u, _mm256_broadcast_ss(vp + 6), a6);
        a7 = _mm256_fmadd_ps(u, _mm256_broadcast_ss(vp + 7), a7);
    }
    _mm256_storeu_ps(M + 0 * OC + oc, a0); _mm256_storeu_ps(M + 1 * OC + oc, a1);
    _mm256_storeu_ps(M + 2 * OC + oc, a2); _mm256_storeu_ps(M + 3 * OC + oc, a3);
    _mm256_storeu_ps(M + 4 * OC + oc, a4); _mm256_storeu_ps(M + 5 * OC + oc, a5);
    _mm256_storeu_ps(M + 6 * OC + oc, a6); _mm256_storeu_ps(M + 7 * OC + oc, a7);
}

static inline void gemm_block_avx2_t8(const float* U, const float* V, float* M,
                                      int IC, int OC) {
    int oc = 0;
    for (; oc + 24 <= OC; oc += 24) { gemm_k24x4(U, V, M, IC, OC, oc, 0); gemm_k24x4(U, V, M, IC, OC, oc, 1); }
    for (; oc + 16 <= OC; oc += 16) { gemm_k16x4(U, V, M, IC, OC, oc, 0); gemm_k16x4(U, V, M, IC, OC, oc, 1); }
    for (; oc + 8  <= OC; oc += 8)  { gemm_k8x8(U, V, M, IC, OC, oc); }
    for (; oc < OC; ++oc) {              // OC % 8 scalar tail, 8 tiles unrolled
        float c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0;
        const float* up = U + oc;
        for (int ic = 0; ic < IC; ++ic) {
            const float uu = up[(size_t)ic * OC];
            const float* vp = V + (size_t)ic * TB;
            c0 += uu * vp[0]; c1 += uu * vp[1]; c2 += uu * vp[2]; c3 += uu * vp[3];
            c4 += uu * vp[4]; c5 += uu * vp[5]; c6 += uu * vp[6]; c7 += uu * vp[7];
        }
        M[0 * OC + oc] = c0; M[1 * OC + oc] = c1; M[2 * OC + oc] = c2; M[3 * OC + oc] = c3;
        M[4 * OC + oc] = c4; M[5 * OC + oc] = c5; M[6 * OC + oc] = c6; M[7 * OC + oc] = c7;
    }
}
#endif

#if defined(FD_WINO_HAVE_AVX512)
// Full-block (TBcur == TB == 8) AVX-512 winograd-domain GEMM microkernel: the
// zmm evolution of the AVX2 gemm_k24x4 register-blocked kernel above. Each
// function carries FD_WINO_AVX512_TARGET = __attribute__((target("avx512f,
// avx512bw,avx512vl"))) so the compiler emits its zmm code even though this TU is
// NOT given a global -mavx512f; wino_gemm_block only calls it when
// wino_use_avx512() (runtime CPUID) is true, so AVX2-only hosts never reach a zmm
// instruction. ggml itself stays AVX2 (GGML_NATIVE=OFF), so the AVX-512 win is
// surgically isolated to the compute-bound (~115 GFLOP/s) winograd GEMM and never
// touches ggml's bandwidth-bound ops (where global AVX-512 regressed earlier).
//
// Shape mirrors the AVX2 kernel but with 16-wide zmm OC lanes instead of 8-wide
// ymm: the widest tile k48x4 holds a 3(OC-vec=48 ocs) x 4(tile) = 12-zmm block,
// one V broadcast feeds 3 FMAs, twelve independent zmm accumulators hide the
// 4-cycle FMA latency across Zen5's two 512-bit FMA pipes. Each (t,oc) lane sums
// in ascending ic with _mm512_fmadd_ps over the identical operand sequence as the
// AVX2 _mm256_fmadd_ps kernel, so the result is bit-identical to the AVX2 path
// (FMA is deterministic; widening the SIMD lane never reassociates a per-lane
// reduction). OC tails fall to 32- then 16-wide blocks, then a single masked
// 16-wide block (_mm512_maskz_loadu_ps / _mm512_mask_storeu_ps, mirroring the
// existing AVX-512 transform tails) for OC % 16 -- no scalar OC tail.

// 3 OC-vectors (48 ocs) x 4 tiles (group g -> tiles 4g..4g+3): 12-zmm block.
static inline FD_WINO_AVX512_TARGET void gemm_k48x4(const float* U, const float* V, float* M,
                              int IC, int OC, int oc, int g) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps();
    __m512 b2 = _mm512_setzero_ps(), b3 = _mm512_setzero_ps();
    __m512 c0 = _mm512_setzero_ps(), c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps(), c3 = _mm512_setzero_ps();
    const float* up = U + oc;
    const int base = g * 4;
    for (int ic = 0; ic < IC; ++ic) {
        const float* ur = up + (size_t)ic * OC;
        const __m512 u0 = _mm512_loadu_ps(ur);
        const __m512 u1 = _mm512_loadu_ps(ur + 16);
        const __m512 u2 = _mm512_loadu_ps(ur + 32);
        const float* vp = V + (size_t)ic * TB + base;
        __m512 bc;
        bc = _mm512_set1_ps(vp[0]); a0 = _mm512_fmadd_ps(u0, bc, a0); b0 = _mm512_fmadd_ps(u1, bc, b0); c0 = _mm512_fmadd_ps(u2, bc, c0);
        bc = _mm512_set1_ps(vp[1]); a1 = _mm512_fmadd_ps(u0, bc, a1); b1 = _mm512_fmadd_ps(u1, bc, b1); c1 = _mm512_fmadd_ps(u2, bc, c1);
        bc = _mm512_set1_ps(vp[2]); a2 = _mm512_fmadd_ps(u0, bc, a2); b2 = _mm512_fmadd_ps(u1, bc, b2); c2 = _mm512_fmadd_ps(u2, bc, c2);
        bc = _mm512_set1_ps(vp[3]); a3 = _mm512_fmadd_ps(u0, bc, a3); b3 = _mm512_fmadd_ps(u1, bc, b3); c3 = _mm512_fmadd_ps(u2, bc, c3);
    }
    float* m;
    m = M + (size_t)(base + 0) * OC + oc; _mm512_storeu_ps(m, a0); _mm512_storeu_ps(m + 16, b0); _mm512_storeu_ps(m + 32, c0);
    m = M + (size_t)(base + 1) * OC + oc; _mm512_storeu_ps(m, a1); _mm512_storeu_ps(m + 16, b1); _mm512_storeu_ps(m + 32, c1);
    m = M + (size_t)(base + 2) * OC + oc; _mm512_storeu_ps(m, a2); _mm512_storeu_ps(m + 16, b2); _mm512_storeu_ps(m + 32, c2);
    m = M + (size_t)(base + 3) * OC + oc; _mm512_storeu_ps(m, a3); _mm512_storeu_ps(m + 16, b3); _mm512_storeu_ps(m + 32, c3);
}

// 2 OC-vectors (32 ocs) x 4 tiles (group g): 8-zmm block, for the 32-wide tail.
static inline FD_WINO_AVX512_TARGET void gemm_k32x4(const float* U, const float* V, float* M,
                              int IC, int OC, int oc, int g) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps();
    __m512 b2 = _mm512_setzero_ps(), b3 = _mm512_setzero_ps();
    const float* up = U + oc;
    const int base = g * 4;
    for (int ic = 0; ic < IC; ++ic) {
        const float* ur = up + (size_t)ic * OC;
        const __m512 u0 = _mm512_loadu_ps(ur);
        const __m512 u1 = _mm512_loadu_ps(ur + 16);
        const float* vp = V + (size_t)ic * TB + base;
        __m512 bc;
        bc = _mm512_set1_ps(vp[0]); a0 = _mm512_fmadd_ps(u0, bc, a0); b0 = _mm512_fmadd_ps(u1, bc, b0);
        bc = _mm512_set1_ps(vp[1]); a1 = _mm512_fmadd_ps(u0, bc, a1); b1 = _mm512_fmadd_ps(u1, bc, b1);
        bc = _mm512_set1_ps(vp[2]); a2 = _mm512_fmadd_ps(u0, bc, a2); b2 = _mm512_fmadd_ps(u1, bc, b2);
        bc = _mm512_set1_ps(vp[3]); a3 = _mm512_fmadd_ps(u0, bc, a3); b3 = _mm512_fmadd_ps(u1, bc, b3);
    }
    float* m;
    m = M + (size_t)(base + 0) * OC + oc; _mm512_storeu_ps(m, a0); _mm512_storeu_ps(m + 16, b0);
    m = M + (size_t)(base + 1) * OC + oc; _mm512_storeu_ps(m, a1); _mm512_storeu_ps(m + 16, b1);
    m = M + (size_t)(base + 2) * OC + oc; _mm512_storeu_ps(m, a2); _mm512_storeu_ps(m + 16, b2);
    m = M + (size_t)(base + 3) * OC + oc; _mm512_storeu_ps(m, a3); _mm512_storeu_ps(m + 16, b3);
}

// 1 OC-vector (16 ocs) x 8 tiles: 8-zmm block, for the 16-wide tail.
static inline FD_WINO_AVX512_TARGET void gemm_k16x8(const float* U, const float* V, float* M,
                              int IC, int OC, int oc) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    __m512 a4 = _mm512_setzero_ps(), a5 = _mm512_setzero_ps();
    __m512 a6 = _mm512_setzero_ps(), a7 = _mm512_setzero_ps();
    const float* up = U + oc;
    for (int ic = 0; ic < IC; ++ic) {
        const __m512 u = _mm512_loadu_ps(up + (size_t)ic * OC);
        const float* vp = V + (size_t)ic * TB;
        a0 = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[0]), a0);
        a1 = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[1]), a1);
        a2 = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[2]), a2);
        a3 = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[3]), a3);
        a4 = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[4]), a4);
        a5 = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[5]), a5);
        a6 = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[6]), a6);
        a7 = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[7]), a7);
    }
    _mm512_storeu_ps(M + 0 * OC + oc, a0); _mm512_storeu_ps(M + 1 * OC + oc, a1);
    _mm512_storeu_ps(M + 2 * OC + oc, a2); _mm512_storeu_ps(M + 3 * OC + oc, a3);
    _mm512_storeu_ps(M + 4 * OC + oc, a4); _mm512_storeu_ps(M + 5 * OC + oc, a5);
    _mm512_storeu_ps(M + 6 * OC + oc, a6); _mm512_storeu_ps(M + 7 * OC + oc, a7);
}

// Masked 1 OC-vector (OC % 16 ocs) x 8 tiles: 8-zmm block, the OC tail. Masked
// load/store keep reads/writes inside [oc,OC) (no scalar remainder, no overrun
// of the U/M buffers on the final ic), mirroring the existing AVX-512 transform
// tails. Same ascending-ic FMA chain as the full-width blocks.
static inline FD_WINO_AVX512_TARGET void gemm_k16x8_masked(const float* U, const float* V, float* M,
                                     int IC, int OC, int oc, __mmask16 mask) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    __m512 a4 = _mm512_setzero_ps(), a5 = _mm512_setzero_ps();
    __m512 a6 = _mm512_setzero_ps(), a7 = _mm512_setzero_ps();
    const float* up = U + oc;
    for (int ic = 0; ic < IC; ++ic) {
        const __m512 u = _mm512_maskz_loadu_ps(mask, up + (size_t)ic * OC);
        const float* vp = V + (size_t)ic * TB;
        a0 = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[0]), a0);
        a1 = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[1]), a1);
        a2 = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[2]), a2);
        a3 = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[3]), a3);
        a4 = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[4]), a4);
        a5 = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[5]), a5);
        a6 = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[6]), a6);
        a7 = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[7]), a7);
    }
    _mm512_mask_storeu_ps(M + 0 * OC + oc, mask, a0); _mm512_mask_storeu_ps(M + 1 * OC + oc, mask, a1);
    _mm512_mask_storeu_ps(M + 2 * OC + oc, mask, a2); _mm512_mask_storeu_ps(M + 3 * OC + oc, mask, a3);
    _mm512_mask_storeu_ps(M + 4 * OC + oc, mask, a4); _mm512_mask_storeu_ps(M + 5 * OC + oc, mask, a5);
    _mm512_mask_storeu_ps(M + 6 * OC + oc, mask, a6); _mm512_mask_storeu_ps(M + 7 * OC + oc, mask, a7);
}

static inline FD_WINO_AVX512_TARGET void gemm_block_avx512_t8(const float* U, const float* V, float* M,
                                        int IC, int OC) {
    int oc = 0;
    for (; oc + 48 <= OC; oc += 48) { gemm_k48x4(U, V, M, IC, OC, oc, 0); gemm_k48x4(U, V, M, IC, OC, oc, 1); }
    for (; oc + 32 <= OC; oc += 32) { gemm_k32x4(U, V, M, IC, OC, oc, 0); gemm_k32x4(U, V, M, IC, OC, oc, 1); }
    for (; oc + 16 <= OC; oc += 16) { gemm_k16x8(U, V, M, IC, OC, oc); }
    if (oc < OC) {
        const __mmask16 mask = (__mmask16)((1u << (OC - oc)) - 1u);
        gemm_k16x8_masked(U, V, M, IC, OC, oc, mask);
    }
}
#endif

static inline void wino_gemm_block(const float* U, const float* V, float* M,
                                   int IC, int OC, int TBcur) {
    // Full-block fast path (all SCRFD/ArcFace tile counts are multiples of TB, so
    // this takes essentially every block). Runtime-dispatched: the AVX-512 zmm
    // microkernel only when the CPU advertises AVX-512 (wino_use_avx512()), else
    // the AVX2 microkernel -- one binary, no SIGILL on AVX2-only hosts.
#if defined(__AVX2__)
    if (TBcur == TB) {
#if defined(FD_WINO_HAVE_AVX512)
        if (wino_use_avx512()) { gemm_block_avx512_t8(U, V, M, IC, OC); return; }
#endif
        gemm_block_avx2_t8(U, V, M, IC, OC);
        return;
    }
#endif
    // Generic per-TBcur path for the rare non-full block. Compile-time ISA only
    // (AVX-512 here is reached solely in a global -mavx512f build); the default
    // portable build runs the AVX2 generic path, which is safe everywhere.
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
#elif defined(__AVX2__)
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
    if (oc < OC) {                    // scalar tail for OC % 8 (same accum order)
        for (int t = 0; t < TBcur; ++t)
            for (int o = oc; o < OC; ++o) M[(size_t)t * OC + o] = 0.0f;
        for (int ic = 0; ic < IC; ++ic) {
            const float* up = U + (size_t)ic * OC;
            const float* vp = V + (size_t)ic * TB;
            for (int t = 0; t < TBcur; ++t) {
                const float vv = vp[t];
                float* mt = M + (size_t)t * OC;
                for (int o = oc; o < OC; ++o) mt[o] += up[o] * vv;
            }
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
// Original F(2x2) per-tile path (mode "f2").
// ------------------------------------------------------------------------
static void compute_f2_gemv(WinogradState* st, ggml_tensor* dst, int ith, int nth,
                            const float* bias) {
    const ggml_tensor* xt = dst->src[0];
    const float* x = (const float*)xt->data;
    float* y = (float*)dst->data;
    const bool fuse = st->fuse_relu;

    const int W = st->W, H = st->H, IC = st->IC, OC = st->OC, pad = st->pad;
    const int Wout = st->Wout, Hout = st->Hout;
    const int tilesX = st->tilesX, tilesY = st->tilesY;
    const float* U = st->U.data();

    const int ntiles = tilesX * tilesY;
    const int64_t total = (int64_t)st->N * ntiles;
    const int64_t beg = total * ith / nth;
    const int64_t end = total * (ith + 1) / nth;

    std::vector<float> Vbuf((size_t)16 * IC);
    std::vector<float> Mbuf((size_t)16 * OC);
    float dpatch[16], vpatch[16], mpatch[16], ypatch[4];

    for (int64_t idx = beg; idx < end; ++idx) {
        const int n  = (int)(idx / ntiles);
        const int t  = (int)(idx % ntiles);
        const int ty = t / tilesX, tx = t % tilesX;
        const int oy0 = ty * 2, ox0 = tx * 2;
        const int iy0 = oy0 - pad, ix0 = ox0 - pad;
        const float* xn = x + (size_t)n * IC * H * W;

        for (int ic = 0; ic < IC; ++ic) {
            const float* xc = xn + (size_t)ic * H * W;
            for (int i = 0; i < 4; ++i) {
                const int yy = iy0 + i;
                const bool yok = (yy >= 0 && yy < H);
                const float* row = yok ? (xc + (size_t)yy * W) : nullptr;
                for (int j = 0; j < 4; ++j) {
                    const int xx = ix0 + j;
                    dpatch[i*4 + j] = (yok && xx >= 0 && xx < W) ? row[xx] : 0.0f;
                }
            }
            F2Policy::inp(dpatch, vpatch);
            for (int pos = 0; pos < 16; ++pos) Vbuf[(size_t)pos * IC + ic] = vpatch[pos];
        }
        for (int pos = 0; pos < 16; ++pos)
            wino_gemv(U + (size_t)pos * IC * OC, Vbuf.data() + (size_t)pos * IC,
                      Mbuf.data() + (size_t)pos * OC, IC, OC);

        float* yn = y + (size_t)n * OC * Hout * Wout;
        for (int oc = 0; oc < OC; ++oc) {
            for (int pos = 0; pos < 16; ++pos) mpatch[pos] = Mbuf[(size_t)pos * OC + oc];
            F2Policy::outp(mpatch, ypatch);
            // Fuse bias[oc] + relu into the store (relu exact, bias one f32 add).
            const float bo = (fuse && bias) ? bias[oc] : 0.0f;
            float* yc = yn + (size_t)oc * Hout * Wout;
            for (int i = 0; i < 2; ++i) {
                const int oy = oy0 + i; if (oy >= Hout) continue;
                for (int j = 0; j < 2; ++j) {
                    const int ox = ox0 + j; if (ox >= Wout) continue;
                    float v = ypatch[i*2 + j];
                    if (fuse) { v += bo; if (v < 0.0f) v = 0.0f; }
                    yc[(size_t)oy * Wout + ox] = v;
                }
            }
        }
    }
}

// ------------------------------------------------------------------------
// Blocked path (modes "f2b" / "f4"): batch TB tiles per winograd-domain GEMM.
// ------------------------------------------------------------------------
template<class Pol>
static void compute_blocked(WinogradState* st, ggml_tensor* dst, int ith, int nth,
                            const float* bias) {
    constexpr int IT = Pol::IT, OT = Pol::OT, P = Pol::P;
    const ggml_tensor* xt = dst->src[0];
    const float* x = (const float*)xt->data;
    float* y = (float*)dst->data;
    const bool fuse = st->fuse_relu;

    const int W = st->W, H = st->H, IC = st->IC, OC = st->OC, pad = st->pad;
    const int Wout = st->Wout, Hout = st->Hout;
    const int tilesX = st->tilesX, tilesY = st->tilesY;
    const float* U = st->U.data();

    const int ntiles = tilesX * tilesY;
    const int64_t total = (int64_t)st->N * ntiles;
    const int64_t nblocks = (total + TB - 1) / TB;
    const int64_t bbeg = nblocks * ith / nth;
    const int64_t bend = nblocks * (ith + 1) / nth;

    std::vector<float> Vblk((size_t)P * IC * TB);   // V[pos][ic][t]
    std::vector<float> Mblk((size_t)P * TB * OC);   // M[pos][t][oc]
    float dpatch[IT*IT], vpatch[P], mpatch[P], ypatch[OT*OT];

    for (int64_t b = bbeg; b < bend; ++b) {
        const int64_t t0 = b * TB;
        const int TBcur = (int)std::min<int64_t>(TB, total - t0);

        // 1. Input transform for each tile in the block -> Vblk[pos][ic][t].
        for (int tb = 0; tb < TBcur; ++tb) {
            const int64_t idx = t0 + tb;
            const int n  = (int)(idx / ntiles);
            const int t  = (int)(idx % ntiles);
            const int ty = t / tilesX, tx = t % tilesX;
            const int iy0 = ty * OT - pad, ix0 = tx * OT - pad;
            const float* xn = x + (size_t)n * IC * H * W;
            int ic = 0;
#if defined(__AVX2__)
            // Vectorize the input transform across 8 channels at once: the tile's
            // spatial gather pattern (and zero-padding mask) is channel-independent,
            // so the same (i,j) gate drives all 8 lanes. inp_avx2 runs the exact
            // per-lane transform; the scatter writes lane k -> channel ic+k.
            const size_t HW = (size_t)H * W;
            for (; ic + 8 <= IC; ic += 8) {
                __m256 dvec[IT*IT];
                for (int i = 0; i < IT; ++i) {
                    const int yy = iy0 + i;
                    const bool yok = (yy >= 0 && yy < H);
                    for (int j = 0; j < IT; ++j) {
                        const int xx = ix0 + j;
                        if (yok && xx >= 0 && xx < W) {
                            const float* base = xn + (size_t)ic * HW + (size_t)yy * W + xx;
                            float lane[8];
                            for (int k = 0; k < 8; ++k) lane[k] = base[(size_t)k * HW];
                            dvec[i*IT + j] = _mm256_loadu_ps(lane);
                        } else {
                            dvec[i*IT + j] = _mm256_setzero_ps();
                        }
                    }
                }
                __m256 vvec[P];
                Pol::inp_avx2(dvec, vvec);
                float* vbase = Vblk.data() + (size_t)ic * TB + tb;
                for (int pos = 0; pos < P; ++pos) {
                    float lane[8];
                    _mm256_storeu_ps(lane, vvec[pos]);
                    float* vp = vbase + (size_t)pos * IC * TB;
                    for (int k = 0; k < 8; ++k) vp[(size_t)k * TB] = lane[k];
                }
            }
#endif
            for (; ic < IC; ++ic) {
                const float* xc = xn + (size_t)ic * H * W;
                for (int i = 0; i < IT; ++i) {
                    const int yy = iy0 + i;
                    const bool yok = (yy >= 0 && yy < H);
                    const float* row = yok ? (xc + (size_t)yy * W) : nullptr;
                    for (int j = 0; j < IT; ++j) {
                        const int xx = ix0 + j;
                        dpatch[i*IT + j] = (yok && xx >= 0 && xx < W) ? row[xx] : 0.0f;
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
            const int n  = (int)(idx / ntiles);
            const int t  = (int)(idx % ntiles);
            const int ty = t / tilesX, tx = t % tilesX;
            const int oy0 = ty * OT, ox0 = tx * OT;
            float* yn = y + (size_t)n * OC * Hout * Wout;
            const size_t OHW = (size_t)Hout * Wout;
            int oc = 0;
#if defined(__AVX2__)
            // Vectorize the output transform across 8 ocs: M is laid out oc-innermost
            // so mpatch[pos] for 8 ocs is one contiguous load. outp_avx2 runs the
            // exact per-lane transform; the spatial scatter mask is oc-independent.
            for (; oc + 8 <= OC; oc += 8) {
                __m256 mvec[P];
                const float* mbase = Mblk.data() + (size_t)tb * OC + oc;
                for (int pos = 0; pos < P; ++pos)
                    mvec[pos] = _mm256_loadu_ps(mbase + (size_t)pos * TB * OC);
                __m256 yvec[OT*OT];
                Pol::outp_avx2(mvec, yvec);
                // Fuse bias[oc..oc+7] + relu in-register before the spatial
                // scatter: yvec[p] lane k holds output[pos p][oc+k], so one bias
                // vector broadcasts across all OT*OT positions. Removes a separate
                // ggml_add + ggml_relu pass over the whole [Wout,Hout,OC,N] output.
                if (fuse) {
                    const __m256 bvec = bias ? _mm256_loadu_ps(bias + oc)
                                             : _mm256_setzero_ps();
                    const __m256 zero = _mm256_setzero_ps();
                    for (int p = 0; p < OT*OT; ++p)
                        yvec[p] = _mm256_max_ps(_mm256_add_ps(yvec[p], bvec), zero);
                }
                float ybuf[OT*OT][8];
                for (int p = 0; p < OT*OT; ++p) _mm256_storeu_ps(ybuf[p], yvec[p]);
                for (int k = 0; k < 8; ++k) {
                    float* yc = yn + (size_t)(oc + k) * OHW;
                    for (int i = 0; i < OT; ++i) {
                        const int oy = oy0 + i; if (oy >= Hout) continue;
                        for (int j = 0; j < OT; ++j) {
                            const int ox = ox0 + j; if (ox >= Wout) continue;
                            yc[(size_t)oy * Wout + ox] = ybuf[i*OT + j][k];
                        }
                    }
                }
            }
#endif
            for (; oc < OC; ++oc) {
                const float* mbase = Mblk.data() + (size_t)tb * OC + oc;
                for (int pos = 0; pos < P; ++pos)
                    mpatch[pos] = mbase[(size_t)pos * TB * OC];
                Pol::outp(mpatch, ypatch);
                const float bo = (fuse && bias) ? bias[oc] : 0.0f;
                float* yc = yn + (size_t)oc * Hout * Wout;
                for (int i = 0; i < OT; ++i) {
                    const int oy = oy0 + i; if (oy >= Hout) continue;
                    for (int j = 0; j < OT; ++j) {
                        const int ox = ox0 + j; if (ox >= Wout) continue;
                        float yv = ypatch[i*OT + j];
                        if (fuse) { yv += bo; if (yv < 0.0f) yv = 0.0f; }
                        yc[(size_t)oy * Wout + ox] = yv;
                    }
                }
            }
        }
    }
}

static void winograd_compute(ggml_tensor* dst, int ith, int nth, void* userdata) {
    WinogradState* st = (WinogradState*)userdata;
    const ggml_tensor* wt = dst->src[1];
    const float* w = (const float*)wt->data;
    // Fused bias rides as the optional third src (see winograd_conv3x3); only
    // consulted when st->fuse_relu. May be null (relu-only fusion -> bias 0).
    const float* bias = (st->fuse_relu && dst->src[2]) ? (const float*)dst->src[2]->data
                                                       : nullptr;

    switch (st->mode) {
        case Mode::F4:
            std::call_once(st->once, [&]{ build_U<F4Policy>(st, w); });
            compute_blocked<F4Policy>(st, dst, ith, nth, bias);
            break;
        case Mode::F2B:
            std::call_once(st->once, [&]{ build_U<F2Policy>(st, w); });
            compute_blocked<F2Policy>(st, dst, ith, nth, bias);
            break;
        case Mode::F2:
        default:
            std::call_once(st->once, [&]{ build_U<F2Policy>(st, w); });
            compute_f2_gemv(st, dst, ith, nth, bias);
            break;
    }
}

// ------------------------------------------------------------------------
// Keyed cache of op states (U transformed once per (filter,shape,mode)).
// ------------------------------------------------------------------------
struct StateKey {
    const void* wdata; int W, H, IC, OC, N, pad; int mode; int fuse;
    bool operator==(const StateKey& o) const {
        return wdata == o.wdata && W == o.W && H == o.H && IC == o.IC &&
               OC == o.OC && N == o.N && pad == o.pad && mode == o.mode &&
               fuse == o.fuse;
    }
};
struct StateKeyHash {
    size_t operator()(const StateKey& k) const {
        size_t h = (size_t)k.wdata;
        auto mix = [&h](int v) { h = h * 1000003u + (size_t)(uint32_t)v; };
        mix(k.W); mix(k.H); mix(k.IC); mix(k.OC); mix(k.N); mix(k.pad); mix(k.mode); mix(k.fuse);
        return h;
    }
};

static std::mutex g_states_mtx;
static std::unordered_map<StateKey, WinogradState*, StateKeyHash> g_states;

static int out_tile(Mode m) { return m == Mode::F4 ? 4 : 2; }

static WinogradState* get_state(ggml_tensor* w, ggml_tensor* x, int pad, Mode mode,
                                bool fuse_relu) {
    const int W = (int)x->ne[0], H = (int)x->ne[1], IC = (int)x->ne[2], N = (int)x->ne[3];
    const int OC = (int)w->ne[3];
    StateKey key{ w->data, W, H, IC, OC, N, pad, (int)mode, fuse_relu ? 1 : 0 };
    std::lock_guard<std::mutex> lk(g_states_mtx);
    auto it = g_states.find(key);
    if (it != g_states.end()) return it->second;
    WinogradState* st = new WinogradState();
    st->mode = mode;
    st->fuse_relu = fuse_relu;
    st->W = W; st->H = H; st->IC = IC; st->OC = OC; st->N = N; st->pad = pad;
    st->Wout = W + 2 * pad - 2;
    st->Hout = H + 2 * pad - 2;
    const int OT = out_tile(mode);
    st->tilesX = (st->Wout + OT - 1) / OT;
    st->tilesY = (st->Hout + OT - 1) / OT;
    st->wdata = w->data;
    g_states[key] = st;
    return st;
}

} // namespace

ggml_tensor* winograd_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x,
                              int pad, bool prefer_f4, ggml_tensor* bias, bool relu) {
    const int OC = (int)w->ne[3];
    const int N  = (int)x->ne[3];
    const int Wout = (int)x->ne[0] + 2 * pad - 2;
    const int Hout = (int)x->ne[1] + 2 * pad - 2;
    const Mode mode = parse_mode(prefer_f4);
    // Fuse bias+relu into the output transform store only when relu is requested
    // (the conv 'activation' is known at graph-build time). bias is optional: when
    // present it rides as a third src so ggml materializes its data; relu-only
    // (bias == nullptr) still fuses the max(0,x).
    WinogradState* st = get_state(w, x, pad, mode, relu);
    if (relu && bias) {
        ggml_tensor* args[3] = { x, w, bias };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, Wout, Hout, OC, N,
                              args, 3, winograd_compute, GGML_N_TASKS_MAX, st);
    }
    ggml_tensor* args[2] = { x, w };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, Wout, Hout, OC, N,
                          args, 2, winograd_compute, GGML_N_TASKS_MAX, st);
}

} // namespace fd

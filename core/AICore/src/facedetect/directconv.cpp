#include "directconv.hpp"
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

// Blocked-layout (nChw16c) AVX-512 register-tiled DIRECT convolution microkernel
// for the ArcFace recognizer's deep-channel/small-spatial 3x3 stride-1 pad-1 body
// (the regime where it beats Winograd at high thread counts; SCRFD's detector
// convs stay on Winograd by default -- see fd::conv2d for the routing gate).
// NO im2col, NO Winograd transform: it
// streams the NCHW activation (broadcast over the 16-OC weight vector) and
// FMA-accumulates a register tile of output-width pixels x 16 output channels
// directly over the IC*KH*KW window. The 16 OC sit in zmm lanes; one aligned zmm
// weight load is reused across the whole output-width strip.
//
// Ship-safety mirrors winograd.cpp EXACTLY: the zmm code carries
// __attribute__((target("avx512f,avx512bw,avx512vl"))) (function multiversioning)
// so it is emitted from an AVX2-only TU (default portable build, GGML_NATIVE=OFF,
// -mavx2 -mfma, NO global -mavx512f); it runs ONLY when __builtin_cpu_supports
// reports AVX-512 at run time, else the AVX2 (ymm, 8c) fallback runs -- so an
// AVX2-only CPU never executes a zmm instruction. A global -mavx512f build
// (GGML_NATIVE=ON) defines __AVX512F__ and the attribute is dropped.
#if defined(__AVX512F__)
#  define FD_DCONV_HAVE_AVX512 1
#  define FD_DCONV_AVX512_TARGET   /* ISA already global; no per-fn attribute */
#elif defined(FACEDETECT_DCONV_AVX512) && defined(__AVX2__)
#  define FD_DCONV_HAVE_AVX512 1
#  define FD_DCONV_AVX512_TARGET __attribute__((target("avx512f,avx512bw,avx512vl")))
#endif

namespace fd {
namespace {

// A/B knob for the stride>1 register-tiled flat downsample path. Default ON; set
// FACEDETECT_DCONV_STRIDE2=col to force the legacy per-column path (one zmm
// accumulator) for honest before/after benchmarking in a single binary.
static bool dconv_stride2_flat() {
    static const bool flat = [] {
        const char* e = std::getenv("FACEDETECT_DCONV_STRIDE2");
        return !(e && (!std::strcmp(e, "col") || !std::strcmp(e, "0")));
    }();
    return flat;
}

#if defined(FD_DCONV_HAVE_AVX512)
// Cached decision: run the AVX-512 microkernel? True only when the running CPU
// advertises avx512f+bw+vl (so zmm code never executes on an AVX2-only host) AND
// the FACEDETECT_DISABLE_AVX512 test hook is unset (shared with winograd.cpp;
// forcing it lets us exercise + parity-check the AVX2 fallback on an AVX-512 box).
// FACEDETECT_DCONV_VERBOSE prints the one-time selection for ship-safety proofs.
static bool dconv_use_avx512() {
    static const bool use512 = [] {
        const char* off = std::getenv("FACEDETECT_DISABLE_AVX512");
        const bool disabled = off && off[0] != '\0' && off[0] != '0';
        const bool supported = __builtin_cpu_supports("avx512f")
                            && __builtin_cpu_supports("avx512bw")
                            && __builtin_cpu_supports("avx512vl");
        const bool sel = supported && !disabled;
        if (std::getenv("FACEDETECT_DCONV_VERBOSE"))
            std::fprintf(stderr,
                         "[directconv] microkernel: %s (avx512 supported=%d, "
                         "disabled=%d)\n", sel ? "AVX-512(16c)" : "AVX2(8c)",
                         (int)supported, (int)disabled);
        return sel;
    }();
    return use512;
}
#endif

// Output-width register-tile width. OWB acc strips (each = LANE output channels
// for one output pixel) + 1 weight register must fit the vector register file.
// AVX-512: 32 zmm -> OWB up to ~28; start 14 (15 zmm live). AVX2: 16 ymm.
#ifndef FD_DCONV_OWB
#define FD_DCONV_OWB 14
#endif
constexpr int OWB = FD_DCONV_OWB;

// Column-tile width for the 2-OC-block kernel. 2*OWB2 acc + 2 weight + 1 bcast
// must fit 32 zmm; OWB2=12 -> 24+3 = 27.
#ifndef FD_DCONV_OWB2
#define FD_DCONV_OWB2 12
#endif
constexpr int OWB2 = FD_DCONV_OWB2;

// ------------------------------------------------------------------------
// Persistent per-op state. Caches the blocked-packed weights (computed once from
// w->data). LANE = 16 (AVX-512) or 8 (AVX2/scalar). Weights packed
// [OCB][KH][KW][IC][LANE] so the kh,kw,ic inner loop walks them contiguously and
// each (ocb,kh,kw,ic) yields one aligned LANE-wide load of LANE output channels.
// ------------------------------------------------------------------------
struct DConvState {
    int W = 0, H = 0, IC = 0, OC = 0, N = 0, pad = 0;
    int stride = 1;         // spatial stride (1 or 2); used by the blocked path
    int kind = 0;           // 0 = 3x3, 1 = 1x1 (blocked downsample)
    int lane = 16;          // OC block width
    int OCB = 0;            // OC / lane
    const void* wdata = nullptr;
    bool has_bias = false;      // fused bias add (bias is src[2])
    bool do_relu = false;       // fused ReLU after bias
    std::vector<float> Wpack;   // 3x3: [OCB*KH*KW*IC*lane]; 1x1: [OCB*IC*lane]
    std::vector<float> Bpack;   // [OCB*lane] bias broadcast pack (if has_bias)
    std::once_flag once;
};

// Pack per-channel bias into a [OCB*lane] broadcast buffer (pad lanes -> 0).
static void build_pack_bias(DConvState* st, const float* b) {
    const int OC = st->OC, lane = st->lane, OCB = st->OCB;
    st->Bpack.assign((size_t)OCB * lane, 0.0f);
    for (int ocb = 0; ocb < OCB; ++ocb)
        for (int l = 0; l < lane; ++l) {
            const int oc = ocb * lane + l;
            st->Bpack[(size_t)ocb * lane + l] = (oc < OC) ? b[oc] : 0.0f;
        }
}

static void build_pack(DConvState* st, const float* w) {
    const int IC = st->IC, OC = st->OC, lane = st->lane, OCB = st->OCB;
    // ggml w ne = [KW=3, KH=3, IC, OC]: w[((oc*IC+ic)*3 + kh)*3 + kw].
    st->Wpack.assign((size_t)OCB * 9 * IC * lane, 0.0f);
    for (int ocb = 0; ocb < OCB; ++ocb)
        for (int kh = 0; kh < 3; ++kh)
            for (int kw = 0; kw < 3; ++kw)
                for (int ic = 0; ic < IC; ++ic) {
                    float* dst = st->Wpack.data() +
                        ((((size_t)ocb * 9 + kh * 3 + kw) * IC) + ic) * lane;
                    for (int l = 0; l < lane; ++l) {
                        const int oc = ocb * lane + l;
                        dst[l] = (oc < OC)
                            ? w[(((size_t)oc * IC + ic) * 3 + kh) * 3 + kw]
                            : 0.0f;
                    }
                }
}

// 1x1 weight pack [OCB][IC][lane]: ggml w ne = [1, 1, IC, OC], so
// w[oc*IC + ic]. dst[(ocb*IC + ic)*lane + l] = w[(ocb*lane+l)*IC + ic].
static void build_pack_1x1(DConvState* st, const float* w) {
    const int IC = st->IC, OC = st->OC, lane = st->lane, OCB = st->OCB;
    st->Wpack.assign((size_t)OCB * IC * lane, 0.0f);
    for (int ocb = 0; ocb < OCB; ++ocb)
        for (int ic = 0; ic < IC; ++ic) {
            float* dst = st->Wpack.data() + ((size_t)ocb * IC + ic) * lane;
            for (int l = 0; l < lane; ++l) {
                const int oc = ocb * lane + l;
                dst[l] = (oc < OC) ? w[(size_t)oc * IC + ic] : 0.0f;
            }
        }
}

// ========================================================================
// AVX-512 microkernel: one output row (ocb, ih) -> blocked row buffer
// outrow[W*16] (outrow[ow*16 + lane]). Interior columns [pad, W-1-pad] run the
// no-branch register-tiled path; the edge columns run a masked-guard path.
// ========================================================================
#if defined(FD_DCONV_HAVE_AVX512)
// One interior strip of CW output columns. CW is compile-time so the j loops
// FULLY UNROLL: no per-FMA branch, weight loaded once per ic and reused across
// the strip via embedded-broadcast FMA (vfmadd231ps {1to16}).
template<int CW>
static inline FD_DCONV_AVX512_TARGET void dconv_strip_avx512(
        const float* x, const float* wocb, float* outrow,
        int W, int H, int IC, int ih, int pad, int ow0) {
    const size_t HW = (size_t)H * W;
    __m512 acc[CW];
    for (int j = 0; j < CW; ++j) acc[j] = _mm512_setzero_ps();
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = ih + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * 16;
            const float* in = x + (size_t)iy * W + (ow0 + kw - pad);
            for (int ic = 0; ic < IC; ++ic) {
                const __m512 wv = _mm512_loadu_ps(wk + (size_t)ic * 16);
                const float* inc = in + (size_t)ic * HW;
                for (int j = 0; j < CW; ++j)
                    acc[j] = _mm512_fmadd_ps(_mm512_set1_ps(inc[j]), wv, acc[j]);
            }
        }
    }
    for (int j = 0; j < CW; ++j)
        _mm512_store_ps(outrow + (size_t)(ow0 + j) * 16, acc[j]);
}

// Two-OC-block strip: each input broadcast feeds TWO weight vectors (the two OC
// blocks ocb0/ocb1), halving the broadcast-load pressure per FMA -> pushes the
// kernel from load-bound toward FMA-bound. 2*CW acc + 2 weight + 1 bcast regs.
template<int CW>
static inline FD_DCONV_AVX512_TARGET void dconv_strip2_avx512(
        const float* x, const float* wocb0, const float* wocb1,
        float* outrow0, float* outrow1,
        int W, int H, int IC, int ih, int pad, int ow0) {
    const size_t HW = (size_t)H * W;
    __m512 a0[CW], a1[CW];
    for (int j = 0; j < CW; ++j) { a0[j] = _mm512_setzero_ps(); a1[j] = _mm512_setzero_ps(); }
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = ih + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const size_t off = ((size_t)kh * 3 + kw) * IC * 16;
            const float* wk0 = wocb0 + off;
            const float* wk1 = wocb1 + off;
            const float* in = x + (size_t)iy * W + (ow0 + kw - pad);
            for (int ic = 0; ic < IC; ++ic) {
                const __m512 w0 = _mm512_loadu_ps(wk0 + (size_t)ic * 16);
                const __m512 w1 = _mm512_loadu_ps(wk1 + (size_t)ic * 16);
                const float* inc = in + (size_t)ic * HW;
                for (int j = 0; j < CW; ++j) {
                    const __m512 b = _mm512_set1_ps(inc[j]);
                    a0[j] = _mm512_fmadd_ps(b, w0, a0[j]);
                    a1[j] = _mm512_fmadd_ps(b, w1, a1[j]);
                }
            }
        }
    }
    for (int j = 0; j < CW; ++j) {
        _mm512_store_ps(outrow0 + (size_t)(ow0 + j) * 16, a0[j]);
        _mm512_store_ps(outrow1 + (size_t)(ow0 + j) * 16, a1[j]);
    }
}

// One edge column for a single OC block (full bounds checks).
static inline FD_DCONV_AVX512_TARGET void dconv_edge_avx512(
        const float* x, const float* wocb, float* outrow,
        int W, int H, int IC, int ih, int pad, int ow) {
    const size_t HW = (size_t)H * W;
    __m512 acc = _mm512_setzero_ps();
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = ih + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const int ix = ow + kw - pad;
            if (ix < 0 || ix >= W) continue;
            const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * 16;
            const float* in = x + (size_t)iy * W + ix;
            for (int ic = 0; ic < IC; ++ic)
                acc = _mm512_fmadd_ps(_mm512_set1_ps(in[(size_t)ic * HW]),
                                      _mm512_loadu_ps(wk + (size_t)ic * 16), acc);
        }
    }
    _mm512_store_ps(outrow + (size_t)ow * 16, acc);
}

// Two-OC-block output row: edges per block, interior via the shared-broadcast
// 2-block strips.
static inline FD_DCONV_AVX512_TARGET void dconv_row2_avx512(
        const float* x, const float* Wpack, float* outrow0, float* outrow1,
        int W, int H, int IC, int ocb0, int ocb1, int ih, int pad) {
    const float* w0 = Wpack + (size_t)ocb0 * 9 * IC * 16;
    const float* w1 = Wpack + (size_t)ocb1 * 9 * IC * 16;
    for (int e = 0; e < 2 * pad; ++e) {
        const int ow = (e < pad) ? e : (W - pad + (e - pad));
        if (ow < 0 || ow >= W) continue;
        if (e >= pad && ow < pad) continue;
        dconv_edge_avx512(x, w0, outrow0, W, H, IC, ih, pad, ow);
        dconv_edge_avx512(x, w1, outrow1, W, H, IC, ih, pad, ow);
    }
    const int hi = W - 1 - pad;
    int ow0 = pad;
    for (; ow0 + OWB2 - 1 <= hi; ow0 += OWB2)
        dconv_strip2_avx512<OWB2>(x, w0, w1, outrow0, outrow1, W, H, IC, ih, pad, ow0);
    int rem = hi - ow0 + 1;
    while (rem >= 8) { dconv_strip2_avx512<8>(x, w0, w1, outrow0, outrow1, W, H, IC, ih, pad, ow0); ow0 += 8; rem -= 8; }
    while (rem >= 4) { dconv_strip2_avx512<4>(x, w0, w1, outrow0, outrow1, W, H, IC, ih, pad, ow0); ow0 += 4; rem -= 4; }
    while (rem >= 2) { dconv_strip2_avx512<2>(x, w0, w1, outrow0, outrow1, W, H, IC, ih, pad, ow0); ow0 += 2; rem -= 2; }
    while (rem >= 1) { dconv_strip2_avx512<1>(x, w0, w1, outrow0, outrow1, W, H, IC, ih, pad, ow0); ow0 += 1; rem -= 1; }
}

static inline FD_DCONV_AVX512_TARGET void dconv_row_avx512(
        const float* x, const float* Wpack, float* outrow,
        int W, int H, int IC, int ocb, int ih, int pad) {
    const size_t HW = (size_t)H * W;
    const float* wocb = Wpack + (size_t)ocb * 9 * IC * 16;

    // Edge columns (ow in [0,pad) and [W-pad,W)): full bounds checks. Two ranges
    // walked as one loop so the zmm intrinsics stay in this target-tagged body
    // (a lambda would not inherit the function-multiversioning target).
    for (int e = 0; e < 2 * pad; ++e) {
        const int ow = (e < pad) ? e : (W - pad + (e - pad));
        if (ow < 0 || ow >= W) continue;
        if (e >= pad && ow < pad) continue;   // overlap when W < 2*pad
        __m512 acc = _mm512_setzero_ps();
        for (int kh = 0; kh < 3; ++kh) {
            const int iy = ih + kh - pad;
            if (iy < 0 || iy >= H) continue;
            for (int kw = 0; kw < 3; ++kw) {
                const int ix = ow + kw - pad;
                if (ix < 0 || ix >= W) continue;
                const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * 16;
                const float* in = x + (size_t)iy * W + ix;
                for (int ic = 0; ic < IC; ++ic)
                    acc = _mm512_fmadd_ps(_mm512_set1_ps(in[(size_t)ic * HW]),
                                          _mm512_loadu_ps(wk + (size_t)ic * 16), acc);
            }
        }
        _mm512_store_ps(outrow + (size_t)ow * 16, acc);
    }

    // Interior columns ow in [pad, W-1-pad]: ix = ow + kw - pad in [0, W-1] always
    // (holds for pad >= 1, which every SCRFD/ArcFace 3x3 conv uses). Bulk strips
    // run the fully-unrolled fixed-OWB kernel; the last partial strip is split
    // into smaller fixed-width kernels (8/4/2/1) so every column stays branchless.
    const int hi = W - 1 - pad;             // last interior column
    int ow0 = pad;
    for (; ow0 + OWB - 1 <= hi; ow0 += OWB)
        dconv_strip_avx512<OWB>(x, wocb, outrow, W, H, IC, ih, pad, ow0);
    int rem = hi - ow0 + 1;
    while (rem >= 8) { dconv_strip_avx512<8>(x, wocb, outrow, W, H, IC, ih, pad, ow0); ow0 += 8; rem -= 8; }
    while (rem >= 4) { dconv_strip_avx512<4>(x, wocb, outrow, W, H, IC, ih, pad, ow0); ow0 += 4; rem -= 4; }
    while (rem >= 2) { dconv_strip_avx512<2>(x, wocb, outrow, W, H, IC, ih, pad, ow0); ow0 += 2; rem -= 2; }
    while (rem >= 1) { dconv_strip_avx512<1>(x, wocb, outrow, W, H, IC, ih, pad, ow0); ow0 += 1; rem -= 1; }
}
#endif

// ========================================================================
// AVX2 microkernel: LANE=8 ymm. Same structure.
// ========================================================================
#if defined(__AVX2__)
static inline void dconv_row_avx2(
        const float* x, const float* Wpack, float* outrow,
        int W, int H, int IC, int ocb, int ih, int pad) {
    const size_t HW = (size_t)H * W;
    const float* wocb = Wpack + (size_t)ocb * 9 * IC * 8;
    auto edge_col = [&](int ow) {
        __m256 acc = _mm256_setzero_ps();
        for (int kh = 0; kh < 3; ++kh) {
            const int iy = ih + kh - pad;
            if (iy < 0 || iy >= H) continue;
            for (int kw = 0; kw < 3; ++kw) {
                const int ix = ow + kw - pad;
                if (ix < 0 || ix >= W) continue;
                const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * 8;
                const float* in = x + (size_t)iy * W + ix;
                for (int ic = 0; ic < IC; ++ic)
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(in[(size_t)ic * HW]),
                                          _mm256_loadu_ps(wk + (size_t)ic * 8), acc);
            }
        }
        _mm256_store_ps(outrow + (size_t)ow * 8, acc);
    };
    for (int ow = 0; ow < pad && ow < W; ++ow) edge_col(ow);
    for (int ow = std::max(W - pad, pad); ow < W; ++ow) edge_col(ow);
    for (int ow0 = pad; ow0 <= W - 1 - pad; ow0 += OWB) {
        const int owc = std::min(OWB, (W - pad) - ow0);
        __m256 acc[OWB];
        for (int j = 0; j < owc; ++j) acc[j] = _mm256_setzero_ps();
        for (int kh = 0; kh < 3; ++kh) {
            const int iy = ih + kh - pad;
            if (iy < 0 || iy >= H) continue;
            for (int kw = 0; kw < 3; ++kw) {
                const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * 8;
                const float* in = x + (size_t)iy * W + (ow0 + kw - pad);
                for (int ic = 0; ic < IC; ++ic) {
                    const __m256 wv = _mm256_loadu_ps(wk + (size_t)ic * 8);
                    const float* inc = in + (size_t)ic * HW;
                    for (int j = 0; j < owc; ++j)
                        acc[j] = _mm256_fmadd_ps(_mm256_set1_ps(inc[j]), wv, acc[j]);
                }
            }
        }
        for (int j = 0; j < owc; ++j)
            _mm256_store_ps(outrow + (size_t)(ow0 + j) * 8, acc[j]);
    }
}
#endif

// Scalar fallback (non-x86): LANE-wide blocked row, plain loops.
static void dconv_row_scalar(const float* x, const float* Wpack, float* outrow,
                             int W, int H, int IC, int lane, int ocb, int ih, int pad) {
    const size_t HW = (size_t)H * W;
    const float* wocb = Wpack + (size_t)ocb * 9 * IC * lane;
    for (int ow = 0; ow < W; ++ow) {
        float* o = outrow + (size_t)ow * lane;
        for (int l = 0; l < lane; ++l) o[l] = 0.0f;
        for (int kh = 0; kh < 3; ++kh) {
            const int iy = ih + kh - pad;
            if (iy < 0 || iy >= H) continue;
            for (int kw = 0; kw < 3; ++kw) {
                const int ix = ow + kw - pad;
                if (ix < 0 || ix >= W) continue;
                const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * lane;
                const float* in = x + (size_t)iy * W + ix;
                for (int ic = 0; ic < IC; ++ic) {
                    const float s = in[(size_t)ic * HW];
                    const float* wv = wk + (size_t)ic * lane;
                    for (int l = 0; l < lane; ++l) o[l] += s * wv[l];
                }
            }
        }
    }
}

// Transpose one blocked output row [W*lane] -> NCHW dst planes:
// dst[(ocb*lane + l)*HW + ih*W + ow] = outrow[ow*lane + l].
static inline void scatter_row(float* y, const float* outrow,
                               int W, size_t HW, int lane, int ocb, int ih, int OC) {
    const size_t base = ih * (size_t)W;
    for (int l = 0; l < lane; ++l) {
        const int oc = ocb * lane + l;
        if (oc >= OC) break;
        float* yc = y + (size_t)oc * HW + base;
        for (int ow = 0; ow < W; ++ow) yc[ow] = outrow[(size_t)ow * lane + l];
    }
}

static void dconv_compute(ggml_tensor* dst, int ith, int nth, void* userdata) {
    DConvState* st = (DConvState*)userdata;
    const ggml_tensor* xt = dst->src[0];
    const ggml_tensor* wt = dst->src[1];
    const float* x = (const float*)xt->data;
    float* y = (float*)dst->data;

    std::call_once(st->once, [&] { build_pack(st, (const float*)wt->data); });

    const int W = st->W, H = st->H, IC = st->IC, OC = st->OC;
    const int pad = st->pad, lane = st->lane, OCB = st->OCB;
    const float* Wpack = st->Wpack.data();
    const size_t HW = (size_t)H * W;

    // Per-thread blocked output-row scratch (aligned for vector stores). Two rows
    // so the AVX-512 path can emit a pair of OC blocks per pass.
    std::vector<float> rowbuf((size_t)W * lane * 2 + 32);
    float* outrow0 = (float*)(((uintptr_t)rowbuf.data() + 63) & ~(uintptr_t)63);
    float* outrow1 = outrow0 + (size_t)W * lane;

#if defined(FD_DCONV_HAVE_AVX512)
    if (lane == 16) {
        // Work items = (OC-block PAIR, ih). Each pass shares one input broadcast
        // across two OC blocks. Odd trailing block falls back to the 1-block row.
        const int npairs = (OCB + 1) / 2;
        const int64_t total = (int64_t)npairs * H;
        const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
        for (int64_t idx = beg; idx < end; ++idx) {
            const int p  = (int)(idx / H);
            const int ih = (int)(idx % H);
            const int ocb0 = 2 * p, ocb1 = 2 * p + 1;
            if (ocb1 < OCB) {
                dconv_row2_avx512(x, Wpack, outrow0, outrow1, W, H, IC, ocb0, ocb1, ih, pad);
                scatter_row(y, outrow0, W, HW, 16, ocb0, ih, OC);
                scatter_row(y, outrow1, W, HW, 16, ocb1, ih, OC);
            } else {
                dconv_row_avx512(x, Wpack, outrow0, W, H, IC, ocb0, ih, pad);
                scatter_row(y, outrow0, W, HW, 16, ocb0, ih, OC);
            }
        }
        return;
    }
#endif
    // AVX2 / scalar: single OC block per (ocb, ih) work item.
    const int64_t total = (int64_t)OCB * H;
    const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
    for (int64_t idx = beg; idx < end; ++idx) {
        const int ocb = (int)(idx / H);
        const int ih  = (int)(idx % H);
#if defined(__AVX2__)
        if (lane == 8) { dconv_row_avx2(x, Wpack, outrow0, W, H, IC, ocb, ih, pad); }
        else
#endif
        { dconv_row_scalar(x, Wpack, outrow0, W, H, IC, lane, ocb, ih, pad); }
        scatter_row(y, outrow0, W, HW, lane, ocb, ih, OC);
    }
}

// ========================================================================
// BLOCKED-LAYOUT (nChw16c) ISLAND: ops that consume AND produce the blocked
// buffer ne = [16, W, H, CB] (flat index ((cb*H + h)*W + w)*16 + l, channel
// c = cb*16 + l). They let the whole ArcFace IResNet body stay blocked between
// ONE reorder-in (after the stem) and ONE reorder-out (before the output head),
// instead of the per-conv NCHW<->blocked round trip the directconv path pays on
// every layer. The IResNet stage widths (64/128/256/512) are all multiples of 16,
// so CB = C/16 is exact and there are no padding lanes for buffalo_l/m.
// ========================================================================
constexpr int BLK = 16;   // nChw16c block width (the island is AVX-512 sized)

#if defined(FD_DCONV_HAVE_AVX512)
// Conv epilogue: optional fused bias add (bvec = 16 bias floats for this OC block,
// or nullptr) + optional ReLU, then store. Fusing bias/ReLU into the conv's own
// register tile removes the separate full-buffer blocked-bias / blocked-relu
// passes.
static inline FD_DCONV_AVX512_TARGET void bemit_avx512(
        float* p, __m512 acc, const float* bvec, bool relu) {
    if (bvec) acc = _mm512_add_ps(acc, _mm512_loadu_ps(bvec));
    if (relu) acc = _mm512_max_ps(acc, _mm512_setzero_ps());
    _mm512_storeu_ps(p, acc);
}

// --- blocked 3x3 conv: one output column (ocb, oh, ow), all bounds checked.
// Reads blocked input, writes the 16-OC tile straight into the blocked output;
// kh,kw,ic accumulation order matches dconv_*  so the result is bit-identical to
// the NCHW direct conv (layout-only change).
static inline FD_DCONV_AVX512_TARGET void bconv3x3_col_avx512(
        const float* xb, const float* wocb, float* orow,
        int W, int H, int IC, int oh, int ow, int pad, int stride,
        const float* bvec = nullptr, bool relu = false) {
    __m512 acc = _mm512_setzero_ps();
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = oh * stride + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const int ix = ow * stride + kw - pad;
            if (ix < 0 || ix >= W) continue;
            const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * BLK;
            const float* col = xb + (((size_t)iy * W) + ix) * BLK;  // (cb=0,l=0) base
            for (int ic = 0; ic < IC; ++ic) {
                const int cb = ic >> 4, l = ic & 15;
                const float s = col[(size_t)cb * H * W * BLK + l];
                acc = _mm512_fmadd_ps(_mm512_set1_ps(s),
                                      _mm512_loadu_ps(wk + (size_t)ic * BLK), acc);
            }
        }
    }
    bemit_avx512(orow + (size_t)ow * BLK, acc, bvec, relu);
}

// stride-1 interior register-tiled strip (CW output columns), fully unrolled.
// Blocked input column stride is BLK floats (vs 1 for NCHW). ix = ow + kw - pad.
template<int CW>
static inline FD_DCONV_AVX512_TARGET void bconv3x3_strip_s1_avx512(
        const float* xb, const float* wocb, float* orow,
        int W, int H, int IC, int ih, int pad, int ow0,
        const float* bvec = nullptr, bool relu = false) {
    __m512 acc[CW];
    for (int j = 0; j < CW; ++j) acc[j] = _mm512_setzero_ps();
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = ih + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * BLK;
            const int ix0 = ow0 + kw - pad;
            const float* base = xb + (((size_t)iy * W) + ix0) * BLK;  // cb=0,l=0
            for (int ic = 0; ic < IC; ++ic) {
                const int cb = ic >> 4, l = ic & 15;
                const float* inc = base + (size_t)cb * H * W * BLK + l;
                const __m512 wv = _mm512_loadu_ps(wk + (size_t)ic * BLK);
                for (int j = 0; j < CW; ++j)
                    acc[j] = _mm512_fmadd_ps(_mm512_set1_ps(inc[(size_t)j * BLK]), wv, acc[j]);
            }
        }
    }
    for (int j = 0; j < CW; ++j)
        bemit_avx512(orow + (size_t)(ow0 + j) * BLK, acc[j], bvec, relu);
}

// stride-1 interior 2-OC-block strip: each (strided, expensive) blocked-input
// broadcast feeds TWO OC-block weight vectors, halving the broadcast-load count
// per FMA (the master directconv's winning lever, here aimed at the blocked
// layout's width-strided input reads). Writes both OC blocks' columns.
template<int CW>
static inline FD_DCONV_AVX512_TARGET void bconv3x3_strip2_s1_avx512(
        const float* xb, const float* wocb0, const float* wocb1,
        float* orow0, float* orow1,
        int W, int H, int IC, int ih, int pad, int ow0,
        const float* bvec0 = nullptr, const float* bvec1 = nullptr, bool relu = false) {
    __m512 a0[CW], a1[CW];
    for (int j = 0; j < CW; ++j) { a0[j] = _mm512_setzero_ps(); a1[j] = _mm512_setzero_ps(); }
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = ih + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const size_t off = ((size_t)kh * 3 + kw) * IC * BLK;
            const float* wk0 = wocb0 + off;
            const float* wk1 = wocb1 + off;
            const int ix0 = ow0 + kw - pad;
            const float* base = xb + (((size_t)iy * W) + ix0) * BLK;
            for (int ic = 0; ic < IC; ++ic) {
                const int cb = ic >> 4, l = ic & 15;
                const float* inc = base + (size_t)cb * H * W * BLK + l;
                const __m512 w0 = _mm512_loadu_ps(wk0 + (size_t)ic * BLK);
                const __m512 w1 = _mm512_loadu_ps(wk1 + (size_t)ic * BLK);
                for (int j = 0; j < CW; ++j) {
                    const __m512 b = _mm512_set1_ps(inc[(size_t)j * BLK]);
                    a0[j] = _mm512_fmadd_ps(b, w0, a0[j]);
                    a1[j] = _mm512_fmadd_ps(b, w1, a1[j]);
                }
            }
        }
    }
    for (int j = 0; j < CW; ++j) {
        bemit_avx512(orow0 + (size_t)(ow0 + j) * BLK, a0[j], bvec0, relu);
        bemit_avx512(orow1 + (size_t)(ow0 + j) * BLK, a1[j], bvec1, relu);
    }
}

// stride-1 2-OC-block output-row driver: edges per block (single-OC col kernel),
// interior via the shared-broadcast 2-block strips.
static inline FD_DCONV_AVX512_TARGET void bconv3x3_row2_s1_avx512(
        const float* xb, const float* Wpack, float* y,
        int W, int H, int IC, int Wout, int Hout, int ocb0, int ocb1, int oh, int pad,
        const float* Bpack, bool relu) {
    const float* w0 = Wpack + (size_t)ocb0 * 9 * IC * BLK;
    const float* w1 = Wpack + (size_t)ocb1 * 9 * IC * BLK;
    const float* b0 = Bpack ? Bpack + (size_t)ocb0 * BLK : nullptr;
    const float* b1 = Bpack ? Bpack + (size_t)ocb1 * BLK : nullptr;
    float* orow0 = y + (((size_t)ocb0 * Hout + oh) * Wout) * BLK;
    float* orow1 = y + (((size_t)ocb1 * Hout + oh) * Wout) * BLK;
    int ow = 0;
    for (; ow < pad && ow < Wout; ++ow) {
        bconv3x3_col_avx512(xb, w0, orow0, W, H, IC, oh, ow, pad, 1, b0, relu);
        bconv3x3_col_avx512(xb, w1, orow1, W, H, IC, oh, ow, pad, 1, b1, relu);
    }
    const int hi = Wout - 1 - pad;
    int ow0 = std::max(pad, ow);
    for (; ow0 + OWB2 - 1 <= hi; ow0 += OWB2)
        bconv3x3_strip2_s1_avx512<OWB2>(xb, w0, w1, orow0, orow1, W, H, IC, oh, pad, ow0, b0, b1, relu);
    int rem = hi - ow0 + 1;
    while (rem >= 8) { bconv3x3_strip2_s1_avx512<8>(xb, w0, w1, orow0, orow1, W, H, IC, oh, pad, ow0, b0, b1, relu); ow0 += 8; rem -= 8; }
    while (rem >= 4) { bconv3x3_strip2_s1_avx512<4>(xb, w0, w1, orow0, orow1, W, H, IC, oh, pad, ow0, b0, b1, relu); ow0 += 4; rem -= 4; }
    while (rem >= 2) { bconv3x3_strip2_s1_avx512<2>(xb, w0, w1, orow0, orow1, W, H, IC, oh, pad, ow0, b0, b1, relu); ow0 += 2; rem -= 2; }
    while (rem >= 1) { bconv3x3_strip2_s1_avx512<1>(xb, w0, w1, orow0, orow1, W, H, IC, oh, pad, ow0, b0, b1, relu); ow0 += 1; rem -= 1; }
    for (int owe = std::max(Wout - pad, ow0); owe < Wout; ++owe) {
        bconv3x3_col_avx512(xb, w0, orow0, W, H, IC, oh, owe, pad, 1, b0, relu);
        bconv3x3_col_avx512(xb, w1, orow1, W, H, IC, oh, owe, pad, 1, b1, relu);
    }
}

// ------------------------------------------------------------------------
// SMALL-SPATIAL (flattened) stride-1 3x3 kernel. On small output maps (Wout <=
// the register-tile width) the per-ROW driver above cannot fill its accumulator
// budget: a 7-wide row's interior is 5 columns, so it runs CW=4 + CW=1 strips
// (4 and 1 live accumulators of the 24-30 the zmm file holds) and the FMA units
// stall. These flattened strips instead treat the output as a single flat
// sequence of Hout*Wout pixels and tile CW *consecutive pixels across row
// boundaries*, so a 7x7=49-pixel map fills width-12 (2-OC) tiles -> 24 live
// accumulators regardless of how narrow each row is.
//
// The 3x3 input window's row/column crossing is handled in the per-pixel input
// gather: for each (kh,kw) tap we precompute, per tiled pixel, the blocked-input
// base offset and a 0/1 validity gate (a pixel near a map edge has some taps out
// of bounds). The gate multiplies the broadcast input by 0.0f for an invalid
// tap, which adds an EXACT +0.0 to that accumulator -- bit-identical to the
// per-row/per-column path's "skip out-of-bounds tap" (same kh,kw,ic order, same
// fp adds), so parity is unchanged. The 16-OC weight vector is still loaded once
// per ic and reused across the whole pixel strip (the master kernel's lever).
#if defined(FD_DCONV_HAVE_AVX512)
// flat 2-OC-block strip: CW consecutive output pixels (flat index f0..f0+CW-1),
// two OC blocks sharing each input broadcast.
template<int CW>
static inline FD_DCONV_AVX512_TARGET void bconv3x3_flat2_avx512(
        const float* xb, const float* wocb0, const float* wocb1,
        float* y0, float* y1,
        int W, int H, int IC, int Wout, int pad, int f0, int stride,
        const float* bvec0 = nullptr, const float* bvec1 = nullptr, bool relu = false) {
    const size_t HWB = (size_t)H * W * BLK;
    __m512 a0[CW], a1[CW];
    int oh[CW], ow[CW];
    for (int j = 0; j < CW; ++j) {
        a0[j] = _mm512_setzero_ps(); a1[j] = _mm512_setzero_ps();
        const int f = f0 + j; oh[j] = f / Wout; ow[j] = f % Wout;
    }
    for (int kh = 0; kh < 3; ++kh) {
        for (int kw = 0; kw < 3; ++kw) {
            size_t base[CW]; float gate[CW];
            for (int j = 0; j < CW; ++j) {
                const int iy = oh[j] * stride + kh - pad, ix = ow[j] * stride + kw - pad;
                const bool v = (unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)W;
                base[j] = v ? (((size_t)iy * W) + ix) * BLK : 0;
                gate[j] = v ? 1.0f : 0.0f;
            }
            const size_t off = ((size_t)kh * 3 + kw) * IC * BLK;
            const float* wk0 = wocb0 + off;
            const float* wk1 = wocb1 + off;
            for (int ic = 0; ic < IC; ++ic) {
                const size_t coff = (size_t)(ic >> 4) * HWB + (ic & 15);
                const __m512 w0 = _mm512_loadu_ps(wk0 + (size_t)ic * BLK);
                const __m512 w1 = _mm512_loadu_ps(wk1 + (size_t)ic * BLK);
                for (int j = 0; j < CW; ++j) {
                    const __m512 b = _mm512_set1_ps(xb[base[j] + coff] * gate[j]);
                    a0[j] = _mm512_fmadd_ps(b, w0, a0[j]);
                    a1[j] = _mm512_fmadd_ps(b, w1, a1[j]);
                }
            }
        }
    }
    for (int j = 0; j < CW; ++j) {
        bemit_avx512(y0 + (size_t)(f0 + j) * BLK, a0[j], bvec0, relu);
        bemit_avx512(y1 + (size_t)(f0 + j) * BLK, a1[j], bvec1, relu);
    }
}

// flat single-OC-block strip (odd trailing OC block).
template<int CW>
static inline FD_DCONV_AVX512_TARGET void bconv3x3_flat_avx512(
        const float* xb, const float* wocb, float* y0,
        int W, int H, int IC, int Wout, int pad, int f0, int stride,
        const float* bvec = nullptr, bool relu = false) {
    const size_t HWB = (size_t)H * W * BLK;
    __m512 acc[CW];
    int oh[CW], ow[CW];
    for (int j = 0; j < CW; ++j) {
        acc[j] = _mm512_setzero_ps();
        const int f = f0 + j; oh[j] = f / Wout; ow[j] = f % Wout;
    }
    for (int kh = 0; kh < 3; ++kh) {
        for (int kw = 0; kw < 3; ++kw) {
            size_t base[CW]; float gate[CW];
            for (int j = 0; j < CW; ++j) {
                const int iy = oh[j] * stride + kh - pad, ix = ow[j] * stride + kw - pad;
                const bool v = (unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)W;
                base[j] = v ? (((size_t)iy * W) + ix) * BLK : 0;
                gate[j] = v ? 1.0f : 0.0f;
            }
            const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * BLK;
            for (int ic = 0; ic < IC; ++ic) {
                const size_t coff = (size_t)(ic >> 4) * HWB + (ic & 15);
                const __m512 wv = _mm512_loadu_ps(wk + (size_t)ic * BLK);
                for (int j = 0; j < CW; ++j)
                    acc[j] = _mm512_fmadd_ps(
                        _mm512_set1_ps(xb[base[j] + coff] * gate[j]), wv, acc[j]);
            }
        }
    }
    for (int j = 0; j < CW; ++j)
        bemit_avx512(y0 + (size_t)(f0 + j) * BLK, acc[j], bvec, relu);
}

// flat 2-OC driver: walk a half-open flat-pixel range [pbeg,pend) in fixed-CW
// strips (12 -> 8/4/2/1 ladder), all branchless and register-tiled.
static inline FD_DCONV_AVX512_TARGET void bconv3x3_flat2_range_avx512(
        const float* xb, const float* w0, const float* w1, float* y0, float* y1,
        int W, int H, int IC, int Wout, int pad, int stride, int pbeg, int pend,
        const float* b0, const float* b1, bool relu) {
    int f = pbeg;
    for (; f + OWB2 <= pend; f += OWB2)
        bconv3x3_flat2_avx512<OWB2>(xb, w0, w1, y0, y1, W, H, IC, Wout, pad, f, stride, b0, b1, relu);
    int rem = pend - f;
    while (rem >= 8) { bconv3x3_flat2_avx512<8>(xb, w0, w1, y0, y1, W, H, IC, Wout, pad, f, stride, b0, b1, relu); f += 8; rem -= 8; }
    while (rem >= 4) { bconv3x3_flat2_avx512<4>(xb, w0, w1, y0, y1, W, H, IC, Wout, pad, f, stride, b0, b1, relu); f += 4; rem -= 4; }
    while (rem >= 2) { bconv3x3_flat2_avx512<2>(xb, w0, w1, y0, y1, W, H, IC, Wout, pad, f, stride, b0, b1, relu); f += 2; rem -= 2; }
    while (rem >= 1) { bconv3x3_flat2_avx512<1>(xb, w0, w1, y0, y1, W, H, IC, Wout, pad, f, stride, b0, b1, relu); f += 1; rem -= 1; }
}

// flat single-OC driver (odd trailing OC block).
static inline FD_DCONV_AVX512_TARGET void bconv3x3_flat_range_avx512(
        const float* xb, const float* w0, float* y0,
        int W, int H, int IC, int Wout, int pad, int stride, int pbeg, int pend,
        const float* b0, bool relu) {
    int f = pbeg;
    for (; f + OWB <= pend; f += OWB)
        bconv3x3_flat_avx512<OWB>(xb, w0, y0, W, H, IC, Wout, pad, f, stride, b0, relu);
    int rem = pend - f;
    while (rem >= 8) { bconv3x3_flat_avx512<8>(xb, w0, y0, W, H, IC, Wout, pad, f, stride, b0, relu); f += 8; rem -= 8; }
    while (rem >= 4) { bconv3x3_flat_avx512<4>(xb, w0, y0, W, H, IC, Wout, pad, f, stride, b0, relu); f += 4; rem -= 4; }
    while (rem >= 2) { bconv3x3_flat_avx512<2>(xb, w0, y0, W, H, IC, Wout, pad, f, stride, b0, relu); f += 2; rem -= 2; }
    while (rem >= 1) { bconv3x3_flat_avx512<1>(xb, w0, y0, W, H, IC, Wout, pad, f, stride, b0, relu); f += 1; rem -= 1; }
}
#endif

// blocked 1x1 strided conv: one output column. pad 0, single tap.
static inline FD_DCONV_AVX512_TARGET void bconv1x1_col_avx512(
        const float* xb, const float* wocb, float* orow,
        int W, int H, int IC, int oh, int ow, int stride,
        const float* bvec = nullptr, bool relu = false) {
    const int iy = oh * stride, ix = ow * stride;
    const float* col = xb + (((size_t)iy * W) + ix) * BLK;
    __m512 acc = _mm512_setzero_ps();
    for (int ic = 0; ic < IC; ++ic) {
        const int cb = ic >> 4, l = ic & 15;
        acc = _mm512_fmadd_ps(_mm512_set1_ps(col[(size_t)cb * H * W * BLK + l]),
                              _mm512_loadu_ps(wocb + (size_t)ic * BLK), acc);
    }
    bemit_avx512(orow + (size_t)ow * BLK, acc, bvec, relu);
}

// FLATTENED-SPATIAL strided 1x1: the IResNet downsample shortcut. Like the 3x3
// flat kernel, tiles CW consecutive OUTPUT pixels and shares each (strided)
// blocked-input broadcast across TWO OC blocks, so the zmm accumulator file fills
// regardless of how narrow each output row is. A 1x1 has a single tap always in
// bounds (output pixel maps to a valid input pixel for pad-0 stride-s), so no
// edge gate is needed. Same ic order as bconv1x1_col_avx512 -> bit-identical.
template<int CW>
static inline FD_DCONV_AVX512_TARGET void bconv1x1_flat2_avx512(
        const float* xb, const float* wocb0, const float* wocb1,
        float* y0, float* y1,
        int W, int H, int IC, int Wout, int stride, int f0,
        const float* bvec0 = nullptr, const float* bvec1 = nullptr, bool relu = false) {
    const size_t HWB = (size_t)H * W * BLK;
    __m512 a0[CW], a1[CW];
    size_t base[CW];
    for (int j = 0; j < CW; ++j) {
        a0[j] = _mm512_setzero_ps(); a1[j] = _mm512_setzero_ps();
        const int f = f0 + j, oh = f / Wout, ow = f % Wout;
        base[j] = (((size_t)(oh * stride) * W) + (size_t)(ow * stride)) * BLK;
    }
    for (int ic = 0; ic < IC; ++ic) {
        const size_t coff = (size_t)(ic >> 4) * HWB + (ic & 15);
        const __m512 w0 = _mm512_loadu_ps(wocb0 + (size_t)ic * BLK);
        const __m512 w1 = _mm512_loadu_ps(wocb1 + (size_t)ic * BLK);
        for (int j = 0; j < CW; ++j) {
            const __m512 b = _mm512_set1_ps(xb[base[j] + coff]);
            a0[j] = _mm512_fmadd_ps(b, w0, a0[j]);
            a1[j] = _mm512_fmadd_ps(b, w1, a1[j]);
        }
    }
    for (int j = 0; j < CW; ++j) {
        bemit_avx512(y0 + (size_t)(f0 + j) * BLK, a0[j], bvec0, relu);
        bemit_avx512(y1 + (size_t)(f0 + j) * BLK, a1[j], bvec1, relu);
    }
}
template<int CW>
static inline FD_DCONV_AVX512_TARGET void bconv1x1_flat_avx512(
        const float* xb, const float* wocb, float* y0,
        int W, int H, int IC, int Wout, int stride, int f0,
        const float* bvec = nullptr, bool relu = false) {
    const size_t HWB = (size_t)H * W * BLK;
    __m512 acc[CW];
    size_t base[CW];
    for (int j = 0; j < CW; ++j) {
        acc[j] = _mm512_setzero_ps();
        const int f = f0 + j, oh = f / Wout, ow = f % Wout;
        base[j] = (((size_t)(oh * stride) * W) + (size_t)(ow * stride)) * BLK;
    }
    for (int ic = 0; ic < IC; ++ic) {
        const size_t coff = (size_t)(ic >> 4) * HWB + (ic & 15);
        const __m512 wv = _mm512_loadu_ps(wocb + (size_t)ic * BLK);
        for (int j = 0; j < CW; ++j)
            acc[j] = _mm512_fmadd_ps(_mm512_set1_ps(xb[base[j] + coff]), wv, acc[j]);
    }
    for (int j = 0; j < CW; ++j)
        bemit_avx512(y0 + (size_t)(f0 + j) * BLK, acc[j], bvec, relu);
}
static inline FD_DCONV_AVX512_TARGET void bconv1x1_flat2_range_avx512(
        const float* xb, const float* w0, const float* w1, float* y0, float* y1,
        int W, int H, int IC, int Wout, int stride, int pbeg, int pend,
        const float* b0, const float* b1, bool relu) {
    int f = pbeg;
    for (; f + OWB2 <= pend; f += OWB2)
        bconv1x1_flat2_avx512<OWB2>(xb, w0, w1, y0, y1, W, H, IC, Wout, stride, f, b0, b1, relu);
    int rem = pend - f;
    while (rem >= 8) { bconv1x1_flat2_avx512<8>(xb, w0, w1, y0, y1, W, H, IC, Wout, stride, f, b0, b1, relu); f += 8; rem -= 8; }
    while (rem >= 4) { bconv1x1_flat2_avx512<4>(xb, w0, w1, y0, y1, W, H, IC, Wout, stride, f, b0, b1, relu); f += 4; rem -= 4; }
    while (rem >= 2) { bconv1x1_flat2_avx512<2>(xb, w0, w1, y0, y1, W, H, IC, Wout, stride, f, b0, b1, relu); f += 2; rem -= 2; }
    while (rem >= 1) { bconv1x1_flat2_avx512<1>(xb, w0, w1, y0, y1, W, H, IC, Wout, stride, f, b0, b1, relu); f += 1; rem -= 1; }
}
static inline FD_DCONV_AVX512_TARGET void bconv1x1_flat_range_avx512(
        const float* xb, const float* w0, float* y0,
        int W, int H, int IC, int Wout, int stride, int pbeg, int pend,
        const float* b0, bool relu) {
    int f = pbeg;
    for (; f + OWB <= pend; f += OWB)
        bconv1x1_flat_avx512<OWB>(xb, w0, y0, W, H, IC, Wout, stride, f, b0, relu);
    int rem = pend - f;
    while (rem >= 8) { bconv1x1_flat_avx512<8>(xb, w0, y0, W, H, IC, Wout, stride, f, b0, relu); f += 8; rem -= 8; }
    while (rem >= 4) { bconv1x1_flat_avx512<4>(xb, w0, y0, W, H, IC, Wout, stride, f, b0, relu); f += 4; rem -= 4; }
    while (rem >= 2) { bconv1x1_flat_avx512<2>(xb, w0, y0, W, H, IC, Wout, stride, f, b0, relu); f += 2; rem -= 2; }
    while (rem >= 1) { bconv1x1_flat_avx512<1>(xb, w0, y0, W, H, IC, Wout, stride, f, b0, relu); f += 1; rem -= 1; }
}
#endif

// scalar fallback column for the 3x3 (works for any stride; lane-wide blocked).
static inline void bconv3x3_col_scalar(
        const float* xb, const float* wocb, float* orow,
        int W, int H, int IC, int oh, int ow, int pad, int stride, int lane,
        const float* bvec = nullptr, bool relu = false) {
    float* o = orow + (size_t)ow * lane;
    for (int l = 0; l < lane; ++l) o[l] = 0.0f;
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = oh * stride + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const int ix = ow * stride + kw - pad;
            if (ix < 0 || ix >= W) continue;
            const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * lane;
            const float* col = xb + (((size_t)iy * W) + ix) * lane;
            for (int ic = 0; ic < IC; ++ic) {
                const int cb = ic / lane, ll = ic % lane;
                const float s = col[(size_t)cb * H * W * lane + ll];
                const float* wv = wk + (size_t)ic * lane;
                for (int l = 0; l < lane; ++l) o[l] += s * wv[l];
            }
        }
    }
    for (int l = 0; l < lane; ++l) {
        if (bvec) o[l] += bvec[l];
        if (relu && o[l] < 0.0f) o[l] = 0.0f;
    }
}
static inline void bconv1x1_col_scalar(
        const float* xb, const float* wocb, float* orow,
        int W, int H, int IC, int oh, int ow, int stride, int lane,
        const float* bvec = nullptr, bool relu = false) {
    const int iy = oh * stride, ix = ow * stride;
    const float* col = xb + (((size_t)iy * W) + ix) * lane;
    float* o = orow + (size_t)ow * lane;
    for (int l = 0; l < lane; ++l) o[l] = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        const int cb = ic / lane, ll = ic % lane;
        const float s = col[(size_t)cb * H * W * lane + ll];
        const float* wv = wocb + (size_t)ic * lane;
        for (int l = 0; l < lane; ++l) o[l] += s * wv[l];
    }
    for (int l = 0; l < lane; ++l) {
        if (bvec) o[l] += bvec[l];
        if (relu && o[l] < 0.0f) o[l] = 0.0f;
    }
}

// Blocked 3x3 conv compute: blocked in -> blocked out. dst ne = [16,Wout,Hout,OCB].
static void bconv3x3_compute(ggml_tensor* dst, int ith, int nth, void* userdata) {
    DConvState* st = (DConvState*)userdata;
    const ggml_tensor* xt = dst->src[0];   // blocked input [16,W,H,CB]
    const ggml_tensor* wt = dst->src[1];
    const float* xb = (const float*)xt->data;
    float* y = (float*)dst->data;
    std::call_once(st->once, [&] {
        build_pack(st, (const float*)wt->data);
        if (st->has_bias) build_pack_bias(st, (const float*)dst->src[2]->data);
    });

    const int W = st->W, H = st->H, IC = st->IC, OC = st->OC;
    const int pad = st->pad, stride = st->stride, lane = st->lane, OCB = st->OCB;
    const int Wout = (int)dst->ne[1], Hout = (int)dst->ne[2];
    const float* Wpack = st->Wpack.data();
    const float* Bpack = st->has_bias ? st->Bpack.data() : nullptr;
    const bool relu = st->do_relu;
    (void)OC;

#if defined(FD_DCONV_HAVE_AVX512)
    if (lane == 16 && stride == 1) {
        // SMALL-SPATIAL dispatch: when the output is narrower than the register
        // tile, the per-row driver below leaves most accumulators idle. Route to
        // the FLATTENED-SPATIAL kernel, which tiles consecutive pixels across rows
        // so the tile fills regardless of row width. Threshold env-overridable
        // (FACEDETECT_DCONV_FLAT_MAXW); default 13: flatten only when a row's
        // interior (Wout-2*pad) cannot fill the 2-OC tile (OWB2=12), i.e. Wout<14.
        // So the IResNet 7x7 stage flattens (big recovery), while 14x14 -- whose
        // interior is exactly 12 and already fills one full per-row strip -- and
        // the larger 28x28/56x56 stages keep the (already-saturated) per-row path.
        // Parity is bit-identical either way.
        static const int flat_maxw = [] {
            const char* e = std::getenv("FACEDETECT_DCONV_FLAT_MAXW");
            return (e && *e) ? atoi(e) : 13;
        }();
        if (Wout <= flat_maxw) {
            const int npairs = (OCB + 1) / 2;
            const int Npix = Wout * Hout;
            // Split the JOINT (pair x pixel) space contiguously so each thread
            // gets one large contiguous run -- it then tiles into full CW strips
            // (per-pair splitting would hand each thread only Npix/nth pixels,
            // which on a 7x7 map is below the tile width and re-underfills).
            const int64_t total = (int64_t)npairs * Npix;
            const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
            if (beg >= end) return;
            for (int p = (int)(beg / Npix); p < npairs && (int64_t)p * Npix < end; ++p) {
                const int ocb0 = 2 * p, ocb1 = 2 * p + 1;
                const int pbeg = (int)std::max<int64_t>(0, beg - (int64_t)p * Npix);
                const int pend = (int)std::min<int64_t>(Npix, end - (int64_t)p * Npix);
                if (pbeg >= pend) continue;
                const float* w0 = Wpack + (size_t)ocb0 * 9 * IC * lane;
                const float* b0 = Bpack ? Bpack + (size_t)ocb0 * lane : nullptr;
                float* y0 = y + (size_t)ocb0 * Npix * lane;
                if (ocb1 < OCB) {
                    const float* w1 = Wpack + (size_t)ocb1 * 9 * IC * lane;
                    const float* b1 = Bpack ? Bpack + (size_t)ocb1 * lane : nullptr;
                    float* y1 = y + (size_t)ocb1 * Npix * lane;
                    bconv3x3_flat2_range_avx512(xb, w0, w1, y0, y1, W, H, IC, Wout,
                                                pad, /*stride=*/1, pbeg, pend, b0, b1, relu);
                } else {
                    bconv3x3_flat_range_avx512(xb, w0, y0, W, H, IC, Wout,
                                               pad, /*stride=*/1, pbeg, pend, b0, relu);
                }
            }
            return;
        }
        // Work items = (OC-block PAIR, oh): each pass shares one (width-strided)
        // blocked-input broadcast across two OC blocks. Odd trailing block falls
        // back to the single-OC strip path.
        const int npairs = (OCB + 1) / 2;
        const int64_t total = (int64_t)npairs * Hout;
        const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
        for (int64_t idx = beg; idx < end; ++idx) {
            const int p  = (int)(idx / Hout);
            const int oh = (int)(idx % Hout);
            const int ocb0 = 2 * p, ocb1 = 2 * p + 1;
            if (ocb1 < OCB) {
                bconv3x3_row2_s1_avx512(xb, Wpack, y, W, H, IC, Wout, Hout, ocb0, ocb1, oh, pad, Bpack, relu);
            } else {
                float* orow = y + (((size_t)ocb0 * Hout + oh) * Wout) * lane;
                const float* wocb = Wpack + (size_t)ocb0 * 9 * IC * lane;
                const float* bvec = Bpack ? Bpack + (size_t)ocb0 * lane : nullptr;
                int ow = 0;
                for (; ow < pad && ow < Wout; ++ow)
                    bconv3x3_col_avx512(xb, wocb, orow, W, H, IC, oh, ow, pad, 1, bvec, relu);
                const int hi = Wout - 1 - pad;
                int ow0 = std::max(pad, ow);
                for (; ow0 + OWB - 1 <= hi; ow0 += OWB)
                    bconv3x3_strip_s1_avx512<OWB>(xb, wocb, orow, W, H, IC, oh, pad, ow0, bvec, relu);
                int rem = hi - ow0 + 1;
                while (rem >= 8) { bconv3x3_strip_s1_avx512<8>(xb, wocb, orow, W, H, IC, oh, pad, ow0, bvec, relu); ow0 += 8; rem -= 8; }
                while (rem >= 4) { bconv3x3_strip_s1_avx512<4>(xb, wocb, orow, W, H, IC, oh, pad, ow0, bvec, relu); ow0 += 4; rem -= 4; }
                while (rem >= 2) { bconv3x3_strip_s1_avx512<2>(xb, wocb, orow, W, H, IC, oh, pad, ow0, bvec, relu); ow0 += 2; rem -= 2; }
                while (rem >= 1) { bconv3x3_strip_s1_avx512<1>(xb, wocb, orow, W, H, IC, oh, pad, ow0, bvec, relu); ow0 += 1; rem -= 1; }
                for (int owe = std::max(Wout - pad, ow0); owe < Wout; ++owe)
                    bconv3x3_col_avx512(xb, wocb, orow, W, H, IC, oh, owe, pad, 1, bvec, relu);
            }
        }
        return;
    }
    // STRIDE-2 (downsample) register-tiled path. The stride>1 case previously ran
    // bconv3x3_col_avx512 per output column -- ONE zmm accumulator, no width tiling
    // and no 2-OC broadcast sharing -- so the four IResNet stage-transition
    // downsample convs ran at ~40 GFLOP/s (~1/5 of the stride-1 small-spatial
    // rate). Route them through the SAME flattened-spatial register-tiled kernel
    // the small-spatial stride-1 path uses (now stride-parameterized): it tiles
    // CW consecutive OUTPUT pixels with 2-OC broadcast sharing and gates out-of-
    // bounds taps with an exact +0.0, so it is bit-identical to the per-column
    // path (same kh,kw,ic order) while filling the zmm accumulator file. The
    // stride>1 output maps (56/28/14/7) are exactly the small-output regime the
    // flat tiling was built for, so no per-row variant is needed.
    if (lane == 16 && stride > 1 && dconv_stride2_flat()) {
        const int npairs = (OCB + 1) / 2;
        const int Npix = Wout * Hout;
        const int64_t total = (int64_t)npairs * Npix;
        const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
        if (beg >= end) return;
        for (int p = (int)(beg / Npix); p < npairs && (int64_t)p * Npix < end; ++p) {
            const int ocb0 = 2 * p, ocb1 = 2 * p + 1;
            const int pbeg = (int)std::max<int64_t>(0, beg - (int64_t)p * Npix);
            const int pend = (int)std::min<int64_t>(Npix, end - (int64_t)p * Npix);
            if (pbeg >= pend) continue;
            const float* w0 = Wpack + (size_t)ocb0 * 9 * IC * lane;
            const float* b0 = Bpack ? Bpack + (size_t)ocb0 * lane : nullptr;
            float* y0 = y + (size_t)ocb0 * Npix * lane;
            if (ocb1 < OCB) {
                const float* w1 = Wpack + (size_t)ocb1 * 9 * IC * lane;
                const float* b1 = Bpack ? Bpack + (size_t)ocb1 * lane : nullptr;
                float* y1 = y + (size_t)ocb1 * Npix * lane;
                bconv3x3_flat2_range_avx512(xb, w0, w1, y0, y1, W, H, IC, Wout,
                                            pad, stride, pbeg, pend, b0, b1, relu);
            } else {
                bconv3x3_flat_range_avx512(xb, w0, y0, W, H, IC, Wout,
                                           pad, stride, pbeg, pend, b0, relu);
            }
        }
        return;
    }
#endif
    const int64_t total = (int64_t)OCB * Hout;
    const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
    for (int64_t idx = beg; idx < end; ++idx) {
        const int ocb = (int)(idx / Hout);
        const int oh  = (int)(idx % Hout);
        float* orow = y + (((size_t)ocb * Hout + oh) * Wout) * lane;
        const float* wocb = Wpack + (size_t)ocb * 9 * IC * lane;
        const float* bvec = Bpack ? Bpack + (size_t)ocb * lane : nullptr;
#if defined(FD_DCONV_HAVE_AVX512)
        if (lane == 16) {   // stride > 1 (AVX-512 build with global flag, no flat path above)
            for (int ow = 0; ow < Wout; ++ow)
                bconv3x3_col_avx512(xb, wocb, orow, W, H, IC, oh, ow, pad, stride, bvec, relu);
            continue;
        }
#endif
        for (int ow = 0; ow < Wout; ++ow)
            bconv3x3_col_scalar(xb, wocb, orow, W, H, IC, oh, ow, pad, stride, lane, bvec, relu);
    }
}

// Blocked 1x1 strided conv compute. dst ne = [16,Wout,Hout,OCB].
static void bconv1x1_compute(ggml_tensor* dst, int ith, int nth, void* userdata) {
    DConvState* st = (DConvState*)userdata;
    const ggml_tensor* xt = dst->src[0];
    const ggml_tensor* wt = dst->src[1];
    const float* xb = (const float*)xt->data;
    float* y = (float*)dst->data;
    std::call_once(st->once, [&] {
        build_pack_1x1(st, (const float*)wt->data);
        if (st->has_bias) build_pack_bias(st, (const float*)dst->src[2]->data);
    });

    const int W = st->W, H = st->H, IC = st->IC;
    const int stride = st->stride, lane = st->lane, OCB = st->OCB;
    const int Wout = (int)dst->ne[1], Hout = (int)dst->ne[2];
    const float* Wpack = st->Wpack.data();
    const float* Bpack = st->has_bias ? st->Bpack.data() : nullptr;
    const bool relu = st->do_relu;

#if defined(FD_DCONV_HAVE_AVX512)
    if (lane == 16 && dconv_stride2_flat()) {
        // FLATTENED-SPATIAL register-tiled path (same recovery as the 3x3 stride-2
        // downsample): tile CW consecutive output pixels with 2-OC broadcast
        // sharing instead of the per-column single-zmm path, which left the IResNet
        // 1x1 shortcuts at ~40 GFLOP/s. The downsample 1x1s output small maps
        // (56/28/14/7), exactly the regime that underfills a per-row tile.
        const int npairs = (OCB + 1) / 2;
        const int Npix = Wout * Hout;
        const int64_t total = (int64_t)npairs * Npix;
        const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
        if (beg >= end) return;
        for (int p = (int)(beg / Npix); p < npairs && (int64_t)p * Npix < end; ++p) {
            const int ocb0 = 2 * p, ocb1 = 2 * p + 1;
            const int pbeg = (int)std::max<int64_t>(0, beg - (int64_t)p * Npix);
            const int pend = (int)std::min<int64_t>(Npix, end - (int64_t)p * Npix);
            if (pbeg >= pend) continue;
            const float* w0 = Wpack + (size_t)ocb0 * IC * lane;
            const float* b0 = Bpack ? Bpack + (size_t)ocb0 * lane : nullptr;
            float* y0 = y + (size_t)ocb0 * Npix * lane;
            if (ocb1 < OCB) {
                const float* w1 = Wpack + (size_t)ocb1 * IC * lane;
                const float* b1 = Bpack ? Bpack + (size_t)ocb1 * lane : nullptr;
                float* y1 = y + (size_t)ocb1 * Npix * lane;
                bconv1x1_flat2_range_avx512(xb, w0, w1, y0, y1, W, H, IC, Wout, stride, pbeg, pend, b0, b1, relu);
            } else {
                bconv1x1_flat_range_avx512(xb, w0, y0, W, H, IC, Wout, stride, pbeg, pend, b0, relu);
            }
        }
        return;
    }
#endif
    const int64_t total = (int64_t)OCB * Hout;
    const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
    for (int64_t idx = beg; idx < end; ++idx) {
        const int ocb = (int)(idx / Hout);
        const int oh  = (int)(idx % Hout);
        float* orow = y + (((size_t)ocb * Hout + oh) * Wout) * lane;
        const float* wocb = Wpack + (size_t)ocb * IC * lane;
        const float* bvec = Bpack ? Bpack + (size_t)ocb * lane : nullptr;
        for (int ow = 0; ow < Wout; ++ow) {
#if defined(FD_DCONV_HAVE_AVX512)
            if (lane == 16) { bconv1x1_col_avx512(xb, wocb, orow, W, H, IC, oh, ow, stride, bvec, relu); continue; }
#endif
            bconv1x1_col_scalar(xb, wocb, orow, W, H, IC, oh, ow, stride, lane, bvec, relu);
        }
    }
}

// reorder NCHW [W,H,C,1] -> blocked [16,W,H,CB] (zero-pads channels C..CB*16-1).
static void reorder_in_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor* xt = dst->src[0];
    const float* x = (const float*)xt->data;
    float* y = (float*)dst->data;
    const int W = (int)xt->ne[0], H = (int)xt->ne[1], C = (int)xt->ne[2];
    const int CB = (int)dst->ne[3];
    const size_t HW = (size_t)H * W;
    const int64_t total = (int64_t)CB * H;
    const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
    for (int64_t idx = beg; idx < end; ++idx) {
        const int cb = (int)(idx / H), h = (int)(idx % H);
        for (int w = 0; w < W; ++w) {
            float* o = y + (((size_t)cb * H + h) * W + w) * BLK;
            for (int l = 0; l < BLK; ++l) {
                const int c = cb * BLK + l;
                o[l] = (c < C) ? x[(size_t)c * HW + (size_t)h * W + w] : 0.0f;
            }
        }
    }
}

// reorder blocked [16,W,H,CB] -> NCHW [W,H,C,1] (drops padding lanes).
static void reorder_out_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor* xt = dst->src[0];
    const float* xb = (const float*)xt->data;
    float* y = (float*)dst->data;
    const int W = (int)dst->ne[0], H = (int)dst->ne[1], C = (int)dst->ne[2];
    const size_t HW = (size_t)H * W;
    const int64_t beg = (int64_t)C * ith / nth, end = (int64_t)C * (ith + 1) / nth;
    for (int64_t c = beg; c < end; ++c) {
        const int cb = (int)c >> 4, l = (int)c & 15;
        float* yc = y + (size_t)c * HW;
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                yc[(size_t)h * W + w] = xb[(((size_t)cb * H + h) * W + w) * BLK + l];
    }
}

// blocked per-channel bias add (channel c = cb*16+l; skip padding lanes c>=C).
static void bbias_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor* xt = dst->src[0];
    const ggml_tensor* bt = dst->src[1];
    const float* xb = (const float*)xt->data;
    const float* b = (const float*)bt->data;
    float* y = (float*)dst->data;
    const int W = (int)dst->ne[1], H = (int)dst->ne[2], CB = (int)dst->ne[3];
    const int C = (int)bt->ne[0];
    const size_t HW = (size_t)H * W;
    const int64_t beg = (int64_t)CB * ith / nth, end = (int64_t)CB * (ith + 1) / nth;
    for (int64_t cb = beg; cb < end; ++cb) {
        for (size_t p = 0; p < HW; ++p) {
            float* o = y + ((size_t)cb * HW + p) * BLK;
            const float* in = xb + ((size_t)cb * HW + p) * BLK;
            for (int l = 0; l < BLK; ++l) {
                const int c = (int)cb * BLK + l;
                o[l] = in[l] + ((c < C) ? b[c] : 0.0f);
            }
        }
    }
}

// blocked per-channel affine y = x*scale[c] + shift[c] (the IResNet pre-activation
// BatchNorm, host-folded gamma/var into scale and beta/mean into shift). Channel
// c = cb*16+l; padding lanes c>=C pass through unchanged (they carry no signal).
static void bscale_shift_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor* xt = dst->src[0];
    const ggml_tensor* st_ = dst->src[1];
    const ggml_tensor* sh = dst->src[2];
    const float* xb = (const float*)xt->data;
    const float* sc = (const float*)st_->data;
    const float* shf = (const float*)sh->data;
    float* y = (float*)dst->data;
    const int W = (int)dst->ne[1], H = (int)dst->ne[2], CB = (int)dst->ne[3];
    const int C = (int)st_->ne[0];
    const size_t HW = (size_t)H * W;
    const int64_t beg = (int64_t)CB * ith / nth, end = (int64_t)CB * (ith + 1) / nth;
    for (int64_t cb = beg; cb < end; ++cb) {
        for (size_t p = 0; p < HW; ++p) {
            float* o = y + ((size_t)cb * HW + p) * BLK;
            const float* in = xb + ((size_t)cb * HW + p) * BLK;
            for (int l = 0; l < BLK; ++l) {
                const int c = (int)cb * BLK + l;
                o[l] = (c < C) ? in[l] * sc[c] + shf[c] : in[l];
            }
        }
    }
}

// blocked per-channel PReLU y = x>0 ? x : alpha[c]*x. Channel c = cb*16+l; padding
// lanes c>=C pass through (zero in, zero out either branch).
static void bprelu_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor* xt = dst->src[0];
    const ggml_tensor* at = dst->src[1];
    const float* xb = (const float*)xt->data;
    const float* a = (const float*)at->data;
    float* y = (float*)dst->data;
    const int W = (int)dst->ne[1], H = (int)dst->ne[2], CB = (int)dst->ne[3];
    const int C = (int)ggml_nelements(at);
    const size_t HW = (size_t)H * W;
    const int64_t beg = (int64_t)CB * ith / nth, end = (int64_t)CB * (ith + 1) / nth;
    for (int64_t cb = beg; cb < end; ++cb) {
        for (size_t p = 0; p < HW; ++p) {
            float* o = y + ((size_t)cb * HW + p) * BLK;
            const float* in = xb + ((size_t)cb * HW + p) * BLK;
            for (int l = 0; l < BLK; ++l) {
                const int c = (int)cb * BLK + l;
                const float v = in[l];
                const float slope = (c < C) ? a[c] : 0.0f;
                o[l] = v > 0.0f ? v : slope * v;
            }
        }
    }
}

// blocked ReLU (whole buffer, no channel awareness needed).
static void brelu_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor* xt = dst->src[0];
    const float* x = (const float*)xt->data;
    float* y = (float*)dst->data;
    const int64_t n = ggml_nelements(dst);
    const int64_t beg = n * ith / nth, end = n * (ith + 1) / nth;
    for (int64_t i = beg; i < end; ++i) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

// blocked elementwise add (residual): dst = src0 + src1.
static void badd_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const float* a = (const float*)dst->src[0]->data;
    const float* b = (const float*)dst->src[1]->data;
    float* y = (float*)dst->data;
    const int64_t n = ggml_nelements(dst);
    const int64_t beg = n * ith / nth, end = n * (ith + 1) / nth;
    for (int64_t i = beg; i < end; ++i) y[i] = a[i] + b[i];
}

// fused residual add + ReLU: dst = max(0, src0 + src1). One pass instead of two.
static void badd_relu_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const float* a = (const float*)dst->src[0]->data;
    const float* b = (const float*)dst->src[1]->data;
    float* y = (float*)dst->data;
    const int64_t n = ggml_nelements(dst);
    const int64_t beg = n * ith / nth, end = n * (ith + 1) / nth;
    for (int64_t i = beg; i < end; ++i) { float v = a[i] + b[i]; y[i] = v > 0.0f ? v : 0.0f; }
}

struct StateKey {
    const void* wdata; int W, H, IC, OC, N, pad, stride, kind, flags;
    bool operator==(const StateKey& o) const {
        return wdata == o.wdata && W == o.W && H == o.H && IC == o.IC &&
               OC == o.OC && N == o.N && pad == o.pad &&
               stride == o.stride && kind == o.kind && flags == o.flags;
    }
};
struct StateKeyHash {
    size_t operator()(const StateKey& k) const {
        size_t h = (size_t)k.wdata;
        auto mix = [&h](int v) { h = h * 1000003u + (size_t)(uint32_t)v; };
        mix(k.W); mix(k.H); mix(k.IC); mix(k.OC); mix(k.N); mix(k.pad);
        mix(k.stride); mix(k.kind); mix(k.flags);
        return h;
    }
};
static std::mutex g_states_mtx;
static std::unordered_map<StateKey, DConvState*, StateKeyHash> g_states;

static int select_lane() {
#if defined(FD_DCONV_HAVE_AVX512)
    if (dconv_use_avx512()) return 16;
#endif
#if defined(__AVX2__)
    return 8;
#else
    return 8;   // scalar still uses an 8-wide pack
#endif
}

static DConvState* get_state(ggml_tensor* w, ggml_tensor* x, int pad) {
    const int W = (int)x->ne[0], H = (int)x->ne[1], IC = (int)x->ne[2], N = (int)x->ne[3];
    const int OC = (int)w->ne[3];
    StateKey key{ w->data, W, H, IC, OC, N, pad, 1, 0, 0 };
    std::lock_guard<std::mutex> lk(g_states_mtx);
    auto it = g_states.find(key);
    if (it != g_states.end()) return it->second;
    DConvState* st = new DConvState();
    st->W = W; st->H = H; st->IC = IC; st->OC = OC; st->N = N; st->pad = pad;
    st->lane = select_lane();
    st->OCB = (OC + st->lane - 1) / st->lane;
    st->wdata = w->data;
    g_states[key] = st;
    return st;
}

// Blocked-island state: input is the blocked buffer [16,W,H,CB] so W,H come from
// ne[1],ne[2]; IC is the TRUE input channel count (the weight's ne[2]); lane is
// ALWAYS 16 (nChw16c). kind 0 = 3x3, 1 = 1x1. flags encodes fused bias/relu so a
// reused weight with a different epilogue gets its own packed state.
static DConvState* get_state_blocked(ggml_tensor* w, ggml_tensor* xb,
                                     int pad, int stride, int kind,
                                     bool has_bias, bool do_relu) {
    const int W = (int)xb->ne[1], H = (int)xb->ne[2];
    const int IC = (int)w->ne[2], OC = (int)w->ne[3], N = 1;
    const int flags = (has_bias ? 1 : 0) | (do_relu ? 2 : 0);
    StateKey key{ w->data, W, H, IC, OC, N, pad, stride, kind, flags };
    std::lock_guard<std::mutex> lk(g_states_mtx);
    auto it = g_states.find(key);
    if (it != g_states.end()) return it->second;
    DConvState* st = new DConvState();
    st->W = W; st->H = H; st->IC = IC; st->OC = OC; st->N = N;
    st->pad = pad; st->stride = stride; st->kind = kind;
    st->has_bias = has_bias; st->do_relu = do_relu;
    st->lane = BLK;
    st->OCB = (OC + BLK - 1) / BLK;
    st->wdata = w->data;
    g_states[key] = st;
    return st;
}

} // namespace

ggml_tensor* directconv_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, int pad) {
    const int OC = (int)w->ne[3];
    const int N  = (int)x->ne[3];
    const int Wout = (int)x->ne[0] + 2 * pad - 2;
    const int Hout = (int)x->ne[1] + 2 * pad - 2;
    DConvState* st = get_state(w, x, pad);
    ggml_tensor* args[2] = { x, w };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, Wout, Hout, OC, N,
                          args, 2, dconv_compute, GGML_N_TASKS_MAX, st);
}

// ---- blocked-island public ops --------------------------------------------

bool directconv_blocked_available() {
#if defined(FD_DCONV_HAVE_AVX512)
    return dconv_use_avx512();   // the blocked island only has a zmm fast path;
#else
    return false;                // non-AVX512 hosts keep the per-conv directconv
#endif                           // (which has an AVX2 path), avoiding a scalar regression.
}

ggml_tensor* blocked_reorder_in(ggml_context* ctx, ggml_tensor* x_nchw) {
    const int W = (int)x_nchw->ne[0], H = (int)x_nchw->ne[1], C = (int)x_nchw->ne[2];
    const int CB = (C + BLK - 1) / BLK;
    ggml_tensor* args[1] = { x_nchw };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, BLK, W, H, CB,
                          args, 1, reorder_in_compute, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor* blocked_reorder_out(ggml_context* ctx, ggml_tensor* xb, int C) {
    const int W = (int)xb->ne[1], H = (int)xb->ne[2];
    ggml_tensor* args[1] = { xb };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, W, H, C, 1,
                          args, 1, reorder_out_compute, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor* blocked_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* xb,
                             int pad, int stride, ggml_tensor* bias, bool do_relu) {
    const int W = (int)xb->ne[1], H = (int)xb->ne[2];
    const int OC = (int)w->ne[3];
    const int Wout = (W + 2 * pad - 3) / stride + 1;
    const int Hout = (H + 2 * pad - 3) / stride + 1;
    const int OCB = (OC + BLK - 1) / BLK;
    DConvState* st = get_state_blocked(w, xb, pad, stride, /*kind=*/0,
                                       bias != nullptr, do_relu);
    ggml_tensor* args[3] = { xb, w, bias };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, BLK, Wout, Hout, OCB,
                          args, bias ? 3 : 2, bconv3x3_compute, GGML_N_TASKS_MAX, st);
}

ggml_tensor* blocked_conv1x1(ggml_context* ctx, ggml_tensor* w, ggml_tensor* xb,
                             int stride, ggml_tensor* bias, bool do_relu) {
    const int W = (int)xb->ne[1], H = (int)xb->ne[2];
    const int OC = (int)w->ne[3];
    const int Wout = (W - 1) / stride + 1;
    const int Hout = (H - 1) / stride + 1;
    const int OCB = (OC + BLK - 1) / BLK;
    DConvState* st = get_state_blocked(w, xb, /*pad=*/0, stride, /*kind=*/1,
                                       bias != nullptr, do_relu);
    ggml_tensor* args[3] = { xb, w, bias };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, BLK, Wout, Hout, OCB,
                          args, bias ? 3 : 2, bconv1x1_compute, GGML_N_TASKS_MAX, st);
}

ggml_tensor* blocked_bias(ggml_context* ctx, ggml_tensor* xb, ggml_tensor* bias) {
    ggml_tensor* args[2] = { xb, bias };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, xb->ne[0], xb->ne[1], xb->ne[2], xb->ne[3],
                          args, 2, bbias_compute, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor* blocked_relu(ggml_context* ctx, ggml_tensor* xb) {
    ggml_tensor* args[1] = { xb };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, xb->ne[0], xb->ne[1], xb->ne[2], xb->ne[3],
                          args, 1, brelu_compute, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor* blocked_add(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    ggml_tensor* args[2] = { a, b };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, a->ne[0], a->ne[1], a->ne[2], a->ne[3],
                          args, 2, badd_compute, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor* blocked_add_relu(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    ggml_tensor* args[2] = { a, b };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, a->ne[0], a->ne[1], a->ne[2], a->ne[3],
                          args, 2, badd_relu_compute, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor* blocked_scale_shift(ggml_context* ctx, ggml_tensor* xb,
                                 ggml_tensor* scale, ggml_tensor* shift) {
    ggml_tensor* args[3] = { xb, scale, shift };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, xb->ne[0], xb->ne[1], xb->ne[2], xb->ne[3],
                          args, 3, bscale_shift_compute, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor* blocked_prelu(ggml_context* ctx, ggml_tensor* xb, ggml_tensor* alpha) {
    ggml_tensor* args[2] = { xb, alpha };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, xb->ne[0], xb->ne[1], xb->ne[2], xb->ne[3],
                          args, 2, bprelu_compute, GGML_N_TASKS_MAX, nullptr);
}

} // namespace fd

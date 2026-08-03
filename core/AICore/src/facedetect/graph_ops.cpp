#include "graph_ops.hpp"
#include "backend.hpp"
#include "winograd.hpp"
#include "directconv.hpp"
#include "model_loader.hpp"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace fd {

namespace {
// One folded BatchNorm affine: the inference-mode BN reduces to a per-channel
// y = x*scale + shift with scale = gamma/sqrt(var+eps) and shift = beta -
// mean*scale. These depend only on the (constant) running stats, so deriving
// them as LIVE graph nodes (add1 + sqrt + div + sub + mul, ~5 ops + 4 weight
// leaves) on EVERY compute is pure graph-size + launch overhead. We fold them to
// host constants ONCE per loaded model and reuse them.
struct BNFold { std::vector<float> scale, shift; };

// Process-lifetime cache keyed by the gamma weight tensor pointer (unique and
// stable per loaded model). Node-based map storage keeps the float buffers'
// addresses stable, so the host pointers handed to graph_input_tensor safely
// outlive the deferred host->device input copy of any later compute.
std::unordered_map<const ggml_tensor*, BNFold> g_bn_fold;
std::mutex g_bn_fold_mu;

// Whether the in-progress graph build permits the F(4x4,3x3) Winograd variant on
// its large 3x3 s1p1 convs (SCRFD only; see graph_ops.hpp). thread_local so
// parallel graph builds stay isolated, and so an unwound build (WinogradF4Scope
// destructor) leaves the next build's default (F2B) intact.
thread_local bool g_wino_f4_scope = false;
} // namespace

void set_winograd_f4_scope(bool on) { g_wino_f4_scope = on; }
bool winograd_f4_scope() { return g_wino_f4_scope; }

void bn_fold_params(const ModelLoader& ml, const char* prefix, float eps,
                    const float** scale, const float** shift, int64_t* count) {
    const std::string p = prefix;
    ggml_tensor* gamma = clone_weight(nullptr, ml, (p + ".weight").c_str());
    {
        std::lock_guard<std::mutex> lk(g_bn_fold_mu);
        auto it = g_bn_fold.find(gamma);
        if (it != g_bn_fold.end()) {
            *scale = it->second.scale.data();
            *shift = it->second.shift.data();
            *count = (int64_t)it->second.scale.size();
            return;
        }
    }
    ggml_tensor* beta = clone_weight(nullptr, ml, (p + ".bias").c_str());
    ggml_tensor* mean = clone_weight(nullptr, ml, (p + ".running_mean").c_str());
    ggml_tensor* var  = clone_weight(nullptr, ml, (p + ".running_var").c_str());
    const int64_t C = ggml_nelements(gamma);
    // 1-D BN stats stay F32 in the GGUF (converter skips quant for ndim<2), so a
    // raw float readback reproduces the reference exactly. ggml_backend_tensor_get
    // is backend-agnostic (works for the CPU-from-ptr buffer and a device buffer).
    std::vector<float> g(C), b(C), m(C), v(C);
    ggml_backend_tensor_get(gamma, g.data(), 0, C * sizeof(float));
    ggml_backend_tensor_get(beta,  b.data(), 0, C * sizeof(float));
    ggml_backend_tensor_get(mean,  m.data(), 0, C * sizeof(float));
    ggml_backend_tensor_get(var,   v.data(), 0, C * sizeof(float));
    BNFold f;
    f.scale.resize(C);
    f.shift.resize(C);
    for (int64_t i = 0; i < C; ++i) {
        // Same f32 op order as the prior live-node path: sqrtf(var+eps), divide,
        // then beta - mean*scale. Algebraic identity, no numeric reordering.
        const float s = g[i] / std::sqrt(v[i] + eps);
        f.scale[i] = s;
        f.shift[i] = b[i] - m[i] * s;
    }
    std::lock_guard<std::mutex> lk(g_bn_fold_mu);
    const BNFold& kept = g_bn_fold.emplace(gamma, std::move(f)).first->second;
    *scale = kept.scale.data();
    *shift = kept.shift.data();
    *count = (int64_t)kept.scale.size();
}

ggml_tensor* conv2d(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                    const char* w, const char* b, int stride, int pad, bool relu) {
    ggml_tensor* W = clone_weight(ctx, ml, w);
    // Conv kernel routing (shared by SCRFD detector + ArcFace recognizer, both
    // dominated by 3x3 stride-1 convs). Two CPU conv kernels trade off by memory
    // bandwidth pressure:
    //   - ggml_conv_2d (im2col + mul_mat): expands each output into a 9x patch
    //     buffer, then a tinyBLAS GEMM. The GEMM is very fast per FLOP, but the
    //     9x buffer is a large memory stream.
    //   - ggml_conv_2d_direct: no patch buffer, far less memory traffic, but a
    //     less throughput-dense kernel.
    // Measured crossover (buffalo_l 640x640 detect, min-of-5, this 16-core box):
    // im2col wins at 1/2/4 threads (spare bandwidth -> GEMM throughput dominates),
    // direct wins ~24% at 8 threads (the 9x stream saturates memory bandwidth and
    // direct's lighter traffic pulls ahead: 135 vs 177 ms). The process-global
    // default is min(hw,8) threads, so on an 8+-logical-core box (the common
    // deployment target) the default lands in direct's win region; smaller boxes
    // default below the crossover and keep im2col. 1x1 convs are pure GEMM (no 9x
    // expansion) so they always stay on im2col. On GPU the im2col + mul_mat
    // (cuBLAS-class) kernel beats the basic direct CUDA conv, so keep im2col
    // there too. Mirrors depth-anything.cpp's da::conv2d device-aware routing,
    // with the extra thread-count gate this model's smaller feature maps need.
    // A/B override via FACEDETECT_CONV: "direct" | "im2col" | "auto" (default).
    constexpr int kDirectConvMinThreads = 8;  // bandwidth-saturation crossover
    const bool kgt1 = (W->ne[0] > 1 || W->ne[1] > 1);
    bool direct = backend_is_cpu() && kgt1 &&
                  global_backend().n_threads() >= kDirectConvMinThreads;
    if (const char* mode = std::getenv("FACEDETECT_CONV")) {
        if (!std::strcmp(mode, "direct"))      direct = true;
        else if (!std::strcmp(mode, "im2col")) direct = false;
        // "auto" (or anything else) keeps the device+thread default above.
    }

    // AVX2/AVX-512 Winograd F(2x2,3x3) for 3x3 stride-1 pad-1 convs, which
    // dominate BOTH the SCRFD detector (backbone/neck/head 3x3 stems) and the
    // ArcFace IResNet body. Winograd does ~2.25x fewer multiplies than
    // direct/im2col and feeds the runtime-CPUID-dispatched winograd-domain GEMM
    // microkernel (a double win), so it beats tinyBLAS sgemm here once the map is
    // big enough to amortize the input/output transforms. Only valid for CPU,
    // KW==KH==3, stride==1, pad==1, F32. The bias add + relu below are unchanged
    // (Winograd applies neither). A/B override via FACEDETECT_WINOGRAD: "on" (all
    // eligible) | "off" | "auto".
    //
    // Threshold (kWinoMinSize): map's min spatial dim must reach this to route
    // through Winograd. Swept 80->56->28->14->7 on buffalo_l (this 16-core box,
    // GGML_NATIVE=OFF), gating BOTH ArcFace recognizer and SCRFD detect at 1t/8t.
    // Lowering it monotonically improves both at 1t (recognizer 135->78ms,
    // detect 148->121ms) and at 8t down to 14, where the recognizer body's 14x14
    // map still pays. 7x7 maps do NOT pay at the production default of min(hw,8)
    // threads (recognizer 8t regresses 20.6->24ms, detect flat), so 14 is the
    // optimum: best-or-tied at 8t for both models AND a large 1t win. It does not
    // regress SCRFD - detect improves at every thread count - so the recognizer
    // and detector share one lowered threshold (no per-context gate needed).
    constexpr int64_t kWinoMinSize = 14;
    const bool k3 = (W->ne[0] == 3 && W->ne[1] == 3 && W->ne[2] == x->ne[2]);
    const bool s1p1 = (stride == 1 && pad == 1);
    const int64_t inMin = std::min(x->ne[0], x->ne[1]);
    bool wino = backend_is_cpu() && k3 && s1p1 && inMin >= kWinoMinSize;
    if (const char* wmode = std::getenv("FACEDETECT_WINOGRAD")) {
        if (!std::strcmp(wmode, "on"))       wino = backend_is_cpu() && k3 && s1p1;
        else if (!std::strcmp(wmode, "off")) wino = false;
        // "auto" (or anything else) keeps the size-gated default above.
    }

    // PROBE: route the SCRFD detector's large early maps (320/160/80, where ~88%
    // of detector FLOPs sit) through F(4x4,3x3) instead of F(2x2). F4 does ~4x
    // fewer multiplies than direct (vs F2's 2.25x), but its 1/6,1/24 transform
    // entries condition f32 worse, so it is gated by the detect 1px + embedding
    // cosine parity suite and settled by the A/B bench. ONLY SCRFD opts in (it
    // sets the F4 scope around its graph build): the ArcFace recognizer and every
    // small map stay on the parity-exact F(2x2) path. The recognizer parity is
    // independent of this flag, since its golden-landmark gate bypasses detection.
    // (prefer_f4 is finalized from use_wino once the kernel choice below settles.)

    // Kernel choice within the Winograd-eligible 3x3-s1-p1 set. The MLAS-class
    // blocked nChw16c register-tiled DIRECT kernel (directconv_conv3x3, ported
    // from the voice-detect.cpp spike) beats Winograd ONLY on deep-channel,
    // small-spatial maps at high thread counts: the ArcFace recognizer body
    // (28x28/14x14 stages) gains ~+27% @8t. SCRFD's detector convs MUST stay on
    // Winograd at every thread count -- its large-spatial/low-channel maps favor
    // Winograd's 2.25x fewer multiplies, and directconv regresses SCRFD detect
    // (-46% @1t / -12% @8t). Two-part gate keeps the two models apart:
    //   1. SCRFD-exclusion (!winograd_f4_scope()): SCRFD builds its ENTIRE graph
    //      inside the F4 Winograd scope, so this guarantees no SCRFD conv ever
    //      reaches directconv. A pure shape gate could not -- SCRFD's stride-32
    //      backbone/head maps (20x20, 224/256ch) are themselves small-spatial +
    //      deep-channel and would be wrongly captured, regressing the detector.
    //   2. shape gate: among the remaining (non-SCRFD) eligible convs, take
    //      directconv only for the deep-channel/small-spatial regime where it
    //      wins -- input spatial min <= kDConvMaxSpatial AND input channels >=
    //      kDConvMinChannels. The recognizer's 28x28/14x14 body lands here; its
    //      56x56 (64ch) stem stays on Winograd (large-spatial favors Winograd).
    // directconv needs F32 weights + N==1; the eligible convs satisfy both, else
    // fall back to Winograd. Overrides via FACEDETECT_CONV: "winograd" forces
    // Winograd for ALL eligible convs (directconv off); "directconv" forces
    // directconv for ALL eligible convs (incl. SCRFD, for A/B only); "direct" or
    // "im2col" drop BOTH specialized kernels so the stock ggml kernel runs
    // everywhere. directconv carries its own runtime-CPUID AVX-512/AVX2 dispatch
    // (FACEDETECT_DISABLE_AVX512 forces its AVX2 path, shared with winograd.cpp).
    constexpr int64_t kDConvMaxSpatial  = 28;   // input spatial min upper bound
    constexpr int64_t kDConvMinChannels = 64;   // input channel lower bound
    const bool dconv_shape = inMin <= kDConvMaxSpatial && x->ne[2] >= kDConvMinChannels;
    bool dconv = wino && !winograd_f4_scope() && dconv_shape;
    if (const char* cmode = std::getenv("FACEDETECT_CONV")) {
        if (!std::strcmp(cmode, "winograd"))        dconv = false;
        else if (!std::strcmp(cmode, "directconv")) dconv = wino;   // force all eligible
        else if (!std::strcmp(cmode, "direct") ||
                 !std::strcmp(cmode, "im2col"))      { dconv = false; wino = false; }
        // "auto" (or anything else) keeps the shape-gated default above.
    }
    if (dconv && (W->type != GGML_TYPE_F32 || x->ne[3] != 1)) dconv = false;
    const bool use_wino = wino && !dconv;
    const bool prefer_f4 = use_wino && winograd_f4_scope();

    // CUDA + cuDNN: emit GGML_OP_CONV_2D (ggml_conv_2d_direct) instead of
    // ggml_conv_2d's im2col + cuBLAS/cutlass SGEMM. ggml-cuda routes that op to a
    // cuDNN implicit-GEMM (cudnn-conv.cu): the convolution is streamed from the
    // NCHW activations with NO im2col global-memory spill (that one im2col kernel
    // is ~50% of SCRFD detect's GPU time, nsys). ggml's conv tensors are already
    // in cuDNN NCHW order, so consecutive convs stay NCHW with zero transpose tax.
    // On GPU the CPU kernels (winograd/directconv/direct) are all gated off, so
    // this captures every detector/recognizer conv (3x3 backbone/neck/head stems
    // and the 1x1 lateral/shortcut convs alike). The output layout [W,H,OC,N]
    // matches ggml_conv_2d, so the bias-add + relu epilogue below is unchanged.
    // A/B: FACEDETECT_CONV=im2col forces the legacy path (never emit the op);
    // the kernel-level kill switch GGML_CUDA_USE_CUDNN=0 makes ggml-cuda fall back
    // to the native conv on the same binary.
    bool cudnn = false;
#ifdef FACEDETECT_GGML_CUDNN
    cudnn = !backend_is_cpu();
    if (const char* mode = std::getenv("FACEDETECT_CONV")) {
        if (!std::strcmp(mode, "im2col")) cudnn = false;
        else if (!std::strcmp(mode, "cudnn")) cudnn = !backend_is_cpu();
        // "auto" (or anything else) keeps the CUDA default above.
    }
#endif

    ggml_tensor* B = clone_weight_opt(ctx, ml, b);
    // Bias+ReLU fusion: for relu-activated Winograd convs, fold bias + max(0,x)
    // into the winograd output (inverse) transform store, removing the separate
    // ggml_add + ggml_relu bandwidth pass over the whole [Wout,Hout,OC,N] output.
    // The activation is known here at graph-build time. Only the Winograd path
    // fuses; directconv/im2col/direct keep the post-conv add+relu below.
    const bool wino_fused = use_wino && relu;
    if (dconv)            x = directconv_conv3x3(ctx, W, x, pad);
    else if (use_wino)    x = winograd_conv3x3(ctx, W, x, pad, prefer_f4,
                                               wino_fused ? B : nullptr, relu);
    else if (cudnn)       x = ggml_conv_2d_direct(ctx, W, x, stride, stride, pad, pad, 1, 1);
    else if (direct)      x = ggml_conv_2d_direct(ctx, W, x, stride, stride, pad, pad, 1, 1);
    else                  x = ggml_conv_2d(ctx, W, x, stride, stride, pad, pad, 1, 1);
    if (!wino_fused) {
        if (B) {
            const int64_t OC = x->ne[2];
            x = ggml_add(ctx, x, ggml_reshape_4d(ctx, B, 1, 1, OC, 1));
        }
        if (relu) x = ggml_relu(ctx, x);
    }
    return x;
}

namespace {
// Fused (bias-add + per-channel PReLU) CPU kernel. dst/a are contiguous f32
// [W,H,C,N]; b (bias) and c (alpha) are [C]. Each task strides over the (C*N)
// planes so the per-plane bias/alpha are loaded once and the W*H run is a tight
// FMA-free scalar loop the compiler auto-vectorizes.
void fused_prelu_bias_cpu(ggml_tensor* dst, const ggml_tensor* a,
                          const ggml_tensor* b, const ggml_tensor* c,
                          int ith, int nth, void* /*userdata*/) {
    const int64_t plane = a->ne[0] * a->ne[1];
    const int64_t C     = a->ne[2];
    const int64_t rows  = C * a->ne[3];
    const float* ad = (const float*)a->data;
    const float* bd = (const float*)b->data;
    const float* cd = (const float*)c->data;
    float*       dd = (float*)dst->data;
    for (int64_t r = ith; r < rows; r += nth) {
        const int64_t ch = r % C;
        const float bias = bd[ch], alpha = cd[ch];
        const float* ar = ad + r * plane;
        float*       dr = dd + r * plane;
        for (int64_t i = 0; i < plane; ++i) {
            const float v = ar[i] + bias;
            dr[i] = v > 0.0f ? v : alpha * v;
        }
    }
}
} // namespace

ggml_tensor* fused_prelu_bias(ggml_context* ctx, ggml_tensor* x,
                              ggml_tensor* bias, ggml_tensor* alpha) {
    return ggml_map_custom3(ctx, x, bias, alpha, fused_prelu_bias_cpu,
                            GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor* prelu(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha) {
    const int64_t C = x->ne[2];
    ggml_tensor* pos = ggml_relu(ctx, x);             // max(x,0)
    ggml_tensor* neg = ggml_sub(ctx, x, pos);         // min(x,0) = x - max(x,0)
    ggml_tensor* a4  = ggml_reshape_4d(ctx, alpha, 1, 1, C, 1);
    return ggml_add(ctx, pos, ggml_mul(ctx, neg, a4));
}

ggml_tensor* batchnorm(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                       const char* prefix, float eps,
                       std::vector<std::vector<float>>& keep) {
    (void)keep;  // affine is host-folded now; no per-call eps scalar buffer needed
    // scale = gamma/sqrt(var+eps), shift = beta-mean*scale, folded ONCE on the
    // host (bn_fold_params caches per model). The graph collapses from 4 weight
    // leaves + add1/sqrt/div/sub/mul to two constant leaves + mul + add.
    const float* scale; const float* shift; int64_t C = 0;
    bn_fold_params(ml, prefix, eps, &scale, &shift, &C);

    const int64_t ne[4] = {C, 1, 1, 1};
    ggml_tensor* s = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne,
                                        scale, C * sizeof(float));
    ggml_tensor* b = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne,
                                        shift, C * sizeof(float));

    // y = x * scale + shift, broadcast per channel (ne[2]).
    ggml_tensor* s4 = ggml_reshape_4d(ctx, s, 1, 1, C, 1);
    ggml_tensor* b4 = ggml_reshape_4d(ctx, b, 1, 1, C, 1);
    return ggml_add(ctx, ggml_mul(ctx, x, s4), b4);
}

} // namespace fd

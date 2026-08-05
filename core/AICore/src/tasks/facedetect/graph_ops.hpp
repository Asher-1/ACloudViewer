#pragma once

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace fd {

class ModelLoader;

// Shared ggml graph building blocks for the SCRFD detector and ArcFace
// recognizer graphs. These reference the loader's weight tensors DIRECTLY as
// graph leaves (zero per-call copy) via clone_weight, mirroring the convention
// established in scrfd_graph.cpp.

// Conv2d(+per-channel bias)(+relu). The insightface ONNX exports fold the
// BatchNorm that follows a conv into that conv's weight+bias, so a "conv block"
// is just conv -> (+ bias broadcast over OC) -> (optional relu). `w` is the
// conv weight name (ggml ne [KW,KH,IC,OC]); `b` is an optional bias name (ne
// [OC]) or nullptr. `stride`/`pad` are symmetric.
ggml_tensor* conv2d(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                    const char* w, const char* b, int stride, int pad, bool relu);

// Mark whether the CURRENT graph build permits the higher-throughput but less
// numerically precise F(4x4,3x3) Winograd variant on its large 3x3 stride-1
// pad-1 convs. Only SCRFD's large early maps (320/160/80, where the detector
// FLOPs concentrate) opt in; the ArcFace recognizer and every small map keep the
// parity-exact F(2x2) path. This is a build-time signal only (graph construction
// is single-threaded), read by conv2d when it routes a conv to Winograd; the
// chosen mode is then baked into the Winograd op state, so the multi-threaded
// compute is unaffected by the flag. thread_local, so concurrent graph builds on
// different threads do not interfere. Prefer the RAII WinogradF4Scope guard so
// the flag is always cleared (incl. on a compute exception).
void set_winograd_f4_scope(bool on);
bool winograd_f4_scope();

// RAII guard: set the F4 Winograd scope for the lifetime of the graph build and
// clear it on scope exit (exception-safe). Construct one at the top of a SCRFD
// graph-build lambda.
struct WinogradF4Scope {
    explicit WinogradF4Scope(bool on) { set_winograd_f4_scope(on); }
    ~WinogradF4Scope() { set_winograd_f4_scope(false); }
    WinogradF4Scope(const WinogradF4Scope&) = delete;
    WinogradF4Scope& operator=(const WinogradF4Scope&) = delete;
};

// PReLU with a per-channel slope: prelu(x) = max(x,0) + alpha*min(x,0). `alpha`
// is a ggml leaf whose element count equals x's channel count C (ne[2]); it is
// reshaped to [1,1,C,1] and broadcast. Used across the ArcFace IResNet trunk
// (insightface "prelu" activation).
ggml_tensor* prelu(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha);

// CPU-only fused per-channel (bias-add + PReLU) in a SINGLE pass over the map:
//   y = let v = x + bias[c];  v > 0 ? v : alpha[c]*v
// After a Conv+BN fold the per-layer tail is a broadcast bias-add (1 full-map
// pass) followed by PReLU (4 passes: relu, sub, mul, add). On the bandwidth-bound
// MobileFaceNet/SFace recognizer those 5 elementwise streams are pure memory
// traffic; this collapses them to one read + one write via a parallel
// ggml_map_custom3 (GGML_N_TASKS_MAX). `bias` and `alpha` are [C] leaves (C =
// x->ne[2]). CPU only - on GPU the native ggml elementwise ops already run on the
// device, so the caller keeps the add+prelu node form there (a custom CPU op
// would force a per-layer device<->host split).
ggml_tensor* fused_prelu_bias(ggml_context* ctx, ggml_tensor* x,
                              ggml_tensor* bias, ggml_tensor* alpha);

// Standalone per-channel BatchNorm in inference mode:
//   y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
// The four stats are loaded from `prefix`+{.weight,.bias,.running_mean,
// .running_var} (e.g. prefix "rec.layer1.0.bn1"); each is a [C] leaf matching x's
// channel count (ne[2]). Unlike the conv-folded post-conv BNs, the IResNet
// IBasicBlock's pre-activation `bn1` survives as an explicit BN node (it has no
// preceding conv to fold into), so this realises it directly. `eps` is the ONNX
// BatchNormalization epsilon (insightface w600k_r50 uses 1e-5). `keep` owns the
// host-side eps scalar buffer registered as a graph input (it must outlive the
// compute); the caller passes the trunk's shared keep vector.
ggml_tensor* batchnorm(ggml_context* ctx, const ModelLoader& ml, ggml_tensor* x,
                       const char* prefix, float eps,
                       std::vector<std::vector<float>>& keep);

// Fold an inference-mode BatchNorm's per-channel affine to HOST CONSTANTS once
// per loaded model and return cached pointers: scale = gamma/sqrt(var+eps),
// shift = beta - mean*scale (each `*count` long, over `prefix`+{.weight,.bias,
// .running_mean,.running_var}). The returned buffers live for the process
// lifetime, so they safely back a deferred graph_input_tensor host->device copy.
// Used by both the conv-style batchnorm above (broadcast over channels) and the
// ArcFace head's batchnorm1d (broadcast over features), replacing the per-call
// add1/sqrt/div/sub/mul live-node chain. `eps` is the ONNX BN epsilon.
void bn_fold_params(const ModelLoader& ml, const char* prefix, float eps,
                    const float** scale, const float** shift, int64_t* count);

} // namespace fd

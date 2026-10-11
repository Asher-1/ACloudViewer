// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// AICore `ocr` task runtime (P1a: recognition path).
//
// The runtime executes the audited Paddle inference topology embedded in
// each GGUF (`ocr.graph.ops`) instead of a hand-written layer sequence, so
// the executed graph is exactly the source graph the conversion pinned.
// Tensor layout: Paddle stores activations and weights in NCHW; the ggml
// counterpart of a numpy (N,C,H,W) array is ne={W,H,C,N} — the raw byte
// streams are identical, so weights load without permutation.

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "json/json.h"

namespace aicore {
namespace ocr {

struct GraphOp {
    struct OpInput {
        std::string param;              // "X", "Filter", ... (may be empty)
        std::vector<std::string> args;  // tensor names
    };
    std::string type;
    std::vector<OpInput> inputs;
    std::vector<std::string> outputs;  // result tensor names, in order
    Json::Value attrs;                 // parsed attribute object
};

struct OcrProgram {
    std::vector<GraphOp> ops;
    std::map<std::string, int> varRanks;  // static logical rank per tensor
    std::map<std::string, std::vector<int64_t>> staticShapes;  // VarDesc dims
    std::string feedInput;                                     // "x"
    std::string fetchOutput;  // "save_infer_model/scale_0.tmp_0"
    bool parse(const std::string& json, std::string* error);
};

/** Probe-only knobs (parity tests): truncate the program after op K and
 *  optionally print every op's output shape. Not part of the runtime. */
struct OcrProbeOptions {
    int fetchAt = -1;  // program-order op index (0-based)
    bool dumpShapes = false;
};

/** Graph-build options. `cpu_conv` routes 3x3 stride-1 convolutions through
 *  the shared Winograd custom op (common/winograd.hpp) instead of
 *  im2col+mul_mat; set it only when the session resolved to a CPU backend
 *  (the custom op is CPU-only — on a GPU session it would force a
 *  host<->device round trip per conv). `blocked_island` routes eligible
 *  conv/activation/SE chains through the shared blocked-layout (nChw16c)
 *  island (common/blocked_conv.hpp) — CPU-only and only when the host
 *  advertises the AVX-512 fast path (aicore::blk::blocked_available()). */
struct OcrBuildOptions {
    bool cpu_conv = false;
    bool blocked_island = false;
    /** GPU dense direct convolution for the recognizer (A/B knob, off in
     *  production). Decoded text matches the im2col route, but the batched
     *  rec graph measured slower (29.3 vs 25.7 ms, CUDA f32, 100 boxes) and
     *  its fixture max|diff| is 2.3x larger (0.0139 vs 0.0059). */
    bool gpu_rec_direct_conv = false;
    /** GPU depthwise convolutions evaluate directly for every graph role.
     *  The depthwise kernel has no cross-channel reduction (9 products per
     *  output in a fixed order), so unlike the dense direct route it cannot
     *  move the DB threshold edges; it removes the K^2*OH*OW*C*N im2col
     *  expansion that dominated the det graph (14 im2col launches, ~27 ms
     *  per 2560x1440 frame on CUDA, mostly memory traffic). */
    bool gpu_dw_direct_conv = false;
    /** Vulkan execution needs this to avoid the detector's unsupported direct
     *  convolution route (see the conv2d route note in ocr_runtime.cpp). */
    bool vk_single = false;
    /** Dense conv GEMMs request GGML_PREC_F32. The affine folding scales the
     *  conv weights by the batch-norm gain and cancels the folded mean in the
     *  bias, so the GEMM partial sums grow by that gain while the layer
     *  output does not; an f16 accumulator (Vulkan's default F16xF32
     *  pipeline) keeps the enlarged absolute error through the cancellation
     *  and the f32 fixture gates fail (rec 0.016, layout 0.022 vs 0.003 /
     *  0.010 on CUDA, whose cuBLAS 16F path keeps f32 k-tile partials).
     *  F32 accumulation restores CUDA-level numerics. Vulkan only: on CUDA
     *  PREC_F32 drops to strict SGEMM with no tensor cores and the default
     *  route already passes every gate. */
    bool conv_gemm_f32acc = false;
};

/** One graph build's folded constants and their device mirrors. The pool
 *  is created by OcrWeights::resetFoldPool, filled by buildGraph (batch-norm
 *  mul/add, metadata scalars, shape constants) and mirrored to the device by
 *  prepareGraph; it is owned jointly by the weights (current build) and by
 *  every cached graph that was built against it, and freed with the last
 *  owner. The members are the host pool, its zero-copy CPU buffer view, the
 *  aligned per-tensor device mirror and the unaligned static-view copies. */
struct OcrFoldPool {
    OcrFoldPool() = default;
    ~OcrFoldPool();
    OcrFoldPool(const OcrFoldPool&) = delete;
    OcrFoldPool& operator=(const OcrFoldPool&) = delete;

    ggml_context* ctx = nullptr;                // host constants (data)
    ggml_backend_buffer_t hostBuf = nullptr;    // CPU buffer view of ctx
    ggml_context* mirrorCtx = nullptr;          // aligned mirror shapes
    ggml_backend_buffer_t deviceBuf = nullptr;  // device mirror of ctx
    ggml_context* viewMirrorCtx = nullptr;      // static-view copies
    ggml_backend_buffer_t viewMirrorBuf = nullptr;
};

class OcrWeights {
public:
    OcrWeights() = default;
    ~OcrWeights();
    OcrWeights(const OcrWeights&) = delete;
    OcrWeights& operator=(const OcrWeights&) = delete;

    /** Load a rec/det GGUF. The model files are small enough to keep the
     *  weight memory in the host pool (no_alloc=false), which lets the CPU
     *  path run directly and constant folding (batch-norm) read tensors
     *  without a backend round-trip. `foldAffine` runs foldAffineChains on
     *  the parsed program (production default); probes that bisect against
     *  the unfolded numpy reference by op index pass false. */
    bool load(const std::string& path,
              std::string* error,
              bool foldAffine = true);

    const ggml_tensor* weight(const std::string& name) const;

    /** Host-pool view of a weight: valid across realizeToDevice, which
     *  swaps the main map to device copies. The CPU-side folding in
     *  buildGraph (batch-norm statistics) must read through this. */
    const ggml_tensor* hostWeight(const std::string& name) const;

    /** Wrap the two host weight pools (GGUF `meta` pool + folded-constant
     *  `constCtx` pool) as zero-copy CPU backend buffers. The multi-backend
     *  copy path requires every graph leaf to carry a buffer; without this
     *  a GPU split cannot copy host leaves (ggml_backend_tensor_copy
     *  dereferences src->buffer). Idempotent. */
    bool realizeHostBuffers();

    /** Tag host leaves created after realizeHostBuffers (folded constants
     *  are emitted during every buildGraph call) with their pool buffer.
     *  Cheap; run after each graph build on the GPU path. */
    void tagHostLeafs();

    /** GPU execution: mirror every weight tensor named in `tensors` into a
     *  buffer allocated on `backend` and repoint the name map (idempotent,
     *  so the weights upload exactly once per context). The folded-constant
     *  context stays host-side; the scheduler copies the small per-channel
     *  constants each run. Returns false with `error` set on failure. */
    bool realizeToDevice(ggml_backend_t backend, std::string* error);

    gguf_context* gguf = nullptr;
    ggml_context* meta = nullptr;
    /** Host-data context for folded constants (batch-norm mul/add, the
     *  metadata scalar chains): tensors created here carry real data. */
    ggml_context* constCtx = nullptr;
    std::map<std::string, const ggml_tensor*> tensors;
    std::string role;  // "det" | "rec"
    std::string arch;
    std::vector<std::string> dictionary;  // rec: embedded vocabulary
    std::vector<std::string> labels;      // layout: embedded class table
    OcrProgram program;
    // q8_0 folded kernels: gguf tensor name -> original numpy dims.
    std::map<std::string, std::vector<int64_t>> reshapedDims;
    // Pre-realize host view of `tensors` (captured by realizeToDevice;
    // the host pools outlive the device copies).
    std::map<std::string, const ggml_tensor*> hostTensors;

    /** Start a fresh folded-constant pool for the next buildGraph. Every
     *  build emits its own fold constants (batch-norm statistics, metadata
     *  scalars) and the pool, its host view and its device mirrors belong
     *  to THAT graph: the pool is reference-counted (foldPool()) so a
     *  graph cache can keep several built graphs alive at once, each
     *  holding the pool it was built against. The previous pool is
     *  released here only if no graph still holds it. Call before each
     *  buildGraph. */
    void resetFoldPool();

    /** The pool the next/last buildGraph emitted into; a graph cache entry
     *  stores this handle for as long as its graph is alive. */
    std::shared_ptr<OcrFoldPool> foldPool() const { return fold; }

    /** Mirror the CURRENT fold pool to a single device buffer (one bulk
     *  upload) and repoint every fold tensor at it. Without this the
     *  sched re-uploads the whole fold pool (~11 MB of BN-folded weights
     *  and per-op vectors) on EVERY compute — measured at ~12 ms per rec
     *  chunk, i.e. more than all rec kernels combined. Safe because fold
     *  constants are never host-read after the build (BN folding already
     *  happened) and never reused across builds (fresh pool per build);
     *  the raw-pool mirror preserves view offsets exactly. */
    bool realizeFoldPoolToDevice(ggml_backend_t backend, std::string* error);

    /** Point every host-resident named weight leaf of `graphCtx` at its
     *  device copy (see realizeToDevice) and re-derive view pointers. For
     *  single-backend Vulkan there is no sched to re-upload or re-bind the
     *  leafs: the graph was built against the HOST originals (BN folding
     *  needs host dereference), so without this the VK dispatch reads host
     *  pointers as device addresses. Idempotent; call after
     *  realizeFoldPoolToDevice and before gallocr_alloc_graph. */
    void retargetGraphToDevice(struct ggml_cgraph* graph);

    /** Materialize only static mul_mat source views whose byte offset cannot
     *  be represented by the backend's storage-buffer alignment. The copied
     *  tensor preserves the view's exact logical byte interval and lives until
     *  the current graph cache is discarded. */
    bool materializeUnalignedStaticViews(struct ggml_cgraph* graph,
                                         ggml_backend_t backend,
                                         std::string* error);

    /** Per-build fold pool (null until the first resetFoldPool); the
     *  builder routes folded constants there. */
    ggml_context* foldContext() const { return fold ? fold->ctx : nullptr; }

private:
    /** Load-time IR rewrite: absorb the affine ops adjacent to every conv
     *  (batch_norm, per-channel/scalar elementwise add/mul) into the conv's
     *  Filter and Bias. See the definition for the algebra and gates. */
    void foldAffineChains();
    ggml_backend_buffer_t hostMetaBuf = nullptr;
    ggml_backend_buffer_t hostConstBuf = nullptr;
    std::shared_ptr<OcrFoldPool> fold;  // current build's constants
    ggml_context* deviceCtx = nullptr;
    ggml_backend_buffer_t deviceBuf = nullptr;

public:
    /** WEIGHTS-usage tagging gate. Tagging the weight buffers is what lets
     *  the backend sched offload conv/matmul ops to the GPU (the offload
     *  routing requires usage == WEIGHTS); without it the whole graph
     *  silently runs on the CPU secondary even for a GPU session. The
     *  Vulkan path keeps this off for now: routing ops to ggml-vulkan
     *  exposed an independent vk subbuffer fault on some det geometries
     *  (352x544), so VK sessions keep the legacy all-CPU routing until
     *  that backend issue is fixed. */
    bool sched_gpu_tagging = true;
};

/** Build the ggml graph for the loaded program. `input` must be laid out
 *  as ne={W,H,3,N}. Returns the fetch tensor. CPU-only folding of the
 *  metadata ops (shape/slice/fill_constant) is performed inline. */
ggml_tensor* buildGraph(ggml_context* ctx,
                        const OcrWeights& weights,
                        ggml_tensor* input,
                        std::string* error,
                        const OcrProbeOptions* probe = nullptr,
                        std::vector<ggml_tensor*>* fetchList = nullptr,
                        const OcrBuildOptions* options = nullptr);

/** Where an OCR graph executes and how it is allocated. One owner for the
 *  CPU / mixed-sched / single-device decision so the production session
 *  (ocr_capi) and the fixture probe (test_ocr_runtime) run the SAME graph:
 *  a probe that allocates through a different route measures a graph that
 *  never ships (and, on Vulkan, faults on host-buffered views). */
struct OcrExecPlan {
    ggml_backend_t gpu = nullptr;  // null for a CPU session
    ggml_backend_t cpu = nullptr;  // session CPU backend or GPU secondary
    bool cpuExecution = false;     // every node runs on `cpu`
    bool singleBackend = false;    // whole graph in one device gallocr
    bool vkSingle = false;         // Vulkan conv-route restriction
};

/** CUDA/Vulkan: single device. Other GPUs: mixed sched when the weights
 *  tagging gate is open, else CPU execution. CPU backend: CPU execution. */
OcrExecPlan makeExecPlan(ggml_backend_t gpu,
                         ggml_backend_t cpu,
                         bool schedGpuTagging);

/** Production build options for the plan (conv routes, Vulkan flag,
 *  blocked island on AVX-512 CPU hosts). */
OcrBuildOptions execBuildOptions(const OcrExecPlan& plan);

/** Allocation state of one prepared graph. */
struct OcrGraphExec {
    ggml_gallocr_t alloc = nullptr;
    ggml_backend_sched_t sched = nullptr;
    void free();
};

/** After buildGraph: realize host/device weights and the fold pool as the
 *  plan requires, build the forward graph over `outputs`, allocate it and
 *  (single device) retarget leafs and rebind every static view to device
 *  storage. On failure `exec` is freed and `error` set. */
bool prepareGraph(OcrWeights& weights,
                  ggml_context* ctx,
                  const OcrExecPlan& plan,
                  const std::vector<ggml_tensor*>& outputs,
                  ggml_cgraph** graph,
                  OcrGraphExec& exec,
                  std::string& error);

/** Run a prepared graph on the backend the plan selected. */
void computeGraph(const OcrExecPlan& plan,
                  const OcrGraphExec& exec,
                  ggml_cgraph* graph);

/** CTC greedy decode over a (T, vocab) probability row set. Index 0 is the
 *  CTC blank, the last index is the word separator; repeated labels
 *  collapse. Returns UTF-8 text and the mean non-blank score. */
std::string ctcGreedyDecode(const float* probs,
                            int timeSteps,
                            int vocab,
                            const std::vector<std::string>& dictionary,
                            float* meanScore);

/** Same decode over a pre-reduced row set: `bestIndex[t]` is the argmax
 *  class and `bestProb[t]` its probability (the two scalars the decoder
 *  actually consumes), as produced by the in-graph argmax/get_rows head on
 *  GPU sessions. Shares the collapse/dictionary logic with ctcGreedyDecode. */
std::string ctcGreedyDecodeIndexed(const int32_t* bestIndex,
                                   const float* bestProb,
                                   int timeSteps,
                                   int vocab,
                                   const std::vector<std::string>& dictionary,
                                   float* meanScore);

/** Recognition preprocessing: resize the input to height 48 preserving
 *  aspect (clamped to `targetWidth`), normalize with the PP-OCR ImageNet
 *  statistics, and emit an NCHW float buffer. */
std::vector<float> preprocessRec(const unsigned char* rgba,
                                 int width,
                                 int height,
                                 int channels,
                                 int targetWidth,
                                 int* outW);
}  // namespace ocr
}  // namespace aicore

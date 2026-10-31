// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// C API implementation of the `ocr` task inference contract
// (core/AICore/include/aicore/ocr_capi.h). The pipeline is:
//
//   image_view -> bilinear resize (limit side, multiple of 32) -> NCHW
//   normalize -> det interpreter -> DB probability map -> threshold +
//   connected components (BFS) -> boxes + unclip offset -> per-box bilinear
//   crop -> 48xH rec batch -> rec interpreter -> CTC greedy decode ->
//   typed aicore_ocr_box{quad, text, score}.
//
// The DB post-process is hand-written (no OpenCV in the AICore tree) and
// mirrors Paddle's integer Clipper/JT_ROUND unclip plus final pixel rounding.

#include "aicore/ocr_capi.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "aicore/runtime_capi.h"
#include "common/blocked_conv.hpp"
#include "common/ggml_backend_registry.hpp"
#include "common/ggml_backend_utils.hpp"
#include "common/image_utils.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ocr_db_postprocess.hpp"
#include "ocr_rec_crop.hpp"
#include "ocr_runtime.hpp"

namespace {

thread_local std::string g_lastError;

void setLastError(const std::string& message) { g_lastError = message; }

bool hasOptionField(const aicore_ocr_options* options,
                    size_t offset,
                    size_t size) {
    return options && options->struct_size >= offset + size;
}

constexpr size_t kOcrOptionsInitialSize = offsetof(aicore_ocr_options, device);
constexpr float kDetDefaultMaxSide = 960.0f;
// PaddleOCR 3.0.1's public OCR pipeline resolves the PP-OCRv4 mobile det
// parameters to limit_type="min", limit_side_len=64. The model inference.yml
// describes engine profiles, not the pipeline's runtime default.
constexpr float kDetOfficialLimitSide = 64.0f;
constexpr float kDetOfficialMaxSideLimit = 4000.0f;
constexpr float kDetDefaultBoxThresh = 0.3f;
// Upstream box_thresh is the Python double 0.6, compared against cv2.mean's
// double; 0.6f would move the gate to 0.60000002.
constexpr double kDetDefaultBoxScoreThresh = 0.6;
constexpr float kDetDefaultUnclipRatio = 1.5f;
constexpr int kRecHeight = 48;
// Official rec_image_shape width (3, 48, 320): the FLOOR of the per-crop
// padded width, not a cap. paddlex resize(): max_wh_ratio = max(320/48,
// w/h); imgW = int(48 * max_wh_ratio); resized_w = ceil(48 * w/h) clipped
// to imgW; the crop is never compressed below its aspect ratio until the
// 3200 px limit (max_imgW). The historical clamp(round(48*w/h), 1, 320)
// squeezed every line longer than 6.67:1 (a 254x15 body line by 2.5x) into
// garbage on every backend.
constexpr int kRecMaxWidth = 320;
constexpr int kRecLimitWidth = 3200;  // paddlex ResizeNormImg.max_imgW
// Official text_recognition_batch_size (PaddleX OCR pipeline config). The
// pipeline sorts a frame's crops by aspect ratio and pads every group of
// this many consecutive crops to the group's widest padded width. The pad
// width is observable: the SVTR self-attention mixes the pad columns into
// every output position, so a crop recognized at another pad width can
// decode to another text (measured on ICDAR2015: every text difference
// against Paddle CPU on identical boxes was reproduced by Paddle itself at
// the former 320/480/640/... bucket widths).
constexpr int kRecPadGroupDefault = 6;
// Distinct (width, rows) builds kept per rec session. Crops with an aspect
// ratio up to 320/48 share the 320 build; only wider lines add builds. The
// activation budget per build is held constant by recRowsCapForWidth so the
// cache's device footprint is bounded by capacity x one 48-row 320-wide
// build.
constexpr size_t kRecGraphCacheCapacity = 6;
constexpr int kDetAlign = 32;

// PP-OCRv4 preprocessing constants (official DetPreProcess/RecPreProcess).
// NormalizeImage applies mean[c]/std[c] to channel c of the cv2 BGR frame
// (the ImageNet RGB-order constants on BGR planes, exactly as upstream).
constexpr float kImageMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImageStd[3] = {0.229f, 0.224f, 0.225f};
constexpr float kRecMean = 0.5f;
constexpr float kRecStd = 0.5f;
constexpr float kRecMean3[3] = {kRecMean, kRecMean, kRecMean};
constexpr float kRecStd3[3] = {kRecStd, kRecStd, kRecStd};
// Fixed rec batch chunk (run- and recognize-phase): see the run-phase note
// for the measured scaling curve and the A/B rationale. The linear segment
// reaches N=72 (VRAM ~3 GiB); the cliff is between 72 and 96, where the
// depthwise im2col workspace pushes the allocator against VRAM (7.6+ GiB
// resident, 20 GiB peak at 96) and every compute re-warms the CUDA graphs.
constexpr int64_t kRecChunkBatch = 48;
// Vulkan sessions run the direct-conv route (no im2col materialization), so
// the workspace scales LINEARLY with N and there are no CUDA graphs to
// re-warm. The per-chunk cost there is dominated by fixed per-op dispatch
// overhead (~58.6 ms per 40-row chunk, of which only ~14 ms is kernel time),
// so the cap is raised to cover a dense image in a single chunk: 1 compute
// instead of 3 cuts the fixed overhead proportionally (measured this
// session: 3x40 rows -> 176 ms rec, 1x104 rows -> 81 ms rec).
constexpr int64_t kRecChunkBatchVulkan = 112;
// Chunk cap per execution backend. CUDA sessions keep 48 (the im2col VRAM
// cliff); Vulkan single-backend sessions keep 112. CPU sessions never batch
// (the blocked island wins there), so the cap value is irrelevant.
inline int64_t recChunkBatchCap(ggml_backend_t gpuBackend) {
    if (gpuBackend && !ggml_common::is_cpu_backend(gpuBackend)) {
        const char* name = ggml_backend_name(gpuBackend);
        if (name && std::string(name).find("Vulkan") != std::string::npos)
            return kRecChunkBatchVulkan;
    }
    return kRecChunkBatch;
}
// Row cap for a wider chunk: rows x width is held at cap x 320 so a 1280
// wide chunk carries a quarter of the rows and the activation / im2col
// workspace (both linear in rows x width) never exceeds the measured
// 48x320 CUDA budget. Multiples of 4 keep the build key set small.
inline int64_t recRowsCapForWidth(int64_t cap, int width) {
    const int64_t rows = cap * kRecMaxWidth / width;
    return std::max<int64_t>(4, rows / 4 * 4);
}
// Official rec pad (paddlex text_recognition/processors.py resize_norm_img
// L61-62): padding_im = np.zeros((imgC, imgH, imgW)) in the NORMALIZED
// domain, i.e. the fill is 0.0 = pixel 127.5 = (0.5 - mean) / std. A black
// (-1) pad is out-of-distribution: the model was trained on ratio-bucketed
// batches whose right side is exactly this gray fill, and the SVTR
// self-attention mixes pad keys into every output position.
constexpr float kRecPadFill = (0.5f - kRecMean) / kRecStd;

struct OcrGraph {
    ggml_context* ctx = nullptr;  // graph tensors (no_alloc)
    ggml_cgraph* graph = nullptr;
    ggml_tensor* input = nullptr;
    ggml_tensor* output = nullptr;
    // Optional in-graph CTC reduction over `output` (rec graphs on GPU
    // execution only): argmax class per (t, n) row and its probability.
    ggml_tensor* argmaxIndex = nullptr;
    ggml_tensor* argmaxProb = nullptr;
    bool argmaxHead = false;
    aicore::ocr::OcrGraphExec exec;  // gallocr (CPU / one device) or sched
    // The folded-constant generation this build was emitted into. The
    // build's constants (and their device mirrors) live in the pool, so the
    // graph co-owns it: the weights move on to a fresh pool at the next
    // build while every cached graph keeps its own generation alive.
    std::shared_ptr<aicore::ocr::OcrFoldPool> fold;
    int64_t inputC = 0;
    int64_t inputH = 0;
    int64_t inputW = 0;
    int64_t inputN = 0;
    uint64_t lastUse = 0;  // LRU clock (OcrGraphCache)

    bool matches(int64_t c, int64_t h, int64_t w) const {
        return graph && inputC == c && inputH == h && inputW == w;
    }

    void clear() {
        exec.free();
        releaseGraph();
    }

    // Drops the build but keeps `exec`: the slot's next build reuses the
    // gallocr and its compute buffer (see aicore::ocr::OcrGraphExec).
    void releaseGraph() {
        if (ctx) ggml_free(ctx);
        // After the graph context; the pool frees the device mirrors when
        // the last owner lets go.
        fold.reset();
        ctx = nullptr;
        graph = nullptr;
        input = nullptr;
        output = nullptr;
        argmaxIndex = nullptr;
        argmaxProb = nullptr;
        argmaxHead = false;
        inputC = inputH = inputW = inputN = 0;
        lastUse = 0;
    }
};

// Bounded set of cached builds keyed by (C, H, W, N, argmax head). The rec
// graph needs one build per (pad width, row count) actually used by a
// frame — a document mixes short labels (320) with full-width body lines
// (wider groups) — and a frame sequence revisits the same handful of keys, so a
// small LRU replaces the historical single entry. Eviction releases the
// graph (and its fold generation) but keeps the slot's allocator; the caller
// rebuilds into the slot.
struct OcrGraphCache {
    std::vector<OcrGraph> entries;
    size_t capacity = 1;
    uint64_t clock = 0;

    OcrGraph& acquire(
            int64_t c, int64_t h, int64_t w, int64_t n, bool argmaxHead) {
        ++clock;
        for (OcrGraph& g : entries) {
            if (g.matches(c, h, w) && g.inputN == n &&
                g.argmaxHead == argmaxHead) {
                g.lastUse = clock;
                return g;
            }
        }
        if (entries.size() < capacity) {
            entries.emplace_back();
            entries.back().lastUse = clock;
            return entries.back();
        }
        OcrGraph* victim = &entries.front();
        for (OcrGraph& g : entries)
            if (g.lastUse < victim->lastUse) victim = &g;
        victim->releaseGraph();
        victim->lastUse = clock;
        return *victim;
    }

    void clear() {
        for (OcrGraph& g : entries) g.clear();
        entries.clear();
    }
};

// Reduced rec output: the two scalars per timestep the CTC decoder consumes
// (argmax class, its probability), laid out [t + T * n]. When `active` the
// full (V, T, N) softmax was NOT read back. The vectors are frame-scratch
// owned by the context.
struct RecArgmaxOut {
    std::vector<int32_t> index;
    std::vector<float> prob;
    bool active = false;
};

// --- Stage helpers (all pure functions over plain buffers) -----------------

// CHW f32 packing with the official PP-OCR normalization. dstW is the
// packed row width: the rec path passes the fixed kRecMaxWidth so every row
// is laid out as [content | right-side pad] exactly like the official
// resize_norm_img (padding_im[:, :, ..., w:] = 0 in normalized space). The
// pack must emit the final stride IN one pass: appending the pad after a
// w-wide pack would scramble channels/rows the moment dstW > w (the packed
// (3,h,w) re-read as (3,h,dstW) interleaves planes) — the root cause of the
// historical empty/garbled rec text on short crops. mean/stdDev are per
// packed plane (plane c reads Image channel c, BGR when img.bgr is set).
void packNchwImageTo(const aicore::img::Image& img,
                     const float (&mean)[3],
                     const float (&stdDev)[3],
                     int dstW,
                     float padValue,
                     float* nchw) {
    const int w = img.width;
    const int h = img.height;
    const float scale = 1.0f / 255.0f;
    // Pre-fill the pad columns once per plane, then overwrite the content
    // region (the pad lives at the row END: [content | pad] per row).
    const size_t plane = static_cast<size_t>(h) * dstW;
    for (int ch = 0; ch < 3; ++ch) {
        float* p = nchw + static_cast<size_t>(ch) * plane;
        for (int y = 0; y < h; ++y) {
            float* row = p + static_cast<size_t>(y) * dstW;
            for (int x = w; x < dstW; ++x) row[x] = padValue;
        }
    }
    // One image row -> three plane rows. The expression is kept verbatim
    // so the packed floats are bit-identical to the historical serial loop;
    // only the source addressing is hoisted (Image::row / channel_index).
    const int scn = img.channels;
    const int cmap[3] = {img.channel_index(0), img.channel_index(1),
                         img.channel_index(2)};
    const auto packRow = [&](int y) {
        const uint8_t* srow = img.row(y);
        for (int ch = 0; ch < 3; ++ch) {
            const uint8_t* s = srow + cmap[ch];
            float* row = nchw + static_cast<size_t>(ch) * plane +
                         static_cast<size_t>(y) * dstW;
            const float m = mean[ch], sd = stdDev[ch];
            for (int x = 0; x < w; ++x)
                row[x] = (s[static_cast<size_t>(x) * scn] * scale - m) / sd;
        }
    };
    // Same gate as cv_resize_linear_u8: only the det/layout frame (h in the
    // hundreds-to-thousands) fans out; rec crops (h = 48) stay serial because
    // they are already packed from inside the per-box worker pool.
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    if (hw > 1 && h >= 256 && static_cast<unsigned>(h) > hw) {
        std::atomic<int> nextRow{0};
        std::vector<std::thread> pool;
        pool.reserve(hw - 1);
        for (unsigned i = 1; i < hw; ++i)
            pool.emplace_back([&] {
                for (;;) {
                    const int y = nextRow.fetch_add(1);
                    if (y >= h) return;
                    packRow(y);
                }
            });
        for (;;) {
            const int y = nextRow.fetch_add(1);
            if (y >= h) break;
            packRow(y);
        }
        for (auto& t : pool) t.join();
    } else {
        for (int y = 0; y < h; ++y) packRow(y);
    }
}

void packNchwImage(const aicore::img::Image& img,
                   const float (&mean)[3],
                   const float (&stdDev)[3],
                   int dstW,
                   float padValue,
                   std::vector<float>& nchw) {
    // resize, not assign: packNchwImageTo writes every element (content and
    // pad columns), so a pre-fill is a second full pass over the buffer.
    // When the caller hands in a reused scratch vector of the same size the
    // resize is a no-op and no page faults are taken.
    nchw.resize(static_cast<size_t>(3) * img.height * dstW);
    packNchwImageTo(img, mean, stdDev, dstW, padValue, nchw.data());
}

// Batched-rec enablement: automatic per execution side. Sessions whose
// graphs execute on a GPU run the chunked batch — CUDA (2.6x dense-image
// e2e, fixture-exact at every measured N with per-slab bitwise
// self-consistency) and Vulkan alike: the single-backend VK route
// (AICORE_OCR_VK_SINGLE) executes the rec graph on the device, and the
// per-crop fallback there is dispatch-bound (~98 x 2.2-3.9 ms of per-op
// vkCmdSubmit overhead; the probe measured 14.1 ms for a 40-row direct-conv
// chunk, i.e. 0.35 ms/row amortized). CPU-executing sessions keep per-crop
// N=1 forwards (the blocked island wins there, and CPU batch measured
// slower).
bool ocrBatchedRecEnabled(ggml_backend_t gpuBackend, bool cpuExecution) {
    // A session whose graphs execute on the CPU (plain CPU session, or a
    // Vulkan session with the sched_gpu_tagging gate closed and the
    // single-backend route off) keeps the per-crop path: the batched
    // graph's strided mul_mat is slower on the CPU, and a gated-off VK
    // batch would execute on the CPU secondary anyway.
    if (cpuExecution) return false;
    if (!gpuBackend || ggml_common::is_cpu_backend(gpuBackend)) return false;
    return true;
}

// Execute one graph over a freshly packed input, reusing the cached build
// for a stable input shape. The build (300+ op walk, folded constants,
// allocation) costs far more than a steady-state compute and dominated real
// images with dozens of rec crops, so the det graph is cached per geometry
// and the rec graph per (pad width, rows) in a small LRU (OcrGraphCache).
// Backend routing (CPU gallocr / mixed sched / single device) is owned by
// aicore::ocr::makeExecPlan + prepareGraph, shared with the fixture probe so
// the probe measures exactly the graph that ships.
bool runGraphCached(aicore_ocr_ctx* session,
                    aicore::ocr::OcrWeights& weights,
                    OcrGraph& cached,
                    ggml_backend_t gpuBackend,
                    ggml_backend_t cpuBackend,
                    int64_t c,
                    int64_t h,
                    int64_t w,
                    int64_t n,
                    const float* nchwData,
                    size_t nchwFloats,
                    std::vector<float>& output,
                    int64_t& outC,
                    int64_t& outH,
                    int64_t& outW,
                    int64_t& outN,
                    double& inferenceMs,
                    double& inputUploadMs,
                    double& outputReadbackMs,
                    std::string& error,
                    RecArgmaxOut* argmaxOut = nullptr) {
    ggml_backend_t backend = gpuBackend ? gpuBackend : cpuBackend;
    if (!backend) {
        error = "ocr: no ggml backend available";
        return false;
    }
    const aicore::ocr::OcrExecPlan plan = aicore::ocr::makeExecPlan(
            gpuBackend, cpuBackend, weights.sched_gpu_tagging);
    const bool cpuExecution = plan.cpuExecution;
    // In-graph CTC reduction, GPU execution only. The CTC decoder consumes
    // two scalars per timestep (argmax class + its probability); reading the
    // full (V, T, N) softmax back (51 MB per 100-box frame, ~10 ms D2H) to
    // compute them on the host was the largest single post stage. ggml's
    // CPU argmax is single-threaded (ith != 0 returns) and would regress a
    // CPU session, which keeps the AVX2 host sweep.
    const bool wantArgmaxHead = argmaxOut && !cpuExecution;
    if (!cached.matches(c, h, w) || cached.inputN != n ||
        cached.argmaxHead != wantArgmaxHead) {
        cached.releaseGraph();
        // Fresh fold pool per build: every build emits new constants and
        // the pool would exhaust; the previous graph is discarded above.
        weights.resetFoldPool();
        const size_t tensorSize = ggml_tensor_overhead() * 500000;
        cached.ctx = ggml_init({tensorSize * 2, nullptr,
                                /*no_alloc=*/true});
        if (!cached.ctx) {
            error = "ocr: ggml_init failed";
            return false;
        }
        cached.input =
                ggml_new_tensor_4d(cached.ctx, GGML_TYPE_F32, w, h, c, n);
        // Marked as the user input: the scheduler assigns it to the CPU
        // side and copies it before the first split; the gallocr path
        // simply allocates it like any other leaf.
        ggml_set_input(cached.input);
        const aicore::ocr::OcrBuildOptions buildOptions =
                aicore::ocr::execBuildOptions(plan);
        cached.output = aicore::ocr::buildGraph(cached.ctx, weights,
                                                cached.input, &error, nullptr,
                                                nullptr, &buildOptions);
        if (!cached.output) {
            cached.clear();
            return false;
        }
        std::vector<ggml_tensor*> outputs{cached.output};
        if (wantArgmaxHead) {
            // Rows are the V-wide class vectors of every (t, n): the rec
            // fetch is ne={V, T, 1, N}, so a contiguous 2D view is [V, T*N].
            // argmax -> i32 [T*N]; get_rows over the [1, V, T*N] view with
            // the [1, T*N] index gathers p[argmax] per row -> f32 [1, 1, T*N].
            ggml_tensor* out = cached.output;
            if (!ggml_is_contiguous(out)) out = ggml_cont(cached.ctx, out);
            const int64_t vocab = out->ne[0];
            const int64_t rows = ggml_nelements(out) / vocab;
            ggml_tensor* flat = ggml_reshape_2d(cached.ctx, out, vocab, rows);
            cached.argmaxIndex = ggml_argmax(cached.ctx, flat);
            cached.argmaxProb = ggml_get_rows(
                    cached.ctx,
                    ggml_reshape_3d(cached.ctx, out, 1, vocab, rows),
                    ggml_reshape_2d(cached.ctx, cached.argmaxIndex, 1, rows));
            ggml_set_output(cached.argmaxIndex);
            ggml_set_output(cached.argmaxProb);
            outputs.push_back(cached.argmaxIndex);
            outputs.push_back(cached.argmaxProb);
        }
        cached.argmaxHead = wantArgmaxHead;
        if (!aicore::ocr::prepareGraph(weights, cached.ctx, plan, outputs,
                                       &cached.graph, cached.exec, error)) {
            cached.clear();
            return false;
        }
        // Co-own this build's fold generation (see OcrGraph::fold).
        cached.fold = weights.foldPool();
        cached.inputC = c;
        cached.inputH = h;
        cached.inputW = w;
        cached.inputN = n;
    }
    const auto inputUploadStart = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(cached.input, nchwData, 0,
                            sizeof(float) * nchwFloats);
    inputUploadMs = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - inputUploadStart)
                            .count();
    const auto start = std::chrono::steady_clock::now();
    aicore::ocr::computeGraph(plan, cached.exec, cached.graph);
    const auto stop = std::chrono::steady_clock::now();
    inferenceMs =
            std::chrono::duration<double, std::milli>(stop - start).count();
    outC = cached.output->ne[2];
    outH = cached.output->ne[1];
    outW = cached.output->ne[0];
    outN = cached.output->ne[3];
    const auto readbackStart = std::chrono::steady_clock::now();
    if (cached.argmaxHead) {
        // Only the reduced head crosses the bus; `output` is left untouched
        // (its stale contents are never read when argmaxOut->active).
        const size_t rows =
                static_cast<size_t>(ggml_nelements(cached.argmaxIndex));
        argmaxOut->index.resize(rows);
        argmaxOut->prob.resize(rows);
        ggml_backend_tensor_get(cached.argmaxIndex, argmaxOut->index.data(), 0,
                                sizeof(int32_t) * rows);
        ggml_backend_tensor_get(cached.argmaxProb, argmaxOut->prob.data(), 0,
                                sizeof(float) * rows);
        argmaxOut->active = true;
    } else {
        if (argmaxOut) argmaxOut->active = false;
        output.resize(static_cast<size_t>(ggml_nelements(cached.output)));
        ggml_backend_tensor_get(cached.output, output.data(), 0,
                                sizeof(float) * output.size());
    }
    outputReadbackMs = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - readbackStart)
                               .count();
    return true;
}

// PicoDet layout graph: same scheduling as runGraph, but every fetch
// (4 cls + 4 reg heads per FPN stride) is expanded into the graph and
// read back as a flat vector.
bool runLayoutHeads(aicore::ocr::OcrWeights& weights,
                    aicore::ocr::OcrGraphExec& exec,
                    ggml_backend_t gpuBackend,
                    ggml_backend_t cpuBackend,
                    int64_t c,
                    int64_t h,
                    int64_t w,
                    const std::vector<float>& nchw,
                    std::vector<std::vector<float>>& heads,
                    double& inferenceMs,
                    std::string& error) {
    const size_t tensorSize = ggml_tensor_overhead() * 500000;
    ggml_context* ctx = ggml_init({tensorSize * 2, nullptr,
                                   /*no_alloc=*/true});
    if (!ctx) {
        error = "ocr: ggml_init failed";
        return false;
    }
    ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, h, c, 1);
    ggml_set_input(input);
    // Fresh fold pool per frame (see runGraph).
    weights.resetFoldPool();
    // Same routing owner as det/rec (makeExecPlan): CUDA/Vulkan one-device,
    // CPU gallocr, otherwise mixed sched behind the tagging gate.
    const aicore::ocr::OcrExecPlan plan = aicore::ocr::makeExecPlan(
            gpuBackend, cpuBackend, weights.sched_gpu_tagging);
    const aicore::ocr::OcrBuildOptions buildOptions =
            aicore::ocr::execBuildOptions(plan);
    std::vector<ggml_tensor*> fetchList;
    ggml_tensor* outputTensor = aicore::ocr::buildGraph(
            ctx, weights, input, &error, nullptr, &fetchList, &buildOptions);
    if (!outputTensor || fetchList.size() != 8) {
        if (error.empty()) error = "ocr: layout graph fetch mismatch";
        ggml_free(ctx);
        return false;
    }
    ggml_cgraph* graph = nullptr;
    bool ok = false;
    do {
        if (!aicore::ocr::prepareGraph(weights, ctx, plan, fetchList, &graph,
                                       exec, error))
            break;
        ggml_backend_tensor_set(input, nchw.data(), 0,
                                sizeof(float) * nchw.size());
        const auto start = std::chrono::steady_clock::now();
        aicore::ocr::computeGraph(plan, exec, graph);
        const auto stop = std::chrono::steady_clock::now();
        inferenceMs =
                std::chrono::duration<double, std::milli>(stop - start).count();
        heads.clear();
        heads.reserve(fetchList.size());
        for (ggml_tensor* t : fetchList) {
            std::vector<float> data(ggml_nelements(t));
            ggml_backend_tensor_get(t, data.data(), 0,
                                    sizeof(float) * data.size());
            heads.push_back(std::move(data));
        }
        ok = true;
    } while (false);
    if (!ok) exec.free();
    ggml_free(ctx);
    return ok;
}

// ---- PicoDet GFL decode + per-class hard NMS (PP-Structure
// PicoDetPostProcess, the layout -> table/OCR consumer of this model) ----

constexpr int kLayoutSizeH = 800;  // official PicoDet layout input
constexpr int kLayoutSizeW = 608;
constexpr int kGflRegMax = 8;
// PP-Structure layout_score_threshold / layout_nms_threshold defaults.
constexpr float kLayoutScoreThresh = 0.5f;
constexpr float kLayoutNmsThresh = 0.5f;
constexpr int kLayoutNmsTopK = 1000;       // per-stride pre-NMS anchors
constexpr int kLayoutCandidateSize = 200;  // hard_nms candidate_size
constexpr int kLayoutKeepTopK = 100;       // per class
constexpr int kLayoutStrides[4] = {8, 16, 32, 64};

struct LayoutRegion {
    float score;
    int classId;
    float x1, y1, x2, y2;
};

// Anchors surviving the per-stride top-K, with every class probability.
struct LayoutCandidates {
    int classes = 0;
    std::vector<std::array<float, 4>> boxes;  // layout input pixels
    std::vector<float> scores;                // boxes.size() x classes
};

// One stride: the exported cls head is already sigmoid-activated (the
// upstream post-process thresholds it as-is), the reg head carries the
// (4, reg_max) DFL logits. Keeps the kLayoutNmsTopK anchors with the
// highest max-class probability, then decodes their boxes (no clip: the
// reference clips only after NMS). O(A log A) for the top-K sort.
void gflDecode(const std::vector<float>& clsHead,
               const std::vector<float>& regHead,
               int stride,
               LayoutCandidates& out) {
    const int fh = (kLayoutSizeH + stride - 1) / stride;
    const int fw = (kLayoutSizeW + stride - 1) / stride;
    const int anchors = fh * fw;
    const int classes = static_cast<int>(clsHead.size()) / anchors;
    out.classes = classes;
    std::vector<float> best(anchors);
    std::vector<int> order(anchors);
    for (int a = 0; a < anchors; ++a) {
        const float* row = &clsHead[static_cast<size_t>(a) * classes];
        best[a] = *std::max_element(row, row + classes);
        order[a] = a;
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](int l, int r) { return best[l] > best[r]; });
    order.resize(std::min(anchors, kLayoutNmsTopK));
    for (const int a : order) {
        const float* regRow =
                &regHead[static_cast<size_t>(a) * (4 * kGflRegMax)];
        float dist[4] = {0.f, 0.f, 0.f, 0.f};
        for (int edge = 0; edge < 4; ++edge) {
            const float* bins = regRow + edge * kGflRegMax;
            const float maxv = *std::max_element(bins, bins + kGflRegMax);
            float sum = 0.f, acc = 0.f;
            for (int bin = 0; bin < kGflRegMax; ++bin) {
                const float e = std::exp(bins[bin] - maxv);
                sum += e;
                acc += e * bin;
            }
            dist[edge] = acc / sum * stride;
        }
        const float cx = (a % fw + 0.5f) * stride;
        const float cy = (a / fw + 0.5f) * stride;
        out.boxes.push_back(
                {cx - dist[0], cy - dist[1], cx + dist[2], cy + dist[3]});
        const float* row = &clsHead[static_cast<size_t>(a) * classes];
        out.scores.insert(out.scores.end(), row, row + classes);
    }
}

// picodet_postprocess iou_of: clipped extents, eps 1e-5.
float layoutIou(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    const auto area = [](float x1, float y1, float x2, float y2) {
        return std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    };
    const float inter = area(std::max(a[0], b[0]), std::max(a[1], b[1]),
                             std::min(a[2], b[2]), std::min(a[3], b[3]));
    return inter / (area(a[0], a[1], a[2], a[3]) +
                    area(b[0], b[1], b[2], b[3]) - inter + 1e-5f);
}

// Per class: probability > threshold, hard NMS over the 200 best
// candidates, at most kLayoutKeepTopK kept. Output is class-major, each
// class in descending score order (the reference concatenation order).
// O(C * K^2) with K <= kLayoutCandidateSize.
std::vector<LayoutRegion> layoutNms(const LayoutCandidates& cand) {
    std::vector<LayoutRegion> keep;
    const size_t n = cand.boxes.size();
    for (int cat = 0; cat < cand.classes; ++cat) {
        std::vector<size_t> idx;
        for (size_t i = 0; i < n; ++i)
            if (cand.scores[i * cand.classes + cat] > kLayoutScoreThresh)
                idx.push_back(i);
        const auto score = [&](size_t i) {
            return cand.scores[i * cand.classes + cat];
        };
        std::stable_sort(idx.begin(), idx.end(), [&](size_t l, size_t r) {
            return score(l) > score(r);
        });
        if (idx.size() > static_cast<size_t>(kLayoutCandidateSize))
            idx.resize(kLayoutCandidateSize);
        std::vector<bool> removed(idx.size(), false);
        int kept = 0;
        for (size_t i = 0; i < idx.size() && kept < kLayoutKeepTopK; ++i) {
            if (removed[i]) continue;
            const auto& box = cand.boxes[idx[i]];
            keep.push_back(
                    {score(idx[i]), cat, box[0], box[1], box[2], box[3]});
            ++kept;
            for (size_t j = i + 1; j < idx.size(); ++j)
                if (!removed[j] &&
                    layoutIou(box, cand.boxes[idx[j]]) > kLayoutNmsThresh)
                    removed[j] = true;
        }
    }
    // warp_boxes: clip to the layout input before the image mapping.
    for (LayoutRegion& r : keep) {
        r.x1 = std::clamp(r.x1, 0.0f, static_cast<float>(kLayoutSizeW));
        r.x2 = std::clamp(r.x2, 0.0f, static_cast<float>(kLayoutSizeW));
        r.y1 = std::clamp(r.y1, 0.0f, static_cast<float>(kLayoutSizeH));
        r.y2 = std::clamp(r.y2, 0.0f, static_cast<float>(kLayoutSizeH));
    }
    return keep;
}

}  // namespace

struct aicore_ocr_ctx {
    aicore::ocr::OcrWeights detWeights;
    aicore::ocr::OcrWeights recWeights;
    bool recLoaded = false;
    aicore::ocr::OcrWeights clsWeights;
    bool clsLoaded = false;
    float clsThreshold = 0.9f;
    int clsBatch = 6;
    int detMaxCandidates = 1000;
    OcrGraphCache clsGraphs{{}, 2, 0};
    std::vector<float> clsBatchScratch, clsOutputScratch;
    // Optional PP-Structure layout head (PicoDet CDLA).
    aicore::ocr::OcrWeights layoutWeights;
    bool layoutLoaded = false;
    std::string language = "ch";
    float detMaxSide = kDetOfficialLimitSide;
    int detLimitType = 0;  // 0 = official "min" resize, 1 = legacy "max" cap
    float boxThresh = kDetDefaultBoxThresh;
    double boxScoreThresh = kDetDefaultBoxScoreThresh;
    float unclipRatio = kDetDefaultUnclipRatio;
    int recPadGroup = kRecPadGroupDefault;
    // Session-owned physical backends, acquired once from the process-wide
    // runtime lease registry (GPU/auto primary + CPU secondary for the
    // scheduler, or a single CPU lease) and reused by every runGraph call;
    // the weights realize against the GPU backend on first use, so its
    // lifetime must span the whole context. The lease keeps the handle
    // shared with other task sessions without handing out raw ownership.
    aicore::runtime::BackendLease primaryLease;
    aicore::runtime::BackendLease cpuLease;
    std::string resolvedDevice;
    // Det graph parameters are shape-stable only per resolution, so the
    // graph is rebuilt whenever the resized input geometry changes.
    OcrGraph det;
    int64_t detC = 0;
    int64_t detH = 0;
    int64_t detW = 0;
    // Rec builds keyed by (pad width, rows): see OcrGraphCache.
    OcrGraphCache recGraphs{{}, kRecGraphCacheCapacity, 0};
    // Layout graphs are rebuilt per frame (fresh fold pool); the allocator
    // persists so the fixed-size compute buffer is reused.
    aicore::ocr::OcrGraphExec layoutExec;
    // Frame scratch owned by the context (one owning thread, see the public
    // header). The det input (10 MB u8 + 40 MB f32), det map (13 MB) and
    // rec batch/output (19 MB / 51 MB) were fresh vectors every frame: the
    // allocation page-faulted and the fill ran serially before any real
    // work (measured 11.5 ms for the det NCHW alone against 2.9 ms for the
    // resize that feeds it). Reusing the capacity makes only the first
    // frame pay; a geometry change simply regrows the vectors.
    std::vector<uint8_t> detRgbScratch;
    std::vector<float> detNchwScratch;
    std::vector<float> detOutputScratch;
    std::vector<int8_t> detContourScratch;  // findContours bordered image
    std::vector<float> recBatchScratch;
    std::vector<float> recOutputScratch;
    RecArgmaxOut recArgmaxScratch;
    bool ready = false;
};

struct aicore_ocr_result {
    std::vector<aicore_ocr_box> boxes;
    std::vector<aicore_ocr_region> regions;
    std::vector<aicore_ocr_relation> relations;
    std::vector<aicore_ocr_orientation> orientations;
    aicore_pipeline_timings timings{};
    aicore_ocr_post_timing_breakdown postTimings{};
    // Text storage backing aicore_ocr_box::text pointers. MUST be a deque:
    // the fill sites take c_str() pointers into these strings as they push,
    // and a vector's push_back reallocation would move every SSO string
    // body, dangling ALL previously handed-out box.text pointers. That
    // use-after-free read stale-but-intact heap in the common case (which
    // is why it looked benign) and flipped single-box transcriptions
    // between empty/garbled whenever a later malloc reused the freed
    // buffer — the root cause behind the "threaded CTC race" and the
    // 98<->97 single-box drift. deque never invalidates element pointers
    // on push_back.
    std::deque<std::string> textStorage;
};

namespace {

// ---- Rec crop geometry, chunk plan and batched forward (shared by the
// det->rec frame path and the recognize-only quad path) ------------------

// One rec crop under its source contract: PaddleX CropByPolys re-fits the
// integer polygon, while X-AnyLabeling warps the quad directly. Both use
// get_rotate_crop_image at int() float32 edge norms, cubic warp, rot90 when
// h/w >= 1.5) followed by resize_norm_img (resized_w = ceil(48 * w/h),
// padded to the chunk width with the normalized-zero fill).
struct RecCropGeom {
    float quad[4][2] = {};  // reported box (det box / caller quad)
    float box[4][2] = {};   // re-fit crop box the warp samples
    int cropW = 0;
    int cropH = 0;
    bool rotate = false;
    bool rotate180 = false;
    aicore_ocr_orientation orientation{};
    double ratio = 0.0;  // w / h after the rot90, the official sort key
    int padW = 0;        // own padded width: int(48 * max(320/48, ratio))
    int recW = 0;        // resized content width
    int chunkW = 0;      // padded width of the crop's ratio group
    int chunk = -1;      // slot assigned by planRecChunks
    int row = -1;
    float detScore = 0.f;
    bool valid = false;
};

// `poly` is the crop source in image pixels, finite and inside the int32
// range (det/legacy callers clamp it; selected regions retain border pixels).
void recCropGeometry(RecCropGeom& g, const float poly[4][2], bool directQuad) {
    if (directQuad) {
        // X-AnyLabeling TextSystem's det_box_type="quad" warps the
        // detector/caller quad directly. Re-fitting it with minAreaRect
        // moves even almost-axis-aligned edges and changes rec pixels.
        for (int i = 0; i < 4; ++i) std::copy(poly[i], poly[i] + 2, g.box[i]);
    } else {
        // PaddleX CropByPolys first re-fits its integer polygon.
        int pts[4][2];
        for (int i = 0; i < 4; ++i) {
            pts[i][0] = static_cast<int>(poly[i][0]);
            pts[i][1] = static_cast<int>(poly[i][1]);
        }
        aicore::ocr::minAreaRectBox(pts, 4, g.box);
    }
    aicore::ocr::rotateCropSize(g.box, &g.cropW, &g.cropH);
    // Upstream warps any non-empty size (an empty dsize would make cv2 fall
    // back to the source size, which no det box reaches).
    g.valid = g.cropW >= 1 && g.cropH >= 1;
    if (!g.valid) return;
    g.rotate = static_cast<double>(g.cropH) / g.cropW >= 1.5;
    const int w = g.rotate ? g.cropH : g.cropW;
    const int h = g.rotate ? g.cropW : g.cropH;
    const double ratio = static_cast<double>(w) / h;
    const double maxRatio =
            std::max(static_cast<double>(kRecMaxWidth) / kRecHeight, ratio);
    const int imgW = static_cast<int>(kRecHeight * maxRatio);
    int resizedW = static_cast<int>(std::ceil(kRecHeight * ratio));
    if (imgW > kRecLimitWidth)
        resizedW = kRecLimitWidth;
    else if (resizedW > imgW)
        resizedW = imgW;
    g.ratio = ratio;
    g.padW = std::min(imgW, kRecLimitWidth);
    g.recW = std::max(1, resizedW);
    g.chunkW = g.padW;
}

// The official batch pad (pipeline sorted(key=w/h), stable; ToBatch pads
// each group of `group` consecutive crops to its widest padW). Rows are
// independent inside a forward, so only the width has to match; crops of
// different groups with the same width still share a chunk.
// O(n log n) for the sort.
void assignRecPadWidths(std::vector<RecCropGeom>& geoms,
                        int group,
                        bool sourceBatchResize) {
    std::vector<int> order;
    order.reserve(geoms.size());
    for (size_t i = 0; i < geoms.size(); ++i)
        if (geoms[i].valid) order.push_back(static_cast<int>(i));
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return geoms[a].ratio < geoms[b].ratio;
    });
    for (size_t at = 0; at < order.size(); at += group) {
        const size_t end = std::min(order.size(), at + group);
        int width = 0;
        for (size_t k = at; k < end; ++k)
            width = std::max(width, geoms[order[k]].padW);
        for (size_t k = at; k < end; ++k) {
            RecCropGeom& g = geoms[order[k]];
            g.chunkW = width;
            if (sourceBatchResize) {
                // TextRecognizer.resize_norm_img clips ceil(48*w/h) to
                // the batch width. Clipping to this crop's own floor width
                // first shrinks non-maximal wide crops by one pixel.
                g.recW = std::max(
                        1, std::min(width, static_cast<int>(std::ceil(
                                                   kRecHeight * g.ratio))));
            }
        }
    }
}

struct RecChunk {
    int width = 0;
    int64_t rows = 0;   // graph N (count + pad tail)
    int count = 0;      // real crops
    int first = 0;      // index into RecChunkPlan::order
    size_t offset = 0;  // float offset of row 0 in the packed buffer
};

struct RecChunkPlan {
    std::vector<RecChunk> chunks;
    std::vector<int> order;  // valid crop indices grouped by chunk
    size_t totalFloats = 0;
};

// Group the valid crops by chunk width and cut each group into chunks.
// Batched: rows x width stays within the per-backend budget
// (recRowsCapForWidth) and the row count is quantized so a stable box
// count keeps hitting the same cached build. Per-crop (CPU execution):
// one single-row chunk per crop at its group width.
RecChunkPlan planRecChunks(std::vector<RecCropGeom>& geoms,
                           int padGroup,
                           int64_t cap,
                           bool batched,
                           bool sourceBatchResize) {
    assignRecPadWidths(geoms, padGroup, sourceBatchResize);
    RecChunkPlan plan;
    plan.order.reserve(geoms.size());
    for (size_t i = 0; i < geoms.size(); ++i)
        if (geoms[i].valid) plan.order.push_back(static_cast<int>(i));
    std::stable_sort(plan.order.begin(), plan.order.end(), [&](int a, int b) {
        return geoms[a].chunkW < geoms[b].chunkW;
    });
    const size_t rowFloats = static_cast<size_t>(3) * kRecHeight;
    size_t pos = 0;
    while (pos < plan.order.size()) {
        const int width = geoms[plan.order[pos]].chunkW;
        size_t end = pos;
        while (end < plan.order.size() &&
               geoms[plan.order[end]].chunkW == width)
            ++end;
        const int64_t groupCount = static_cast<int64_t>(end - pos);
        int64_t rowsPerChunk = 1;
        if (batched) {
            const int64_t capW = recRowsCapForWidth(cap, width);
            const int64_t chunksWanted = (groupCount + capW - 1) / capW;
            const int64_t rows = (groupCount + chunksWanted - 1) / chunksWanted;
            const int64_t q = capW >= 8 ? 8 : 4;
            rowsPerChunk = std::min(capW, (rows + q - 1) / q * q);
        }
        for (size_t at = pos; at < end;) {
            RecChunk c;
            c.width = width;
            c.rows = rowsPerChunk;
            c.count = static_cast<int>(std::min<size_t>(
                    end - at, static_cast<size_t>(rowsPerChunk)));
            c.first = static_cast<int>(at);
            c.offset = plan.totalFloats;
            for (int r = 0; r < c.count; ++r) {
                RecCropGeom& g = geoms[plan.order[at + r]];
                g.chunk = static_cast<int>(plan.chunks.size());
                g.row = r;
            }
            plan.totalFloats += static_cast<size_t>(c.rows) * rowFloats * width;
            plan.chunks.push_back(c);
            at += c.count;
        }
        pos = end;
    }
    return plan;
}

// Pad-tail fill for every chunk (rows [count, rows)). Content rows are
// written in full by packNchwImageTo (content + right pad columns), so
// this is the only fill the buffer needs; it is tiny next to the frame.
void fillRecPadRows(const RecChunkPlan& plan, float* batch) {
    const size_t rowFloats = static_cast<size_t>(3) * kRecHeight;
    for (const RecChunk& c : plan.chunks) {
        if (c.count >= c.rows) continue;
        const size_t planeSz = rowFloats * c.width;
        std::fill(batch + c.offset + planeSz * c.count,
                  batch + c.offset + planeSz * c.rows, kRecPadFill);
    }
}

// Perspective crop + (rot90) + resize + BGR + normalized pack of one crop
// straight into its chunk slot. Pure over the borrowed view; distinct
// slots per crop keep the parallel callers race-free.
aicore::img::Image warpRecCrop(const aicore::img::Image& src,
                               const RecCropGeom& g) {
    std::vector<uint8_t> quadRgb(static_cast<size_t>(g.cropW) * g.cropH * 3);
    aicore::ocr::warpPerspectiveCubicReplicate(src, g.box, g.cropW, g.cropH,
                                               quadRgb.data());
    aicore::img::Image warped;
    warped.channels = 3;
    if (g.rotate) {
        // np.rot90: counter-clockwise, out(i, j) = in(j, W-1-i), shape
        // (W, H) -> the text runs left-to-right again for the rec head.
        const int W = g.cropW, H = g.cropH;
        std::vector<uint8_t> rot(static_cast<size_t>(W) * H * 3);
        for (int i = 0; i < W; ++i) {
            uint8_t* orow = rot.data() + static_cast<size_t>(i) * H * 3;
            for (int j = 0; j < H; ++j) {
                const uint8_t* s =
                        quadRgb.data() +
                        (static_cast<size_t>(j) * W + (W - 1 - i)) * 3;
                orow[j * 3 + 0] = s[0];
                orow[j * 3 + 1] = s[1];
                orow[j * 3 + 2] = s[2];
            }
        }
        warped.width = H;
        warped.height = W;
        warped.rgb = std::move(rot);
    } else {
        warped.width = g.cropW;
        warped.height = g.cropH;
        warped.rgb = std::move(quadRgb);
    }
    if (g.rotate180) {
        const size_t pixels = static_cast<size_t>(warped.width) * warped.height;
        for (size_t i = 0; i < pixels / 2; ++i)
            for (size_t ch = 0; ch < 3; ++ch)
                std::swap(warped.rgb[i * 3 + ch],
                          warped.rgb[(pixels - 1 - i) * 3 + ch]);
    }
    return warped;
}

void cropAndPackRec(const aicore::img::Image& src,
                    const RecCropGeom& g,
                    const RecChunkPlan& plan,
                    float* batch) {
    const auto warped = warpRecCrop(src, g);
    aicore::img::Image cropImage;
    cropImage.width = g.recW;
    cropImage.height = kRecHeight;
    cropImage.channels = 3;
    cropImage.bgr = true;  // official cv2 channel order
    cropImage.rgb.resize(static_cast<size_t>(g.recW) * kRecHeight * 3);
    aicore::img::cv_resize_linear_u8(warped, cropImage.rgb.data(), g.recW,
                                     kRecHeight, 3);
    const RecChunk& c = plan.chunks[static_cast<size_t>(g.chunk)];
    float* slot = batch + c.offset +
                  static_cast<size_t>(g.row) * 3 * kRecHeight * c.width;
    packNchwImageTo(cropImage, kRecMean3, kRecStd3, c.width, kRecPadFill, slot);
}

// Phase A driver: crop/pack every planned crop over a bounded pool.
void cropAndPackAll(const aicore::img::Image& src,
                    const std::vector<RecCropGeom>& geoms,
                    const RecChunkPlan& plan,
                    float* batch) {
    const int count = static_cast<int>(plan.order.size());
    if (count == 0) return;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    if (hw > static_cast<unsigned>(count)) hw = static_cast<unsigned>(count);
    std::atomic<int> next{0};
    auto worker = [&]() {
        for (;;) {
            const int k = next.fetch_add(1);
            if (k >= count) return;
            cropAndPackRec(src, geoms[static_cast<size_t>(plan.order[k])], plan,
                           batch);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(hw - 1);
    for (unsigned i = 1; i < hw; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
}

struct RecRunTimings {
    double inferenceMs = 0.0;
    double inputUploadMs = 0.0;
    double outputReadbackMs = 0.0;
    double ctcDecodeMs = 0.0;
};

// Upstream TextClassifier: stable aspect order, 3x48x192 BGR, normalized
// zero padding, batches of six, and a strict >0.9 rotation gate. Keep only
// one finite batch of crops; regenerate the warp for rec rather than retain
// a frame-sized collection of decoded crops.
bool classifyCrops(aicore_ocr_ctx* ctx,
                   ggml_backend_t gpuBackend,
                   ggml_backend_t cpuBackend,
                   const aicore::img::Image& src,
                   std::vector<RecCropGeom>& geoms,
                   RecRunTimings& timings,
                   double& packMs,
                   std::string& error) {
    if (!ctx->clsLoaded) return true;
    std::vector<size_t> order;
    for (size_t i = 0; i < geoms.size(); ++i)
        if (geoms[i].valid) order.push_back(i);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return geoms[a].ratio < geoms[b].ratio;
    });
    constexpr int width = 192, height = 48;
    constexpr size_t rowFloats = 3 * height * width;
    for (size_t at = 0; at < order.size(); at += ctx->clsBatch) {
        const size_t rows = std::min<size_t>(ctx->clsBatch, order.size() - at);
        const auto packStart = std::chrono::steady_clock::now();
        ctx->clsBatchScratch.resize(rows * rowFloats);
        for (size_t row = 0; row < rows; ++row) {
            const auto& g = geoms[order[at + row]];
            const auto warped = warpRecCrop(src, g);
            const int resizedWidth = std::clamp(
                    static_cast<int>(std::ceil(height * g.ratio)), 1, width);
            aicore::img::Image resized;
            resized.width = resizedWidth;
            resized.height = height;
            resized.channels = 3;
            resized.bgr = true;
            resized.rgb.resize(static_cast<size_t>(height) * resizedWidth * 3);
            aicore::img::cv_resize_linear_u8(warped, resized.rgb.data(),
                                             resizedWidth, height, 3);
            packNchwImageTo(resized, kRecMean3, kRecStd3, width, kRecPadFill,
                            ctx->clsBatchScratch.data() + row * rowFloats);
        }
        packMs += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - packStart)
                          .count();
        auto& cached = ctx->clsGraphs.acquire(3, height, width, rows, false);
        int64_t c = 0, h = 0, w = 0, n = 0;
        double compute = 0, upload = 0, readback = 0;
        if (!runGraphCached(ctx, ctx->clsWeights, cached, gpuBackend,
                            cpuBackend, 3, height, width, rows,
                            ctx->clsBatchScratch.data(), rows * rowFloats,
                            ctx->clsOutputScratch, c, h, w, n, compute, upload,
                            readback, error))
            return false;
        if (w != 2 || ctx->clsOutputScratch.size() != rows * 2) {
            error = "ocr: classifier output must have two probabilities per "
                    "crop";
            return false;
        }
        timings.inferenceMs += compute;
        timings.inputUploadMs += upload;
        timings.outputReadbackMs += readback;
        for (size_t row = 0; row < rows; ++row) {
            auto& g = geoms[order[at + row]];
            const float p0 = ctx->clsOutputScratch[row * 2];
            const float p180 = ctx->clsOutputScratch[row * 2 + 1];
            if (!std::isfinite(p0) || !std::isfinite(p180)) {
                error = "ocr: classifier produced non-finite probabilities";
                return false;
            }
            g.orientation.angle = p180 > p0 ? 180 : 0;
            g.orientation.score = std::max(p0, p180);
            g.rotate180 = g.orientation.angle == 180 &&
                          g.orientation.score > ctx->clsThreshold;
            g.orientation.rotated = g.rotate180 ? 1 : 0;
        }
    }
    return true;
}

// Phase B: one forward per chunk against the (width, rows) cached build,
// then the per-row CTC decode in parallel. texts/scores are indexed by
// crop (geoms index); unplanned crops are left untouched.
bool runRecChunks(aicore_ocr_ctx* ctx,
                  ggml_backend_t gpuBackend,
                  ggml_backend_t cpuBackend,
                  const std::vector<RecCropGeom>& geoms,
                  const RecChunkPlan& plan,
                  const float* batch,
                  std::vector<std::string>& texts,
                  std::vector<float>& scores,
                  RecRunTimings& timings,
                  std::string& error) {
    const bool cpuExecution =
            aicore::ocr::makeExecPlan(gpuBackend, cpuBackend,
                                      ctx->recWeights.sched_gpu_tagging)
                    .cpuExecution;
    // Same gate as runGraphCached's wantArgmaxHead: the cache key must
    // select the build that the call will produce.
    const bool wantArgmaxHead = !cpuExecution;
    std::vector<float> recOutput = std::move(ctx->recOutputScratch);
    RecArgmaxOut& recArgmax = ctx->recArgmaxScratch;
    const size_t rowFloats = static_cast<size_t>(3) * kRecHeight;
    for (const RecChunk& c : plan.chunks) {
        OcrGraph& cached = ctx->recGraphs.acquire(3, kRecHeight, c.width,
                                                  c.rows, wantArgmaxHead);
        int64_t outC = 0, outT = 0, outV = 0, outN = 0;
        double inferenceMs = 0.0, uploadMs = 0.0, readbackMs = 0.0;
        if (!runGraphCached(ctx, ctx->recWeights, cached, gpuBackend,
                            cpuBackend, 3, kRecHeight, c.width, c.rows,
                            batch + c.offset,
                            static_cast<size_t>(c.rows) * rowFloats * c.width,
                            recOutput, outC, outT, outV, outN, inferenceMs,
                            uploadMs, readbackMs, error, &recArgmax)) {
            ctx->recOutputScratch = std::move(recOutput);
            return false;
        }
        timings.inferenceMs += inferenceMs;
        timings.inputUploadMs += uploadMs;
        timings.outputReadbackMs += readbackMs;
        // Per-row CTC decode in parallel: rows are independent pure
        // functions over the chunk output; each worker writes ONLY its
        // crop's slot. Worker cap 8: spawning hardware_concurrency threads
        // per chunk costs more in pthread_create than the extra
        // parallelism returns at these row counts.
        const auto ctcStart = std::chrono::steady_clock::now();
        std::atomic<int> nextRow{0};
        const unsigned workers = std::min<unsigned>(
                {8u, std::max(1u, std::thread::hardware_concurrency()),
                 static_cast<unsigned>(c.count)});
        auto decoder = [&]() {
            for (;;) {
                const int r = nextRow.fetch_add(1);
                if (r >= c.count) return;
                const int crop = plan.order[static_cast<size_t>(c.first + r)];
                float meanScore = 0.0f;
                if (recArgmax.active) {
                    // Reduced head: rows are [t + T * n].
                    const size_t rowBase =
                            static_cast<size_t>(r) * static_cast<size_t>(outT);
                    texts[crop] = aicore::ocr::ctcGreedyDecodeIndexed(
                            recArgmax.index.data() + rowBase,
                            recArgmax.prob.data() + rowBase,
                            static_cast<int>(outT), static_cast<int>(outV),
                            ctx->recWeights.dictionary, &meanScore);
                } else {
                    const float* rowBase = recOutput.data() +
                                           static_cast<size_t>(r) *
                                                   static_cast<size_t>(outV) *
                                                   static_cast<size_t>(outT);
                    texts[crop] = aicore::ocr::ctcGreedyDecode(
                            rowBase, static_cast<int>(outT),
                            static_cast<int>(outV), ctx->recWeights.dictionary,
                            &meanScore);
                }
                scores[crop] = meanScore;
            }
        };
        if (workers > 1) {
            std::vector<std::thread> pool;
            pool.reserve(workers - 1);
            for (unsigned i = 1; i < workers; ++i) pool.emplace_back(decoder);
            decoder();
            for (auto& t : pool) t.join();
        } else {
            decoder();
        }
        timings.ctcDecodeMs +=
                std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - ctcStart)
                        .count();
    }
    ctx->recOutputScratch = std::move(recOutput);
    return true;
}

}  // namespace

extern "C" {

aicore_ocr_ctx* aicore_ocr_context_create(const aicore_ocr_options* options) {
    if (!options || options->struct_size < kOcrOptionsInitialSize) {
        setLastError("ocr: invalid options (struct_size)");
        return nullptr;
    }
    if (options->abi_version != AICORE_OCR_ABI_VERSION) {
        setLastError("ocr: abi version mismatch");
        return nullptr;
    }
    if (!options->det_model_path || !options->rec_model_path) {
        setLastError("ocr: det/rec model paths are required");
        return nullptr;
    }
    if (options->kie_model_path && options->kie_model_path[0]) {
        setLastError("ocr: KIE models are not supported by this ABI version");
        return nullptr;
    }
    const bool hasCls = hasOptionField(
            options, offsetof(aicore_ocr_options, cls_model_path),
            sizeof(options->cls_model_path));
    const bool hasClsThreshold =
            hasOptionField(options, offsetof(aicore_ocr_options, cls_thresh),
                           sizeof(options->cls_thresh));
    const bool hasClsBatch =
            hasOptionField(options, offsetof(aicore_ocr_options, cls_max_batch),
                           sizeof(options->cls_max_batch));
    const bool hasMaxCandidates = hasOptionField(
            options, offsetof(aicore_ocr_options, det_max_candidates),
            sizeof(options->det_max_candidates));
    if ((hasClsThreshold &&
         (!std::isfinite(options->cls_thresh) || options->cls_thresh < 0 ||
          options->cls_thresh > 1)) ||
        (hasClsBatch &&
         (options->cls_max_batch < 0 || options->cls_max_batch > 1024)) ||
        (hasMaxCandidates && options->det_max_candidates < 0)) {
        setLastError("ocr: invalid classifier or candidate options");
        return nullptr;
    }
    auto* ctx = new (std::nothrow) aicore_ocr_ctx();
    if (!ctx) {
        setLastError("ocr: context allocation failed");
        return nullptr;
    }
    std::string error;
    if (!ctx->detWeights.load(options->det_model_path, &error)) {
        setLastError("ocr: det model load failed: " + error);
        delete ctx;
        return nullptr;
    }
    if (!ctx->recWeights.load(options->rec_model_path, &error)) {
        setLastError("ocr: rec model load failed: " + error);
        delete ctx;
        return nullptr;
    }
    ctx->recLoaded = true;
    if (hasCls && options->cls_model_path && options->cls_model_path[0]) {
        const bool loaded =
                ctx->clsWeights.load(options->cls_model_path, &error);
        if (loaded && ctx->clsWeights.role != "cls")
            error = "expected cls model role";
        if (!loaded || ctx->clsWeights.role != "cls") {
            setLastError("ocr: classifier model load failed: " + error);
            delete ctx;
            return nullptr;
        }
        ctx->clsLoaded = true;
    }
    if (hasClsThreshold && options->cls_thresh > 0)
        ctx->clsThreshold = options->cls_thresh;
    if (hasClsBatch && options->cls_max_batch > 0)
        ctx->clsBatch = options->cls_max_batch;
    if (hasMaxCandidates && options->det_max_candidates > 0)
        ctx->detMaxCandidates = options->det_max_candidates;
    // Optional layout model: the PicoDet CDLA pack with the embedded
    // class table; regions stay empty when absent.
    if (options->layout_model_path) {
        if (!ctx->layoutWeights.load(options->layout_model_path, &error)) {
            setLastError("ocr: layout model load failed: " + error);
            delete ctx;
            return nullptr;
        }
        if (ctx->layoutWeights.labels.empty()) {
            setLastError("ocr: layout model has no embedded labels");
            delete ctx;
            return nullptr;
        }
        ctx->layoutLoaded = true;
    }
    // Script tag from the rec catalog filename (ppocr-rec-<lang>-*).
    const std::string recPath = options->rec_model_path;
    const auto langPos = recPath.find("ppocr-rec-");
    if (langPos != std::string::npos) {
        std::string tail = recPath.substr(langPos + std::strlen("ppocr-rec-"));
        const auto dash = tail.find('-');
        if (dash != std::string::npos) tail = tail.substr(0, dash);
        if (!tail.empty()) ctx->language = tail;
    }
    // det_limit_type 0 (default) aligns with PaddleOCR 3.0.1's public OCR
    // pipeline: "min"-type resize with limit_side_len 64. det_max_side
    // overrides that limit; 1 selects the table pipeline's long-side cap.
    const bool hasDetLimitType = hasOptionField(
            options, offsetof(aicore_ocr_options, det_limit_type),
            sizeof(options->det_limit_type));
    const bool hasBoxScoreThreshold = hasOptionField(
            options, offsetof(aicore_ocr_options, det_box_score_thresh),
            sizeof(options->det_box_score_thresh));
    ctx->detLimitType = hasDetLimitType ? options->det_limit_type : 0;
    ctx->detMaxSide =
            ctx->detLimitType == 0
                    ? (options->det_max_side > 0
                               ? static_cast<float>(options->det_max_side)
                               : kDetOfficialLimitSide)
                    : (options->det_max_side > 0
                               ? static_cast<float>(options->det_max_side)
                               : kDetDefaultMaxSide);
    ctx->boxThresh = options->det_box_thresh > 0 ? options->det_box_thresh
                                                 : kDetDefaultBoxThresh;
    ctx->boxScoreThresh =
            hasBoxScoreThreshold && options->det_box_score_thresh > 0
                    ? options->det_box_score_thresh
                    : kDetDefaultBoxScoreThresh;
    ctx->unclipRatio = options->det_unclip_ratio > 0 ? options->det_unclip_ratio
                                                     : kDetDefaultUnclipRatio;
    ctx->recPadGroup = options->rec_max_batch > 0 ? options->rec_max_batch
                                                  : kRecPadGroupDefault;
    // One backend pair for the whole context, resolved from the caller's
    // device request. load_backends_once() is mandatory in GGML_BACKEND_DL
    // builds: without it no device is registered and every init silently
    // returns nullptr. "auto" follows the platform order (CUDA -> Vulkan ->
    // CPU on Linux/Windows, Metal -> CPU on macOS) and already includes the
    // CPU fallback; an explicit unavailable request is an honest failure.
    ggml_common::load_backends_once();
    const bool hasDevice =
            hasOptionField(options, offsetof(aicore_ocr_options, device),
                           sizeof(options->device));
    const bool hasThreads =
            hasOptionField(options, offsetof(aicore_ocr_options, n_threads),
                           sizeof(options->n_threads));
    const std::string deviceRequest =
            (hasDevice && options->device && options->device[0])
                    ? options->device
                    : "auto";
    // The OCR graphs are hundreds of small nodes; beyond a handful of CPU
    // threads the per-node pool barrier dominates (measured: rec graph
    // 94 ms at 32 threads vs 61 ms at 8). Cap the default like the
    // facedetect runtime does; an explicit options->n_threads still wins.
    int nThreads = hasThreads ? options->n_threads : 0;
    if (nThreads <= 0) {
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 4;
        nThreads = static_cast<int>(std::min(hw, 8u));
    }
    std::string leaseError;
    ctx->primaryLease = aicore::runtime::acquire_backend_lease(
            deviceRequest, nThreads, &leaseError);
    if (!ctx->primaryLease) {
        setLastError("ocr: requested device unavailable (" + deviceRequest +
                     "): " + leaseError);
        delete ctx;
        return nullptr;
    }
    ctx->resolvedDevice = ctx->primaryLease.device();
    if (!ctx->primaryLease.is_cpu()) {
        // WEIGHTS tagging gate: CUDA keeps the sched GPU offload (5-8x per
        // forward); Vulkan keeps the legacy all-CPU routing for now because
        // offloaded det graphs trip an independent ggml-vulkan subbuffer
        // fault on some geometries (352x544).
        const bool isVulkan =
                !ctx->primaryLease.is_cpu() &&
                std::string(ggml_backend_name(ctx->primaryLease.handle()))
                                .find("Vulkan") != std::string::npos;
        ctx->detWeights.sched_gpu_tagging = !isVulkan;
        // Measured reversal (this session): routing rec through the sched
        // offload on Vulkan is SLOWER than the single-backend route
        // (73-77 ms vs 58.6 ms per 40-row chunk). The sched shards the
        // ~600-node rec graph into ~172 splits (small nodes below the
        // offload threshold alternate CPU/VK) and every split pays a
        // queue submit + weight-check + barrier; the plain gallocr pays
        // one dispatch per op instead. The probe numbers that suggested
        // sched was 5x faster were mislabeled: the probe's
        // init_by_name("Vulkan") falls back to init_by_type and returns
        // CUDA0, so "backend=Vulkan device=CUDA0" probes measured CUDA.
        // rec keeps the gate closed like det/layout and runs through
        // AICORE_OCR_VK_SINGLE.
        ctx->recWeights.sched_gpu_tagging = !isVulkan;
        ctx->clsWeights.sched_gpu_tagging = !isVulkan;
        if (ctx->layoutLoaded) {
            ctx->layoutWeights.sched_gpu_tagging = !isVulkan;
        }
        // GPU runs schedule against a CPU secondary (see runGraphCached); the
        // parallel lease keeps independent CPU graphs concurrent.
        ctx->cpuLease = aicore::runtime::acquire_parallel_backend_lease(
                "cpu", nThreads, &leaseError);
        if (!ctx->cpuLease) {
            setLastError("ocr: CPU secondary backend unavailable: " +
                         leaseError);
            delete ctx;
            return nullptr;
        }
        // NOTE: the weights are realized after each graph build (see
        // runGraphCached). Realizing before the build was measured as a net
        // regression: the scheduler assigns more nodes to the GPU, which
        // both slowed the small-input case ~5x (90 vs 19 ms det fixture)
        // and drifted the f32 det parity beyond its 0.01 gate at high
        // resolutions (0.0139 vs 0.0012 max diff) through the changed
        // op->backend mapping.
    }
    // Only the rec graph carries the embedded dictionary (the det graph is
    // a pure DBNet CNN with no vocabulary).
    if (ctx->recWeights.dictionary.empty()) {
        setLastError("ocr: rec model has no embedded dictionary");
        delete ctx;
        return nullptr;
    }
    ctx->ready = true;
    return ctx;
}

void aicore_ocr_context_destroy(aicore_ocr_ctx* ctx) {
    if (!ctx) return;
    ctx->det.clear();
    ctx->recGraphs.clear();
    ctx->clsGraphs.clear();
    ctx->layoutExec.free();
    // Release the leases after the graphs; the weights' device buffer is
    // released by the OcrWeights destructor and must outlive neither.
    ctx->cpuLease.reset();
    ctx->primaryLease.reset();
    delete ctx;
}

int aicore_ocr_context_ready(const aicore_ocr_ctx* ctx) {
    return ctx && ctx->ready ? 1 : 0;
}

const char* aicore_ocr_context_device(const aicore_ocr_ctx* ctx) {
    return ctx ? ctx->resolvedDevice.c_str() : "";
}

aicore_ocr_result* aicore_ocr_run(aicore_ocr_ctx* ctx,
                                  const aicore_image_view* image) {
    if (!ctx || !ctx->ready || !image || !image->data) {
        setLastError("ocr: run called with an invalid context or image");
        return nullptr;
    }
    const auto e2eStart = std::chrono::steady_clock::now();
    const auto preprocessStart = std::chrono::steady_clock::now();

    int channels = 3;
    bool swapRB = false;
    switch (image->format) {
        case AICORE_IMAGE_RGB8:
            channels = 3;
            swapRB = false;
            break;
        case AICORE_IMAGE_RGBA8:
            channels = 4;
            swapRB = false;
            break;
        case AICORE_IMAGE_BGR8:
            channels = 3;
            swapRB = true;
            break;
        case AICORE_IMAGE_BGRA8:
            channels = 4;
            swapRB = true;
            break;
        case AICORE_IMAGE_GRAY8:
            channels = 1;
            swapRB = false;
            break;
        default:
            setLastError("ocr: unsupported image format");
            return nullptr;
    }
    if (image->width <= 0 || image->height <= 0) {
        setLastError("ocr: empty image");
        return nullptr;
    }

    // --- det preprocessing: limit the long side, align to 32, normalize.
    // The resize goes through the shared OpenCV-aligned bilinear
    // (common/image_utils); channel mapping and the borrowed stride are
    // handled by the Image view wrapper.
    aicore::img::Image srcImage;
    if (!aicore::img::image_from_view(*image, srcImage)) {
        setLastError("ocr: unsupported image view");
        return nullptr;
    }
    srcImage.bgr = swapRB;  // BGR/BGRA views are channel-swapped on read
    // Upstream DetResizeForTest. det_limit_type 0 (default) replicates the
    // PaddleOCR 3.0.1 OCR pipeline (limit_type="min", limit_side_len=64,
    // max_side_limit=4000). det_limit_type 1 selects the table pipeline's
    // never-upscale long-side cap. Either way each side TRUNCATES to int,
    // then rounds to the nearest multiple of 32 with a floor at 32 (the
    // network needs 32-stride alignment); the historical ceil() overshoot
    // by up to 31 px skewed the det map ratio and the box mapping.
    float scale;
    if (ctx->detLimitType == 0) {
        const int minSide = std::min(image->width, image->height);
        scale = minSide < static_cast<int>(ctx->detMaxSide)
                        ? ctx->detMaxSide / static_cast<float>(minSide)
                        : 1.0f;
    } else {
        scale = std::min(
                1.0f, ctx->detMaxSide / static_cast<float>(std::max(
                                                image->width, image->height)));
    }
    const int resizedW = static_cast<int>(image->width * scale);
    const int resizedH = static_cast<int>(image->height * scale);
    // Python round() is round-half-to-even; replicate it so pinned inputs
    // land on the exact upstream geometry.
    const auto roundHalfEven = [](double v) {
        const double f = std::floor(v);
        const double d = v - f;
        if (d > 0.5) return f + 1.0;
        if (d < 0.5) return f;
        return std::fmod(f, 2.0) != 0.0 ? f + 1.0 : f;
    };
    int detW = std::max(static_cast<int>(roundHalfEven(resizedW / 32.0)) * 32,
                        kDetAlign);
    int detH = std::max(static_cast<int>(roundHalfEven(resizedH / 32.0)) * 32,
                        kDetAlign);
    if (ctx->detLimitType == 0) {
        // Official max_side_limit (4000): if either rounded side exceeds
        // it, rescale both sides to fit before the 32 alignment stands.
        const int longSide = std::max(detW, detH);
        if (longSide > static_cast<int>(kDetOfficialMaxSideLimit)) {
            const float down =
                    kDetOfficialMaxSideLimit / static_cast<float>(longSide);
            detW = std::max(static_cast<int>(roundHalfEven(
                                    static_cast<double>(detW) * down / 32.0)) *
                                    32,
                            kDetAlign);
            detH = std::max(static_cast<int>(roundHalfEven(
                                    static_cast<double>(detH) * down / 32.0)) *
                                    32,
                            kDetAlign);
        }
    }
    aicore::img::Image detImage;
    detImage.width = detW;
    detImage.height = detH;
    detImage.channels = 3;
    detImage.rgb = std::move(ctx->detRgbScratch);
    detImage.rgb.resize(static_cast<size_t>(detW) * detH * 3);
    aicore::img::cv_resize_linear_u8(srcImage, detImage.rgb.data(), detW, detH,
                                     3);
    // The official pipeline feeds the networks BGR (cv2 order, never
    // converted); align the channel order here. Color-sensitive rec
    // features and the det map both shift without this. The flag makes
    // Image::channel read plane 2-c, which is exactly what the historical
    // in-place byte swap produced, minus one full pass over the frame.
    detImage.bgr = true;
    std::vector<float> detNchw = std::move(ctx->detNchwScratch);
    packNchwImage(detImage, kImageMean, kImageStd, detW, 0.f, detNchw);
    ctx->detRgbScratch = std::move(detImage.rgb);
    const auto preprocessStop = std::chrono::steady_clock::now();

    // --- det inference: DB probability map (1,1,H,W). The graph time is
    // captured by runGraphCached (detInferenceMs), not a wall clock, so
    // the inference stage can stay truthful when rec/layout also run.
    // Session backend handles: GPU primary + CPU secondary, or a single
    // CPU backend when the lease resolved to CPU.
    ggml_backend_t gpuBackend =
            ctx->primaryLease.is_cpu() ? nullptr : ctx->primaryLease.handle();
    ggml_backend_t cpuBackend = ctx->primaryLease.is_cpu()
                                        ? ctx->primaryLease.handle()
                                        : ctx->cpuLease.handle();

    std::vector<float> detOutput = std::move(ctx->detOutputScratch);
    int64_t detOutC = 0, detOutH = 0, detOutW = 0, detOutN = 0;
    double detInferenceMs = 0.0;
    double detInputUploadMs = 0.0;
    double detOutputReadbackMs = 0.0;
    // rec/layout graph time is accumulated separately so the stage timing
    // stays truthful: inference_ms = det + rec + layout graph compute,
    // postprocess_ms excludes it.
    double recLayoutInferenceMs = 0.0;
    std::string error;
    const bool detOk = runGraphCached(
            ctx, ctx->detWeights, ctx->det, gpuBackend, cpuBackend, 3, detH,
            detW, 1, detNchw.data(), static_cast<size_t>(3) * detH * detW,
            detOutput, detOutC, detOutH, detOutW, detOutN, detInferenceMs,
            detInputUploadMs, detOutputReadbackMs, error);
    // The input is uploaded synchronously inside runGraphCached, so its
    // buffer can go back to the context right here (even on failure).
    ctx->detNchwScratch = std::move(detNchw);
    if (!detOk) {
        setLastError(error);
        return nullptr;
    }

    // --- det post-process: DBPostProcess.boxes_from_bitmap + SortQuadBoxes
    // (exact ports, ocr_db_postprocess.cpp): integer boxes in image pixels,
    // in the official reading order.
    const auto postStart = std::chrono::steady_clock::now();
    const auto dbBoxesStart = postStart;
    aicore::ocr::DbParams dbParams;
    dbParams.thresh = ctx->boxThresh;
    dbParams.boxThresh = ctx->boxScoreThresh;
    dbParams.unclipRatio = ctx->unclipRatio;
    dbParams.maxCandidates = ctx->detMaxCandidates;
    auto boxes = aicore::ocr::dbBoxesFromBitmap(
            detOutput.data(), static_cast<int>(detOutW),
            static_cast<int>(detOutH), image->width, image->height, dbParams,
            ctx->detContourScratch);
    if (ctx->detWeights.program.onnxLowered)
        aicore::ocr::filterSourceDetectorBoxes(boxes, image->width,
                                               image->height);
    aicore::ocr::sortQuadBoxes(boxes);
    const auto dbBoxesStop = std::chrono::steady_clock::now();

    auto* result = new (std::nothrow) aicore_ocr_result();
    if (!result) {
        setLastError("ocr: result allocation failed");
        return nullptr;
    }
    result->boxes.reserve(boxes.size());

    // Packed full-resolution RGB for the perspective crops (the det map
    // is a downscaled working image; the official pipeline warps from the
    // original image).
    const auto cropResizePackStart = std::chrono::steady_clock::now();
    // Two-phase batched recognition (shared with the recognize-only path):
    //   A) per-box unclip/map + official crop geometry, chunk plan, then
    //      parallel crop/pack straight into the planned slots;
    //   B) one forward per (width, rows) chunk + per-row CTC decode.
    // Batched rec: automatic per execution side (see ocrBatchedRecEnabled).
    const bool recCpuExecution =
            aicore::ocr::makeExecPlan(gpuBackend, cpuBackend,
                                      ctx->recWeights.sched_gpu_tagging)
                    .cpuExecution;
    const bool batchedRec = ocrBatchedRecEnabled(gpuBackend, recCpuExecution);
    // Batch chunking: measured single-forward scaling on the idle RTX 4090
    // is linear to N=72 (10/30/84/91/100/120 ms at N=4/16/48/56/64/72,
    // fixture-exact at every N) and collapses by N=96 (15.5 s) because the
    // depthwise im2col workspace pushes the gallocr buffer against VRAM
    // (20.2/24.5 GiB) and every compute re-warms the CUDA graphs. The 48
    // row cap (at 320 wide; recRowsCapForWidth scales it down for wider
    // chunks) stands on a same-load A/B: 48 rows at 215 ms e2e vs 64 rows
    // at 257 ms for the 74-box bench frame. Pad rows are computed at full
    // cost, so the row count is quantized (planRecChunks) rather than
    // filled to the cap, and a stable box count hits the same build.
    std::vector<RecCropGeom> geoms(boxes.size());
    for (size_t bi = 0; bi < boxes.size(); ++bi) {
        RecCropGeom& g = geoms[bi];
        for (int i = 0; i < 4; ++i) {
            g.quad[i][0] = static_cast<float>(boxes[bi].quad[i][0]);
            g.quad[i][1] = static_cast<float>(boxes[bi].quad[i][1]);
        }
        g.detScore = static_cast<float>(boxes[bi].score);
        recCropGeometry(g, g.quad, ctx->recWeights.program.onnxLowered);
    }
    RecRunTimings clsTimings;
    double clsPackMs = 0.0;
    const auto clsStart = std::chrono::steady_clock::now();
    if (!classifyCrops(ctx, gpuBackend, cpuBackend, srcImage, geoms, clsTimings,
                       clsPackMs, error)) {
        setLastError(error);
        delete result;
        return nullptr;
    }
    const double clsElapsedMs =
            std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - clsStart)
                    .count();
    const RecChunkPlan recPlan =
            planRecChunks(geoms, ctx->recPadGroup, recChunkBatchCap(gpuBackend),
                          batchedRec, ctx->recWeights.program.onnxLowered);
    std::vector<float> batchNchw = std::move(ctx->recBatchScratch);
    // resize, not assign: every content row is written in full by the pack
    // (content + right pad columns) and the pad tails by fillRecPadRows.
    batchNchw.resize(recPlan.totalFloats);
    fillRecPadRows(recPlan, batchNchw.data());
    // Sample with quads in IMAGE pixel coordinates (the unclipped region
    // mapped out of the det map). The det-map coordinate space is 2x the
    // image under the official "min"/736 resize; sampling the full-res
    // image with det-map coordinates produces edge-clamped garbage.
    cropAndPackAll(srcImage, geoms, recPlan, batchNchw.data());
    const auto cropResizePackStop = std::chrono::steady_clock::now();
    double inputUploadMs = detInputUploadMs;
    double outputReadbackMs = detOutputReadbackMs;
    std::vector<std::string> recTexts(boxes.size());
    std::vector<float> recScores(boxes.size(), 0.0f);
    RecRunTimings recTimings;
    if (!runRecChunks(ctx, gpuBackend, cpuBackend, geoms, recPlan,
                      batchNchw.data(), recTexts, recScores, recTimings,
                      error)) {
        setLastError(error);
        delete result;
        return nullptr;
    }
    recLayoutInferenceMs += recTimings.inferenceMs + clsTimings.inferenceMs;
    inputUploadMs += recTimings.inputUploadMs + clsTimings.inputUploadMs;
    outputReadbackMs +=
            recTimings.outputReadbackMs + clsTimings.outputReadbackMs;
    const double ctcDecodeMs = recTimings.ctcDecodeMs;
    // Box fill in det order (the chunk plan reorders by width only for the
    // forward); textStorage pushes stay ordered so the deque-backed
    // box.text pointers never dangle.
    const auto assemblyStart = std::chrono::steady_clock::now();
    for (size_t bi = 0; bi < boxes.size(); ++bi) {
        const RecCropGeom& g = geoms[bi];
        if (!g.valid) continue;
        aicore_ocr_box box{};
        std::memcpy(box.quad, g.quad, sizeof(box.quad));
        result->textStorage.push_back(recTexts[bi]);
        box.text = result->textStorage.back().c_str();
        // This is recognition confidence, including zero for an all-blank
        // CTC decode. Substituting detector confidence lets empty text pass
        // the caller's drop_score filter, unlike the source TextSystem.
        box.score = recScores[bi];
        std::strncpy(box.language, ctx->language.c_str(),
                     sizeof(box.language) - 1);
        result->boxes.push_back(box);
        if (ctx->clsLoaded) result->orientations.push_back(g.orientation);
    }
    double resultAssemblyMs =
            std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - assemblyStart)
                    .count();
    // Hand the frame buffers back to the context for the next frame (error
    // returns above simply drop them; the next frame reallocates).
    ctx->detOutputScratch = std::move(detOutput);
    ctx->recBatchScratch = std::move(batchNchw);

    // --- optional layout stage: PicoDet CDLA heads -> regions.
    if (ctx->layoutLoaded) {
        const auto layoutAssemblyStart = std::chrono::steady_clock::now();
        aicore::img::Image layoutImage;
        layoutImage.width = kLayoutSizeW;
        layoutImage.height = kLayoutSizeH;
        layoutImage.channels = 3;
        layoutImage.rgb.resize(static_cast<size_t>(kLayoutSizeW) *
                               kLayoutSizeH * 3);
        aicore::img::cv_resize_linear_u8(srcImage, layoutImage.rgb.data(),
                                         kLayoutSizeW, kLayoutSizeH, 3);
        // PP-Structure LayoutPredictor: cv2 BGR frame, per-channel
        // NormalizeImage (same planes as the det input).
        layoutImage.bgr = true;
        std::vector<float> layoutNchw;
        packNchwImage(layoutImage, kImageMean, kImageStd, kLayoutSizeW, 0.f,
                      layoutNchw);
        std::vector<std::vector<float>> heads;
        double layoutMs = 0.0;
        std::string layoutError;
        if (!runLayoutHeads(
                    ctx->layoutWeights, ctx->layoutExec,
                    ctx->primaryLease.is_cpu() ? nullptr
                                               : ctx->primaryLease.handle(),
                    ctx->primaryLease.is_cpu() ? ctx->primaryLease.handle()
                                               : ctx->cpuLease.handle(),
                    3, kLayoutSizeH, kLayoutSizeW, layoutNchw, heads, layoutMs,
                    layoutError)) {
            // The layout stage is an enhancement: a failure downgrades the
            // result to boxes-only (with the reason on last_error for the
            // caller's log) instead of discarding the recognition pass.
            setLastError("ocr: layout stage failed: " + layoutError);
        } else {
            recLayoutInferenceMs += layoutMs;
            LayoutCandidates candidates;
            for (int si = 0; si < 4; ++si) {
                // Fetch order: transpose_0..7 = cls(8,16,32,64) then
                // reg(8,16,32,64).
                gflDecode(heads[si], heads[4 + si], kLayoutStrides[si],
                          candidates);
            }
            for (const LayoutRegion& r : layoutNms(candidates)) {
                aicore_ocr_region region{};
                region.class_id = r.classId;
                region.reading_order = -1;
                region.quad[0][0] = r.x1;
                region.quad[0][1] = r.y1;
                region.quad[1][0] = r.x2;
                region.quad[1][1] = r.y1;
                region.quad[2][0] = r.x2;
                region.quad[2][1] = r.y2;
                region.quad[3][0] = r.x1;
                region.quad[3][1] = r.y2;
                region.score = r.score;
                result->regions.push_back(region);
            }
            // Reading order: top-to-bottom band, left-to-right within a band
            // (the upstream PP-Structure sort, adequate for single-column
            // documents and refined per-class in a later round). The band
            // test and the comparison run in the layout input space before
            // the coordinate mapping below.
            std::vector<int> order(result->regions.size());
            for (int i = 0; i < static_cast<int>(order.size()); ++i)
                order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                const auto& ra = result->regions[a];
                const auto& rb = result->regions[b];
                const float ay = (ra.quad[0][1] + ra.quad[3][1]) * 0.5f;
                const float by = (rb.quad[0][1] + rb.quad[3][1]) * 0.5f;
                if (std::fabs(ay - by) > 0.02f * kLayoutSizeH) return ay < by;
                return ra.quad[0][0] < rb.quad[0][0];
            });
            for (int rank = 0; rank < static_cast<int>(order.size()); ++rank)
                result->regions[order[rank]].reading_order = rank;
            // Map the region quads from the 608x800 layout input space back
            // to source image pixels (the same ratio walk the upstream
            // PP-Structure visualizer applies). Regions share the box
            // contract: image pixel coordinates.
            const float regionToImageX = static_cast<float>(image->width) /
                                         static_cast<float>(kLayoutSizeW);
            const float regionToImageY = static_cast<float>(image->height) /
                                         static_cast<float>(kLayoutSizeH);
            for (auto& region : result->regions) {
                for (int c = 0; c < 4; ++c) {
                    region.quad[c][0] =
                            std::clamp(region.quad[c][0] * regionToImageX, 0.0f,
                                       static_cast<float>(image->width));
                    region.quad[c][1] =
                            std::clamp(region.quad[c][1] * regionToImageY, 0.0f,
                                       static_cast<float>(image->height));
                }
            }
        }
        resultAssemblyMs +=
                std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - layoutAssemblyStart)
                        .count() -
                layoutMs;
    }
    const auto postStop = std::chrono::steady_clock::now();
    const auto e2eStop = postStop;
    // The det graph ran inside the inference window above; the rec and
    // layout graphs ran inside the post window, so their graph time moves
    // from postprocess into inference and the remainder is the true
    // post-process work (boxes, crops, decode, NMS).
    result->timings.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    result->timings.valid_fields =
            AICORE_TIMING_PREPROCESS | AICORE_TIMING_INFERENCE |
            AICORE_TIMING_POSTPROCESS | AICORE_TIMING_E2E;
    result->timings.preprocess_ms = std::chrono::duration<double, std::milli>(
                                            preprocessStop - preprocessStart)
                                            .count();
    result->timings.inference_ms = detInferenceMs + recLayoutInferenceMs;
    double postMs =
            std::chrono::duration<double, std::milli>(postStop - postStart)
                    .count() -
            recLayoutInferenceMs;
    result->timings.postprocess_ms = postMs > 0.0 ? postMs : 0.0;
    result->timings.e2e_ms =
            std::chrono::duration<double, std::milli>(e2eStop - e2eStart)
                    .count();
    result->postTimings.abi_version = AICORE_OCR_POST_TIMINGS_ABI_VERSION;
    result->postTimings.valid_fields =
            AICORE_OCR_POST_TIMING_DB_BOXES |
            AICORE_OCR_POST_TIMING_CROP_RESIZE_PACK |
            AICORE_OCR_POST_TIMING_CTC_DECODE |
            AICORE_OCR_POST_TIMING_RESULT_ASSEMBLY |
            AICORE_OCR_POST_TIMING_OUTPUT_READBACK |
            AICORE_OCR_POST_TIMING_INPUT_UPLOAD |
            AICORE_OCR_POST_TIMING_DET_GRAPH_COMPUTE |
            AICORE_OCR_POST_TIMING_REC_GRAPH_COMPUTE;
    result->postTimings.db_boxes_ms = std::chrono::duration<double, std::milli>(
                                              dbBoxesStop - dbBoxesStart)
                                              .count();
    result->postTimings.crop_resize_pack_ms =
            std::chrono::duration<double, std::milli>(cropResizePackStop -
                                                      cropResizePackStart)
                    .count() -
            clsElapsedMs + clsPackMs;
    result->postTimings.ctc_decode_ms = ctcDecodeMs;
    result->postTimings.result_assembly_ms = std::max(0.0, resultAssemblyMs);
    result->postTimings.output_readback_ms = outputReadbackMs;
    result->postTimings.input_upload_ms = inputUploadMs;
    // Split inference timing by graph to identify det/rec hotspots instead of
    // guessing from totals.
    result->postTimings.det_graph_compute_ms = detInferenceMs;
    result->postTimings.rec_graph_compute_ms = recTimings.inferenceMs;
    return result;
}

// Skip-detection path: recognize text inside caller-provided quads.
// Mirrors the rec stage of run() (perspective warp -> height-48 resize ->
// CTC) without the det pass; quads are full-image pixel coordinates.
static aicore_ocr_result* recognizeQuads(aicore_ocr_ctx* ctx,
                                         const aicore_image_view* image,
                                         const float* quads,
                                         int32_t count,
                                         bool selectedRegions) {
    if (!ctx || !image || !quads || count <= 0) {
        setLastError("ocr: recognize requires ctx, image and quads");
        return nullptr;
    }
    if (!ctx->ready) {
        setLastError("ocr: context not ready");
        return nullptr;
    }
    aicore::img::Image srcImage;
    int channels = 0;
    bool swapRB = false;
    switch (image->format) {
        case AICORE_IMAGE_RGB8:
            channels = 3;
            swapRB = false;
            break;
        case AICORE_IMAGE_RGBA8:
            channels = 4;
            swapRB = false;
            break;
        case AICORE_IMAGE_BGR8:
            channels = 3;
            swapRB = true;
            break;
        case AICORE_IMAGE_BGRA8:
            channels = 4;
            swapRB = true;
            break;
        case AICORE_IMAGE_GRAY8:
            channels = 1;
            swapRB = false;
            break;
        default:
            setLastError("ocr: unsupported image format");
            return nullptr;
    }
    if (image->width <= 0 || image->height <= 0) {
        setLastError("ocr: empty image");
        return nullptr;
    }
    if (!aicore::img::image_from_view(*image, srcImage)) {
        setLastError("ocr: unsupported image view");
        return nullptr;
    }
    srcImage.bgr = swapRB;
    // The crop samples the borrowed view in place (stride / 1-, 3- or
    // 4-channel / BGR all resolved by Image::row + channel_index).
    const auto cropResizePackStart = std::chrono::steady_clock::now();
    auto* result = new (std::nothrow) aicore_ocr_result();
    if (!result) {
        setLastError("ocr: result allocation failed");
        return nullptr;
    }
    result->boxes.reserve(static_cast<size_t>(count));
    const auto start = std::chrono::steady_clock::now();
    std::string error;
    // Same two-phase recognition as the frame path (recCropGeometry /
    // planRecChunks / cropAndPackAll / runRecChunks): official crop
    // geometry, official ratio-group pad widths, one forward per chunk,
    // parallel CTC.
    ggml_backend_t gpuBackend =
            ctx->primaryLease.is_cpu() ? nullptr : ctx->primaryLease.handle();
    ggml_backend_t cpuBackend = ctx->primaryLease.is_cpu()
                                        ? ctx->primaryLease.handle()
                                        : ctx->cpuLease.handle();
    const bool recCpuExecution =
            aicore::ocr::makeExecPlan(gpuBackend, cpuBackend,
                                      ctx->recWeights.sched_gpu_tagging)
                    .cpuExecution;
    const bool batchedRec = ocrBatchedRecEnabled(gpuBackend, recCpuExecution);
    std::vector<RecCropGeom> geoms(static_cast<size_t>(count));
    for (int32_t q = 0; q < count; ++q) {
        RecCropGeom& g = geoms[static_cast<size_t>(q)];
        std::memcpy(g.quad, quads + static_cast<size_t>(q) * 8, sizeof(g.quad));
        // Same crop as the det path, so recognize(run(img) quads) reproduces
        // run(img). Caller quads are clipped to the image like det boxes
        // (bounds the crop size); a non-finite quad stays invalid.
        float poly[4][2];
        bool finite = true;
        for (int i = 0; i < 4; ++i) {
            finite = finite && std::isfinite(g.quad[i][0]) &&
                     std::isfinite(g.quad[i][1]);
            poly[i][0] = selectedRegions
                                 ? g.quad[i][0]
                                 : std::clamp(g.quad[i][0], 0.0f,
                                              static_cast<float>(image->width));
            poly[i][1] =
                    selectedRegions
                            ? g.quad[i][1]
                            : std::clamp(g.quad[i][1], 0.0f,
                                         static_cast<float>(image->height));
        }
        if (finite)
            recCropGeometry(
                    g, poly,
                    selectedRegions || ctx->recWeights.program.onnxLowered);
    }
    RecRunTimings clsTimings;
    double clsPackMs = 0.0;
    const auto clsStart = std::chrono::steady_clock::now();
    if (!classifyCrops(ctx, gpuBackend, cpuBackend, srcImage, geoms, clsTimings,
                       clsPackMs, error)) {
        setLastError(error);
        delete result;
        return nullptr;
    }
    const double clsElapsedMs =
            std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - clsStart)
                    .count();
    const RecChunkPlan recPlan =
            planRecChunks(geoms, ctx->recPadGroup, recChunkBatchCap(gpuBackend),
                          batchedRec, ctx->recWeights.program.onnxLowered);
    std::vector<float> batchNchw = std::move(ctx->recBatchScratch);
    batchNchw.resize(recPlan.totalFloats);
    fillRecPadRows(recPlan, batchNchw.data());
    cropAndPackAll(srcImage, geoms, recPlan, batchNchw.data());
    const auto cropResizePackStop = std::chrono::steady_clock::now();
    std::vector<std::string> recTexts(static_cast<size_t>(count));
    std::vector<float> recScores(static_cast<size_t>(count), 0.0f);
    RecRunTimings recTimings;
    if (!runRecChunks(ctx, gpuBackend, cpuBackend, geoms, recPlan,
                      batchNchw.data(), recTexts, recScores, recTimings,
                      error)) {
        setLastError(error);
        delete result;
        return nullptr;
    }
    ctx->recBatchScratch = std::move(batchNchw);
    const double recInferenceMs =
            recTimings.inferenceMs + clsTimings.inferenceMs;
    const double inputUploadMs =
            recTimings.inputUploadMs + clsTimings.inputUploadMs;
    const double outputReadbackMs =
            recTimings.outputReadbackMs + clsTimings.outputReadbackMs;
    const double ctcDecodeMs = recTimings.ctcDecodeMs;
    const auto assemblyStart = std::chrono::steady_clock::now();
    for (int32_t q = 0; q < count; ++q) {
        const RecCropGeom& g = geoms[static_cast<size_t>(q)];
        // One box per input quad, in input order: callers map results back
        // by index, so a degenerate quad yields an empty text, score 0.
        aicore_ocr_box box{};
        std::memcpy(box.quad, g.quad, sizeof(box.quad));
        result->textStorage.push_back(g.valid ? recTexts[static_cast<size_t>(q)]
                                              : std::string());
        box.text = result->textStorage.back().c_str();
        box.score = g.valid ? recScores[static_cast<size_t>(q)] : 0.0f;
        std::strncpy(box.language, ctx->language.c_str(),
                     sizeof(box.language) - 1);
        result->boxes.push_back(box);
        if (ctx->clsLoaded) result->orientations.push_back(g.orientation);
    }
    const double resultAssemblyMs =
            std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - assemblyStart)
                    .count();

    const auto stop = std::chrono::steady_clock::now();
    result->timings.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    result->timings.valid_fields = AICORE_TIMING_INFERENCE | AICORE_TIMING_E2E;
    result->timings.inference_ms = recInferenceMs;
    result->timings.e2e_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
    result->postTimings.abi_version = AICORE_OCR_POST_TIMINGS_ABI_VERSION;
    result->postTimings.valid_fields = AICORE_OCR_POST_TIMING_CROP_RESIZE_PACK |
                                       AICORE_OCR_POST_TIMING_CTC_DECODE |
                                       AICORE_OCR_POST_TIMING_RESULT_ASSEMBLY |
                                       AICORE_OCR_POST_TIMING_OUTPUT_READBACK |
                                       AICORE_OCR_POST_TIMING_INPUT_UPLOAD |
                                       AICORE_OCR_POST_TIMING_REC_GRAPH_COMPUTE;
    result->postTimings.crop_resize_pack_ms =
            std::chrono::duration<double, std::milli>(cropResizePackStop -
                                                      cropResizePackStart)
                    .count() -
            clsElapsedMs + clsPackMs;
    result->postTimings.ctc_decode_ms = ctcDecodeMs;
    result->postTimings.result_assembly_ms = resultAssemblyMs;
    result->postTimings.output_readback_ms = outputReadbackMs;
    result->postTimings.input_upload_ms = inputUploadMs;
    result->postTimings.rec_graph_compute_ms = recTimings.inferenceMs;
    return result;
}

aicore_ocr_result* aicore_ocr_recognize(aicore_ocr_ctx* ctx,
                                        const aicore_image_view* image,
                                        const float* quads,
                                        int32_t count) {
    return recognizeQuads(ctx, image, quads, count, false);
}

aicore_ocr_result* aicore_ocr_recognize_regions(
        aicore_ocr_ctx* ctx,
        const aicore_image_view* image,
        const aicore_ocr_text_region* regions,
        int32_t count) {
    const auto start = std::chrono::steady_clock::now();
    if (!ctx || !image || !regions || count <= 0) {
        setLastError(
                "ocr: selected recognition requires ctx, image and regions");
        return nullptr;
    }
    std::vector<float> quads(static_cast<size_t>(count) * 8);
    for (int32_t i = 0; i < count; ++i) {
        const auto& region = regions[i];
        if (!region.points_xy || region.point_count < 3) {
            setLastError("ocr: selected region needs at least three points");
            return nullptr;
        }
        std::vector<std::array<int, 2>> points(region.point_count);
        for (int32_t j = 0; j < region.point_count; ++j) {
            for (int k = 0; k < 2; ++k) {
                const float value = region.points_xy[size_t(j) * 2 + k];
                // Float(INT_MAX) rounds out of range, so compare as double.
                if (!std::isfinite(value) || double(value) < INT32_MIN ||
                    double(value) > INT32_MAX) {
                    setLastError(
                            "ocr: selected region coordinate is not a finite "
                            "int32 pixel");
                    return nullptr;
                }
                points[j][k] = static_cast<int>(value);
            }
        }
        float box[4][2];
        aicore::ocr::minAreaRectBox(
                reinterpret_cast<const int(*)[2]>(points.data()),
                region.point_count, box);
        // Source normalize_custom_detection_boxes: lexsort(y,x), then
        // top pair left-to-right and bottom pair right-to-left.
        int order[4] = {0, 1, 2, 3};
        std::stable_sort(order, order + 4, [&](int a, int b) {
            return box[a][1] < box[b][1] ||
                   (box[a][1] == box[b][1] && box[a][0] < box[b][0]);
        });
        if (box[order[0]][0] > box[order[1]][0]) std::swap(order[0], order[1]);
        if (box[order[2]][0] < box[order[3]][0]) std::swap(order[2], order[3]);
        for (int j = 0; j < 4; ++j)
            std::copy(box[order[j]], box[order[j]] + 2,
                      quads.data() + size_t(i) * 8 + j * 2);
    }
    std::vector<int32_t> order(count);
    for (int32_t i = 0; i < count; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        const float* qa = quads.data() + size_t(a) * 8;
        const float* qb = quads.data() + size_t(b) * 8;
        return qa[1] < qb[1] || (qa[1] == qb[1] && qa[0] < qb[0]);
    });
    // Source sorted_boxes uses this O(count^2) worst-case adjacent correction.
    // Preserve it: reading order determines padding batches and recognized
    // text.
    for (int32_t i = 0; i < count - 1; ++i)
        for (int32_t j = i; j >= 0; --j) {
            const float* a = quads.data() + size_t(order[j]) * 8;
            const float* b = quads.data() + size_t(order[j + 1]) * 8;
            if (std::fabs(b[1] - a[1]) < 10 && b[0] < a[0])
                std::swap(order[j], order[j + 1]);
            else
                break;
        }
    std::vector<float> sorted(quads.size());
    for (int32_t i = 0; i < count; ++i)
        std::copy_n(quads.data() + size_t(order[i]) * 8, 8,
                    sorted.data() + size_t(i) * 8);
    const auto normalized = std::chrono::steady_clock::now();
    auto* result = recognizeQuads(ctx, image, sorted.data(), count, true);
    if (result) {
        // Keep source reading-order batching but return original input order
        // so the UI can update annotations without replacing their geometry.
        const auto assembly = std::chrono::steady_clock::now();
        const auto boxes = result->boxes;
        const auto orientations = result->orientations;
        for (int32_t i = 0; i < count; ++i) {
            result->boxes[order[i]] = boxes[i];
            if (!orientations.empty())
                result->orientations[order[i]] = orientations[i];
        }
        const auto stop = std::chrono::steady_clock::now();
        result->postTimings.crop_resize_pack_ms +=
                std::chrono::duration<double, std::milli>(normalized - start)
                        .count();
        result->postTimings.result_assembly_ms +=
                std::chrono::duration<double, std::milli>(stop - assembly)
                        .count();
        result->timings.e2e_ms =
                std::chrono::duration<double, std::milli>(stop - start).count();
    }
    return result;
}

const aicore_ocr_orientation* aicore_ocr_orientation_at(
        const aicore_ocr_result* result, int index) {
    if (!result || index < 0 ||
        index >= static_cast<int>(result->orientations.size()))
        return nullptr;
    return &result->orientations[index];
}

int aicore_ocr_box_count(const aicore_ocr_result* result) {
    return result ? static_cast<int>(result->boxes.size()) : 0;
}

const aicore_ocr_box* aicore_ocr_box_at(const aicore_ocr_result* result,
                                        int index) {
    if (!result || index < 0 || index >= static_cast<int>(result->boxes.size()))
        return nullptr;
    return &result->boxes[index];
}

int aicore_ocr_label_count(const aicore_ocr_ctx* ctx) {
    return ctx ? static_cast<int>(ctx->layoutWeights.labels.size()) : 0;
}

const char* aicore_ocr_label_at(const aicore_ocr_ctx* ctx, int index) {
    if (!ctx || index < 0 ||
        index >= static_cast<int>(ctx->layoutWeights.labels.size()))
        return nullptr;
    return ctx->layoutWeights.labels[index].c_str();
}

int aicore_ocr_region_count(const aicore_ocr_result* result) {
    return result ? static_cast<int>(result->regions.size()) : 0;
}

const aicore_ocr_region* aicore_ocr_region_at(const aicore_ocr_result* result,
                                              int index) {
    if (!result || index < 0 ||
        index >= static_cast<int>(result->regions.size()))
        return nullptr;
    return &result->regions[index];
}

int aicore_ocr_relation_count(const aicore_ocr_result* result) {
    return result ? static_cast<int>(result->relations.size()) : 0;
}

const aicore_ocr_relation* aicore_ocr_relation_at(
        const aicore_ocr_result* result, int index) {
    if (!result || index < 0 ||
        index >= static_cast<int>(result->relations.size()))
        return nullptr;
    return &result->relations[index];
}

void aicore_ocr_result_release(aicore_ocr_result* result) { delete result; }

aicore_pipeline_timings aicore_ocr_timings(const aicore_ocr_result* result) {
    aicore_pipeline_timings timings{};
    if (!result) return timings;
    timings = result->timings;
    return timings;
}

aicore_ocr_post_timing_breakdown aicore_ocr_post_timings(
        const aicore_ocr_result* result) {
    aicore_ocr_post_timing_breakdown timings{};
    if (!result) return timings;
    timings = result->postTimings;
    return timings;
}

const char* aicore_ocr_last_error(void) { return g_lastError.c_str(); }

void aicore_ocr_runtime_shutdown(void) {
    // Delegates to the process-wide shutdown; live contexts are untouched.
    aicore_runtime_shutdown();
}

}  // extern "C"

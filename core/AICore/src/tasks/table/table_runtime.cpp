// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "table_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "gguf.h"

namespace aicore {
namespace table {

namespace {

// Decoder steps unrolled per graph. A table emits 30-500 tokens; one launch
// per 16 steps keeps the dispatch count at ceil(steps / 16) while the steps
// computed past eos (< 16) are discarded on the host.
constexpr int kStepsPerGraph = 16;

double msSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
            .count();
}

uint32_t readU32(gguf_context* gguf, const char* key, uint32_t fallback) {
    const int idx = gguf_find_key(gguf, key);
    if (idx < 0 || gguf_get_kv_type(gguf, idx) != GGUF_TYPE_UINT32)
        return fallback;
    return gguf_get_val_u32(gguf, idx);
}

ggml_tensor* matmulF32(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    // The decoder feeds its own argmax back, so a GPU backend's f16
    // accumulation could flip a near-tie token and diverge the whole
    // remaining sequence; request f32 accumulation everywhere.
    ggml_tensor* out = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    return out;
}

}  // namespace

void TableStructureModel::Graph::clear() {
    if (alloc) ggml_gallocr_free(alloc);
    if (ctx) ggml_free(ctx);
    alloc = nullptr;
    ctx = nullptr;
    graph = nullptr;
}

TableStructureModel::~TableStructureModel() { release(); }

void TableStructureModel::release() {
    step_.clear();
    prep_.clear();
    stepY_.clear();
    stepFt_ = stepFp_ = stepH0_ = stepY0_ = nullptr;
    stepProbs_ = stepLoc_ = stepH_ = nullptr;
    prepFt_ = prepFp_ = nullptr;
    if (decWeightBuf_) ggml_backend_buffer_free(decWeightBuf_);
    if (decWeightCtx_) ggml_free(decWeightCtx_);
    decWeightBuf_ = nullptr;
    decWeightCtx_ = nullptr;
    decBackend_ = nullptr;
    dw_ = DecoderTensors{};
    encExec_.free();
    if (encCtx_) ggml_free(encCtx_);
    encCtx_ = nullptr;
    encGraph_ = nullptr;
    encInput_ = encOutput_ = nullptr;
    encFold_.reset();
}

bool TableStructureModel::load(const std::string& path, std::string* error) {
    if (!encoder_.load(path, error)) return false;
    if (encoder_.role != "table") {
        if (error)
            *error = "not a table structure GGUF (ocr.role=" + encoder_.role +
                     ")";
        return false;
    }
    gguf_context* gguf = encoder_.gguf;
    const int tokIdx = gguf_find_key(gguf, "ocr.table.tokens");
    if (tokIdx < 0) {
        if (error) *error = "GGUF is missing ocr.table.tokens";
        return false;
    }
    const int n = static_cast<int>(gguf_get_arr_n(gguf, tokIdx));
    vocab_.clear();
    for (int i = 0; i < n; ++i) {
        const char* s = gguf_get_arr_str(gguf, tokIdx, i);
        vocab_.emplace_back(s ? s : "");
    }
    if (vocab_.size() < 3 || vocab_.front() != "sos" ||
        vocab_.back() != "eos") {
        if (error) *error = "ocr.table.tokens must be sos ... eos";
        return false;
    }
    maxSteps_ = static_cast<int>(readU32(gguf, "ocr.table.max_steps", 501));
    locReg_ = static_cast<int>(readU32(gguf, "ocr.table.loc_reg_num", 8));
    maxLen_ = static_cast<int>(readU32(gguf, "ocr.table.max_len", 488));
    padSize_ = static_cast<int>(readU32(gguf, "ocr.table.pad_size", 488));
    if (maxSteps_ <= 0 || locReg_ != 8 || maxLen_ <= 0 || padSize_ < maxLen_) {
        if (error) *error = "invalid ocr.table.* decode constants";
        return false;
    }
    return loadDecoderWeights(error);
}

bool TableStructureModel::loadDecoderWeights(std::string* error) {
    // Paddle numpy dims of every 2-D weight come from the converter's
    // registry; biases are 1-D. Each host tensor is copied once into the
    // ggml mul_mat orientation (A = ne{K, M}) so the step graph needs no
    // transposes.
    const auto dims = [&](const char* name) -> std::vector<int64_t> {
        auto it = encoder_.reshapedDims.find(name);
        return it == encoder_.reshapedDims.end() ? std::vector<int64_t>{}
                                                 : it->second;
    };
    std::string bad;
    const auto data = [&](const char* name, int64_t count) -> const float* {
        const ggml_tensor* t = encoder_.hostWeight(std::string("ocr.") + name);
        if (!t || t->type != GGML_TYPE_F32 || ggml_nelements(t) != count ||
            !t->data) {
            if (bad.empty()) bad = name;
            return nullptr;
        }
        return static_cast<const float*>(t->data);
    };
    const std::vector<int64_t> w0 = dims("linear_0.w_0");
    const std::vector<int64_t> wih = dims("gru_cell_0.w_0");
    if (w0.size() != 2 || wih.size() != 2) {
        if (error) *error = "table decoder weight shapes missing";
        return false;
    }
    featC_ = static_cast<int>(w0[0]);
    hidden_ = static_cast<int>(w0[1]);
    const int64_t C = featC_, H = hidden_;
    const int64_t V = static_cast<int64_t>(vocab_.size());
    const int64_t L = locReg_;
    if (wih[0] != 3 * H || wih[1] != C + V) {
        if (error) *error = "table GRU input width does not match C + vocab";
        return false;
    }
    hostDecoder_.clear();
    bool ok = true;
    // Paddle linear weight W is (in, out): A = W^T, i.e. ne{in, out}.
    const auto transposed = [&](const char* name, int64_t in, int64_t out) {
        HostTensor t;
        t.ne0 = in;
        t.ne1 = out;
        const float* src = data(name, in * out);
        if (!src) {
            ok = false;
            return t;
        }
        t.data.resize(static_cast<size_t>(in * out));
        for (int64_t i = 0; i < in; ++i)
            for (int64_t o = 0; o < out; ++o)
                t.data[static_cast<size_t>(o * in + i)] = src[i * out + o];
        return t;
    };
    const auto flat = [&](const char* name, int64_t count) {
        HostTensor t;
        t.ne0 = count;
        const float* src = data(name, count);
        if (!src) {
            ok = false;
            return t;
        }
        t.data.assign(src, src + count);
        return t;
    };
    hostDecoder_.emplace_back("w0t", transposed("linear_0.w_0", C, H));
    hostDecoder_.emplace_back("w1t", transposed("linear_1.w_0", H, H));
    hostDecoder_.emplace_back("b1", flat("linear_1.b_0", H));
    hostDecoder_.emplace_back("w2t", transposed("linear_2.w_0", H, 1));
    {
        // GRUCell weight_ih (3H, C + V): rows are gates, so the context
        // block is ne{C, 3H} and the token block is gathered by row as
        // ne{3H, V}.
        const float* src = data("gru_cell_0.w_0", 3 * H * (C + V));
        HostTensor ctxBlock, tokBlock;
        ctxBlock.ne0 = C;
        ctxBlock.ne1 = 3 * H;
        tokBlock.ne0 = 3 * H;
        tokBlock.ne1 = V;
        if (!src) {
            ok = false;
        } else {
            ctxBlock.data.resize(static_cast<size_t>(C * 3 * H));
            tokBlock.data.resize(static_cast<size_t>(3 * H * V));
            for (int64_t g = 0; g < 3 * H; ++g) {
                const float* row = src + g * (C + V);
                std::copy(
                        row, row + C,
                        ctxBlock.data.begin() + static_cast<ptrdiff_t>(g * C));
                for (int64_t v = 0; v < V; ++v)
                    tokBlock.data[static_cast<size_t>(v * 3 * H + g)] =
                            row[C + v];
            }
        }
        hostDecoder_.emplace_back("wihC", std::move(ctxBlock));
        hostDecoder_.emplace_back("wihY", std::move(tokBlock));
    }
    hostDecoder_.emplace_back("bih", flat("gru_cell_0.b_0", 3 * H));
    {
        // weight_hh (3H, H) is already ne{H, 3H}.
        HostTensor t = flat("gru_cell_0.w_1", 3 * H * H);
        t.ne0 = H;
        t.ne1 = 3 * H;
        hostDecoder_.emplace_back("whh", std::move(t));
    }
    hostDecoder_.emplace_back("bhh", flat("gru_cell_0.b_1", 3 * H));
    hostDecoder_.emplace_back("w3t", transposed("linear_3.w_0", H, H));
    hostDecoder_.emplace_back("b3", flat("linear_3.b_0", H));
    hostDecoder_.emplace_back("w4t", transposed("linear_4.w_0", H, V));
    hostDecoder_.emplace_back("b4", flat("linear_4.b_0", V));
    hostDecoder_.emplace_back("w5t", transposed("linear_5.w_0", H, H));
    hostDecoder_.emplace_back("b5", flat("linear_5.b_0", H));
    hostDecoder_.emplace_back("w6t", transposed("linear_6.w_0", H, L));
    hostDecoder_.emplace_back("b6", flat("linear_6.b_0", L));
    if (!ok && error)
        *error = "table decoder weight missing or mis-shaped: " + bad;
    // The encoder program never reads the decoder parameters; removing them
    // from its name map keeps realizeToDevice from mirroring them a second
    // time (the decoder uploads its own transposed copies).
    for (const char* name :
         {"linear_0.w_0", "linear_1.w_0", "linear_1.b_0", "linear_2.w_0",
          "gru_cell_0.w_0", "gru_cell_0.w_1", "gru_cell_0.b_0",
          "gru_cell_0.b_1", "linear_3.w_0", "linear_3.b_0", "linear_4.w_0",
          "linear_4.b_0", "linear_5.w_0", "linear_5.b_0", "linear_6.w_0",
          "linear_6.b_0"})
        encoder_.tensors.erase(std::string("ocr.") + name);
    return ok;
}

bool TableStructureModel::ensureDecoder(ggml_backend_t backend,
                                        std::string& error) {
    if (decBackend_ == backend && decWeightBuf_) return true;
    step_.clear();
    prep_.clear();
    stepY_.clear();
    if (decWeightBuf_) ggml_backend_buffer_free(decWeightBuf_);
    if (decWeightCtx_) ggml_free(decWeightCtx_);
    decWeightBuf_ = nullptr;
    decWeightCtx_ =
            ggml_init({ggml_tensor_overhead() * (hostDecoder_.size() + 4),
                       nullptr, /*no_alloc=*/true});
    if (!decWeightCtx_) {
        error = "table: decoder weight context init failed";
        return false;
    }
    std::vector<ggml_tensor*> created;
    for (const auto& entry : hostDecoder_) {
        ggml_tensor* t = ggml_new_tensor_2d(decWeightCtx_, GGML_TYPE_F32,
                                            entry.second.ne0, entry.second.ne1);
        ggml_set_name(t, entry.first.c_str());
        created.push_back(t);
    }
    decWeightBuf_ = ggml_backend_alloc_ctx_tensors(decWeightCtx_, backend);
    if (!decWeightBuf_) {
        error = "table: decoder weight allocation failed";
        return false;
    }
    ggml_backend_buffer_set_usage(decWeightBuf_,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (size_t i = 0; i < created.size(); ++i)
        ggml_backend_tensor_set(created[i], hostDecoder_[i].second.data.data(),
                                0, ggml_nbytes(created[i]));
    const auto get = [&](const char* name) {
        return ggml_get_tensor(decWeightCtx_, name);
    };
    dw_.w0t = get("w0t");
    dw_.w1t = get("w1t");
    dw_.b1 = get("b1");
    dw_.w2t = get("w2t");
    dw_.wihC = get("wihC");
    dw_.wihY = get("wihY");
    dw_.bih = get("bih");
    dw_.whh = get("whh");
    dw_.bhh = get("bhh");
    dw_.w3t = get("w3t");
    dw_.b3 = get("b3");
    dw_.w4t = get("w4t");
    dw_.b4 = get("b4");
    dw_.w5t = get("w5t");
    dw_.b5 = get("b5");
    dw_.w6t = get("w6t");
    dw_.b6 = get("b6");
    decBackend_ = backend;
    return true;
}

bool TableStructureModel::runEncoder(ggml_backend_t gpu,
                                     ggml_backend_t cpu,
                                     const float* nchw,
                                     std::string& error) {
    if (!encGraph_) {
        encPlan_ = paddle::makeExecPlan(gpu, cpu, encoder_.sched_gpu_tagging);
        encoder_.resetFoldPool();
        const size_t tensorSize = ggml_tensor_overhead() * 500000;
        encCtx_ = ggml_init({tensorSize * 2, nullptr, /*no_alloc=*/true});
        if (!encCtx_) {
            error = "table: encoder ggml_init failed";
            return false;
        }
        encInput_ = ggml_new_tensor_4d(encCtx_, GGML_TYPE_F32, padSize_,
                                       padSize_, 3, 1);
        ggml_set_input(encInput_);
        const paddle::OcrBuildOptions options = paddle::execBuildOptions(encPlan_);
        encOutput_ = paddle::buildGraph(encCtx_, encoder_, encInput_, &error,
                                     nullptr, nullptr, &options);
        if (!encOutput_) {
            if (error.empty()) error = "table: encoder graph build failed";
            release();
            return false;
        }
        if (!ggml_is_contiguous(encOutput_))
            encOutput_ = ggml_cont(encCtx_, encOutput_);
        ggml_set_output(encOutput_);
        if (encOutput_->ne[2] != featC_ || encOutput_->ne[3] != 1) {
            error = "table: encoder output channels do not match the decoder";
            release();
            return false;
        }
        if (!paddle::prepareGraph(encoder_, encCtx_, encPlan_, {encOutput_},
                               &encGraph_, encExec_, error)) {
            release();
            return false;
        }
        encFold_ = encoder_.foldPool();
        tokens_ = encOutput_->ne[0] * encOutput_->ne[1];
    }
    ggml_backend_tensor_set(encInput_, nchw, 0, ggml_nbytes(encInput_));
    paddle::computeGraph(encPlan_, encExec_, encGraph_);
    // NCHW (1, C, H', W') is ne{W', H', C}: flattening (W', H') gives
    // Ft = ne{T, C} with t = h * W' + w, the order of Paddle's
    // reshape(-1, C, -1) + transpose(0, 2, 1).
    feature_.resize(static_cast<size_t>(tokens_ * featC_));
    ggml_backend_tensor_get(encOutput_, feature_.data(), 0,
                            sizeof(float) * feature_.size());
    return true;
}

bool TableStructureModel::buildPrepGraph(std::string& error) {
    prep_.clear();
    prep_.ctx = ggml_init({ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
                           nullptr, /*no_alloc=*/true});
    if (!prep_.ctx) {
        error = "table: prep graph init failed";
        return false;
    }
    prepFt_ = ggml_new_tensor_2d(prep_.ctx, GGML_TYPE_F32, tokens_, featC_);
    ggml_set_input(prepFt_);
    ggml_tensor* f = ggml_cont(prep_.ctx, ggml_transpose(prep_.ctx, prepFt_));
    prepFp_ = matmulF32(prep_.ctx, dw_.w0t, f);  // {H, T}
    ggml_set_output(prepFp_);
    prep_.graph = ggml_new_graph(prep_.ctx);
    ggml_build_forward_expand(prep_.graph, prepFp_);
    prep_.alloc =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(decBackend_));
    if (!prep_.alloc || !ggml_gallocr_alloc_graph(prep_.alloc, prep_.graph)) {
        error = "table: prep graph allocation failed";
        prep_.clear();
        return false;
    }
    return true;
}

bool TableStructureModel::buildStepGraph(std::string& error) {
    step_.clear();
    stepY_.clear();
    const int64_t H = hidden_;
    const size_t nodeBudget = static_cast<size_t>(kStepsPerGraph) * 64 + 64;
    step_.ctx =
            ggml_init({ggml_tensor_overhead() * nodeBudget +
                               ggml_graph_overhead_custom(nodeBudget, false),
                       nullptr, /*no_alloc=*/true});
    if (!step_.ctx) {
        error = "table: step graph init failed";
        return false;
    }
    ggml_context* ctx = step_.ctx;
    stepFt_ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tokens_, featC_);
    stepFp_ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, tokens_);
    stepH0_ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    stepY0_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    for (ggml_tensor* t : {stepFt_, stepFp_, stepH0_, stepY0_})
        ggml_set_input(t);
    // The per-image features are uploaded once and read by every chunk;
    // the output flag keeps the allocator from recycling their storage for
    // intermediate nodes after their last in-graph use.
    ggml_set_output(stepFt_);
    ggml_set_output(stepFp_);

    ggml_tensor* h = stepH0_;
    ggml_tensor* y = stepY0_;
    ggml_tensor* probsAll = nullptr;
    ggml_tensor* locAll = nullptr;
    const size_t gate = sizeof(float) * static_cast<size_t>(H);
    for (int k = 0; k < kStepsPerGraph; ++k) {
        // Attention over the T encoder positions.
        ggml_tensor* hp = ggml_add(ctx, matmulF32(ctx, dw_.w1t, h), dw_.b1);
        ggml_tensor* e = ggml_tanh(ctx, ggml_add(ctx, stepFp_, hp));  // {H,T}
        ggml_tensor* s = matmulF32(ctx, dw_.w2t, e);                  // {1,T}
        ggml_tensor* a =
                ggml_soft_max(ctx, ggml_reshape_2d(ctx, s, tokens_, 1));
        ggml_tensor* c = matmulF32(ctx, stepFt_, a);  // {C,1}
        // Paddle GRUCell: gates split as (r, z, c) in both projections.
        ggml_tensor* xg = ggml_add(ctx,
                                   ggml_add(ctx, matmulF32(ctx, dw_.wihC, c),
                                            ggml_get_rows(ctx, dw_.wihY, y)),
                                   dw_.bih);
        ggml_tensor* hg = ggml_add(ctx, matmulF32(ctx, dw_.whh, h), dw_.bhh);
        ggml_tensor* xr = ggml_view_1d(ctx, xg, H, 0);
        ggml_tensor* xz = ggml_view_1d(ctx, xg, H, gate);
        ggml_tensor* xc = ggml_view_1d(ctx, xg, H, 2 * gate);
        ggml_tensor* hr = ggml_view_1d(ctx, hg, H, 0);
        ggml_tensor* hz = ggml_view_1d(ctx, hg, H, gate);
        ggml_tensor* hc = ggml_view_1d(ctx, hg, H, 2 * gate);
        ggml_tensor* r = ggml_sigmoid(ctx, ggml_add(ctx, xr, hr));
        ggml_tensor* z = ggml_sigmoid(ctx, ggml_add(ctx, xz, hz));
        ggml_tensor* cand =
                ggml_tanh(ctx, ggml_add(ctx, xc, ggml_mul(ctx, r, hc)));
        h = ggml_add(ctx, ggml_mul(ctx, ggml_sub(ctx, h, cand), z), cand);
        // Structure and cell-location heads (two linears each, no
        // activation in between, exactly as exported).
        ggml_tensor* logits = ggml_add(
                ctx,
                matmulF32(ctx, dw_.w4t,
                          ggml_add(ctx, matmulF32(ctx, dw_.w3t, h), dw_.b3)),
                dw_.b4);
        ggml_tensor* loc = ggml_sigmoid(
                ctx,
                ggml_add(ctx,
                         matmulF32(ctx, dw_.w6t,
                                   ggml_add(ctx, matmulF32(ctx, dw_.w5t, h),
                                            dw_.b5)),
                         dw_.b6));
        y = ggml_argmax(ctx, logits);  // I32 {1}: next step's token
        ggml_set_output(y);
        stepY_.push_back(y);
        ggml_tensor* prob = ggml_soft_max(ctx, logits);
        probsAll = probsAll ? ggml_concat(ctx, probsAll, prob, 1) : prob;
        locAll = locAll ? ggml_concat(ctx, locAll, loc, 1) : loc;
    }
    stepProbs_ = probsAll;
    stepLoc_ = locAll;
    stepH_ = h;
    for (ggml_tensor* t : {stepProbs_, stepLoc_, stepH_}) ggml_set_output(t);
    step_.graph = ggml_new_graph_custom(ctx, nodeBudget, false);
    ggml_build_forward_expand(step_.graph, stepProbs_);
    ggml_build_forward_expand(step_.graph, stepLoc_);
    ggml_build_forward_expand(step_.graph, stepH_);
    for (ggml_tensor* t : stepY_) ggml_build_forward_expand(step_.graph, t);
    step_.alloc =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(decBackend_));
    if (!step_.alloc || !ggml_gallocr_alloc_graph(step_.alloc, step_.graph)) {
        error = "table: step graph allocation failed";
        step_.clear();
        return false;
    }
    return true;
}

bool TableStructureModel::infer(ggml_backend_t gpu,
                                ggml_backend_t cpu,
                                const float* nchw,
                                TableStructureOutput& out,
                                std::string& error,
                                bool keepProbs) {
    out = TableStructureOutput{};
    if (!cpu || !nchw) {
        error = "table: missing backend or input";
        return false;
    }
    const auto encStart = std::chrono::steady_clock::now();
    if (!runEncoder(gpu, cpu, nchw, error)) return false;
    out.encoderMs = msSince(encStart);

    const auto decStart = std::chrono::steady_clock::now();
    ggml_backend_t backend = encPlan_.gpu ? encPlan_.gpu : cpu;
    if (!ensureDecoder(backend, error)) return false;
    if (!prep_.graph && !buildPrepGraph(error)) return false;
    if (!step_.graph && !buildStepGraph(error)) return false;

    const int64_t H = hidden_;
    const int64_t V = static_cast<int64_t>(vocab_.size());
    std::vector<float> fp(static_cast<size_t>(H * tokens_));
    ggml_backend_tensor_set(prepFt_, feature_.data(), 0,
                            sizeof(float) * feature_.size());
    ggml_backend_graph_compute(decBackend_, prep_.graph);
    ggml_backend_tensor_get(prepFp_, fp.data(), 0, sizeof(float) * fp.size());
    ggml_backend_tensor_set(stepFt_, feature_.data(), 0,
                            sizeof(float) * feature_.size());
    ggml_backend_tensor_set(stepFp_, fp.data(), 0, sizeof(float) * fp.size());

    std::vector<float> hState(static_cast<size_t>(H), 0.0f);
    int32_t yState = 0;  // sos
    std::vector<float> probs(static_cast<size_t>(V * kStepsPerGraph));
    std::vector<float> loc(static_cast<size_t>(locReg_ * kStepsPerGraph));
    const int eos = eosIndex();
    bool done = false;
    int steps = 0;
    while (!done && steps < maxSteps_) {
        ggml_backend_tensor_set(stepH0_, hState.data(), 0,
                                sizeof(float) * hState.size());
        ggml_backend_tensor_set(stepY0_, &yState, 0, sizeof(int32_t));
        ggml_backend_graph_compute(decBackend_, step_.graph);
        ggml_backend_tensor_get(stepProbs_, probs.data(), 0,
                                sizeof(float) * probs.size());
        ggml_backend_tensor_get(stepLoc_, loc.data(), 0,
                                sizeof(float) * loc.size());
        ggml_backend_tensor_get(stepH_, hState.data(), 0,
                                sizeof(float) * hState.size());
        for (int k = 0; k < kStepsPerGraph && steps < maxSteps_; ++k) {
            int32_t token = 0;
            ggml_backend_tensor_get(stepY_[static_cast<size_t>(k)], &token, 0,
                                    sizeof(int32_t));
            const float* row = probs.data() + static_cast<size_t>(k) * V;
            out.tokens.push_back(token);
            out.scores.push_back(*std::max_element(row, row + V));
            out.loc.insert(out.loc.end(), loc.begin() + k * locReg_,
                           loc.begin() + (k + 1) * locReg_);
            if (keepProbs) out.probs.insert(out.probs.end(), row, row + V);
            ++steps;
            yState = token;
            if (token == eos) {
                done = true;
                break;
            }
        }
    }
    out.decoderMs = msSince(decStart);
    return true;
}

}  // namespace table
}  // namespace aicore

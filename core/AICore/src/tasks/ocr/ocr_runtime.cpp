// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ocr_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>

#if (defined(__GNUC__) || defined(__clang__)) && \
        (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define AICORE_OCR_HAS_AVX2_DISPATCH 1
#else
#define AICORE_OCR_HAS_AVX2_DISPATCH 0
#endif
#include <sstream>
#include <unordered_set>

#include "common/blocked_conv.hpp"
#include "common/winograd.hpp"

namespace aicore {
namespace ocr {

namespace {

// CTC argmax is O(T*V), with V=6625 for PP-OCR Chinese. Keep the decode
// state machine scalar and only vectorize the independent max reduction.
// The vector path has exactly the scalar `>` semantics: NaNs never replace
// the current winner and equal values retain their earliest class index.
inline int ctcArgmaxScalar(const float* row, int vocab, float* bestProb) {
    int best = 0;
    float bestValue = row[0];
    for (int i = 1; i < vocab; ++i) {
        if (row[i] > bestValue) {
            bestValue = row[i];
            best = i;
        }
    }
    *bestProb = bestValue;
    return best;
}

#if AICORE_OCR_HAS_AVX2_DISPATCH
__attribute__((target("avx2"))) int ctcArgmaxAvx2(const float* row,
                                                  int vocab,
                                                  float* bestProb) {
    // Scalar starts from row[0], therefore a leading NaN is itself the
    // winner (all later `value > NaN` comparisons are false). Preserve that
    // unusual but defined behavior before entering the vector path.
    if (std::isnan(row[0])) return ctcArgmaxScalar(row, vocab, bestProb);

    const __m256 negInf =
            _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    __m256 laneMax = negInf;
    int i = 0;
    for (; i + 8 <= vocab; i += 8) {
        const __m256 values = _mm256_loadu_ps(row + i);
        // Ordered compare is false for NaNs, matching scalar `>`.
        const __m256 replace = _mm256_cmp_ps(values, laneMax, _CMP_GT_OQ);
        laneMax = _mm256_blendv_ps(laneMax, values, replace);
    }
    alignas(32) float lane[8];
    _mm256_store_ps(lane, laneMax);
    float bestValue = lane[0];
    for (int laneIndex = 1; laneIndex < 8; ++laneIndex)
        if (lane[laneIndex] > bestValue) bestValue = lane[laneIndex];
    for (; i < vocab; ++i)
        if (row[i] > bestValue) bestValue = row[i];

    // A second vector pass finds the earliest equal class, preserving the
    // scalar tie rule without carrying an index through the reduction.
    const __m256 target = _mm256_set1_ps(bestValue);
    i = 0;
    for (; i + 8 <= vocab; i += 8) {
        const int mask = _mm256_movemask_ps(
                _mm256_cmp_ps(_mm256_loadu_ps(row + i), target, _CMP_EQ_OQ));
        if (mask != 0) {
            *bestProb = bestValue;
            return i + __builtin_ctz(static_cast<unsigned>(mask));
        }
    }
    for (; i < vocab; ++i) {
        if (row[i] == bestValue) {
            *bestProb = bestValue;
            return i;
        }
    }
    // Only reachable for an all-NaN row, already handled above; retain a
    // scalar fallback rather than manufacture an index.
    return ctcArgmaxScalar(row, vocab, bestProb);
}

bool ctcAvx2Available() {
    static const bool available = __builtin_cpu_supports("avx2");
    return available;
}
#endif

inline int ctcArgmax(const float* row, int vocab, float* bestProb) {
#if AICORE_OCR_HAS_AVX2_DISPATCH
    if (ctcAvx2Available()) return ctcArgmaxAvx2(row, vocab, bestProb);
#endif
    return ctcArgmaxScalar(row, vocab, bestProb);
}

int64_t attrInt(const Json::Value& attrs,
                const char* name,
                int64_t fallback = 0) {
    if (!attrs.isObject()) return fallback;
    const Json::Value& v = attrs[name];
    return v.isNumeric() ? v.asInt64() : fallback;
}

std::vector<int64_t> attrInts(const Json::Value& attrs, const char* name) {
    std::vector<int64_t> out;
    if (!attrs.isObject()) return out;
    for (const Json::Value& v : attrs[name]) out.push_back(v.asInt64());
    return out;
}

double attrFloat(const Json::Value& attrs,
                 const char* name,
                 double fallback = 0.0) {
    if (!attrs.isObject()) return fallback;
    const Json::Value& v = attrs[name];
    return v.isNumeric() ? v.asDouble() : fallback;
}

std::string attrString(const Json::Value& attrs, const char* name) {
    if (!attrs.isObject()) return {};
    const Json::Value& v = attrs[name];
    return v.isString() ? v.asString() : std::string();
}

bool attrBool(const Json::Value& attrs, const char* name, bool fallback) {
    if (!attrs.isObject()) return fallback;
    const Json::Value& v = attrs[name];
    return v.isBool() ? v.asBool() : (v.isNumeric() ? v.asBool() : fallback);
}

// A numpy transpose perm applied to a rank-n tensor maps to a ggml permute.
// ggml_permute scatters (in axis i -> out axis axis[i]) while numpy gathers
// (out axis i <- in axis p[i]), so the ggml argument is the INVERSE mapping:
// axis[n-1-p[i]] = n-1-i. Unused high axes are size-1 dummies that map to
// themselves (never duplicate a real axis, or ggml_permute aborts).
void numpyPermToGgml(const std::vector<int64_t>& perm, int k[4]) {
    const int n = static_cast<int>(perm.size());
    for (int g = 0; g < 4; ++g) k[g] = g;
    for (int i = 0; i < n && i < 4; ++i)
        k[n - 1 - static_cast<int>(perm[i])] = n - 1 - i;
}

ggml_tensor* scalarF32(ggml_context* ctx, float value) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    static_cast<float*>(t->data)[0] = value;
    return t;
}

// Binary kernels (add/mul) require element-contiguous operands; a
// transposed mul_mat result or permuted view feeding add/mul must be
// materialized first.
ggml_tensor* elemCont(ggml_context* ctx, ggml_tensor* t) {
    if (t->type == GGML_TYPE_F32 && t->nb[0] != sizeof(float))
        return ggml_cont(ctx, t);
    return t;
}

ggml_tensor* addChk(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    return ggml_add(ctx, elemCont(ctx, a), elemCont(ctx, b));
}

static ggml_tensor* mulChk(ggml_context* ctx,
                           ggml_tensor* a,
                           ggml_tensor* b,
                           const char* tag) {
    a = elemCont(ctx, a);
    b = elemCont(ctx, b);
    if (!ggml_can_repeat(b, a)) {
        std::fprintf(stderr,
                     "[mul-mismatch] %s a ne=%lld,%lld,%lld,%lld | "
                     "b ne=%lld,%lld,%lld,%lld\n",
                     tag, (long long)a->ne[0], (long long)a->ne[1],
                     (long long)a->ne[2], (long long)a->ne[3],
                     (long long)b->ne[0], (long long)b->ne[1],
                     (long long)b->ne[2], (long long)b->ne[3]);
        std::fflush(stderr);
    }
    return ggml_mul(ctx, a, b);
}
ggml_tensor* constantTensor(ggml_context* ctx,
                            const std::vector<int64_t>& values) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, values.size());
    auto* data = static_cast<int32_t*>(t->data);
    for (size_t i = 0; i < values.size(); ++i)
        data[i] = static_cast<int32_t>(values[i]);
    return t;
}

// Per-parameter input lookup ("X", "Filter", "Mean", ...) with a positional
// fallback for ops whose inputs carry no parameter names.
struct BuildState {
    ggml_context* ctx;
    const OcrWeights& weights;
    std::map<std::string, ggml_tensor*> values;
    std::map<std::string, std::vector<int64_t>> constants;

    ggml_tensor* resolve(const std::string& name) const {
        auto it = values.find(name);
        if (it != values.end()) return it->second;
        return const_cast<ggml_tensor*>(weights.weight("ocr." + name));
    }
    ggml_tensor* byParam(const GraphOp& op, const char* param) const {
        for (const auto& io : op.inputs)
            if (io.param == param && !io.args.empty())
                return resolve(io.args[0]);
        return nullptr;
    }
    ggml_tensor* firstInput(const GraphOp& op) const {
        for (const auto& io : op.inputs)
            if (!io.args.empty())
                if (ggml_tensor* t = resolve(io.args[0])) return t;
        return nullptr;
    }
    ggml_tensor* broadcastChannel(ggml_tensor* vector) const {
        if (ggml_n_dims(vector) == 1)
            return ggml_reshape_4d(ctx, vector, 1, 1, vector->ne[0], 1);
        return vector;
    }
};

std::vector<float> tensorHostFloats(const ggml_tensor* t) {
    std::vector<float> out(static_cast<size_t>(t->ne[0]));
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), t->data, sizeof(float) * t->ne[0]);
    } else if (t->type == GGML_TYPE_F16) {
        const auto* src = reinterpret_cast<const ggml_fp16_t*>(t->data);
        for (int64_t i = 0; i < t->ne[0]; ++i)
            out[static_cast<size_t>(i)] = ggml_fp16_to_fp32(src[i]);
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// OcrProgram

bool OcrProgram::parse(const std::string& json, std::string* error) {
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errs;
    std::istringstream stream(json);
    if (!Json::parseFromStream(builder, stream, &root, &errs)) {
        if (error) *error = "graph JSON parse failed: " + errs;
        return false;
    }
    ops.clear();
    varRanks.clear();
    staticShapes.clear();
    // Newer exports wrap the op array together with the static logical
    // ranks (from the VarDesc shapes); older exports are a bare array.
    const Json::Value* opArray = &root;
    if (root.isObject()) {
        if (root.isMember("ranks") && root["ranks"].isObject()) {
            for (auto it = root["ranks"].begin(); it != root["ranks"].end();
                 ++it)
                varRanks[it.key().asString()] = it->asInt();
        }
        if (root.isMember("shapes") && root["shapes"].isObject()) {
            for (auto it = root["shapes"].begin(); it != root["shapes"].end();
                 ++it) {
                std::vector<int64_t> dims;
                for (const Json::Value& d : *it) dims.push_back(d.asInt64());
                staticShapes[it.key().asString()] = std::move(dims);
            }
        }
        opArray = &root["ops"];
    }
    for (const Json::Value& op : *opArray) {
        GraphOp entry;
        entry.type = op["type"].asString();
        for (const Json::Value& io : op["inputs"]) {
            GraphOp::OpInput input;
            input.param = io["param"].asString();
            for (const Json::Value& arg : io["args"])
                input.args.push_back(arg.asString());
            entry.inputs.push_back(std::move(input));
        }
        for (const Json::Value& io : op["outputs"])
            for (const Json::Value& arg : io["args"])
                entry.outputs.push_back(arg.asString());
        // The exported attrs are a list of {name, value} entries; flatten
        // them into an object so the builders can index by name.
        Json::Value attrMap(Json::objectValue);
        for (const Json::Value& a : op["attrs"]) {
            if (a["name"].isString())
                attrMap[a["name"].asString()] = a.get("value", Json::Value());
        }
        entry.attrs = std::move(attrMap);
        ops.push_back(std::move(entry));
    }
    return true;
}

// ---------------------------------------------------------------------------
// OcrWeights

OcrWeights::~OcrWeights() {
    if (deviceBuf) ggml_backend_buffer_free(deviceBuf);
    if (deviceCtx) ggml_free(deviceCtx);
    if (viewMirrorBuf) ggml_backend_buffer_free(viewMirrorBuf);
    if (viewMirrorCtx) ggml_free(viewMirrorCtx);
    if (foldDeviceBuf) ggml_backend_buffer_free(foldDeviceBuf);
    if (foldMirrorCtx) ggml_free(foldMirrorCtx);
    if (foldBuf) ggml_backend_buffer_free(foldBuf);
    if (foldCtx) ggml_free(foldCtx);
    if (hostConstBuf) ggml_backend_buffer_free(hostConstBuf);
    if (hostMetaBuf) ggml_backend_buffer_free(hostMetaBuf);
    if (constCtx) ggml_free(constCtx);
    if (meta) ggml_free(meta);
    if (gguf) gguf_free(gguf);
}

void OcrWeights::resetFoldPool() {
    // Drop the previous build's fold constants (its graph is gone by the
    // time the next frame builds) and start a fresh pool; otherwise the
    // load-time pool fills after a few builds and every later tensor
    // comes back with a null data pointer.
    if (viewMirrorBuf) {
        ggml_backend_buffer_free(viewMirrorBuf);
        viewMirrorBuf = nullptr;
    }
    if (viewMirrorCtx) {
        ggml_free(viewMirrorCtx);
        viewMirrorCtx = nullptr;
    }
    if (foldDeviceBuf) {
        ggml_backend_buffer_free(foldDeviceBuf);
        foldDeviceBuf = nullptr;
    }
    if (foldMirrorCtx) {
        ggml_free(foldMirrorCtx);
        foldMirrorCtx = nullptr;
    }
    if (foldBuf) {
        ggml_backend_buffer_free(foldBuf);
        foldBuf = nullptr;
    }
    if (foldCtx) ggml_free(foldCtx);
    foldCtx = ggml_init({96 * 1024 * 1024, nullptr, /*no_alloc=*/false});
}

bool OcrWeights::realizeFoldPoolToDevice(ggml_backend_t backend,
                                         std::string* error) {
    if (!foldCtx || foldDeviceBuf) return true;
    if (!backend) {
        if (error) *error = "null backend";
        return false;
    }
    // Per-tensor ALIGNED mirror. The first cut retargeted every fold tensor
    // into a raw copy of the pool at its original offset, but the ggml pool
    // only guarantees 16-byte alignment while Vulkan storage-buffer offsets
    // must respect minStorageBufferOffsetAlignment — the misaligned weight
    // bindings faulted the VK conv2d dispatch. Each fold tensor instead gets
    // its own aligned slot (alloc_ctx_tensors), its bytes are uploaded
    // individually, and the graph's fold tensors are repointed at the
    // aligned copies. Safe because fold constants are read-only after the
    // build and never reused across builds; reshaped views duplicate their
    // parent bytes (both copies stay valid for read-only use).
    std::vector<ggml_tensor*> foldTensors;
    for (ggml_tensor* t = ggml_get_first_tensor(foldCtx); t != nullptr;
         t = ggml_get_next_tensor(foldCtx, t)) {
        if (!t->data) continue;
        // Canonical strides only: a non-canonical view (permute/transpose)
        // would not round-trip through a canonical-layout mirror.
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] != t->nb[i - 1] * t->ne[i - 1]) {
                if (error) *error = "fold pool holds a non-canonical view";
                return false;
            }
        }
        foldTensors.push_back(t);
    }
    if (foldTensors.empty()) return true;
    foldMirrorCtx =
            ggml_init({ggml_tensor_overhead() * (foldTensors.size() + 8),
                       nullptr, /*no_alloc=*/true});
    if (!foldMirrorCtx) {
        if (error) *error = "fold mirror ctx init failed";
        return false;
    }
    std::vector<ggml_tensor*> mirrors;
    mirrors.reserve(foldTensors.size());
    for (ggml_tensor* t : foldTensors) {
        ggml_tensor* m =
                ggml_new_tensor(foldMirrorCtx, t->type, GGML_MAX_DIMS, t->ne);
        mirrors.push_back(m);
    }
    foldDeviceBuf = ggml_backend_alloc_ctx_tensors(foldMirrorCtx, backend);
    if (!foldDeviceBuf) {
        if (error) *error = "fold device pool alloc failed";
        return false;
    }
    if (sched_gpu_tagging) {
        ggml_backend_buffer_set_usage(foldDeviceBuf,
                                      GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    for (size_t i = 0; i < foldTensors.size(); ++i) {
        ggml_backend_tensor_set(mirrors[i], foldTensors[i]->data, 0,
                                ggml_nbytes(foldTensors[i]));
        // A folded tensor can itself be a canonical view. Keeping its
        // view_src would make ggml-vulkan derive the descriptor offset from
        // the parent allocation, discarding this dedicated aligned slot.
        // Its contiguous bytes were copied above, so detach the view and
        // bind the tensor directly to its aligned device mirror.
        foldTensors[i]->view_src = nullptr;
        foldTensors[i]->view_offs = 0;
        foldTensors[i]->data = mirrors[i]->data;
        foldTensors[i]->buffer = foldDeviceBuf;
    }
    return true;
}
bool OcrWeights::materializeUnalignedStaticViews(struct ggml_cgraph* graph,
                                                 ggml_backend_t backend,
                                                 std::string* error) {
    if (!graph || !backend || viewMirrorBuf) return graph && backend;
    const size_t alignment = ggml_backend_buft_get_alignment(
            ggml_backend_get_default_buffer_type(backend));
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return true;

    struct PendingCopy {
        ggml_tensor* source;
        ggml_tensor* mirror;
        std::vector<uint8_t> bytes;
    };
    std::vector<PendingCopy> pending;
    std::unordered_set<ggml_tensor*> seen;
    std::vector<ggml_tensor*> all;
    std::vector<ggml_tensor*> stack;
    const auto push = [&](ggml_tensor* tensor) {
        if (tensor && seen.insert(tensor).second) stack.push_back(tensor);
    };
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        push(ggml_graph_node(graph, i));
    while (!stack.empty()) {
        ggml_tensor* tensor = stack.back();
        stack.pop_back();
        all.push_back(tensor);
        push(tensor->view_src);
        for (int i = 0; i < GGML_MAX_SRC; ++i) push(tensor->src[i]);
    }

    // A view's descriptor offset is its root slot plus every view_offs in its
    // chain. Only device-resident model/fold roots are static; gallocr-owned
    // activation views must retain their allocator-managed storage.
    seen.clear();
    for (ggml_tensor* source : all) {
        if (!source->view_src || !seen.insert(source).second) continue;
        ggml_tensor* root = source;
        size_t sourceOffset = 0;
        while (root->view_src) {
            sourceOffset += root->view_offs;
            root = root->view_src;
        }
        if ((root->buffer != deviceBuf && root->buffer != foldDeviceBuf) ||
            !root->data || (sourceOffset & (alignment - 1)) == 0)
            continue;
        // The mirror has canonical strides. A non-contiguous view cannot be
        // represented by that layout without changing its element mapping.
        if (!ggml_is_contiguous(source)) {
            if (error) *error = "ocr: unaligned non-contiguous static view";
            return false;
        }
        const size_t byteCount = ggml_nbytes(source);
        if (sourceOffset > ggml_nbytes(root) ||
            byteCount > ggml_nbytes(root) - sourceOffset) {
            if (error) *error = "ocr: static view exceeds its root tensor";
            return false;
        }
        PendingCopy copy{source, nullptr, std::vector<uint8_t>(byteCount)};
        ggml_backend_tensor_get(root, copy.bytes.data(), sourceOffset,
                                copy.bytes.size());
        pending.push_back(std::move(copy));
    }
    if (pending.empty()) return true;

    viewMirrorCtx = ggml_init({ggml_tensor_overhead() * (pending.size() + 8),
                               nullptr, /*no_alloc=*/true});
    if (!viewMirrorCtx) {
        if (error) *error = "view mirror ctx init failed";
        return false;
    }
    for (PendingCopy& copy : pending)
        copy.mirror = ggml_new_tensor(viewMirrorCtx, copy.source->type,
                                      GGML_MAX_DIMS, copy.source->ne);
    viewMirrorBuf = ggml_backend_alloc_ctx_tensors(viewMirrorCtx, backend);
    if (!viewMirrorBuf) {
        if (error) *error = "view mirror alloc failed";
        return false;
    }
    for (PendingCopy& copy : pending) {
        ggml_backend_tensor_set(copy.mirror, copy.bytes.data(), 0,
                                copy.bytes.size());
        copy.source->view_src = nullptr;
        copy.source->view_offs = 0;
        copy.source->data = copy.mirror->data;
        copy.source->buffer = viewMirrorBuf;
    }
    return true;
}

void OcrWeights::retargetGraphToDevice(struct ggml_cgraph* graph) {
    if (!graph) return;
    // Collect every tensor reachable from the graph: the computed nodes,
    // their src chains (the weight leafs live in the meta/const pools, NOT
    // in the graph context, so a ctx walk never sees them), and the view
    // bases. Then retarget in two phases: named host-resident weights to
    // their device copies, and every view re-derived from its base.
    std::vector<ggml_tensor*> all;
    std::unordered_set<ggml_tensor*> seen;
    std::vector<ggml_tensor*> stack;
    auto push = [&](ggml_tensor* t) {
        if (t && seen.insert(t).second) stack.push_back(t);
    };
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        push(ggml_graph_node(graph, i));
    while (!stack.empty()) {
        ggml_tensor* t = stack.back();
        stack.pop_back();
        all.push_back(t);
        if (t->view_src) push(t->view_src);
        for (int i = 0; i < GGML_MAX_SRC; ++i) push(t->src[i]);
    }
    // Phase 1: named weight leafs still bound to the host pools (the graph
    // was built against the host originals) move to their device copies.
    for (ggml_tensor* t : all) {
        if (t->view_src) continue;
        if (t->buffer == hostMetaBuf || t->buffer == hostConstBuf ||
            t->buffer == nullptr) {
            auto it = tensors.find(t->name ? t->name : "");
            if (it != tensors.end() && it->second && it->second->data) {
                // The device mirror owns its tensor slot, so the host-pool
                // byte offset must not be added a second time.
                t->data = it->second->data;
                t->buffer = it->second->buffer;
                t->view_offs = 0;
            }
        }
    }
    // Phase 2 (x2 for chained views): re-derive every view's data pointer
    // from its (possibly just-retargeted) base. A graph may have been built
    // while that base belonged to a host pool, so its view can still carry a
    // valid-looking host buffer. Clear every view binding here: gallocr sees
    // the null buffer and recreates the view from the retargeted device base
    // during allocation. Leaving the stale binding intact makes Vulkan bind a
    // host-pool subrange in mul_mat, whose offset need not satisfy
    // minStorageBufferOffsetAlignment.
    for (int pass = 0; pass < 2; ++pass) {
        for (ggml_tensor* t : all) {
            if (!t->view_src) continue;
            t->buffer = nullptr;
            t->data = (char*)t->view_src->data + t->view_offs;
        }
    }
}
bool OcrWeights::realizeHostBuffers() {
    if (hostMetaBuf) return true;
    // Zero-copy CPU buffer views over the two host pools: the GGUF weights
    // (meta, no_alloc=false) and the folded constants (constCtx). The
    // scheduler requires a buffer on every leaf to route cross-backend
    // copies; these views make the host leaves look like ordinary CPU
    // weights without moving any bytes.
    hostMetaBuf = ggml_backend_cpu_buffer_from_ptr(ggml_get_mem_buffer(meta),
                                                   ggml_get_mem_size(meta));
    if (!hostMetaBuf) return false;
    // WEIGHTS usage: the sched offloads weight-consuming ops to the GPU
    // only when the (possibly host-resident) buffer carries this tag.
    if (sched_gpu_tagging) {
        ggml_backend_buffer_set_usage(hostMetaBuf,
                                      GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    if (constCtx) {
        hostConstBuf = ggml_backend_cpu_buffer_from_ptr(
                ggml_get_mem_buffer(constCtx), ggml_get_mem_size(constCtx));
        if (!hostConstBuf) return false;
        if (sched_gpu_tagging) {
            ggml_backend_buffer_set_usage(hostConstBuf,
                                          GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
    }
    return true;
}

void OcrWeights::tagHostLeafs() {
    // GGUF weights live in the meta pool (f32 files) or were materialized
    // into the constCtx pool at load time (f16/q8_0 dequant); folded
    // constants are appended to the per-build fold pool on every build.
    // The pointer-range test picks the owning pool view without tracking
    // it per tensor.
    const auto inPool = [](ggml_context* pool, const void* p) {
        if (!pool) return false;
        const char* base = static_cast<const char*>(ggml_get_mem_buffer(pool));
        return p >= base &&
               p < base + static_cast<ptrdiff_t>(ggml_get_mem_size(pool));
    };
    for (auto& kv : tensors) {
        ggml_tensor* t = const_cast<ggml_tensor*>(kv.second);
        if (t->buffer) continue;
        if (inPool(meta, t->data))
            t->buffer = hostMetaBuf;
        else if (inPool(constCtx, t->data))
            t->buffer = hostConstBuf;
        else if (inPool(foldCtx, t->data)) {
            if (!foldBuf) {
                foldBuf = ggml_backend_cpu_buffer_from_ptr(
                        ggml_get_mem_buffer(foldCtx),
                        ggml_get_mem_size(foldCtx));
                // Fold constants are what the graph consumes: tag WEIGHTS so
                // the sched can offload conv/matmul ops to the GPU.
                if (sched_gpu_tagging) {
                    ggml_backend_buffer_set_usage(
                            foldBuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                }
            }
            t->buffer = foldBuf;
        }
    }
    if (hostConstBuf) {
        for (ggml_tensor* t = ggml_get_first_tensor(constCtx); t != nullptr;
             t = ggml_get_next_tensor(constCtx, t)) {
            if (!t->buffer) t->buffer = hostConstBuf;
        }
    }
    // The per-build fold pool's constants are fresh every build: tag
    // every one of them with the pool view (the tensors map only sees
    // the weight leaves).
    if (foldCtx) {
        if (!foldBuf) {
            foldBuf = ggml_backend_cpu_buffer_from_ptr(
                    ggml_get_mem_buffer(foldCtx), ggml_get_mem_size(foldCtx));
            // Fold constants (BN-fused conv weights) are what the graph
            // actually consumes: tag the pool view WEIGHTS so the sched
            // offloads conv/matmul ops to the GPU (copies are inserted).
            if (sched_gpu_tagging) {
                ggml_backend_buffer_set_usage(
                        foldBuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        for (ggml_tensor* t = ggml_get_first_tensor(foldCtx); t != nullptr;
             t = ggml_get_next_tensor(foldCtx, t)) {
            if (!t->buffer) t->buffer = foldBuf;
        }
    }
}

bool OcrWeights::realizeToDevice(ggml_backend_t backend, std::string* error) {
    if (deviceBuf) return true;
    if (!backend) {
        if (error) *error = "null backend";
        return false;
    }
    const auto fail = [&](const char* what) {
        if (error) *error = what;
        return false;
    };
    // Mirror the name->tensor map into a no_alloc context, allocate that
    // context on the device backend, upload each tensor's bytes, and swap
    // the map to the device copies (the facedetect ModelLoader pattern).
    const size_t n = tensors.size();
    deviceCtx = ggml_init({ggml_tensor_overhead() * (n + 8), nullptr,
                           /*no_alloc=*/true});
    if (!deviceCtx) return fail("device ctx init failed");
    std::vector<std::pair<ggml_tensor*, const void*>> uploads;
    uploads.reserve(n);
    std::map<std::string, const ggml_tensor*> devmap;
    for (const auto& kv : tensors) {
        const ggml_tensor* s = kv.second;
        ggml_tensor* d =
                ggml_new_tensor(deviceCtx, s->type, GGML_MAX_DIMS, s->ne);
        ggml_set_name(d, kv.first.c_str());
        devmap.emplace(kv.first, d);
        uploads.emplace_back(d, s->data);
    }
    deviceBuf = ggml_backend_alloc_ctx_tensors(deviceCtx, backend);
    if (!deviceBuf) return fail("device weights alloc failed");
    // Without the WEIGHTS usage tag the backend sched treats these as
    // ordinary host buffers: pass-1 never routes weight-consuming ops to
    // the GPU (offload branch requires usage == WEIGHTS) and the whole
    // graph silently runs on the CPU secondary (zero CUDA kernels).
    if (sched_gpu_tagging) {
        ggml_backend_buffer_set_usage(deviceBuf,
                                      GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    for (const auto& up : uploads)
        ggml_backend_tensor_set(up.first, up.second, 0, ggml_nbytes(up.first));
    // Keep the host view alive: every later buildGraph must fold the
    // batch-norm statistics on the CPU (the device copies are not
    // dereferenceable from the host).
    hostTensors = tensors;
    tensors.swap(devmap);
    return true;
}

const ggml_tensor* OcrWeights::weight(const std::string& name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : it->second;
}

const ggml_tensor* OcrWeights::hostWeight(const std::string& name) const {
    auto it = hostTensors.find(name);
    if (it != hostTensors.end()) return it->second;
    // Before the device realization the main map is the host view.
    return weight(name);
}

bool OcrWeights::load(const std::string& path, std::string* error) {
    ggml_context* metaCtx = nullptr;
    struct gguf_init_params params = {
            /*.no_alloc =*/false,  // small models: keep weights host-readable
            /*.ctx =*/&metaCtx,
    };
    gguf = gguf_init_from_file(path.c_str(), params);
    if (!gguf) {
        if (error) *error = "failed to open GGUF: " + path;
        return false;
    }
    meta = metaCtx;
    constCtx = ggml_init({96 * 1024 * 1024, nullptr,
                          /*no_alloc=*/false});

    auto readString = [&](const char* key) -> std::string {
        const int idx = gguf_find_key(gguf, key);
        if (idx < 0 || gguf_get_kv_type(gguf, idx) != GGUF_TYPE_STRING)
            return {};
        return gguf_get_val_str(gguf, idx);
    };

    role = readString("ocr.role");
    arch = readString("ocr.arch");
    if (role.empty()) {
        if (error) *error = "GGUF is missing ocr.role";
        return false;
    }

    const int dictIdx = gguf_find_key(gguf, "ocr.dictionary");
    if (dictIdx >= 0) {
        const int32_t n = gguf_get_arr_n(gguf, dictIdx);
        dictionary.reserve(n);
        for (int32_t i = 0; i < n; ++i) {
            const char* s = gguf_get_arr_str(gguf, dictIdx, i);
            dictionary.emplace_back(s ? s : "");
        }
    }

    // Layout class labels (PicoDet CDLA table), same embedding pattern.
    const int labelIdx = gguf_find_key(gguf, "ocr.labels");
    if (labelIdx >= 0) {
        const int32_t n = gguf_get_arr_n(gguf, labelIdx);
        labels.reserve(n);
        for (int32_t i = 0; i < n; ++i) {
            const char* s = gguf_get_arr_str(gguf, labelIdx, i);
            labels.emplace_back(s ? s : "");
        }
    }

    const std::string opsJson = readString("ocr.graph.ops");
    if (opsJson.empty()) {
        if (error) *error = "GGUF is missing ocr.graph.ops";
        return false;
    }
    if (!program.parse(opsJson, error)) return false;

    const int reshapedIdx = gguf_find_key(gguf, "ocr.quant.reshaped");
    if (reshapedIdx >= 0) {
        const int32_t n = gguf_get_arr_n(gguf, reshapedIdx);
        for (int32_t i = 0; i < n; ++i) {
            const char* entry = gguf_get_arr_str(gguf, reshapedIdx, i);
            if (!entry) continue;
            const std::string text(entry);
            const size_t colon = text.find(':');
            if (colon == std::string::npos) continue;
            std::vector<int64_t> dims;
            size_t start = colon + 1;
            while (start <= text.size()) {
                const size_t comma = text.find(',', start);
                dims.push_back(
                        std::atoll(text.substr(start, comma - start).c_str()));
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            reshapedDims[text.substr(0, colon)] = std::move(dims);
        }
    }

    const size_t count = gguf_get_n_tensors(gguf);
    for (size_t i = 0; i < count; ++i) {
        const char* name = gguf_get_tensor_name(gguf, i);
        ggml_tensor* t = ggml_get_tensor(meta, name);
        if (!t) {
            if (error) *error = std::string("tensor missing: ") + name;
            return false;
        }
        // Q8_0/F16 containers: the CPU kernels require matching f32
        // operands for binary ops, im2col and mul_mat's src1, so every
        // F16 and Q8_0 parameter is materialized as f32 at load time.
        // (Q8_0 block layout: one f16 scale per 32 int8 values.)
        if (t->type == GGML_TYPE_F16) {
            ggml_tensor* f32 =
                    ggml_new_tensor(constCtx, GGML_TYPE_F32, 4, t->ne);
            const auto* src = static_cast<const ggml_fp16_t*>(t->data);
            auto* dst = static_cast<float*>(f32->data);
            const int64_t n = ggml_nelements(t);
            for (int64_t j = 0; j < n; ++j) dst[j] = ggml_fp16_to_fp32(src[j]);
            // An F16 container written from values beyond the F16 range
            // carries Inf; the affected layer would silently collapse to
            // zero, so fail the load instead.
            for (int64_t j = 0; j < n; ++j) {
                if (!std::isfinite(dst[j])) {
                    if (error)
                        *error =
                                std::string("non-finite value in F16 tensor ") +
                                name + " (container exceeds F16 range)";
                    return false;
                }
            }
            t = f32;
        } else if (t->type == GGML_TYPE_Q8_0) {
            constexpr int64_t kBlock = 32;
            const int64_t n = ggml_nelements(t);
            const int64_t blocks = n / kBlock;
            ggml_tensor* f32 = ggml_new_tensor_1d(constCtx, GGML_TYPE_F32, n);
            const uint8_t* src = static_cast<const uint8_t*>(t->data);
            auto* dst = static_cast<float*>(f32->data);
            for (int64_t b = 0; b < blocks; ++b) {
                const uint16_t scaleBits =
                        static_cast<uint16_t>(src[b * (kBlock + 2)]) |
                        static_cast<uint16_t>(src[b * (kBlock + 2) + 1]) << 8;
                const float d = ggml_fp16_to_fp32(scaleBits);
                for (int k = 0; k < kBlock; ++k)
                    dst[b * kBlock + k] =
                            d *
                            static_cast<int8_t>(src[b * (kBlock + 2) + 2 + k]);
            }
            // Stay 1-D: each consuming branch reshapes from the pinned
            // reshapedDims registry (conv and conv_transpose disagree on
            // the Cin/Cout axis order, so the registry is the only
            // consumer-authoritative source).
            t = f32;
        }
        tensors[name] = t;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Graph builder

ggml_tensor* buildGraph(ggml_context* ctx,
                        const OcrWeights& weights,
                        ggml_tensor* input,
                        std::string* error,
                        const OcrProbeOptions* probe,
                        std::vector<ggml_tensor*>* fetchList,
                        const OcrBuildOptions* options) {
    // Probe-only truncation: stop the build after op `fetchAt` and expose
    // its activation as the fetch tensor (parity bisection support).
    const int fetchAt = probe ? probe->fetchAt : -1;
    const bool dumpShapes = probe ? probe->dumpShapes : false;
    // Winograd routing for 3x3 stride-1 convolutions (CPU sessions only —
    // the shared custom op in common/winograd.hpp is a CPU implementation;
    // on a GPU session it would force a host<->device round trip per conv).
    const bool cpuConv = options ? options->cpu_conv : false;
    int probeIndex = 0;
    // Blocked-island routing. Availability (AVX-512 CPUID) is the CALLER's
    // responsibility — production sets blocked_island only when
    // aicore::blk::blocked_available(); the probe may force it on to keep
    // the scalar island kernels parity-covered on AVX2-only hosts.
    const bool islandOn = options && options->blocked_island;
    // Vars whose values currently live in a blocked buffer, mapped to their
    // own blocked tensor + true channel count. Island ops resolve their
    // inputs through this map by var name, so arbitrary branch topologies
    // (the det FPN consumes vars across segments) stay correct.
    std::map<std::string, std::pair<ggml_tensor*, int>> blockedVars;
    struct BuildState {
        ggml_context* ctx;
        const OcrWeights& weights;
        std::map<std::string, ggml_tensor*> values;
        std::map<std::string, std::vector<int64_t>> constants;

        ggml_tensor* resolve(const std::string& name) const {
            auto it = values.find(name);
            if (it != values.end()) return it->second;
            return const_cast<ggml_tensor*>(weights.weight("ocr." + name));
        }
        ggml_tensor* byParam(const GraphOp& op, const char* param) const {
            for (const auto& io : op.inputs)
                if (io.param == param && !io.args.empty())
                    return resolve(io.args[0]);
            return nullptr;
        }
        ggml_tensor* firstInput(const GraphOp& op) const {
            for (const auto& io : op.inputs)
                if (!io.args.empty())
                    if (ggml_tensor* t = resolve(io.args[0])) return t;
            return nullptr;
        }
        ggml_tensor* broadcastChannel(ggml_tensor* vector) const {
            if (ggml_n_dims(vector) == 1)
                return ggml_reshape_4d(ctx, vector, 1, 1, vector->ne[0], 1);
            return vector;
        }
        ggml_tensor* need(const GraphOp& op,
                          const char* param,
                          const char* what,
                          std::string* error) const {
            ggml_tensor* t = byParam(op, param);
            if (!t) *error = std::string(what) + ": missing " + param;
            return t;
        }
    } state{ctx, weights, {}, {}};
    // Blocked-island dataflow model: the det FPN branches interleave, so an
    // op's input is NOT necessarily the previous op's output. Every island
    // op resolves its own Input var: if the var is tracked in blockedVars
    // the blocked tensor is consumed directly; otherwise the NCHW tensor is
    // reordered in. `islandRebindOne` converts a single var back to NCHW
    // (used when a stock op consumes it); `islandRebindAll` converts every
    // tracked var (used when the program truncates).
    struct IslandBase {
        ggml_tensor* t = nullptr;
        int c = 0;
        bool blocked = false;
        std::string name;
    };
    auto islandInput = [&](const GraphOp& op, const char* param) -> IslandBase {
        for (const auto& io : op.inputs) {
            if (io.param != param || io.args.empty()) continue;
            const std::string& n = io.args[0];
            ggml_tensor* t = state.resolve(n);
            auto it = blockedVars.find(n);
            if (it != blockedVars.end())
                return {it->second.first, it->second.second, true, n};
            return {t, t ? static_cast<int>(t->ne[2]) : 0, false, n};
        }
        return {};
    };
    auto islandRebindOne = [&](const std::string& name) {
        auto it = blockedVars.find(name);
        if (it == blockedVars.end()) return;
        state.values[name] = aicore::blk::blocked_reorder_out(
                ctx, it->second.first, it->second.second);
        blockedVars.erase(it);
    };
    auto islandRebindAll = [&]() {
        for (auto& kv : blockedVars)
            state.values[kv.first] = aicore::blk::blocked_reorder_out(
                    ctx, kv.second.first, kv.second.second);
        blockedVars.clear();
    };
    // Island eligibility for a conv. The spatial cap gates ENTRY only (a
    // conv consuming an already-blocked deep map stays in the island).
    auto islandConvEligible = [&](const IslandBase& in, ggml_tensor* w, bool dw,
                                  int64_t groups, int s0, int s1, int p0,
                                  int p1) {
        if (!islandOn) return false;
        if (!in.blocked) {
            // entering the island requires a clean single-image NCHW input
            if (!in.t || in.t->type != GGML_TYPE_F32 || in.t->ne[3] != 1 ||
                !ggml_is_contiguous(in.t))
                return false;
            // deep/small-spatial regime: the blocked direct kernel wins on
            // the LCNet/MobileNet bodies, not on the wide early maps
            constexpr int64_t kIslandMaxSpatial = 64;
            if (std::min(in.t->ne[0], in.t->ne[1]) > kIslandMaxSpatial)
                return false;
        }
        if (w->type != GGML_TYPE_F32) return false;
        if (dw)
            return w->ne[0] == 3 && w->ne[1] == 3 && p0 == p1 && p0 <= 2 &&
                   s0 >= 1 && s0 <= 2 && s1 >= 1 && s1 <= 2;
        if (groups != 1) return false;
        if (w->ne[0] == 1 && w->ne[1] == 1) return s0 == 1 && s1 == 1;
        return w->ne[0] == 3 && w->ne[1] == 3 && p0 == p1 && p0 <= 2 &&
               s0 == s1 && (s0 == 1 || s0 == 2);
    };
    // Blocked-island conv routing. Returns false (after rebinding the
    // op's own blocked inputs) when the conv must run on the stock path.
    auto islandConvRoute = [&](ggml_tensor* w, ggml_tensor** x,
                               const GraphOp& op, ggml_tensor* bias,
                               int64_t groups, int s0, int s1, int p0, int p1,
                               ggml_tensor** out) -> bool {
        IslandBase in = islandInput(op, "Input");
        if (!islandConvEligible(in, w, op.type == "depthwise_conv2d", groups,
                                s0, s1, p0, p1)) {
            islandRebindOne(in.name);
            return false;
        }
        ggml_tensor* base =
                in.blocked ? in.t : aicore::blk::blocked_reorder_in(ctx, in.t);
        if (op.type == "depthwise_conv2d") {
            *out = aicore::blk::blocked_conv3x3_dw(ctx, w, base, p0, s0, s1,
                                                   bias);
        } else if (w->ne[0] == 1) {
            *out = aicore::blk::blocked_conv1x1(ctx, w, base, s0, bias);
        } else {
            *out = aicore::blk::blocked_conv3x3(ctx, w, base, p0, s0, bias);
        }
        *x = in.t;  // the stock path re-resolves after a route failure
        return true;
    };
    // The island kernels assume dilation 1; a dilated conv (mid-chain)
    // exits the island instead, and a dilated conv never enters one.
    auto islandDilationOk = [&](int d0, int d1) { return d0 == 1 && d1 == 1; };
    // Bind a blocked result to the op outputs. Returns true when the caller
    // must break out of the op loop (probe truncation rebinds everything so
    // the fetch stays comparable with the numpy reference).
    auto islandBind = [&](const GraphOp& op, ggml_tensor* xb, int xbC) -> bool {
        for (const auto& name : op.outputs) {
            state.values[name] = xb;
            blockedVars[name] = {xb, xbC};
        }
        if (dumpShapes && !op.outputs.empty()) {
            std::fprintf(stderr,
                         "[shape] [%s] %s ne=%lld,%lld,%lld,%lld (blocked)\n",
                         op.type.c_str(), op.outputs[0].c_str(),
                         (long long)xb->ne[0], (long long)xb->ne[1],
                         (long long)xb->ne[2], (long long)xb->ne[3]);
            std::fflush(stderr);
        }
        if (probeIndex == fetchAt) {
            islandRebindAll();
            state.values["__fetch__"] =
                    op.outputs.empty() ? nullptr : state.values[op.outputs[0]];
            ++probeIndex;
            return true;
        }
        ++probeIndex;
        for (const auto& name : op.outputs)
            if (name.find("scale_0.tmp_0") != std::string::npos)
                state.values["__fetch__"] = xb;  // blocked; the fetch op's
        return false;                            // rebind converts to NCHW
    };
    ggml_context* cctx =
            weights.foldContext() ? weights.foldContext() : weights.constCtx;

    // FPN alignment: each nearest_interp must match the size of the
    // branch it is fused with - either the concat's non-interpolated
    // branch or the other operand of its downstream elementwise_add.
    // Pre-scan the audited graph so the build can resolve per-op
    // target sizes.
    std::map<std::string, std::string> nearestTarget;
    {
        std::set<std::string> nearestOuts;
        for (const auto& o : weights.program.ops)
            if (o.type == "nearest_interp_v2" && !o.outputs.empty())
                nearestOuts.insert(o.outputs.front());
        auto alignAgainst = [&](const GraphOp& o) {
            const std::string* other = nullptr;
            std::vector<const std::string*> nearestVars;
            for (const auto& io : o.inputs) {
                for (const auto& arg : io.args) {
                    if (arg.empty()) continue;
                    if (nearestOuts.count(arg))
                        nearestVars.push_back(&arg);
                    else if (!other)
                        other = &arg;
                }
            }
            if (other && !nearestVars.empty())
                for (const auto* v : nearestVars) nearestTarget[*v] = *other;
        };
        for (const auto& o : weights.program.ops)
            if (o.type == "elementwise_add") alignAgainst(o);
        for (const auto& o : weights.program.ops)
            if (o.type == "concat") alignAgainst(o);
    }

    // Batched-attention support (rec, N>1): the audited SVTR head has TWO
    // serial attention blocks; each packs Q/K/V through a rank-5 reshape
    // (N,T,3,heads,dim) that ggml's 4-D tensors cannot represent, which is
    // what pins the rec phase to per-crop N=1 forwards. For N>1 replace
    // each chain with strided 4-D views of the qkv projection plus
    // batched mul_mat (the llama.cpp attention layout), so ONE forward
    // covers all crops:
    //   Q/K/V views : (d,t,h,n) strided views of qkv (360,T,N)
    //   scores      = K_view x scale(Q_view)             -> (s,t,h,n)
    //   ctx         = cont(transpose(V_view)) x scores   -> (t,d,h,n)
    //   merged      = cont(permute(ctx)) reshaped to     -> (120,T,1,N)
    // Pattern-driven: a rank-5 reshape [0,-1,3,8,15] triggers the bypass;
    // everything until the block-tail reshape [0,-1,120] is skipped and
    // its output replaced by the merged tensor. The qkv projection is
    // located by its 360-wide first axis and validated (contiguous,
    // (360,T,N)); any mismatch is a hard error, never a silent wrong
    // graph. The FFN block between the two attentions is batch-safe and
    // runs normally.
    float attnScale = 0.f;
    for (const auto& o : weights.program.ops) {
        if (o.type == "scale") {
            const double v = attrFloat(o.attrs, "scale", 0.0);
            if (v > 0.0) {
                attnScale = static_cast<float>(v);
                break;
            }
        }
    }
    bool batchAttnSkip = false;
    ggml_tensor* batchAttnMerged = nullptr;

    for (const GraphOp& op : weights.program.ops) {
        const std::string outName =
                op.outputs.empty() ? std::string() : op.outputs.front();

        if (op.type == "reshape2" && !op.outputs.empty()) {
            const auto shape = attrInts(op.attrs, "shape");
            if (shape.size() == 5 && shape[0] == 0 && shape[1] == -1 &&
                shape[2] == 3 && input->ne[3] > 1) {
                // Attention-block head: bypass the whole rank-5 chain.
                ggml_tensor* qkv = nullptr;
                for (const auto& io : op.inputs)
                    for (const auto& arg : io.args) {
                        auto it = state.values.find(arg);
                        if (it != state.values.end() && it->second &&
                            it->second->ne[0] == 360)
                            qkv = it->second;
                    }
                // T is the attention time axis of the qkv projection
                // (qkv (360,T,N)); the rec input's ne[1] is the 48-px
                // image height, not T.
                const int64_t T = qkv->ne[1];
                const int64_t N = input->ne[3];
                if (attnScale <= 0.f) {
                    if (error) *error = "batched attention: scale attr missing";
                    return nullptr;
                }
                if (!qkv || qkv->ne[0] != 360 || qkv->ne[1] != T ||
                    qkv->ne[2] != N || qkv->ne[3] != 1 ||
                    !ggml_is_contiguous(qkv)) {
                    if (error) *error = "batched attention: qkv shape drift";
                    return nullptr;
                }
                const size_t rowBytes = 360 * sizeof(float);
                const size_t nStride = qkv->nb[2];
                // (d,t,h,n) strided views over the packed [q|k|v] rows.
                // Paddle packs e = c*120 + h*15 + d (reshape (N,T,3,8,15)),
                // so the head stride is 15 elements = 60 BYTES and the
                // q/k/v section offsets are 120/240 elements = 480/960
                // bytes. The section offsets and the head stride must be
                // kept distinct: both are numeric 60/240 pairs that swap
                // easily and every past mix-up here scrambled the text.
                ggml_tensor* Qdt = ggml_view_4d(ctx, qkv, 15, T, 8, N, rowBytes,
                                                15 * sizeof(float), nStride, 0);
                ggml_tensor* Kdt =
                        ggml_view_4d(ctx, qkv, 15, T, 8, N, rowBytes,
                                     15 * sizeof(float), nStride, 480);
                // scores(s,t,h,n) = sum_d Q(d,t)K(d,s), batched over (h,n).
                // The 1/sqrt(d) Q scale is linear, so applying it to the
                // product keeps the math identical while letting ggml_scale
                // see the contiguous mul_mat output (a strided view would
                // trip its padded_1d assert).
                // ggml_mul_mat's CUDA kernel does not preserve the
                // element order of strided src0/src1 the way the CPU kernel
                // does (CPU: bitwise-exact vs the N=1 chain; CUDA: garbage),
                // so both attention operands are materialized contiguous
                // first — three (15,T,8,N) copies per block, trivial next to
                // the batched GEMMs they enable.
                ggml_tensor* Kc = ggml_cont(ctx, Kdt);
                ggml_tensor* Qc = ggml_cont(ctx, Qdt);
                ggml_tensor* scores =
                        ggml_scale(ctx, ggml_mul_mat(ctx, Kc, Qc), attnScale);
                ggml_tensor* probs = ggml_soft_max(ctx, scores);
                // ctx(d,t,h,n) = sum_s probs(s,t) V(d,s): V must enter
                // with a contiguous s axis, so materialize (s,d) once.
                ggml_tensor* Vdt =
                        ggml_view_4d(ctx, qkv, 15, T, 8, N, rowBytes,
                                     15 * sizeof(float), nStride, 960);
                ggml_tensor* Vsd = ggml_cont(ctx, ggml_transpose(ctx, Vdt));
                ggml_tensor* attnCtx = ggml_mul_mat(ctx, Vsd, probs);
                // Reorder (d,t,h,n) -> (d,h,t,n): after cont, the (h,d)
                // composite h*15+d becomes the fastest axis with stride 1,
                // which is exactly the block-tail contract out[n,t,h*15+d]
                // = attnCtx[n,h,t,d] once reshaped to (120,T,1,N). The
                // permute index order is easy to get backwards: axes move
                // TO the front, so (0,2,1,3) keeps d first and swaps t/h.
                ggml_tensor* dhtn = ggml_permute(ctx, attnCtx, 0, 2, 1, 3);
                // The audited chain carries a 4-D (N,1,T,120) footprint
                // into the following transpose's rank-4 perm — keep it.
                batchAttnMerged = ggml_reshape_4d(ctx, ggml_cont(ctx, dhtn),
                                                  120, T, 1, N);
                batchAttnSkip = true;
                continue;
            }
            if (batchAttnSkip && shape.size() == 3 && shape[0] == 0 &&
                shape[2] == 120) {
                // Block tail: its output is the merged tensor.
                if (!op.outputs.empty())
                    state.values[op.outputs[0]] = batchAttnMerged;
                batchAttnSkip = false;
                batchAttnMerged = nullptr;
                continue;
            }
        }
        if (batchAttnSkip) continue;

        ggml_tensor* result = nullptr;
        bool isConstant = false;
        // Ops outside the island's op set run stock: rebind their blocked
        // INPUT vars to NCHW (other tracked vars stay blocked for later
        // island consumers — the det FPN consumes vars across segments).
        if (islandOn && !blockedVars.empty()) {
            static const std::set<std::string> kIslandOps = {"conv2d",
                                                             "depthwise_conv2d",
                                                             "elementwise_add",
                                                             "elementwise_mul",
                                                             "relu",
                                                             "hard_swish",
                                                             "hard_sigmoid",
                                                             "swish",
                                                             "pool2d"};
            if (!kIslandOps.count(op.type)) {
                for (const auto& io : op.inputs)
                    for (const auto& arg : io.args)
                        if (!arg.empty()) islandRebindOne(arg);
            }
        }

        if (op.type == "feed") {
            result = input;
        } else if (op.type == "fetch" || op.type == "assign") {
            result = state.firstInput(op);
            // The fetch op is the program's explicit output anchor; the
            // save_infer_model variable name is not always present.
            if (op.type == "fetch") {
                state.values["__fetch__"] = result;
                // Multi-output programs (PicoDet layout exposes 8 heads)
                // collect every fetch in program order.
                if (fetchList) fetchList->push_back(result);
            }
        } else if (op.type == "split") {
            // GFL head channel split: axis is a numpy axis over the
            // (N,C,H,W) contract; the pieces are views, never copies.
            ggml_tensor* x = state.byParam(op, "X");
            if (!x) {
                if (error) *error = "split: missing X";
                return nullptr;
            }
            const int axis = attrInt(op.attrs, "axis", 1);
            auto sections = attrInts(op.attrs, "sections");
            const int num = attrInt(op.attrs, "num", 0);
            // numpy (N,C,H,W) axis a -> ggml ne axis 3-a.
            const int gaxis = 3 - axis;
            if (gaxis < 0 || gaxis > 3) {
                if (error) *error = "split: bad axis";
                return nullptr;
            }
            int64_t piece = 0;
            if (!sections.empty()) {
                piece = sections.front();
            } else if (num > 0) {
                piece = x->ne[gaxis] / num;
            } else {
                if (error) *error = "split: no sections/num";
                return nullptr;
            }
            int64_t offset = 0;
            int idx = 0;
            for (const auto& arg : op.outputs) {
                const int64_t len =
                        sections.empty()
                                ? piece
                                : (idx < (int)sections.size() ? sections[idx]
                                                              : 0);
                int64_t ne[4] = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
                ne[gaxis] = len;
                // Element offset along the split axis; ggml view offsets
                // are byte offsets (nb[gaxis] = per-element stride there).
                ggml_tensor* v = ggml_view_4d(
                        ctx, x, ne[0], ne[1], ne[2], ne[3], x->nb[1], x->nb[2],
                        x->nb[3], static_cast<size_t>(offset * x->nb[gaxis]));
                state.values[arg] = v;
                offset += len;
                ++idx;
            }
            result = state.byParam(op, "X");
            isConstant = false;
            // Fall through to the output binding below: outputs were
            // already bound individually, so skip the default path.
            if (probeIndex == fetchAt) {
                ++probeIndex;
                break;
            }
            ++probeIndex;
            continue;
        } else if (op.type == "conv2d" || op.type == "depthwise_conv2d") {
            ggml_tensor* w = state.byParam(op, "Filter");
            ggml_tensor* x = state.byParam(op, "Input");
            ggml_tensor* bias = state.byParam(op, "Bias");
            if (!w || !x) {
                if (error) *error = op.type + ": missing Filter/Input";
                return nullptr;
            }
            if (ggml_n_dims(w) != 4) {
                std::string filterName;
                for (const auto& io : op.inputs)
                    if (io.param == "Filter" && !io.args.empty())
                        filterName = io.args[0];
                auto it = weights.reshapedDims.find("ocr." + filterName);
                if (it == weights.reshapedDims.end())
                    it = weights.reshapedDims.find(filterName);
                if (it == weights.reshapedDims.end() ||
                    it->second.size() != 4) {
                    if (error)
                        *error = op.type + ": folded kernel without dims (" +
                                 filterName + ")";
                    return nullptr;
                }
                w = ggml_reshape_4d(ctx, w, it->second[3], it->second[2],
                                    it->second[1], it->second[0]);
            }
            const auto strides = attrInts(op.attrs, "strides");
            const auto paddings = attrInts(op.attrs, "paddings");
            const auto dilations = attrInts(op.attrs, "dilations");
            const int64_t groups = attrInt(op.attrs, "groups", 1);
            // im2col requires an element-contiguous input; the head conv
            // consumes a transposed encoder output.
            x = elemCont(ctx, x);
            int64_t ph = paddings.size() > 3 ? paddings[2]
                         : paddings.empty()  ? 0
                                             : paddings[0];
            int64_t pw = paddings.size() > 3   ? paddings[3]
                         : paddings.size() < 2 ? 0
                                               : paddings[1];
            const int s0 =
                    static_cast<int>(strides.size() > 1 ? strides[1] : 1);
            const int s1 = static_cast<int>(strides.empty() ? 1 : strides[0]);
            const int p0 = static_cast<int>(pw);
            const int p1 = static_cast<int>(ph);
            const int d0 =
                    static_cast<int>(dilations.size() > 1 ? dilations[1] : 1);
            const int d1 =
                    static_cast<int>(dilations.empty() ? 1 : dilations[0]);

            // Blocked-island conv routing (see islandConvRoute above).
            ggml_tensor* islandOut = nullptr;
            if (islandDilationOk(d0, d1) &&
                islandConvRoute(w, &x, op, bias, groups, s0, s1, p0, p1,
                                &islandOut)) {
                if (islandBind(op, islandOut, static_cast<int>(w->ne[3])))
                    break;
                continue;
            }
            // re-resolve x after a potential island rebind (the stock path
            // always consumes NCHW)
            x = state.byParam(op, "Input");
            if (!x) {
                if (error) *error = op.type + ": missing Input";
                return nullptr;
            }
            x = elemCont(ctx, x);

            if (op.type == "conv2d" && groups == 1) {
                // CPU sessions route 3x3 stride-1 dilation-1 convolutions
                // through the shared Winograd custom op (2.25x fewer
                // multiplies than direct; facedetect/depth parity-proven).
                // GPU sessions keep im2col+mul_mat (the custom op is
                // CPU-only). Bias stays a separate add: the winograd
                // epilogue only fuses relu, and the OCR activations are
                // hard_swish/swish handled downstream.
                if (cpuConv && w->ne[0] == 3 && w->ne[1] == 3 && s0 == 1 &&
                    s1 == 1 && d0 == 1 && d1 == 1 && p0 == p1 &&
                    w->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(x) && ggml_is_contiguous(w)) {
                    result = aicore::wino::winograd_conv3x3(ctx, w, x, p0);
                } else if (options && options->gpu_rec_direct_conv) {
                    // Direct convolution is limited to the recognizer. Its
                    // fixed 320x48 shapes have passed decoded-text parity;
                    // detector maps remain on im2col because small F32
                    // accumulation changes can move DB threshold edges.
                    result = ggml_conv_2d_direct(ctx, w, x, s0, s1, p0, p1, d0,
                                                 d1);
                    // The CUDA backend uses this marker to select its TF32
                    // IGEMM path only for the parity-validated rec route.
                    ggml_format_name(result, "ocr_rec_direct_conv");
                } else {
                    // GPU sessions route the 3x3/1x1 convs through
                    // im2col+GEMM like the CPU stock path. Every direct
                    // variant tried (pin-native naive, TF32 WMMA igemm, and
                    // an F32-FMA micro-tile written for this task) is
                    // FASTER on the det graph (14 vs 60 ms) but produces a
                    // different F32 summation order, and the det map's
                    // 3e-3-level deltas shift box geometry past the 0.3
                    // threshold edge: end-to-end transcriptions regressed
                    // 98 -> 67 valid rows even though every fixture gate
                    // (argmax, max_diff, self) stayed green. The im2col
                    // expansion's cuBLAS-backed GEMM order is the
                    // transcription-parity reference, so it is the
                    // production route until a kernel matches its numerics.
                    result = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);
                }
            } else if (op.type == "depthwise_conv2d") {
                // Folded q8_0 kernels restore to {kw,kh,1,cin} from the
                // pinned registry.
                if (ggml_n_dims(w) < 4) {
                    std::string filterName;
                    for (const auto& io : op.inputs)
                        if (io.param == "Filter" && !io.args.empty())
                            filterName = io.args[0];
                    auto it = weights.reshapedDims.find("ocr." + filterName);
                    if (it == weights.reshapedDims.end())
                        it = weights.reshapedDims.find(filterName);
                    if (it == weights.reshapedDims.end() ||
                        it->second.size() != 4) {
                        if (error)
                            *error = op.type +
                                     ": folded kernel without dims (" +
                                     filterName + ")";
                        return nullptr;
                    }
                    w = ggml_reshape_4d(ctx, w, it->second[3], it->second[2], 1,
                                        it->second[0]);
                }
                // Footprint guard: the batched GEMM materializes a
                // K*OW*OH*C*N im2col. At det resolutions (960x512) the
                // high-C layers reach multiple GB and starve the CPU
                // allocator, so fall back to the per-channel chain
                // (whose per-channel im2col stays K*OW*OH) whenever
                // the batched buffer would exceed the cap. GPU buffers
                // live in device memory (GBs), so the same guard uses a
                // much larger cap there.
                const auto convOut = [](int64_t in, int64_t k, int s, int p,
                                        int d) {
                    return (in + 2 * p - d * (k - 1) - 1) / s + 1;
                };
                const int64_t dwOW = convOut(x->ne[0], w->ne[0], s0, p0, d0);
                const int64_t dwOH = convOut(x->ne[1], w->ne[1], s1, p1, d1);
                const int64_t im2colBytes = w->ne[0] * w->ne[1] * dwOW * dwOH *
                                            x->ne[2] * x->ne[3] * sizeof(float);
                constexpr int64_t kMaxIm2colBytesCpu = 256ll << 20;
                constexpr int64_t kMaxIm2colBytesGpu = 2048ll << 20;
                const int64_t kMaxIm2colBytes = (options && options->cpu_conv)
                                                        ? kMaxIm2colBytesCpu
                                                        : kMaxIm2colBytesGpu;
                if (options && options->gpu_rec_direct_conv) {
                    // The depthwise kernel has no cross-channel reduction,
                    // so its direct 3x3 evaluation preserves the recognizer
                    // result while avoiding the K^2*OH*OW*C*N im2col buffer.
                    result = ggml_conv_2d_dw_direct(ctx, w, x, s0, s1, p0, p1,
                                                    d0, d1);
                } else if (im2colBytes > kMaxIm2colBytes) {
                    std::vector<ggml_tensor*> parts;
                    for (int64_t c = 0; c < x->ne[2]; ++c) {
                        ggml_tensor* xc = ggml_view_4d(
                                ctx, x, x->ne[0], x->ne[1], 1, x->ne[3],
                                x->nb[1], x->nb[2], x->nb[3], c * x->nb[2]);
                        ggml_tensor* wc = ggml_view_4d(
                                ctx, w, w->ne[0], w->ne[1], 1, 1, w->nb[1],
                                w->nb[2], w->nb[3], c * w->nb[3]);
                        parts.push_back(ggml_conv_2d(ctx, wc, xc, s0, s1, p0,
                                                     p1, d0, d1));
                    }
                    result = parts.front();
                    for (size_t i = 1; i < parts.size(); ++i)
                        result = ggml_concat(ctx, result, parts[i], 2);
                } else {
                    // Depthwise as one batched GEMM (the ggml_conv_2d_dw
                    // idiom): collapse the channel axis into the batch
                    // axis, im2col once with a single-channel kernel
                    // shape, and multiply by the per-channel filters as
                    // the GEMM's batch entries. Same F32 K accumulation
                    // as the historical per-channel conv chain, but ~6
                    // nodes instead of ~2*C+1 — the old expansion was the
                    // dominant node-count source (14 depthwise ops with
                    // up to 3072 total channels produced tens of
                    // thousands of graph nodes whose launch overhead
                    // dwarfed the arithmetic on every backend).
                    ggml_tensor* xc =
                            ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
                    ggml_tensor* xb =
                            ggml_reshape_4d(ctx, xc, xc->ne[0], xc->ne[1], 1,
                                            xc->ne[2] * xc->ne[3]);
                    // im2col reads only the kernel geometry (KW, KH,
                    // Cin=1) from this operand. A channel-0 VIEW of w
                    // would be non-contiguous for C>1 (the batch stride
                    // keeps the channel gap), which the CUDA backend
                    // rejects for IM2COL; cont(materialize) of the
                    // channel-0 view is a 36-byte copy that stays
                    // contiguous.
                    ggml_tensor* sv = ggml_cont(
                            ctx, ggml_view_4d(ctx, w, w->ne[0], w->ne[1], 1, 1,
                                              w->nb[1], w->nb[2], w->nb[3], 0));
                    ggml_tensor* cols =
                            ggml_im2col(ctx, sv, xb, s0, s1, p0, p1, d0, d1,
                                        true, GGML_TYPE_F32);
                    ggml_tensor* cols3 = ggml_reshape_3d(
                            ctx, cols, cols->ne[0], cols->ne[1] * cols->ne[2],
                            cols->ne[3]);
                    // Per-channel filters as the GEMM's batch entries.
                    // The merged batch axis is i = c + n*C (c fastest,
                    // from the reshape of the contiguous {W,H,C,N}
                    // input), but ggml's mul_mat broadcast maps src0's
                    // batch as i12/r2 (c SLOWEST, ggml-cpu.c), so the
                    // slot-to-filter mapping would scramble. Instead of
                    // relying on broadcast (or paying a full activation
                    // permute+cont), tile the tiny filter tensor to the
                    // full C*N batch: repeat's dst(i) = wr(i % C) is
                    // EXACTLY our slot order, and the GEMM runs with
                    // matching batch counts (no broadcast at all).
                    ggml_tensor* wr = ggml_reshape_3d(
                            ctx, w, w->ne[0] * w->ne[1], 1, w->ne[3]);
                    ggml_tensor* wrBatchTarget = ggml_new_tensor_3d(
                            ctx, GGML_TYPE_F32, wr->ne[0], 1, cols3->ne[2]);
                    ggml_tensor* wrRep = ggml_repeat(ctx, wr, wrBatchTarget);
                    result = ggml_mul_mat(ctx, wrRep, cols3);
                    // The merged batch axis splits back order-preserving:
                    // ne={OW,OH,C,N} enumerates c slower than n, i.e.
                    // exactly the i = c + n*C slot order the GEMM used.
                    result = ggml_reshape_4d(ctx, result, cols->ne[1],
                                             cols->ne[2], x->ne[2], x->ne[3]);
                }
            } else {
                if (error) *error = "grouped conv2d (groups>1) unsupported";
                return nullptr;
            }
            if (bias)
                result = addChk(ctx, result,
                                state.broadcastChannel(ggml_reshape_1d(
                                        ctx, bias, bias->ne[0])));
        } else if (op.type == "nearest_interp_v2") {
            ggml_tensor* x = state.need(op, "X", "nearest_interp", error);
            if (!x) return nullptr;
            // Paddle resolves the unbound OutSize/Scale from the static
            // VarDesc shape. The DBNet head upscales by exact integer
            // factors, which ggml_upscale (nearest) provides.
            // The det input is fully dynamic (-1) so the VarDesc shape
            // carries no sizes. FPN alignment pins the target to the
            // fused branch (which may be a non-integer multiple of the
            // input, e.g. 8 -> 15 after floor-pooled strides), so the
            // op resizes to the exact target instead of an integer
            // factor; standalone nearest ops (the audited DBNet head)
            // upscale 2x.
            int64_t outW = x->ne[0] * 2;
            int64_t outH = x->ne[1] * 2;
            auto refIt = nearestTarget.find(outName);
            if (refIt != nearestTarget.end()) {
                if (ggml_tensor* ref = state.resolve(refIt->second)) {
                    outW = ref->ne[0];
                    outH = ref->ne[1];
                }
            }
            result = ggml_interpolate(
                    ctx, elemCont(ctx, x), static_cast<int>(outW),
                    static_cast<int>(outH), static_cast<int>(x->ne[2]),
                    static_cast<int>(x->ne[3]), GGML_SCALE_MODE_NEAREST);
        } else if (op.type == "conv2d_transpose") {
            ggml_tensor* w = state.byParam(op, "Filter");
            ggml_tensor* x = state.need(op, "Input", "conv2d_transpose", error);
            if (!x) return nullptr;
            if (!w) {
                if (error) *error = "conv2d_transpose: missing Filter";
                return nullptr;
            }
            // Folded q8 kernels restore to {kw,kh,Cout,Cin}; ggml's
            // conv_transpose expects Cout on ne[2] and Cin on ne[3].
            if (ggml_n_dims(w) != 4) {
                std::string filterName;
                for (const auto& io : op.inputs)
                    if (io.param == "Filter" && !io.args.empty())
                        filterName = io.args[0];
                auto it = weights.reshapedDims.find("ocr." + filterName);
                if (it == weights.reshapedDims.end())
                    it = weights.reshapedDims.find(filterName);
                if (it == weights.reshapedDims.end() ||
                    it->second.size() != 4) {
                    if (error)
                        *error = "conv2d_transpose: folded kernel without dims";
                    return nullptr;
                }
                w = ggml_reshape_4d(ctx, w, it->second[3], it->second[2],
                                    it->second[0], it->second[1]);
            }
            x = elemCont(ctx, x);
            const auto strides = attrInts(op.attrs, "strides");
            const int s0 = static_cast<int>(strides.empty() ? 1 : strides[0]);
            result = ggml_conv_transpose_2d_p0(ctx, w, x, s0);
            if (ggml_tensor* biasT = state.byParam(op, "Bias"))
                result = addChk(ctx, result,
                                state.broadcastChannel(ggml_reshape_1d(
                                        ctx, biasT, biasT->ne[0])));
        } else if (op.type == "batch_norm") {
            // Param names from the audited graph: Scale=w_0, Mean=w_1.
            // The four statistics are read on the CPU, so they resolve
            // through the host view (the main map may point into the
            // device buffer after realizeToDevice).
            auto hostParam = [&](const char* param) -> const ggml_tensor* {
                for (const auto& io : op.inputs) {
                    if (io.param == param && !io.args.empty())
                        return weights.hostWeight("ocr." + io.args[0]);
                }
                return nullptr;
            };
            ggml_tensor* x = state.byParam(op, "X");
            const ggml_tensor* biasT = hostParam("Bias");
            const ggml_tensor* scaleT = hostParam("Scale");
            const ggml_tensor* meanT = hostParam("Mean");
            const ggml_tensor* varT = hostParam("Variance");
            if (!x || !biasT || !scaleT || !meanT || !varT) {
                if (error) *error = "batch_norm: missing parameters";
                return nullptr;
            }
            const double eps = attrFloat(op.attrs, "epsilon", 1e-5);
            const auto bias = tensorHostFloats(biasT);
            const auto scale = tensorHostFloats(scaleT);
            const auto mean = tensorHostFloats(meanT);
            const auto var = tensorHostFloats(varT);
            const int64_t C = x->ne[2];
            ggml_tensor* mulT = ggml_new_tensor_1d(cctx, GGML_TYPE_F32, C);
            ggml_tensor* addT = ggml_new_tensor_1d(cctx, GGML_TYPE_F32, C);
            auto* mul = static_cast<float*>(mulT->data);
            auto* add = static_cast<float*>(addT->data);
            for (int64_t c = 0; c < C; ++c) {
                const double stdv = std::sqrt(var[c] + eps);
                mul[c] = static_cast<float>(scale[c] / stdv);
                add[c] = static_cast<float>(bias[c] - mean[c] * mul[c]);
            }
            if (!ggml_can_repeat(state.broadcastChannel(mulT), x)) {
                std::fprintf(stderr,
                             "[bn-mismatch] x ne=%lld,%lld,%lld,%lld "
                             "C=%lld mul ne=%lld,%lld,%lld,%lld\n",
                             (long long)x->ne[0], (long long)x->ne[1],
                             (long long)x->ne[2], (long long)x->ne[3],
                             (long long)C, (long long)mulT->ne[0],
                             (long long)mulT->ne[1], (long long)mulT->ne[2],
                             (long long)mulT->ne[3]);
                std::fflush(stderr);
            }
            result = mulChk(ctx, x, state.broadcastChannel(mulT), "bn_mul");
            result = addChk(ctx, result, state.broadcastChannel(addT));
        } else if (op.type == "elementwise_add" ||
                   op.type == "elementwise_mul") {
            ggml_tensor* a = state.byParam(op, "X");
            ggml_tensor* b = state.byParam(op, "Y");
            if (!a) a = state.firstInput(op);
            if (!b) {
                for (const auto& io : op.inputs) {
                    if (io.args.empty()) continue;
                    ggml_tensor* t = state.resolve(io.args[0]);
                    if (t && t != a) {
                        b = t;
                        break;
                    }
                }
            }
            if (!a || !b) {
                if (error) *error = op.type + ": missing inputs";
                return nullptr;
            }
            const int64_t axis = attrInt(op.attrs, "axis", -1);
            bool stockFallback = false;
            // Blocked-island elementwise routing (per-input-var resolution):
            // 1-D [C] operands become fused bias / SE channel-scale passes;
            // the SE excitation multiplies the body activation by the
            // [16,1,1,CB] vector (both sides may be blocked or NCHW);
            // residual adds run blocked when both operands are blocked or
            // the NCHW one matches the body shape (reorder-in). Anything
            // else rebinds this op's blocked inputs and runs stock.
            if (islandOn) {
                IslandBase xa = islandInput(op, "X");
                IslandBase ya = islandInput(op, "Y");
                if (!xa.t && a) xa = {a, a ? (int)a->ne[2] : 0, false, ""};
                if (!ya.t && b) ya = {b, b ? (int)b->ne[2] : 0, false, ""};
                // body = the blocked SPATIAL operand (the SE vector has
                // 1x1 spatial; when both operands are blocked and one is
                // 1x1, the other one is the body).
                IslandBase* body = nullptr;
                IslandBase* other = nullptr;
                if (xa.blocked && ya.blocked) {
                    const bool xaIsVec = xa.t->ne[1] == 1 && xa.t->ne[2] == 1;
                    const bool yaIsVec = ya.t->ne[1] == 1 && ya.t->ne[2] == 1;
                    if (xaIsVec && !yaIsVec) {
                        body = &ya;
                        other = &xa;
                    } else {
                        body = &xa;
                        other = &ya;
                    }
                } else if (xa.blocked) {
                    body = &xa;
                    other = &ya;
                } else if (ya.blocked) {
                    body = &ya;
                    other = &xa;
                }
                if (body && other && body->t) {
                    ggml_tensor* o = other->t;
                    const bool channelVec =
                            o && ggml_n_dims(o) == 1 && o->ne[0] == body->c &&
                            o->ne[0] > 1 &&
                            (axis == 1 || body->c != (int)body->t->ne[1]);
                    const bool seVecBlocked =
                            other->blocked && ggml_n_dims(o) == 4 &&
                            o->ne[0] == 16 && o->ne[1] == 1 && o->ne[2] == 1 &&
                            o->ne[3] == body->t->ne[3];
                    const bool seVecNchw =
                            !other->blocked && o->type == GGML_TYPE_F32 &&
                            ggml_n_dims(o) == 4 && o->ne[0] == 1 &&
                            o->ne[1] == body->c && o->ne[2] == 1 &&
                            o->ne[3] == 1;
                    const bool sameShape =
                            o && ggml_n_dims(o) == 4 && o->ne[3] == 1 &&
                            o->ne[0] == body->t->ne[1] &&
                            o->ne[1] == body->t->ne[2] && o->ne[2] == body->c;
                    const bool seMul = op.type == "elementwise_mul" &&
                                       (seVecBlocked ||
                                        (seVecNchw && ggml_is_contiguous(o)));
                    const bool blockedResidual = op.type == "elementwise_add" &&
                                                 other->blocked && sameShape;
                    const bool residual = op.type == "elementwise_add" &&
                                          !other->blocked && sameShape &&
                                          o->type == GGML_TYPE_F32 &&
                                          ggml_is_contiguous(o);
                    ggml_tensor* xb = body->t;
                    if (channelVec) {
                        xb = op.type == "elementwise_add"
                                     ? aicore::blk::blocked_bias(ctx, xb, o)
                                     : aicore::blk::blocked_mul_channel(ctx, xb,
                                                                        o);
                    } else if (seMul) {
                        ggml_tensor* vec =
                                seVecBlocked ? o
                                             : aicore::blk::blocked_reorder_in(
                                                       ctx, o);
                        xb = aicore::blk::blocked_mul_channel(ctx, xb, vec);
                    } else if (blockedResidual) {
                        xb = aicore::blk::blocked_add(ctx, xb, o);
                    } else if (residual) {
                        ggml_tensor* ob =
                                aicore::blk::blocked_reorder_in(ctx, o);
                        xb = aicore::blk::blocked_add(ctx, xb, ob);
                    } else {
                        // unsupported combination: rebind this op's blocked
                        // inputs and run the stock broadcast path
                        islandRebindOne(xa.name);
                        islandRebindOne(ya.name);
                        stockFallback = true;
                    }
                    if (!stockFallback) {
                        if (islandBind(op, xb, body->c)) break;
                        continue;
                    }
                }
            }
            if (stockFallback) {
                // The stock path needs NCHW operands; re-resolve a/b from
                // the (possibly rebound) value map.
                a = state.byParam(op, "X");
                if (!a) a = state.firstInput(op);
                b = state.byParam(op, "Y");
                if (!b) {
                    for (const auto& io : op.inputs) {
                        if (io.args.empty()) continue;
                        ggml_tensor* t = state.resolve(io.args[0]);
                        if (t && t != a) {
                            b = t;
                            break;
                        }
                    }
                }
                if (!a || !b) {
                    if (error) *error = op.type + ": missing inputs";
                    return nullptr;
                }
            }
            // ggml broadcasts `b` into `a`'s shape; Paddle may place the
            // scalar/vector operand first, so normalize the order here.
            if (ggml_nelements(a) < ggml_nelements(b)) std::swap(a, b);
            // Explicit Paddle axis: a 1-D Y with axis=1 sits on the
            // channel axis (ggml ne2). The can_repeat fallback below
            // cannot infer this - when 16 divides the 320-wide W axis it
            // silently broadcasts over W instead and corrupts every
            // element of the layer.
            if (axis == 1 && ggml_n_dims(b) == 1 && b->ne[0] > 1)
                b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
            if (ggml_n_dims(b) == 1 && b->ne[0] != 1 &&
                !ggml_can_repeat(b, a)) {
                // Paddle's automatic alignment: place the vector on the
                // dimension of `a` whose size matches the vector length
                // (axis=1 lands on the channel axis, axis=-1 infers it).
                int matchDim = -1;
                for (int d = 0; d < 4; ++d)
                    if (a->ne[d] == b->ne[0]) {
                        matchDim = d;
                        break;
                    }
                if (matchDim >= 0) {
                    int64_t ne[4] = {1, 1, 1, 1};
                    ne[matchDim] = b->ne[0];
                    b = ggml_reshape_4d(ctx, b, ne[0], ne[1], ne[2], ne[3]);
                }
            }
            if (!ggml_can_repeat(b, a) &&
                ggml_nelements(a) == ggml_nelements(b)) {
                // Same graph, same element order, two different high-dim
                // axis assignments (batch on ne[2] on the conv-flatten
                // branch vs ne[3] through the batched-matmul branch).
                // Element counts match, so reshaping b onto a's footprint
                // is order-preserving in the layout contract.
                b = ggml_reshape_4d(ctx, b, a->ne[0], a->ne[1], a->ne[2],
                                    a->ne[3]);
            }
            if (!ggml_can_repeat(b, a)) {
                std::string xName, yName;
                for (const auto& io : op.inputs) {
                    if (io.param == "X" && !io.args.empty()) xName = io.args[0];
                    if (io.param == "Y" && !io.args.empty()) yName = io.args[0];
                }
                std::fprintf(
                        stderr,
                        "[ew-mismatch] %s axis=%lld X=%s Y=%s a ne=%lld,%lld,"
                        "%lld,%lld | b ne=%lld,%lld,%lld,%lld\n",
                        op.type.c_str(), (long long)axis, xName.c_str(),
                        yName.c_str(), (long long)a->ne[0], (long long)a->ne[1],
                        (long long)a->ne[2], (long long)a->ne[3],
                        (long long)b->ne[0], (long long)b->ne[1],
                        (long long)b->ne[2], (long long)b->ne[3]);
                std::fflush(stderr);
            }
            result = op.type == "elementwise_add"
                             ? addChk(ctx, a, b)
                             : mulChk(ctx, a, b, "elementwise_mul");
        } else if (op.type == "relu") {
            if (islandOn) {
                IslandBase in = islandInput(op, "X");
                if (in.blocked) {
                    ggml_tensor* xb = aicore::blk::blocked_relu(ctx, in.t);
                    if (islandBind(op, xb, in.c)) break;
                    continue;
                }
            }
            result = ggml_relu(ctx, state.need(op, "X", "relu", error));
            if (!result) return nullptr;
        } else if (op.type == "sigmoid") {
            result = ggml_sigmoid(ctx, state.need(op, "X", "sigmoid", error));
            if (!result) return nullptr;
        } else if (op.type == "softmax") {
            ggml_tensor* x = state.need(op, "X", "softmax", error);
            if (!x) return nullptr;
            // ggml_soft_max requires contiguous storage; the attention
            // scores arrive as a transposed mul_mat view.
            if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
            result = ggml_soft_max(ctx, x);
            if (!result) return nullptr;
        } else if (op.type == "hard_swish") {
            ggml_tensor* x = state.need(op, "X", "hard_swish", error);
            if (!x) return nullptr;
            const double scale = attrFloat(op.attrs, "scale", 6.0);
            const double offset = attrFloat(op.attrs, "offset", 3.0);
            if (islandOn) {
                IslandBase in = islandInput(op, "X");
                if (in.blocked && scale == 6.0 && offset == 3.0) {
                    ggml_tensor* xb =
                            aicore::blk::blocked_hard_swish(ctx, in.t);
                    if (islandBind(op, xb, in.c)) break;
                    continue;
                }
                if (in.blocked) islandRebindOne(in.name);  // unsupported form
            }
            if (scale == 6.0 && offset == 3.0) {
                // Native single-node kernel (same piecewise formula);
                // measured BETTER high-resolution parity than the manual
                // chain (0.0012 vs 0.0139 max diff at 512x960 det) and
                // ~6 fewer graph nodes per activation.
                result = ggml_hardswish(ctx, x);
            } else {
                result = mulChk(
                        ctx, x,
                        ggml_div(
                                ctx,
                                ggml_clamp(
                                        ctx,
                                        ggml_relu(
                                                ctx,
                                                ggml_add(
                                                        ctx, x,
                                                        scalarF32(
                                                                cctx,
                                                                static_cast<
                                                                        float>(
                                                                        offset)))),
                                        0.0f, static_cast<float>(scale)),
                                scalarF32(cctx, static_cast<float>(scale))),
                        "hard_swish");
            }
        } else if (op.type == "hard_sigmoid") {
            ggml_tensor* x = state.need(op, "X", "hard_sigmoid", error);
            if (!x) return nullptr;
            const double slope = attrFloat(op.attrs, "slope", 0.2);
            const double offset = attrFloat(op.attrs, "offset", 0.5);
            if (islandOn) {
                IslandBase in = islandInput(op, "X");
                if (in.blocked) {
                    // any slope/offset: the island kernel takes them
                    ggml_tensor* xb = aicore::blk::blocked_hard_sigmoid(
                            ctx, in.t, static_cast<float>(slope),
                            static_cast<float>(offset));
                    if (islandBind(op, xb, in.c)) break;
                    continue;
                }
            }
            result = ggml_clamp(
                    ctx,
                    addChk(ctx,
                           mulChk(ctx, x,
                                  scalarF32(cctx, static_cast<float>(slope)),
                                  "hsigmoid_inner"),
                           scalarF32(cctx, static_cast<float>(offset))),
                    0.0f, 1.0f);
        } else if (op.type == "swish") {
            ggml_tensor* x = state.need(op, "X", "swish", error);
            if (!x) return nullptr;
            const double beta = attrFloat(op.attrs, "beta", 1.0);
            if (islandOn) {
                IslandBase in = islandInput(op, "X");
                if (in.blocked && beta == 1.0) {
                    ggml_tensor* xb = aicore::blk::blocked_silu(ctx, in.t);
                    if (islandBind(op, xb, in.c)) break;
                    continue;
                }
                if (in.blocked) islandRebindOne(in.name);  // unsupported form
            }
            if (beta == 1.0) {
                // swish(x, 1) == silu: native single-node kernel.
                result = ggml_silu(ctx, x);
            } else {
                result = mulChk(
                        ctx, x,
                        ggml_sigmoid(ctx,
                                     mulChk(ctx, x,
                                            scalarF32(cctx,
                                                      static_cast<float>(beta)),
                                            "swish_inner")),
                        "swish_outer");
            }
        } else if (op.type == "layer_norm") {
            ggml_tensor* x = state.need(op, "X", "layer_norm", error);
            if (!x) return nullptr;
            const double eps = attrFloat(op.attrs, "epsilon", 1e-5);
            result = ggml_norm(ctx, elemCont(ctx, x), static_cast<float>(eps));
            if (ggml_tensor* scaleT = state.byParam(op, "Scale"))
                result = mulChk(ctx, result, scaleT, "layer_norm_scale");
            if (ggml_tensor* biasT2 = state.byParam(op, "Bias"))
                result = addChk(ctx, result, biasT2);
        } else if (op.type == "matmul_v2") {
            ggml_tensor* a = state.byParam(op, "X");
            ggml_tensor* b = state.byParam(op, "Y");
            if (!a || !b) {
                if (error) *error = "matmul_v2: missing X/Y";
                return nullptr;
            }
            // ggml_mul_mat(A,B) requires the shared K dimension on ne[0]
            // of both operands, and its batch broadcast requires the
            // SECOND operand to carry the batch dims (src0's high dims
            // must divide src1's). The weight operand is therefore src0
            // and the activation (batched) operand is src1:
            //   Paddle X (M,K), ne={K,M,batch}; Y (K,N), ne={N,K}
            //   Wk = trans_y ? Y : Y^T  -> (K,N) on ne[0]
            //   out(j=m,i=n) = sum_k Wk(k,n) Ax(k,m) -> ne={M,N,batch}
            // which transposes back onto the contract layout
            // (batch,M,N) = ne={N,M,batch}, same as the N=1 path.
            ggml_tensor* Ax = a;
            ggml_tensor* Wk = b;
            if (attrBool(op.attrs, "trans_x", false))
                Ax = ggml_transpose(ctx, a);
            if (!attrBool(op.attrs, "trans_y", false))
                Wk = ggml_transpose(ctx, b);
            // The CPU mul_mat kernel needs row-contiguous operands.
            if (ggml_is_transposed(Wk) || !ggml_is_contiguous(Wk))
                Wk = ggml_cont(ctx, Wk);
            if (!ggml_is_contiguous(Ax)) Ax = ggml_cont(ctx, Ax);
            // ggml_mul_mat out ne = {src0.ne[1], src1.ne[1], batch...}:
            // with the weight on src0 (ne1 = N_out) the product lands
            // directly on the contract layout — no trailing transpose.
            result = ggml_mul_mat(ctx, Wk, Ax);
        } else if (op.type == "transpose2") {
            ggml_tensor* x = state.need(op, "X", "transpose2", error);
            if (!x) return nullptr;
            auto perm = attrInts(op.attrs, "axis");
            if (perm.empty()) perm = attrInts(op.attrs, "permute");
            if (perm.empty()) {
                // AnyLayout exports omit the attribute: the convention is
                // NCHW -> NHWC, generalized to ND as (0, 2, 1, 3, ...).
                // The numpy rank of the source disambiguates 3D heads.
                int rank = 4;
                for (const auto& io : op.inputs) {
                    if (io.param == "X" && !io.args.empty()) {
                        auto it = weights.program.varRanks.find(io.args[0]);
                        if (it != weights.program.varRanks.end())
                            rank = static_cast<int>(it->second);
                    }
                }
                if (rank == 3) {
                    perm = {0, 2, 1};
                } else {
                    perm = {0, 2, 3, 1};
                }
            }
            // Rank-5 qkv layouts (batch, T, 3, heads, dim) are folded to
            // four axes by dropping the size-1 batch axis on both sides:
            // remaining output axes keep their order, each surviving
            // source axis renumbers past the removed one.
            if (perm.size() == 5) {
                std::vector<int64_t> folded;
                for (int64_t p : perm)
                    if (p != 0) folded.push_back(p - 1);
                perm = folded;
            }
            int k[4];
            numpyPermToGgml(perm, k);
            result = ggml_permute(ctx, x, k[0], k[1], k[2], k[3]);
        } else if (op.type == "slice") {
            // Metadata constants flow through slices without a ggml node;
            // tensor inputs become strided views.
            std::string constKey;
            for (const auto& io : op.inputs)
                if (io.param == "Input" && !io.args.empty())
                    constKey = io.args[0];
            const auto starts = attrInts(op.attrs, "starts");
            const auto ends = attrInts(op.attrs, "ends");
            const auto axes = attrInts(op.attrs, "axes");
            const auto decrease = attrInts(op.attrs, "decrease_axis");
            auto constIt = state.constants.find(constKey);
            if (!constKey.empty() && constIt != state.constants.end()) {
                // Shape constants are 1D value vectors. Paddle slices the
                // value range v[s:e] on axis 0 (starts/ends address the
                // vector); decrease_axis drops the sliced (size-1) tensor
                // axis, which is already implied by the range extraction.
                // The old erase/decrease combination deleted entries
                // instead of selecting them and dropped every element
                // after the first decrease hit.
                const std::vector<int64_t>& values = constIt->second;
                const int64_t s =
                        starts.empty()
                                ? 0
                                : (starts[0] < 0 ? static_cast<int64_t>(
                                                           values.size()) +
                                                           starts[0]
                                                 : starts[0]);
                int64_t e = ends.empty()
                                    ? static_cast<int64_t>(values.size())
                                    : (ends[0] < 0 ? static_cast<int64_t>(
                                                             values.size()) +
                                                             ends[0]
                                                   : ends[0]);
                e = std::min(e, static_cast<int64_t>(values.size()));
                std::vector<int64_t> kept;
                if (s >= 0 && s <= e &&
                    s <= static_cast<int64_t>(values.size()))
                    kept.assign(values.begin() + s, values.begin() + e);
                state.constants[outName] = kept;
                isConstant = true;
                result = constantTensor(cctx, kept);
            } else {
                ggml_tensor* x = state.need(op, "Input", "slice", error);
                if (!x) return nullptr;
                int64_t ne[4] = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
                int64_t offset = 0;
                for (size_t i = 0; i < axes.size(); ++i) {
                    const int dim = 3 - static_cast<int>(axes[i]);
                    const int64_t size = x->ne[dim];
                    int64_t start = i < starts.size() ? starts[i] : 0;
                    int64_t end = i < ends.size() ? ends[i] : size;
                    if (start < 0) start += size;
                    if (end < 0) end += size;
                    end = std::min(end, size);
                    ne[dim] = end - start;
                    offset += start * x->nb[dim];
                }
                result = ggml_view_4d(ctx, x, ne[0], ne[1], ne[2], ne[3],
                                      x->nb[1], x->nb[2], x->nb[3], offset);
            }
        } else if (op.type == "reshape2" || op.type == "squeeze2" ||
                   op.type == "flatten_contiguous_range") {
            ggml_tensor* x = state.byParam(op, "X");
            if (!x) {
                if (error) *error = op.type + ": missing X";
                return nullptr;
            }
            std::vector<int64_t> shape;
            if (op.type == "reshape2") {
                shape = attrInts(op.attrs, "shape");
                int st = 0;
                for (const auto& io : op.inputs) {
                    if (io.param != "ShapeTensor") continue;
                    for (const auto& arg : io.args) {
                        auto it = state.constants.find(arg);
                        if (it != state.constants.end() && !it->second.empty())
                            shape[st] = it->second[0];
                        ++st;
                    }
                }
            } else if (op.type == "squeeze2") {
                // Collect in numpy axis order (numpy axis j lives at
                // ggml ne[3-j]); the shared reshape logic downstream
                // treats the shape vector as numpy-ordered.
                const auto axes = attrInts(op.attrs, "axes");
                for (int j = 0; j < 4; ++j) {
                    bool squeezed = false;
                    for (int64_t a : axes)
                        if (a == j) squeezed = true;
                    if (!squeezed || x->ne[3 - j] != 1)
                        shape.push_back(x->ne[3 - j]);
                }
                if (shape.empty()) shape.push_back(1);
            } else {
                // flatten_contiguous_range merges [axis, stop_axis] and
                // keeps the remaining axes. SVTR's x.flatten(2) maps
                // (1,C,H,1) to (1,C,H) - not a full flatten; collapsing
                // to (1,C*H) breaks the later transpose into layer_norm
                // (begin_norm_axis must land on the ggml fastest axis).
                int64_t first = attrInt(op.attrs, "start_axis", 0);
                int64_t last = attrInt(op.attrs, "stop_axis", first);
                if (first < 0) first += 4;
                if (last < 0) last += 4;
                last = std::max(first, last);
                std::vector<int64_t> dims;
                for (int d = 0; d < 4; ++d) dims.push_back(x->ne[3 - d]);
                shape.clear();
                int64_t merged = 1;
                for (int i = 0; i < 4; ++i)
                    if (i >= first && i <= last) merged *= dims[i];
                bool mergedPushed = false;
                for (int i = 0; i < 4; ++i) {
                    if (i < first || i > last) {
                        shape.push_back(dims[i]);
                    } else if (!mergedPushed) {
                        shape.push_back(merged);
                        mergedPushed = true;
                    }
                }
            }
            int64_t total = 1;
            for (int d = 0; d < 4; ++d) total *= x->ne[d];
            const int numpyRank = static_cast<int>(shape.size());
            std::vector<int64_t> numpyDims(numpyRank, 1);
            for (int i = 0; i < numpyRank; ++i)
                numpyDims[i] =
                        shape[i] == 0 ? (i < 4 ? x->ne[3 - i] : 1) : shape[i];
            int64_t stated = 1;
            int inferAt = -1;
            for (int i = 0; i < numpyRank; ++i) {
                // 0 is Paddle's dynamic-batch placeholder (shape_*.slice_1
                // feeds the runtime batch): infer it from the element count
                // exactly like -1, instead of assuming the batch lives on
                // ggml ne[3] (the conv-flatten branch carries it on ne[2]).
                if (shape[i] == -1 || shape[i] == 0)
                    inferAt = i;
                else
                    stated *= numpyDims[i];
            }
            if (inferAt >= 0) numpyDims[inferAt] = total / stated;
            std::vector<int64_t> kept;
            for (int i = 0; i < numpyRank; ++i)
                if (numpyDims[i] != 1 || kept.size() < 4)
                    kept.push_back(numpyDims[i]);
            while (kept.size() > 4) kept.erase(kept.begin());
            while (kept.size() < 4) kept.insert(kept.begin(), 1);
            int64_t ne[4];
            for (int d = 0; d < 4; ++d) ne[d] = kept[3 - d];
            int64_t product = 1;
            for (int d = 0; d < 4; ++d) product *= ne[d];
            if (product != total) {
                std::fprintf(stderr, "[reshape-mismatch] %s out=%s shape=",
                             op.type.c_str(), outName.c_str());
                for (int64_t v : shape)
                    std::fprintf(stderr, "%lld,", (long long)v);
                std::fprintf(stderr, " x ne=%lld,%lld,%lld,%lld\n",
                             (long long)x->ne[0], (long long)x->ne[1],
                             (long long)x->ne[2], (long long)x->ne[3]);
                std::fflush(stderr);
                if (error) *error = "reshape: element count mismatch";
                return nullptr;
            }
            // ggml_reshape requires contiguous storage; a permuted view
            // (e.g. the attention output transpose feeding the projector
            // reshape) must be materialized first.
            if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
            result = ggml_reshape_4d(ctx, x, ne[0], ne[1], ne[2], ne[3]);
        } else if (op.type == "concat") {
            const int64_t axis = attrInt(op.attrs, "axis", 1);
            ggml_tensor* acc = nullptr;
            for (const auto& io : op.inputs) {
                if (io.param != "X") continue;
                for (const auto& arg : io.args) {
                    ggml_tensor* t = state.resolve(arg);
                    if (!t) continue;
                    if (!acc) {
                        acc = t;
                        continue;
                    }
                    acc = ggml_concat(ctx, acc, t, 3 - static_cast<int>(axis));
                }
            }
            result = acc;
        } else if (op.type == "pool2d") {
            ggml_tensor* x = state.byParam(op, "X");
            if (!x) {
                if (error) *error = "pool2d: missing X";
                return nullptr;
            }
            const auto ksize = attrInts(op.attrs, "ksize");
            const auto strides = attrInts(op.attrs, "strides");
            const auto paddings = attrInts(op.attrs, "paddings");
            const bool adaptive = attrBool(op.attrs, "adaptive", false);
            const bool global = attrBool(op.attrs, "global_pooling", false);
            const std::string poolType = attrString(op.attrs, "pooling_type");
            if (islandOn) {
                IslandBase in = islandInput(op, "X");
                if (in.blocked &&
                    (global || (adaptive && ksize.size() == 2 &&
                                ksize[0] == 1 && ksize[1] == 1)) &&
                    poolType == "avg") {
                    // SE squeeze: global average pool over the blocked body.
                    ggml_tensor* xb = aicore::blk::blocked_gap(ctx, in.t);
                    if (islandBind(op, xb, in.c)) break;
                    continue;
                }
                if (in.blocked) islandRebindOne(in.name);  // non-global pool
            }
            const ggml_op_pool pool =
                    poolType == "avg" ? GGML_OP_POOL_AVG : GGML_OP_POOL_MAX;
            const int k0 =
                    (global || adaptive)
                            ? static_cast<int>(x->ne[0])
                            : static_cast<int>(ksize.size() > 1 ? ksize[1] : 1);
            const int k1 =
                    (global || adaptive)
                            ? static_cast<int>(x->ne[1])
                            : static_cast<int>(ksize.empty() ? 1 : ksize[0]);
            const int s0 =
                    adaptive ? k0
                             : static_cast<int>(strides.size() > 1 ? strides[1]
                                                                   : 1);
            const int s1 = adaptive ? k1
                                    : static_cast<int>(
                                              strides.empty() ? 1 : strides[0]);
            const int p0 =
                    paddings.size() > 1 ? static_cast<int>(paddings[1]) : 0;
            const int p1 = paddings.empty() ? 0 : static_cast<int>(paddings[0]);
            result = ggml_pool_2d(ctx, x, pool, k0, k1, s0, s1, p0, p1);
        } else if (op.type == "scale") {
            ggml_tensor* x = state.need(op, "X", "scale", error);
            if (!x) return nullptr;
            const double scaleV = attrFloat(op.attrs, "scale", 1.0);
            const double biasV = attrFloat(op.attrs, "bias", 0.0);
            result = mulChk(ctx, x, scalarF32(cctx, static_cast<float>(scaleV)),
                            "scale");
            if (biasV != 0.0)
                result = addChk(ctx, result,
                                scalarF32(cctx, static_cast<float>(biasV)));
        } else if (op.type == "dropout") {
            // is_test inference graphs: identity.
            result = state.byParam(op, "X");
            if (!result) {
                if (error) *error = "dropout: missing X";
                return nullptr;
            }
        } else if (op.type == "shape") {
            ggml_tensor* x = state.byParam(op, "Input");
            if (!x) x = state.byParam(op, "X");
            if (!x) {
                if (error) *error = "shape: missing X";
                return nullptr;
            }
            std::vector<int64_t> dims;
            // Paddle reports every logical axis; the logical rank comes
            // from the exported VarDesc ranks because a ggml shape alone
            // cannot distinguish real size-1 axes from layout padding
            // (e.g. layer_norm's (N,T,C) as ne={C,T,1,1} versus swish_12's
            // true (N,120,1,T)).
            int rank = 4;
            auto rankIt = weights.program.varRanks.find(
                    op.inputs.front().args.front());
            for (const auto& io : op.inputs) {
                if (io.param == "Input" && !io.args.empty()) {
                    rankIt = weights.program.varRanks.find(io.args[0]);
                    break;
                }
            }
            if (rankIt != weights.program.varRanks.end() &&
                rankIt->second >= 1 && rankIt->second <= 4)
                rank = rankIt->second;
            for (int i = 0; i < rank; ++i) dims.push_back(x->ne[rank - 1 - i]);
            state.constants[outName] = dims;
            isConstant = true;
            result = constantTensor(cctx, dims);
        } else if (op.type == "fill_constant") {
            const std::string strValue = attrString(op.attrs, "str_value");
            const int64_t value = strValue.empty()
                                          ? static_cast<int64_t>(attrFloat(
                                                    op.attrs, "value", 0.0))
                                          : std::atoll(strValue.c_str());
            state.constants[outName] = {value};
            isConstant = true;
            result = constantTensor(cctx, {value});
        } else {
            if (error) *error = "unsupported op: " + op.type;
            return nullptr;
        }

        if (!result) {
            if (error) *error = "op produced no tensor: " + op.type;
            return nullptr;
        }
        if (!isConstant)
            for (const auto& name : op.outputs) state.values[name] = result;
        if (dumpShapes && !isConstant && !op.outputs.empty()) {
            std::fprintf(stderr, "[shape] [%s] %s ne=%lld,%lld,%lld,%lld\n",
                         op.type.c_str(), op.outputs[0].c_str(),
                         (long long)result->ne[0], (long long)result->ne[1],
                         (long long)result->ne[2], (long long)result->ne[3]);
            std::fflush(stderr);
        }
        if (probeIndex == fetchAt && !isConstant) {
            // Parity bisection: truncate the program here and expose this
            // activation as the fetch tensor.
            state.values["__fetch__"] = result;
            ++probeIndex;
            break;
        }
        ++probeIndex;
        for (const auto& name : op.outputs)
            if (name.find("scale_0.tmp_0") != std::string::npos)
                state.values["__fetch__"] = result;
    }

    auto it = state.values.find("__fetch__");
    if (it == state.values.end()) {
        if (error) *error = "program produced no fetch tensor";
        return nullptr;
    }
    return it->second;
}

std::string ctcGreedyDecode(const float* probs,
                            int timeSteps,
                            int vocab,
                            const std::vector<std::string>& dictionary,
                            float* meanScore) {
    std::string text;
    double scoreSum = 0.0;
    int scoreCount = 0;
    int previous = 0;
    // Class-index layout over the rec head (PP-OCR CTC): index 0 is the
    // blank and the remaining classes follow one of two published
    // container conventions:
    //   vocab == dict.size() + 2 -> 1..N = dictionary, N+1 = " " (the raw
    //          ppocr_keys_v1 vocabulary; use_space_char appends a trailing
    //          space class at export time),
    //   vocab == dict.size() + 1 -> 1..N = dictionary (the stored table
    //          already carries its own trailing " " entry).
    // The trailing space class is real recognized text, not a separator:
    // unconditionally skipping vocab-1 deleted every recognized space (and
    // the last vocabulary character whenever the container carries no
    // space class).
    const int64_t dictSize = static_cast<int64_t>(dictionary.size());
    const bool spaceTail = static_cast<int64_t>(vocab) - dictSize == 2;
    for (int t = 0; t < timeSteps; ++t) {
        const float* row = probs + static_cast<size_t>(t) * vocab;
        float bestProb = 0.0f;
        const int best = ctcArgmax(row, vocab, &bestProb);
        // 0 = CTC blank; collapse repeats, then map the class through the
        // layout above.
        if (best != 0 && best != previous) {
            if (best == vocab - 1 && spaceTail) {
                text += " ";
                scoreSum += bestProb;
                ++scoreCount;
            } else {
                const int dictIndex = best - 1;
                if (dictIndex >= 0 && dictIndex < static_cast<int>(dictSize)) {
                    text += dictionary[dictIndex];
                    scoreSum += bestProb;
                    ++scoreCount;
                }
            }
        }
        previous = best;
    }
    if (meanScore)
        *meanScore =
                scoreCount ? static_cast<float>(scoreSum / scoreCount) : 0.0f;
    return text;
}

std::vector<float> preprocessRec(const unsigned char* rgb,
                                 int width,
                                 int height,
                                 int targetWidth,
                                 int* outW) {
    // PP-OCRv4 rec: resize to height 48 (aspect preserved), normalize with
    // the PP-OCR ImageNet statistics, NCHW float output.
    constexpr int kHeight = 48;
    int w = std::max(1, static_cast<int>(1LL * width * kHeight / height));
    if (targetWidth > 0) w = std::min(w, targetWidth);
    if (outW) *outW = w;

    std::vector<float> out(3ULL * kHeight * w);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};
    for (int y = 0; y < kHeight; ++y) {
        const int sy = std::min(height - 1, y * height / kHeight);
        for (int x = 0; x < w; ++x) {
            const int sx = std::min(width - 1, x * width / std::max(1, w));
            const unsigned char* pixel =
                    rgb + (static_cast<size_t>(sy) * width + sx) * 3;
            for (int c = 0; c < 3; ++c)
                out[(static_cast<size_t>(c) * kHeight + y) * w + x] =
                        (pixel[c] / 255.0f - mean[c]) / stdv[c];
        }
    }
    return out;
}

}  // namespace ocr
}  // namespace aicore

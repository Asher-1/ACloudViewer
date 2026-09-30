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
#include <set>
#include <sstream>

#include "common/winograd.hpp"

namespace aicore {
namespace ocr {

namespace {

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
    if (foldBuf) {
        ggml_backend_buffer_free(foldBuf);
        foldBuf = nullptr;
    }
    if (foldCtx) ggml_free(foldCtx);
    foldCtx = ggml_init({96 * 1024 * 1024, nullptr, /*no_alloc=*/false});
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
    if (constCtx) {
        hostConstBuf = ggml_backend_cpu_buffer_from_ptr(
                ggml_get_mem_buffer(constCtx), ggml_get_mem_size(constCtx));
        if (!hostConstBuf) return false;
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

    for (const GraphOp& op : weights.program.ops) {
        const std::string outName =
                op.outputs.empty() ? std::string() : op.outputs.front();
        ggml_tensor* result = nullptr;
        bool isConstant = false;

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
                } else {
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
                if (x->ne[3] == 1) {
                    // Footprint guard: the batched GEMM materializes a
                    // K*OW*OH*C im2col. At det resolutions (960x512) the
                    // high-C layers reach multiple GB and starve the
                    // allocator, so fall back to the per-channel chain
                    // (whose per-channel im2col stays K*OW*OH) whenever
                    // the batched buffer would exceed the cap.
                    const auto convOut = [](int64_t in, int64_t k, int s, int p,
                                            int d) {
                        return (in + 2 * p - d * (k - 1) - 1) / s + 1;
                    };
                    const int64_t dwOW =
                            convOut(x->ne[0], w->ne[0], s0, p0, d0);
                    const int64_t dwOH =
                            convOut(x->ne[1], w->ne[1], s1, p1, d1);
                    const int64_t im2colBytes = w->ne[0] * w->ne[1] * dwOW *
                                                dwOH * x->ne[2] * sizeof(float);
                    constexpr int64_t kMaxIm2colBytes = 256ll << 20;
                    if (im2colBytes > kMaxIm2colBytes) {
                        std::vector<ggml_tensor*> parts;
                        for (int64_t c = 0; c < x->ne[2]; ++c) {
                            ggml_tensor* xc = ggml_view_4d(
                                    ctx, x, x->ne[0], x->ne[1], 1, x->ne[3],
                                    x->nb[1], x->nb[2], x->nb[3], c * x->nb[2]);
                            ggml_tensor* wc = ggml_view_4d(
                                    ctx, w, w->ne[0], w->ne[1], 1, 1, w->nb[1],
                                    w->nb[2], w->nb[3], c * w->nb[3]);
                            parts.push_back(ggml_conv_2d(ctx, wc, xc, s0, s1,
                                                         p0, p1, d0, d1));
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
                        ggml_tensor* xb = ggml_reshape_4d(
                                ctx, xc, xc->ne[0], xc->ne[1], 1, xc->ne[2]);
                        // im2col reads only the kernel geometry (KW, KH, Cin=1)
                        // from this operand; a channel-0 view keeps the
                        // per-channel weights intact for the GEMM below.
                        ggml_tensor* sv =
                                ggml_view_4d(ctx, w, w->ne[0], w->ne[1], 1, 1,
                                             w->nb[1], w->nb[2], w->nb[3], 0);
                        ggml_tensor* cols =
                                ggml_im2col(ctx, sv, xb, s0, s1, p0, p1, d0, d1,
                                            true, GGML_TYPE_F32);
                        ggml_tensor* wr = ggml_reshape_3d(
                                ctx, w, w->ne[0] * w->ne[1], 1, w->ne[3]);
                        result = ggml_mul_mat(
                                ctx, wr,
                                ggml_reshape_3d(ctx, cols, cols->ne[0],
                                                cols->ne[1] * cols->ne[2],
                                                cols->ne[3]));
                        result = ggml_reshape_4d(ctx, result, cols->ne[1],
                                                 cols->ne[2], x->ne[2], 1);
                    }
                } else {
                    // Batched input falls back to the per-channel chain.
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
            // of both operands and yields ne={A->ne[1], B->ne[1], ...}.
            // Paddle stores X (M,K) - K already on ne[0] - and Y (K,N),
            // which must be transposed to bring K onto ne[0]. The product
            // comes out as (M,N) in ggml axes while the layout contract
            // stores paddle (batch,M,N) as ne={N,M,batch}, so the result
            // is transposed back onto the contract layout.
            ggml_tensor* A = a;
            ggml_tensor* B = b;
            if (attrBool(op.attrs, "trans_x", false))
                A = ggml_transpose(ctx, a);
            if (!attrBool(op.attrs, "trans_y", false))
                B = ggml_transpose(ctx, b);
            // ggml_mul_mat rejects transposed A operands; a permuted X
            // (e.g. the final head transpose feeding the classifier
            // matmul) must be materialized first. Transposed B is the
            // supported weight path, but the CPU kernel still needs
            // row-contiguous src1, so a permuted B is materialized too.
            if (ggml_is_transposed(A)) A = ggml_cont(ctx, A);
            if (!ggml_is_contiguous(B)) B = ggml_cont(ctx, B);
            result = ggml_transpose(ctx, ggml_mul_mat(ctx, A, B));
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
                if (shape[i] == -1)
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
    for (int t = 0; t < timeSteps; ++t) {
        const float* row = probs + static_cast<size_t>(t) * vocab;
        int best = 0;
        float bestProb = row[0];
        for (int i = 1; i < vocab; ++i) {
            if (row[i] > bestProb) {
                bestProb = row[i];
                best = i;
            }
        }
        // 0 = CTC blank; vocab-1 = the word separator.
        if (best != 0 && best != vocab - 1 && best != previous) {
            const int dictIndex = best - 1;
            if (dictIndex >= 0 &&
                dictIndex < static_cast<int>(dictionary.size()))
                text += dictionary[dictIndex];
            scoreSum += bestProb;
            ++scoreCount;
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

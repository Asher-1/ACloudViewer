// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/facedetect/model_loader.hpp"

#include <cstring>
#include <utility>
#include <vector>

#include "common/ggml_backend_utils.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "tasks/facedetect/backend.hpp"
#include "tasks/facedetect/common.hpp"
namespace fd {

// --- KV helpers (tolerant: return the default when a key is absent) ----------
static uint32_t kv_u32(gguf_context* g, const char* k, uint32_t d = 0) {
    int64_t id = gguf_find_key(g, k);
    return id < 0 ? d : (uint32_t)gguf_get_val_u32(g, id);
}
static float kv_f32(gguf_context* g, const char* k, float d = 0) {
    int64_t id = gguf_find_key(g, k);
    return id < 0 ? d : gguf_get_val_f32(g, id);
}
static bool kv_bool(gguf_context* g, const char* k, bool d = false) {
    int64_t id = gguf_find_key(g, k);
    return id < 0 ? d : gguf_get_val_bool(g, id);
}
static std::string kv_str(gguf_context* g, const char* k, const char* d = "") {
    int64_t id = gguf_find_key(g, k);
    return id < 0 ? std::string(d) : std::string(gguf_get_val_str(g, id));
}
static std::vector<int32_t> kv_i32_arr(gguf_context* g, const char* k) {
    std::vector<int32_t> out;
    int64_t id = gguf_find_key(g, k);
    if (id >= 0 && gguf_get_arr_type(g, id) == GGUF_TYPE_INT32) {
        size_t n = gguf_get_arr_n(g, id);
        const int32_t* a = (const int32_t*)gguf_get_arr_data(g, id);
        out.assign(a, a + n);
    }
    return out;
}
static std::vector<float> kv_f32_arr(gguf_context* g, const char* k) {
    std::vector<float> out;
    int64_t id = gguf_find_key(g, k);
    if (id >= 0 && gguf_get_arr_type(g, id) == GGUF_TYPE_FLOAT32) {
        size_t n = gguf_get_arr_n(g, id);
        const float* a = (const float*)gguf_get_arr_data(g, id);
        out.assign(a, a + n);
    }
    return out;
}
static std::vector<std::string> kv_str_arr(gguf_context* g, const char* k) {
    std::vector<std::string> out;
    int64_t id = gguf_find_key(g, k);
    if (id >= 0 && gguf_get_arr_type(g, id) == GGUF_TYPE_STRING) {
        size_t n = gguf_get_arr_n(g, id);
        for (size_t i = 0; i < n; ++i)
            out.emplace_back(gguf_get_arr_str(g, id, i));
    }
    return out;
}

ModelLoader::~ModelLoader() {
    if (weights_buf_) {
        // Purge any persistent graph-cache entries that reference these weights
        // BEFORE the buffer is freed, so a later model reallocating this
        // address cannot false-hit a stale cached graph (multi-model hosting
        // safety).
        invalidate_graph_cache_for_weights(weights_buf_);
        ggml_backend_buffer_free(weights_buf_);
    }
    if (device_ctx_) ggml_free(device_ctx_);
    if (gguf_) gguf_free(gguf_);
    if (ctx_) ggml_free(ctx_);
}

bool ModelLoader::realize_weights(ggml_backend_t backend) {
    if (weights_buf_) return true;  // idempotent
    if (!backend || !ctx_) {
        FD_LOG("realize_weights: null backend/ctx");
        return false;
    }

    if (ggml_common::is_cpu_backend(backend)) {
        // Fast path: borrow the host ctx memory directly (no copy). The GGUF is
        // loaded no_alloc=false, so every tensor's data lives in one contiguous
        // ctx mem_buffer; wrap it as a CPU backend buffer and repoint every
        // tensor's ->buffer at it so graphs reference the loader tensors
        // directly.
        void* base = ggml_get_mem_buffer(ctx_);
        size_t size = ggml_get_mem_size(ctx_);
        weights_buf_ = ggml_backend_cpu_buffer_from_ptr(base, size);
        if (!weights_buf_) {
            FD_LOG("realize_weights: buffer_from_ptr failed");
            return false;
        }
        for (auto& kv : tensors_) kv.second->buffer = weights_buf_;
        return true;
    }

    // Device path: mirror every weight into a no_alloc ctx, allocate THAT on
    // the backend, upload each tensor's bytes, and repoint the name->tensor
    // map.
    const size_t n = tensors_.size();
    struct ggml_init_params dp = {
            /*.mem_size  =*/ggml_tensor_overhead() * (n + 8),
            /*.mem_buffer=*/nullptr,
            /*.no_alloc  =*/true,
    };
    device_ctx_ = ggml_init(dp);
    if (!device_ctx_) {
        FD_LOG("realize_weights: device ctx init failed");
        return false;
    }

    std::vector<std::pair<ggml_tensor*, const void*>> ups;
    ups.reserve(n);
    std::unordered_map<std::string, ggml_tensor*> devmap;
    devmap.reserve(n);
    for (auto& kv : tensors_) {
        ggml_tensor* s = kv.second;
        ggml_tensor* d =
                ggml_new_tensor(device_ctx_, s->type, GGML_MAX_DIMS, s->ne);
        ggml_set_name(d, kv.first.c_str());
        devmap.emplace(kv.first, d);
        ups.emplace_back(d, s->data);
    }
    weights_buf_ = ggml_backend_alloc_ctx_tensors(device_ctx_, backend);
    if (!weights_buf_) {
        FD_LOG("realize_weights: alloc_ctx_tensors failed");
        return false;
    }
    for (auto& pr : ups)
        ggml_backend_tensor_set(pr.first, pr.second, 0, ggml_nbytes(pr.first));
    tensors_.swap(devmap);
    return true;
}

bool ModelLoader::load(const std::string& path) {
    struct gguf_init_params p { /*no_alloc*/
        false, /*ctx*/ &ctx_
    };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if (!gguf_) {
        FD_LOG("gguf open failed: %s", path.c_str());
        return false;
    }

    cfg_.arch = kv_str(gguf_, "facedetect.arch", "scrfd+arcface");
    // detector
    cfg_.detector = kv_str(gguf_, "facedetect.detector.kind", "scrfd");
    cfg_.det_input_size = kv_u32(gguf_, "facedetect.detector.input_size", 640);
    cfg_.det_strides = kv_i32_arr(gguf_, "facedetect.detector.strides");
    if (cfg_.det_strides.empty()) cfg_.det_strides = {8, 16, 32};
    cfg_.det_num_anchors = kv_u32(gguf_, "facedetect.detector.num_anchors", 2);
    cfg_.det_score_thresh =
            kv_f32(gguf_, "facedetect.detector.score_thresh", 0.5f);
    cfg_.det_nms_thresh = kv_f32(gguf_, "facedetect.detector.nms_thresh", 0.4f);
    // Optional embedded detector topology (buffalo_m det_2.5g / buffalo_s
    // det_500m). Absent for buffalo_l (hand-mapped det_10g path).
    cfg_.det_graph = kv_str_arr(gguf_, "facedetect.detector.graph");
    cfg_.det_graph_input =
            kv_str(gguf_, "facedetect.detector.input", "input.1");
    cfg_.det_graph_outputs = kv_str_arr(gguf_, "facedetect.detector.outputs");
    // recognizer
    cfg_.recognizer = kv_str(gguf_, "facedetect.recognizer.kind", "arcface");
    cfg_.rec_input_size =
            kv_u32(gguf_, "facedetect.recognizer.input_size", 112);
    cfg_.embed_dim = kv_u32(gguf_, "facedetect.recognizer.embed_dim", 512);
    cfg_.verify_threshold =
            kv_f32(gguf_, "facedetect.recognizer.verify_threshold", 0.35f);
    // Optional MobileFaceNet (w600k_mbf) recognizer topology; absent for the
    // IResNet50 packs (buffalo_l/buffalo_m), which use the hand-mapped graph.
    cfg_.rec_graph = kv_str_arr(gguf_, "facedetect.recognizer.graph");
    cfg_.rec_graph_input =
            kv_str(gguf_, "facedetect.recognizer.input", "input");
    cfg_.rec_graph_output =
            kv_str(gguf_, "facedetect.recognizer.output", "output");
    // genderage
    cfg_.genderage_present =
            kv_bool(gguf_, "facedetect.genderage.present", false);
    cfg_.genderage_input_size =
            kv_u32(gguf_, "facedetect.genderage.input_size", 96);
    // anti-spoof (MiniFASNet ensemble)
    cfg_.antispoof_present =
            kv_bool(gguf_, "facedetect.antispoof.present", false);
    cfg_.antispoof_input_size =
            kv_u32(gguf_, "facedetect.antispoof.input_size", 80);
    cfg_.antispoof_scales = kv_f32_arr(gguf_, "facedetect.antispoof.scales");
    // Per-model MiniFASNet graph topology (one entry per scale). The
    // interpreter in antispoof_graph.cpp replays these; absent keys leave the
    // graph empty so a pack without anti-spoof still loads.
    for (size_t i = 0; i < cfg_.antispoof_scales.size(); ++i) {
        const std::string gk =
                "facedetect.antispoof." + std::to_string(i) + ".graph";
        const std::string ok =
                "facedetect.antispoof." + std::to_string(i) + ".output";
        cfg_.antispoof_graphs.push_back(kv_str_arr(gguf_, gk.c_str()));
        cfg_.antispoof_graph_outputs.push_back(
                kv_str(gguf_, ok.c_str(), "output"));
    }
    // Dense-landmark heads (engine-only; no LocalAI RPC consumes these yet).
    auto load_landmark = [&](const char* tag, FaceConfig::LandmarkHead& h) {
        const std::string base =
                std::string("facedetect.landmark.") + tag + ".";
        h.present = kv_bool(gguf_, (base + "present").c_str(), false);
        if (!h.present) return;
        h.input_size = kv_u32(gguf_, (base + "input_size").c_str(), 192);
        h.num_points = kv_u32(gguf_, (base + "num_points").c_str(), 0);
        h.dim = kv_u32(gguf_, (base + "dim").c_str(), 0);
        h.input_mean = kv_f32(gguf_, (base + "input_mean").c_str(), 0.0f);
        h.input_std = kv_f32(gguf_, (base + "input_std").c_str(), 1.0f);
        h.graph = kv_str_arr(gguf_, (base + "graph").c_str());
        h.graph_input = kv_str(gguf_, (base + "input").c_str(), "data");
        h.graph_output = kv_str(gguf_, (base + "output").c_str(), "fc1");
    };
    load_landmark("2d", cfg_.landmark_2d);
    load_landmark("3d", cfg_.landmark_3d);

    // tensors (verbatim names from the source ONNX/state_dict)
    const int64_t nt = gguf_get_n_tensors(gguf_);
    for (int64_t i = 0; i < nt; ++i) {
        const char* nm = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* t = ggml_get_tensor(ctx_, nm);
        if (t) tensors_[nm] = t;
    }
    // A valid pack must at least name an architecture and an embedding dim.
    return !cfg_.arch.empty() && cfg_.embed_dim > 0;
}

ggml_tensor* ModelLoader::tensor(const std::string& n) const {
    auto it = tensors_.find(n);
    return it == tensors_.end() ? nullptr : it->second;
}
}  // namespace fd

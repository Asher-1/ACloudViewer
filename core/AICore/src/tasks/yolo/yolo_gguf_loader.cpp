// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/yolo/yolo_gguf_loader.hpp"

#include <cstring>

#include "gguf.h"

namespace yolo {

namespace {

int64_t key_or(const gguf_context* g, const char* key, int64_t def) {
    int64_t id = gguf_find_key(g, key);
    return id >= 0 ? (int64_t)gguf_get_val_u32(g, id) : def;
}

std::string str_or(const gguf_context* g, const char* key, const char* def) {
    int64_t id = gguf_find_key(g, key);
    return id >= 0 ? gguf_get_val_str(g, id) : def;
}

// Read an i32/u32/f32 KV scalar into int64 depending on stored type.
int64_t scalar_kv(const gguf_context* g, int64_t id) {
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32:
            return (int64_t)gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:
            return (int64_t)gguf_get_val_i32(g, id);
        case GGUF_TYPE_FLOAT32:
            return (int64_t)gguf_get_val_f32(g, id);  // truncated
        default:
            return 0;
    }
}

static void parse_meta(const gguf_context* g, ModelMeta& meta) {
    meta.name = str_or(g, "general.name", "yolo");
    meta.task = str_or(g, "yolo.task", "detect");
    meta.dtype = str_or(g, "yolo.dtype", "?");
    meta.text_model = str_or(g, "yolo.text_model", "");
    meta.nc = (int)key_or(g, "yolo.nc", 80);
    meta.nm = (int)key_or(g, "yolo.nm", 0);
    meta.nk = (int)key_or(g, "yolo.nk", 0);
    meta.ne = (int)key_or(g, "yolo.ne", 0);
    meta.nl = (int)key_or(g, "yolo.nl", 3);
    meta.imgsz = (int)key_or(g, "yolo.imgsz", 640);

    if (int64_t id = gguf_find_key(g, "yolo.kpt_shape");
        id >= 0 && gguf_get_arr_n(g, id) >= 2) {
        const uint32_t* p = (const uint32_t*)gguf_get_arr_data(g, id);
        meta.kpt_ndim = (int)p[1];
    }

    if (int64_t id = gguf_find_key(g, "yolo.strides"); id >= 0) {
        size_t n = gguf_get_arr_n(g, id);
        const uint8_t* p = (const uint8_t*)gguf_get_arr_data(g, id);
        for (size_t i = 0; i < n; i++)
            meta.strides.push_back(((const float*)p)[i]);
    }
    if (int64_t id = gguf_find_key(g, "yolo.class_names"); id >= 0) {
        size_t n = gguf_get_arr_n(g, id);
        for (size_t i = 0; i < n; i++)
            meta.class_names.emplace_back(gguf_get_arr_str(g, id, i));
    }
    meta.has_text_input = key_or(g, "yolo.world", 0) != 0;
    if (meta.strides.empty()) {
        for (int i = 0; i < meta.nl; i++) meta.strides.push_back(float(8 << i));
    }
}

}  // namespace

ModelMeta read_gguf_meta(const std::string& path) {
    gguf_init_params ip{};  // no_alloc: header only, no tensor mapping
    gguf_context* g = gguf_init_from_file(path.c_str(), ip);
    if (!g) {
        YOLO_LOG_ERROR("failed to open GGUF: %s", path.c_str());
        return {};
    }
    ModelMeta meta;
    parse_meta(g, meta);
    gguf_free(g);
    return meta;
}

std::unique_ptr<ModelDef> load_gguf(const std::string& path) {
    ggml_context* weight_ctx = nullptr;
    gguf_init_params ip{};
    ip.no_alloc = false;  // map tensor data directly
    ip.ctx = &weight_ctx;

    gguf_context* g = gguf_init_from_file(path.c_str(), ip);
    if (!g) {
        YOLO_LOG_ERROR("failed to open GGUF: %s", path.c_str());
        return nullptr;
    }

    auto model = std::make_unique<ModelDef>();
    model->gguf_path = path;

    // ---- metadata ----
    parse_meta(g, model->meta);
    const int64_t graph_version = key_or(g, "yolo.op_graph_version", 0);
    // v1/v2: closed-set detect/segment/depth. v3: YOLO-World text-conditioned
    // heads. v4: YOLOE (reprta residual rides on the detect op).
    if (graph_version < 1 || graph_version > 4) {
        YOLO_LOG_ERROR("unsupported yolo.op_graph_version: %lld",
                       (long long)graph_version);
        gguf_free(g);
        ggml_free(weight_ctx);
        return nullptr;
    }

    // ---- op graph ----
    const int64_t n_ops = key_or(g, "yolo.op.count", 0);
    if (n_ops <= 0 || n_ops > 10000) {
        YOLO_LOG_ERROR("invalid yolo.op.count: %lld", (long long)n_ops);
        gguf_free(g);
        ggml_free(weight_ctx);
        return nullptr;
    }
    model->ops.resize(n_ops);
    for (int64_t i = 0; i < n_ops; i++) {
        OpDef& op = model->ops[i];
        std::string prefix = "op." + std::to_string(i);
        op.type = str_or(g, (prefix + ".type").c_str(), "");
        if (op.type.empty()) {
            YOLO_LOG_ERROR("missing op type at index %lld", (long long)i);
            gguf_free(g);
            ggml_free(weight_ctx);
            return nullptr;
        }

        if (int64_t id = gguf_find_key(g, (prefix + ".inputs").c_str());
            id >= 0) {
            size_t n = gguf_get_arr_n(g, id);
            const int32_t* p = (const int32_t*)gguf_get_arr_data(g, id);
            op.inputs.assign(p, p + n);
            for (int input : op.inputs) {
                if (input < -1 || input >= i) {
                    YOLO_LOG_ERROR("invalid input %d for op %lld", input,
                                   (long long)i);
                    gguf_free(g);
                    ggml_free(weight_ctx);
                    return nullptr;
                }
            }
        }

        // Scan all keys under this op prefix for params / tensor references.
        const int64_t n_kv = gguf_get_n_kv(g);
        for (int64_t kv = 0; kv < n_kv; kv++) {
            const char* k = gguf_get_key(g, kv);
            if (strncmp(k, prefix.c_str(), prefix.size()) != 0 ||
                k[prefix.size()] != '.')
                continue;
            const char* field = k + prefix.size() + 1;
            if (!strcmp(field, "type") || !strcmp(field, "inputs")) continue;
            switch (gguf_get_kv_type(g, kv)) {
                case GGUF_TYPE_STRING:
                    op.sparams[field] = gguf_get_val_str(g, kv);
                    break;
                case GGUF_TYPE_ARRAY: {
                    size_t n = gguf_get_arr_n(g, kv);
                    auto& vec = op.aparams[field];
                    vec.resize(n);
                    const uint8_t* p = (const uint8_t*)gguf_get_arr_data(g, kv);
                    for (size_t j = 0; j < n; j++)
                        vec[j] = ((const uint32_t*)p)[j];
                    break;
                }
                case GGUF_TYPE_FLOAT32: {
                    float f = gguf_get_val_f32(g, kv);
                    if (f == (int64_t)f &&
                        (field[0] == 'p' || field[0] == 's' || field[0] == 'd'))
                        op.iparams[field] = (int64_t)
                                f;  // pad/stride/dilation stored as float
                    else
                        op.fparams[field] = f;
                    break;
                }
                default:
                    op.iparams[field] = scalar_kv(g, kv);
                    break;
            }
        }
        if (op.type == "detect" || op.type == "segment" || op.type == "pose" ||
            op.type == "obb" || op.type == "world_detect" ||
            op.type == "world_segment") {
            model->has_detect = true;
            model->detect_op_index = (int)i;
            model->meta.reg_max =
                    (int)key_or(g, (prefix + ".reg_max").c_str(), 16);
            model->meta.end2end =
                    key_or(g, (prefix + ".end2end").c_str(), 0) != 0;
            model->meta.max_det =
                    (int)key_or(g, (prefix + ".max_det").c_str(), 300);
        }
        if (op.type == "max_sigmoid_attn" || op.type == "image_pooling_attn" ||
            op.type == "world_detect" || op.type == "world_segment") {
            model->has_text_input = true;
        }
    }
    const std::string& tail = model->ops.back().type;
    const bool tail_ok =
            (tail == "detect" || tail == "segment" || tail == "pose" ||
             tail == "obb" || tail == "world_detect" ||
             tail == "world_segment") ||
            (model->meta.task == "depth" && tail == "depth") ||
            (model->meta.task == "semantic" && tail == "semantic") ||
            (model->meta.task == "classify" && tail == "classify");
    if (!tail_ok) {
        YOLO_LOG_ERROR("op graph does not contain the declared %s output",
                       model->meta.task.c_str());
        gguf_free(g);
        ggml_free(weight_ctx);
        return nullptr;
    }
    if (model->meta.task != "detect" && model->meta.task != "depth" &&
        model->meta.task != "segment" && model->meta.task != "pose" &&
        model->meta.task != "obb" && model->meta.task != "semantic" &&
        model->meta.task != "classify") {
        YOLO_LOG_ERROR("unsupported task: %s", model->meta.task.c_str());
        gguf_free(g);
        ggml_free(weight_ctx);
        return nullptr;
    }
    if (model->meta.task == "segment" && model->meta.nm <= 0) {
        YOLO_LOG_ERROR("segment model without yolo.nm prototypes");
        gguf_free(g);
        ggml_free(weight_ctx);
        return nullptr;
    }
    if (model->meta.task == "pose" &&
        (model->meta.nk <= 0 ||
         (model->meta.kpt_ndim != 2 && model->meta.kpt_ndim != 3))) {
        YOLO_LOG_ERROR("pose model without valid yolo.nk/kpt_shape");
        gguf_free(g);
        ggml_free(weight_ctx);
        return nullptr;
    }
    if (model->meta.task == "obb" && model->meta.ne <= 0) {
        YOLO_LOG_ERROR("obb model without yolo.ne angle channels");
        gguf_free(g);
        ggml_free(weight_ctx);
        return nullptr;
    }

    // ---- YOLOE visual-prompt (savpe) support ----
    // Resolve the FPN feature ops the savpe encoder consumes by walking the
    // world head's box/embed branch chains back to their shared producer:
    // branch-internal conv outputs have exactly one consumer, while the FPN
    // feature feeds both the box and the embed branch (>= 2 consumers).
    model->has_savpe = key_or(g, "yolo.savpe", 0) != 0;
    if (model->has_savpe) {
        if (!model->has_text_input || model->detect_op_index < 0) {
            YOLO_LOG_ERROR(
                    "yolo.savpe set but the graph has no world head; ignore "
                    "the "
                    "flag or reconvert the checkpoint");
            gguf_free(g);
            ggml_free(weight_ctx);
            return nullptr;
        }
        std::vector<int> consumers(model->ops.size(), 0);
        for (const OpDef& op : model->ops)
            for (int in : op.inputs)
                if (in >= 0) consumers[in]++;
        auto branch_root = [&](int out_idx) -> int {
            int cur = out_idx;
            while (cur >= 0 && cur < (int)model->ops.size()) {
                const OpDef& o = model->ops[cur];
                if ((o.type != "conv" && o.type != "dwconv") ||
                    o.inputs.empty() || o.inputs[0] < 0)
                    return cur;
                const int prev = o.inputs[0];
                if (consumers[prev] >= 2) return prev;  // shared FPN feature
                cur = prev;
            }
            return cur;
        };
        const OpDef& head = model->ops[model->detect_op_index];
        const bool head_masks = head.ip("has_masks", 0) != 0;
        const size_t lv_stride = head_masks ? 3 : 2;
        const size_t lv_count = head_masks
                                        ? (head.inputs.size() - 1) / lv_stride
                                        : head.inputs.size() / lv_stride;
        for (size_t l = 0; l < lv_count && model->savpe_fpn_ops.size() < 3;
             l++) {
            // Per level: [box, embed(, mask)] — box and embed are rooted at
            // the same FPN feature, so the pair cross-checks the walk.
            const int root = branch_root(head.inputs[lv_stride * l]);
            if (root < 0 || root >= (int)model->ops.size() ||
                root != branch_root(head.inputs[lv_stride * l + 1])) {
                YOLO_LOG_ERROR(
                        "savpe: cannot resolve the FPN feature for level %zu",
                        l);
                gguf_free(g);
                ggml_free(weight_ctx);
                return nullptr;
            }
            // Level order follows the head inputs: level 0 = P3 (smallest
            // stride), which is the torch x = [P3, P4, P5] order savpe needs.
            model->savpe_fpn_ops.push_back(root);
        }
        if (model->savpe_fpn_ops.size() != 3) {
            YOLO_LOG_ERROR("savpe: expected 3 FPN levels, resolved %zu",
                           model->savpe_fpn_ops.size());
            gguf_free(g);
            ggml_free(weight_ctx);
            return nullptr;
        }
    }
    if (model->meta.nl <= 0 ||
        model->meta.strides.size() < (size_t)model->meta.nl) {
        YOLO_LOG_ERROR("invalid feature-level metadata");
        gguf_free(g);
        ggml_free(weight_ctx);
        return nullptr;
    }

    // ---- tensors: reference the mapped ggml context data ----
    const int64_t n_tensors = gguf_get_n_tensors(g);
    const size_t data_offset = gguf_get_data_offset(g);
    for (int64_t t = 0; t < n_tensors; t++) {
        const char* name = gguf_get_tensor_name(g, t);
        ggml_tensor* cur = ggml_get_tensor(weight_ctx, name);
        if (!cur) {
            YOLO_LOG_ERROR("tensor %s missing from ggml context", name);
            gguf_free(g);
            ggml_free(weight_ctx);
            return nullptr;
        }
        HostTensor ht;
        ht.name = name;
        ht.type = cur->type;
        ht.file_type = cur->type;
        for (int d = 0; d < 4; d++) ht.ne[d] = cur->ne[d];
        ht.data.resize(ggml_nbytes(cur));
        memcpy(ht.data.data(), cur->data, ggml_nbytes(cur));
        // Absolute file offset for the on-demand reload path
        // (session_ensure_host_weights after a release).
        ht.file_offset = data_offset + gguf_get_tensor_offset(g, t);
        model->tensors[name] = std::move(ht);
    }

    if (int64_t id = gguf_find_key(g, "yolo.vocab_txt"); id >= 0) {
        const size_t n = gguf_get_arr_n(g, id);
        const float* p = (const float*)gguf_get_arr_data(g, id);
        model->vocab_txt.assign(p, p + n);
    }

    // A savpe-flagged GGUF must carry the full savpe weight set; anything
    // else is a truncated conversion and would fail deep inside the graph
    // builder instead. Naming follows the GraphBuilder convention
    // ("savpe.cv1_0_0.w" = w("savpe.cv1_0_0", "w")), see yolo_graph.cpp.
    if (model->has_savpe) {
        for (int i = 0; i < 3; i++) {
            const std::string lv = std::to_string(i);
            const std::string required[] = {
                    "savpe.cv1_" + lv + "_0.w", "savpe.cv1_" + lv + "_0.b",
                    "savpe.cv1_" + lv + "_1.w", "savpe.cv1_" + lv + "_1.b",
                    "savpe.cv2_" + lv + ".w",   "savpe.cv2_" + lv + ".b",
            };
            for (const std::string& name : required) {
                if (!model->tensors.count(name)) {
                    YOLO_LOG_ERROR("yolo.savpe set but tensor %s is missing",
                                   name.c_str());
                    gguf_free(g);
                    ggml_free(weight_ctx);
                    return nullptr;
                }
            }
        }
        for (const std::string& tag : {"savpe.cv3", "savpe.cv4", "savpe.cv5",
                                       "savpe.cv6_0", "savpe.cv6_1"}) {
            if (!model->tensors.count(tag + ".w")) {
                YOLO_LOG_ERROR("yolo.savpe set but tensor %s.w is missing",
                               tag.c_str());
                gguf_free(g);
                ggml_free(weight_ctx);
                return nullptr;
            }
        }
        // Shape gate: the savpe convs must consume the same channel counts
        // as the FPN features the head actually receives. Resolve each
        // level's channel count from the first conv of the box branch (the
        // conv adjacent to the FPN root; its kernel ne[2] = in-channels).
        const OpDef& savpe_head = model->ops[model->detect_op_index];
        const bool savpe_masks = savpe_head.ip("has_masks", 0) != 0;
        const size_t savpe_stride = savpe_masks ? 3 : 2;
        std::vector<int> savpe_consumers(model->ops.size(), 0);
        for (const OpDef& op : model->ops)
            for (int in : op.inputs)
                if (in >= 0) savpe_consumers[in]++;
        for (size_t l = 0; l < model->savpe_fpn_ops.size(); ++l) {
            int conv_idx = savpe_head.inputs[l * savpe_stride];
            while (conv_idx >= 0 && conv_idx < (int)model->ops.size() &&
                   (model->ops[conv_idx].type == "conv" ||
                    model->ops[conv_idx].type == "dwconv") &&
                   !model->ops[conv_idx].inputs.empty() &&
                   model->ops[conv_idx].inputs[0] >= 0 &&
                   savpe_consumers[model->ops[conv_idx].inputs[0]] == 1) {
                conv_idx = model->ops[conv_idx].inputs[0];
            }
            // conv_idx is now the last conv before the FPN root; its kernel
            // in-channels equal the FPN channel count.
            if (conv_idx < 0 || conv_idx >= (int)model->ops.size()) {
                YOLO_LOG_ERROR("savpe: box branch conv for level %zu not found",
                               l);
                gguf_free(g);
                ggml_free(weight_ctx);
                return nullptr;
            }
            const std::string conv_w = "op." + std::to_string(conv_idx) + ".w";
            const auto it = model->tensors.find(conv_w);
            if (it == model->tensors.end() || it->second.ne[2] <= 0) {
                YOLO_LOG_ERROR("savpe: missing weight %s", conv_w.c_str());
                gguf_free(g);
                ggml_free(weight_ctx);
                return nullptr;
            }
            const int64_t fpn_ch = it->second.ne[2];
            const std::string lv = std::to_string(l);
            const std::string cv1 = "savpe.cv1_" + lv + "_0.w";
            const std::string cv2 = "savpe.cv2_" + lv + ".w";
            for (const std::string& name : {cv1, cv2}) {
                const auto sit = model->tensors.find(name);
                if (sit == model->tensors.end() ||
                    sit->second.ne[2] != fpn_ch) {
                    YOLO_LOG_ERROR(
                            "savpe: %s in-channels (%lld) do not match the "
                            "level-%zu FPN feature (%lld); the GGUF was "
                            "converted from a different checkpoint",
                            name.c_str(),
                            (long long)(sit == model->tensors.end()
                                                ? -1
                                                : sit->second.ne[2]),
                            l, (long long)fpn_ch);
                    gguf_free(g);
                    ggml_free(weight_ctx);
                    return nullptr;
                }
            }
        }
    }

    // Extract the tiny F32 scalar constants the graph builder bakes into the
    // graph as build-time constants (max_sigmoid_attn head bias; the world
    // head per-level logit_scale/bias). Reading them back from HostTensor
    // data at graph-build time would break canvas rebuilds after
    // session_release_host_weights dropped the host copies.
    for (size_t i = 0; i < model->ops.size(); i++) {
        const OpDef& op = model->ops[i];
        const std::string prefix = "op." + std::to_string(i);
        const bool maxsig = op.type == "max_sigmoid_attn";
        const bool world_head =
                op.type == "world_detect" || op.type == "world_segment";
        if (!maxsig && !world_head) continue;
        for (const auto& [name, ht] : model->tensors) {
            bool want = false;
            if (maxsig) {
                want = name.rfind(prefix + ".", 0) == 0 && name.size() > 5 &&
                       name.compare(name.size() - 5, 5, ".bias") == 0;
            } else if (name.rfind(prefix + ".cv4_", 0) == 0) {
                want = name.size() > 12 &&
                       name.compare(name.size() - 12, 12, "_logit_scale") == 0;
                if (!want) {
                    want = name.size() > 5 &&
                           name.compare(name.size() - 5, 5, "_bias") == 0;
                }
            }
            if (!want || ht.type != GGML_TYPE_F32) continue;
            const float* p = reinterpret_cast<const float*>(ht.data.data());
            model->scalar_params[name].assign(
                    p, p + ht.data.size() / sizeof(float));
        }
    }

    YOLO_LOG_INFO(
            "loaded %s: %lld ops, %lld tensors, dtype=%s, task=%s, nc=%d, "
            "nm=%d, nk=%d, ne=%d, end2end=%d",
            path.c_str(), (long long)n_ops, (long long)n_tensors,
            model->meta.dtype.c_str(), model->meta.task.c_str(), model->meta.nc,
            model->meta.nm, model->meta.nk, model->meta.ne,
            (int)model->meta.end2end);

    gguf_free(g);
    ggml_free(weight_ctx);
    return model;
}

}  // namespace yolo

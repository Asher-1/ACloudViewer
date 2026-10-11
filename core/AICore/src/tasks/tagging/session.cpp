// Adapted from recognize-anything-ggml cpp_ggml/src/main.cpp (Apache-2.0).
// Numeric graph, WordPiece tokenization and beam search retain upstream
// ordering.
#include "session.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "aicore/runtime_capi.h"
#include "common/ggml_backend_registry.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

namespace {
using Tensor = ggml_tensor;
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
std::vector<std::string> lines(const std::string &path) {
    std::ifstream stream(path);
    require(stream.good(), "cannot open " + path);
    std::vector<std::string> result;
    for (std::string line; std::getline(stream, line);) result.push_back(line);
    return result;
}
struct Model {
    std::string architecture;
    bool caption = false;
    bool is_cpu = false;
    bool is_cuda = false;
    bool cuda_f16_compute = false;
    bool cpu_floating_qkv = false;
    bool profile = false;
    gguf_context *gguf = nullptr;
    ggml_context *host = nullptr;
    ggml_context *weights = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_context *prepared_weights = nullptr;
    ggml_backend_buffer_t prepared_buffer = nullptr;
    Tensor *prepared_labels = nullptr;
    Tensor *prepared_query = nullptr;
    Tensor *prepared_descriptions = nullptr;
    std::map<Tensor *, Tensor *> prepared_matrices;
    aicore::runtime::BackendLease backend_lease, cpu_lease;
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::map<std::string, Tensor *> tensors;
    std::map<std::string, std::vector<float>> constants;
    std::map<std::string, std::vector<int32_t>> indices;
    std::vector<std::string> tag_names;
    std::vector<float> tag_thresholds;
    std::vector<int32_t> delete_tag_indices;
    std::vector<std::string> bert_vocab;
    bool label_override = false;

    ~Model() {
        auto lock = aicore::runtime::lock_backend_leases(
                {backend_lease, cpu_lease});
        if (sched) ggml_backend_sched_free(sched);
        if (prepared_buffer) ggml_backend_buffer_free(prepared_buffer);
        if (prepared_weights) ggml_free(prepared_weights);
        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (weights) ggml_free(weights);
        if (host) ggml_free(host);
        if (gguf) gguf_free(gguf);
    }

    Tensor *get(const std::string &name) const {
        auto it = tensors.find(architecture + "." + name);
        require(it != tensors.end(),
                "missing GGUF tensor " + architecture + "." + name);
        return it->second;
    }

    bool needed(const std::string &name) const {
        const std::string prefix = architecture + ".";
        if (label_override && (name == prefix + "label_embed" ||
                               name == prefix + "label_embed.weight"))
            return false;
        if (caption && architecture == "tag2text" &&
            (name.rfind(prefix + "tag_encoder.", 0) == 0 ||
             name.rfind(prefix + "text_decoder.", 0) == 0))
            return name.find("position_ids") == std::string::npos;
        if (name.rfind(prefix + "visual_encoder.", 0) == 0) {
            return name.find("relative_position_index") == std::string::npos &&
                   name.find("attn_mask") == std::string::npos;
        }
        const std::string decoder =
                architecture == "tag2text" ? "tag_encoder" : "tagging_head";
        if (name.rfind(prefix + decoder + ".encoder.layer.", 0) == 0)
            return name.find(".crossattention.") != std::string::npos ||
                   name.find(".intermediate.") != std::string::npos ||
                   name.find(".output.") != std::string::npos;
        return name == prefix + "label_embed" ||
               name == prefix + "label_embed.weight" ||
               name.rfind(prefix + "wordvec_proj.", 0) == 0 ||
               name.rfind(prefix + "image_proj.", 0) == 0 ||
               name.rfind(prefix + "fc.", 0) == 0;
    }

    static float value(const Tensor *t, size_t i) {
        const auto *bytes = static_cast<const uint8_t *>(t->data);
        if (t->type == GGML_TYPE_F32)
            return reinterpret_cast<const float *>(bytes)[i];
        if (t->type == GGML_TYPE_F16)
            return ggml_fp16_to_fp32(
                    reinterpret_cast<const ggml_fp16_t *>(bytes)[i]);
        throw std::runtime_error(
                std::string("unsupported constant tensor type: ") + t->name);
    }

    void add_constant(const std::string &name,
                      const std::vector<float> &data,
                      int64_t n0,
                      int64_t n1,
                      int64_t n2,
                      int64_t n3) {
        Tensor *dst =
                ggml_new_tensor_4d(weights, GGML_TYPE_F32, n0, n1, n2, n3);
        ggml_set_name(dst, name.c_str());
        tensors.emplace(name, dst);
        constants.emplace(name, data);
    }

    void add_indices(const std::string &name, std::vector<int32_t> data) {
        Tensor *dst = ggml_new_tensor_1d(weights, GGML_TYPE_I32, data.size());
        ggml_set_name(dst, name.c_str());
        tensors.emplace(name, dst);
        indices.emplace(name, std::move(data));
    }

    void read_metadata() {
        const auto read_strings = [&](const char *key) {
            std::vector<std::string> result;
            const int64_t id = gguf_find_key(gguf, key);
            if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY ||
                gguf_get_arr_type(gguf, id) != GGUF_TYPE_STRING)
                return result;
            result.reserve(gguf_get_arr_n(gguf, id));
            for (size_t i = 0; i < gguf_get_arr_n(gguf, id); ++i)
                result.emplace_back(gguf_get_arr_str(gguf, id, i));
            return result;
        };
        const auto read_floats = [&](const char *key) {
            std::vector<float> result;
            const int64_t id = gguf_find_key(gguf, key);
            if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY ||
                gguf_get_arr_type(gguf, id) != GGUF_TYPE_FLOAT32)
                return result;
            const auto *data =
                    static_cast<const float *>(gguf_get_arr_data(gguf, id));
            result.assign(data, data + gguf_get_arr_n(gguf, id));
            return result;
        };
        const auto read_i32 = [&](const char *key) {
            std::vector<int32_t> result;
            const int64_t id = gguf_find_key(gguf, key);
            if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY ||
                gguf_get_arr_type(gguf, id) != GGUF_TYPE_INT32)
                return result;
            const auto *data =
                    static_cast<const int32_t *>(gguf_get_arr_data(gguf, id));
            result.assign(data, data + gguf_get_arr_n(gguf, id));
            return result;
        };
        tag_names = read_strings("ram.tag_list");
        tag_thresholds = read_floats("ram.tag_thresholds");
        delete_tag_indices = read_i32("ram.delete_tag_indices");
        if (caption && architecture == "tag2text")
            bert_vocab = read_strings("ram.bert_vocab");
    }

    void apply_label_space(const std::string &tag_list_path,
                           const std::string &threshold_path,
                           const std::string &embedding_path,
                           float threshold_override) {
        if (!tag_list_path.empty()) tag_names = lines(tag_list_path);
        if (!threshold_path.empty()) {
            tag_thresholds.clear();
            for (const auto &value : lines(threshold_path))
                tag_thresholds.push_back(std::stof(value));
        }
        if (threshold_override >= 0.0f)
            tag_thresholds.assign(tag_names.size(), threshold_override);
        else if (tag_thresholds.size() != tag_names.size())
            tag_thresholds.assign(tag_names.size(), 0.5f);
        if (embedding_path.empty()) return;
        require(!tag_names.empty(), "--label-embedding requires --tag-list");
        std::ifstream file(embedding_path, std::ios::binary | std::ios::ate);
        require(file.good(), "cannot open label embedding " + embedding_path);
        const std::streamsize bytes = file.tellg();
        require(bytes > 0 && bytes % sizeof(float) == 0,
                "label embedding must be a non-empty float32 file");
        file.seekg(0);
        std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
        require(file.read(reinterpret_cast<char *>(data.data()), bytes).good(),
                "cannot read label embedding " + embedding_path);
        require(data.size() % (512 * tag_names.size()) == 0,
                "label embedding shape is not [classes, descriptions, 512]");
        const std::string tensor_name =
                architecture == "tag2text"
                        ? architecture + ".label_embed.weight"
                        : architecture + ".label_embed";
        add_constant(tensor_name, data, 512, data.size() / 512, 1, 1);
        label_override = true;
    }

    ggml_backend_sched_t make_scheduler() const {
        ggml_backend_t backends[2] = {backend, cpu};
        ggml_backend_sched_t result = ggml_backend_sched_new(
                backends, nullptr, is_cpu ? 1 : 2, 20000, false, true);
        require(result, "cannot create ggml backend scheduler");
        return result;
    }

    void load(const std::string &path,
              const std::string &backend_name,
              int threads,
              bool enable_caption = false,
              const std::string &tag_list_path = {},
              const std::string &threshold_path = {},
              const std::string &embedding_path = {},
              float threshold_override = -1.0f) {
        gguf_init_params params{};
        params.ctx = &host;
        gguf = gguf_init_from_file(path.c_str(), params);
        require(gguf && host, "cannot load GGUF " + path);
        int64_t arch = gguf_find_key(gguf, "general.architecture");
        require(arch >= 0, "GGUF architecture is missing");
        architecture = gguf_get_val_str(gguf, arch);
        caption = enable_caption;
        is_cpu = backend_name == "cpu";
        is_cuda = backend_name == "cuda";
        require(architecture == "ram" || architecture == "ram_plus" ||
                        architecture == "tag2text",
                "unsupported GGUF architecture");
        read_metadata();

        std::string error;
        backend_lease = aicore::runtime::acquire_backend_lease(backend_name,
                                                               threads, &error);
        require(bool(backend_lease), error);
        backend = backend_lease.handle();
        cpu_lease =
                aicore::runtime::acquire_backend_lease("cpu", threads, &error);
        require(bool(cpu_lease), error);
        cpu = cpu_lease.handle();
        const std::string resolved = backend_lease.device();
        is_cpu = backend_lease.is_cpu();
        is_cuda = resolved.rfind("cuda", 0) == 0;
        auto lock = aicore::runtime::lock_backend_leases(
                {backend_lease, cpu_lease});

        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 1200 + 1024 * 1024;
        ip.no_alloc = true;
        weights = ggml_init(ip);
        require(weights, "cannot allocate weight metadata");
        label_override = !embedding_path.empty();
        apply_label_space(tag_list_path, threshold_path, embedding_path,
                          threshold_override);
        for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
            std::string name = gguf_get_tensor_name(gguf, i);
            if (!needed(name)) continue;
            Tensor *source = ggml_get_tensor(host, name.c_str());
            require(source != nullptr, "GGUF tensor data missing: " + name);
            const bool caption_text_weight =
                    caption && source->type == GGML_TYPE_F16 &&
                    (name.rfind(architecture + ".tag_encoder.", 0) == 0 ||
                     name.rfind(architecture + ".text_decoder.", 0) == 0);
            Tensor *dst =
                    caption_text_weight
                            ? ggml_new_tensor(weights, GGML_TYPE_F32,
                                              ggml_n_dims(source), source->ne)
                            : ggml_dup_tensor(weights, source);
            ggml_set_name(dst, name.c_str());
            tensors.emplace(name, dst);
            if (caption_text_weight) {
                auto &converted = constants[name];
                converted.resize(ggml_nelements(source));
                ggml_fp16_to_fp32_row(
                        static_cast<const ggml_fp16_t *>(source->data),
                        converted.data(), converted.size());
            }
        }
        const int depths[4] = {2, 2, 18, 2};
        const int heads_large[4] = {6, 12, 24, 48};
        const int heads_base[4] = {4, 8, 16, 32};
        const int *heads =
                architecture == "tag2text" ? heads_base : heads_large;
        for (int stage = 0; stage < 4; ++stage) {
            for (int block = 0; block < depths[stage]; ++block) {
                const std::string base = architecture +
                                         ".visual_encoder.layers." +
                                         std::to_string(stage) + ".blocks." +
                                         std::to_string(block);
                Tensor *table = ggml_get_tensor(
                        host,
                        (base + ".attn.relative_position_bias_table").c_str());
                require(table && table->ne[0] == heads[stage],
                        "bad relative position table");
                const int h = 96 >> stage;
                const int w = std::min(12, h);
                const int n = w * w;
                const int bias_width = 2 * w - 1;
                std::vector<float> bias(n * n * heads[stage]);
                for (int head = 0; head < heads[stage]; ++head)
                    for (int query = 0; query < n; ++query)
                        for (int key = 0; key < n; ++key) {
                            int dy = query / w - key / w + w - 1;
                            int dx = query % w - key % w + w - 1;
                            int index = dy * bias_width + dx;
                            bias[(head * n + query) * n + key] =
                                    value(table, index * heads[stage] + head);
                        }
                add_constant(base + ".computed_bias", bias, n, n, heads[stage],
                             1);
                if (block % 2 && h > w) {
                    const int windows_per_side = h / w;
                    std::vector<float> mask(n * n * windows_per_side *
                                            windows_per_side);
                    auto region = [h, w](int coordinate) {
                        if (coordinate < h - w) return 0;
                        return coordinate < h - w / 2 ? 1 : 2;
                    };
                    for (int wy = 0; wy < windows_per_side; ++wy)
                        for (int wx = 0; wx < windows_per_side; ++wx)
                            for (int query = 0; query < n; ++query)
                                for (int key = 0; key < n; ++key) {
                                    int q_region =
                                            3 * region(wy * w + query / w) +
                                            region(wx * w + query % w);
                                    int k_region =
                                            3 * region(wy * w + key / w) +
                                            region(wx * w + key % w);
                                    mask[(((wy * windows_per_side + wx) * n +
                                           query) *
                                                  n +
                                          key)] =
                                            q_region == k_region ? 0.f : -100.f;
                                }
                    add_constant(base + ".computed_mask", mask, n, n, 1,
                                 windows_per_side * windows_per_side);
                    std::vector<float> attention_bias(mask.size() *
                                                      heads[stage]);
                    for (int window = 0;
                         window < windows_per_side * windows_per_side; ++window)
                        for (int head = 0; head < heads[stage]; ++head)
                            for (int position = 0; position < n * n; ++position)
                                attention_bias[(window * heads[stage] + head) *
                                                       n * n +
                                               position] =
                                        bias[head * n * n + position] +
                                        mask[window * n * n + position];
                    add_constant(base + ".computed_attention_bias",
                                 attention_bias, n, n, heads[stage],
                                 windows_per_side * windows_per_side);
                }
            }
        }
        for (int h : {96, 48, 24}) {
            const int w = 12, windows_per_side = h / w;
            std::vector<int32_t> forward(h * h), inverse(h * h);
            for (int wy = 0; wy < windows_per_side; ++wy)
                for (int wx = 0; wx < windows_per_side; ++wx)
                    for (int ly = 0; ly < w; ++ly)
                        for (int lx = 0; lx < w; ++lx) {
                            int source = (wy * w + ly) * h + wx * w + lx;
                            int target = (wy * windows_per_side + wx) * w * w +
                                         ly * w + lx;
                            forward[target] = source;
                            inverse[source] = target;
                        }
            add_indices(architecture + ".window_index." + std::to_string(h),
                        std::move(forward));
            add_indices(architecture + ".unwindow_index." + std::to_string(h),
                        std::move(inverse));
        }
        weight_buffer = ggml_backend_alloc_ctx_tensors(weights, backend);
        require(weight_buffer, "cannot allocate GGUF weights on backend");
        for (auto &[name, dst] : tensors) {
            auto constant = constants.find(name);
            if (constant != constants.end()) {
                ggml_backend_tensor_set(
                        dst, constant->second.data(), 0,
                        constant->second.size() * sizeof(float));
            } else if (auto index = indices.find(name);
                       index != indices.end()) {
                ggml_backend_tensor_set(dst, index->second.data(), 0,
                                        index->second.size() * sizeof(int32_t));
            } else {
                Tensor *source = ggml_get_tensor(host, name.c_str());
                ggml_backend_tensor_set(dst, source->data, 0,
                                        ggml_nbytes(source));
            }
        }
        sched = make_scheduler();
        ggml_free(host);
        host = nullptr;
        constants.clear();
        indices.clear();
    }
};

struct InferenceGraph {
    Model &model;
    ggml_context *ctx;
    bool diagnostics;
    std::map<std::string, Tensor *> observed;
    Tensor *image_embeddings = nullptr;

    Tensor *get(const std::string &name) { return model.get(name); }
    Tensor *observe(const std::string &name, Tensor *tensor) {
        if (diagnostics) {
            Tensor *output = tensor->type == GGML_TYPE_F32
                                     ? tensor
                                     : ggml_cast(ctx, tensor, GGML_TYPE_F32);
            ggml_set_name(output, name.c_str());
            ggml_set_output(output);
            observed[name] = output;
        }
        return tensor;
    }
    Tensor *f32(Tensor *x) {
        return x->type == GGML_TYPE_F32 ? x : ggml_cast(ctx, x, GGML_TYPE_F32);
    }
    Tensor *reshape(Tensor *x, int64_t a, int64_t b) {
        return ggml_reshape_2d(ctx, x, a, b);
    }
    Tensor *multiply(Tensor *weight, Tensor *x) {
        const auto prepared = model.prepared_matrices.find(weight);
        if (prepared != model.prepared_matrices.end()) {
            weight = prepared->second;
            if (weight->type == GGML_TYPE_F16 && x->type == GGML_TYPE_F32)
                x = ggml_cast(ctx, x, GGML_TYPE_F16);
        }
        // CUDA MMQ quantizes the right-hand activation as well as the GGUF
        // weight. A floating RHS preserves the weight-only Q8 contract.
        if (model.is_cuda && ggml_is_quantized(weight->type) &&
            x->type == GGML_TYPE_F32)
            x = ggml_cast(ctx, x, GGML_TYPE_F16);
        Tensor *product = ggml_mul_mat(ctx, weight, x);
        if (!model.caption) ggml_mul_mat_set_prec(product, GGML_PREC_F32);
        ggml_set_name(product,
                      model.caption ? "ram_caption_matmul" : "ram_tags_matmul");
        return product;
    }
    Tensor *linear(Tensor *x, const std::string &base) {
        Tensor *weight = get(base + ".weight");
        require(weight->ne[0] == x->ne[0],
                "linear input width mismatch at " + base);
        if (model.caption && weight->type != GGML_TYPE_F32)
            weight = f32(weight);
        if (x->type == GGML_TYPE_F16 && weight->type != GGML_TYPE_F16)
            x = f32(x);
        Tensor *y = multiply(weight, x);
        return ggml_add(ctx, y, f32(get(base + ".bias")));
    }
    void prepare_constants() {
        // These subgraphs depend only on the loaded weights and label space.
        // Execute the original backend math once, retaining its dtype and
        // order.
        std::vector<std::pair<Tensor *, Tensor **>> outputs;
        if ((model.is_cuda && model.cuda_f16_compute && !model.caption) ||
            model.cpu_floating_qkv) {
            for (const auto &entry : model.tensors) {
                Tensor *weight = entry.second;
                const bool cpu_qkv =
                        model.cpu_floating_qkv &&
                        weight->type == GGML_TYPE_Q8_0 &&
                        (entry.first.rfind("ram.visual_encoder.layers.0.", 0) ==
                                 0 ||
                         entry.first.rfind("ram.visual_encoder.layers.1.", 0) ==
                                 0) &&
                        entry.first.find(".attn.qkv.weight") !=
                                std::string::npos;
                const bool cuda_matrix =
                        model.is_cuda && model.cuda_f16_compute &&
                        !model.caption && weight->type == GGML_TYPE_F32;
                if ((!cpu_qkv && !cuda_matrix) || ggml_n_dims(weight) != 2 ||
                    entry.first.size() < 7 ||
                    entry.first.compare(entry.first.size() - 7, 7, ".weight") !=
                            0 ||
                    entry.first.find("label_embed") != std::string::npos)
                    continue;
                auto prepared = model.prepared_matrices.emplace(
                        weight,
                        ggml_cast(ctx, weight,
                                  cpu_qkv ? GGML_TYPE_F32 : GGML_TYPE_F16));
                outputs.emplace_back(prepared.first->second,
                                     &prepared.first->second);
            }
        }
        if (model.architecture == "ram_plus") {
            Tensor *source = get("label_embed");
            const int64_t classes = model.tag_names.size();
            require(classes > 0 && source->ne[1] % classes == 0,
                    "RAM++ label embedding does not match tag-list size");
            Tensor *descriptions = ggml_reshape_3d(
                    ctx, source, 512, source->ne[1] / classes, classes);
            outputs.emplace_back(
                    ggml_cont(ctx, ggml_permute(ctx, descriptions, 1, 0, 2, 3)),
                    &model.prepared_descriptions);
        } else {
            const bool tag2text = model.architecture == "tag2text";
            Tensor *labels =
                    get(tag2text ? "label_embed.weight" : "label_embed");
            if (!tag2text)
                labels = ggml_relu(ctx, linear(f32(labels), "wordvec_proj"));
            outputs.emplace_back(labels, &model.prepared_labels);
            outputs.emplace_back(
                    linear(labels, tag2text ? "tag_encoder.encoder.layer.0."
                                              "crossattention.self.query"
                                            : "tagging_head.encoder.layer.0."
                                              "crossattention.self.query"),
                    &model.prepared_query);
        }
        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * outputs.size() + 1024;
        params.no_alloc = true;
        model.prepared_weights = ggml_init(params);
        require(model.prepared_weights,
                "cannot allocate constant graph metadata");
        ggml_cgraph *preparation = ggml_new_graph_custom(ctx, 20000, false);
        for (const auto &entry : outputs) {
            ggml_set_output(entry.first);
            ggml_build_forward_expand(preparation, entry.first);
            *entry.second =
                    ggml_dup_tensor(model.prepared_weights, entry.first);
        }
        model.prepared_buffer = ggml_backend_alloc_ctx_tensors(
                model.prepared_weights, model.backend);
        require(model.prepared_buffer,
                "cannot allocate prepared label tensors");
        ggml_backend_buffer_set_usage(model.prepared_buffer,
                                      GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto preparation_sched =
                std::unique_ptr<ggml_backend_sched,
                                decltype(&ggml_backend_sched_free)>(
                        model.make_scheduler(), ggml_backend_sched_free);
        require(ggml_backend_sched_alloc_graph(preparation_sched.get(),
                                               preparation),
                "cannot allocate constant graph");
        require(ggml_backend_sched_graph_compute(preparation_sched.get(),
                                                 preparation) ==
                        GGML_STATUS_SUCCESS,
                "constant graph compute failed");
        for (const auto &entry : outputs)
            ggml_backend_tensor_copy(entry.first, *entry.second);
    }
    Tensor *norm(Tensor *x, const std::string &base, float eps = 1e-5f) {
        x = ggml_norm(ctx, f32(x), eps);
        x = ggml_mul(ctx, x, f32(get(base + ".weight")));
        return ggml_add(ctx, x, f32(get(base + ".bias")));
    }
    Tensor *slice_channels(Tensor *x, int start, int count) {
        return ggml_view_3d(ctx, x, count, x->ne[1], x->ne[2], x->nb[1],
                            x->nb[2], start * x->nb[0]);
    }
    Tensor *attention(Tensor *q,
                      Tensor *k,
                      Tensor *v,
                      int heads,
                      Tensor *bias = nullptr,
                      Tensor *mask = nullptr) {
        const int dim = q->ne[0] / heads;
        auto split = [&](Tensor *t) {
            t = ggml_view_4d(ctx, t, dim, heads, t->ne[1], t->ne[2],
                             dim * t->nb[0], t->nb[1], t->nb[2], 0);
            return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));
        };
        q = split(q);
        k = split(k);
        v = split(v);
        Tensor *scores = ggml_mul_mat(ctx, k, q);
        if (!model.caption) ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        Tensor *probabilities;
        if (!model.is_cuda && !model.caption) {
            require(!bias || !mask, "fused attention requires a combined bias");
            probabilities = ggml_soft_max_ext(ctx, scores, bias ? bias : mask,
                                              1.f / std::sqrt(float(dim)), 0.f);
        } else {
            scores = ggml_scale(ctx, scores, 1.f / std::sqrt(float(dim)));
            if (bias) scores = ggml_add(ctx, scores, bias);
            if (mask) scores = ggml_add(ctx, scores, mask);
            probabilities = ggml_soft_max(ctx, scores);
        }
        Tensor *vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
        Tensor *out = ggml_mul_mat(ctx, vt, probabilities);
        if (!model.caption) ggml_mul_mat_set_prec(out, GGML_PREC_F32);
        out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
        return ggml_reshape_3d(ctx, out, dim * heads, q->ne[1], q->ne[3]);
    }
    Tensor *windows(Tensor *x, int h, int w) {
        if (h == w) return ggml_reshape_3d(ctx, x, x->ne[0], w * w, 1);
        Tensor *flat = ggml_reshape_2d(ctx, ggml_cont(ctx, x), x->ne[0], h * h);
        Tensor *gathered = ggml_get_rows(
                ctx, flat, get("window_index." + std::to_string(h)));
        return ggml_reshape_3d(ctx, gathered, x->ne[0], w * w, h * h / (w * w));
    }
    Tensor *unwindow(Tensor *x, int h, int w) {
        if (h == w) return ggml_reshape_3d(ctx, x, x->ne[0], h, h);
        Tensor *flat = ggml_reshape_2d(ctx, ggml_cont(ctx, x), x->ne[0], h * h);
        Tensor *gathered = ggml_get_rows(
                ctx, flat, get("unwindow_index." + std::to_string(h)));
        return ggml_reshape_3d(ctx, gathered, x->ne[0], h, h);
    }
    Tensor *swin_block(Tensor *x, int stage, int block, int h, int heads) {
        std::string base = "visual_encoder.layers." + std::to_string(stage) +
                           ".blocks." + std::to_string(block);
        const int c = x->ne[0], w = std::min(12, h),
                  shift = block % 2 && h > w ? w / 2 : 0;
        Tensor *shortcut = x;
        Tensor *y = norm(x, base + ".norm1");
        y = ggml_reshape_3d(ctx, y, c, h, h);
        if (shift) y = ggml_roll(ctx, y, 0, -shift, -shift, 0);
        y = windows(y, h, w);
        Tensor *qkv = linear(y, base + ".attn.qkv");
        Tensor *q = slice_channels(qkv, 0, c);
        Tensor *k = slice_channels(qkv, c, c);
        Tensor *v = slice_channels(qkv, 2 * c, c);
        // Relative bias and shifted-window mask are generated from the
        // checkpoint by graph construction, so every GGUF dtype follows the
        // same attention math.
        Tensor *bias = get(base + ".computed_bias");
        Tensor *mask = shift ? get(base + ".computed_mask") : nullptr;
        if (mask && !model.is_cuda && !model.caption) {
            bias = get(base + ".computed_attention_bias");
            mask = nullptr;
        }
        y = attention(q, k, v, heads, bias, mask);
        y = linear(y, base + ".attn.proj");
        y = unwindow(y, h, w);
        if (shift) y = ggml_roll(ctx, y, 0, shift, shift, 0);
        y = ggml_reshape_2d(ctx, ggml_cont(ctx, y), c, h * h);
        x = ggml_add(ctx, shortcut, y);
        y = norm(x, base + ".norm2");
        y = ggml_gelu_erf(ctx, linear(y, base + ".mlp.fc1"));
        y = linear(y, base + ".mlp.fc2");
        return ggml_add(ctx, x, y);
    }
    Tensor *merge(Tensor *x, int stage, int h) {
        const int c = x->ne[0], half = h / 2;
        x = ggml_reshape_3d(ctx, x, c, h, h);
        Tensor *pieces = nullptr;
        for (int xx = 0; xx < 2; ++xx)
            for (int y = 0; y < 2; ++y) {
                Tensor *part = ggml_view_3d(ctx, x, c, half, half, 2 * x->nb[1],
                                            2 * x->nb[2],
                                            xx * x->nb[1] + y * x->nb[2]);
                part = ggml_cont(ctx, part);
                pieces = pieces ? ggml_concat(ctx, pieces, part, 0) : part;
            }
        std::string base = "visual_encoder.layers." + std::to_string(stage) +
                           ".downsample";
        Tensor *y = ggml_reshape_2d(ctx, pieces, 4 * c, half * half);
        y = norm(y, base + ".norm");
        return multiply(get(base + ".reduction.weight"), y);
    }
    Tensor *build(Tensor *input) {
        const bool tag2text = model.architecture == "tag2text";
        const int patch_width = tag2text ? 128 : 192;
        Tensor *kernel = get("visual_encoder.patch_embed.proj.weight");
        Tensor *patches = ggml_im2col(
                ctx, kernel, input, 4, 4, 0, 0, 1, 1, true,
                kernel->type == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16);
        Tensor *x = ggml_mul_mat(
                ctx,
                ggml_reshape_2d(
                        ctx, patches, patches->ne[0],
                        patches->ne[1] * patches->ne[2] * patches->ne[3]),
                ggml_reshape_2d(ctx, kernel,
                                kernel->ne[0] * kernel->ne[1] * kernel->ne[2],
                                kernel->ne[3]));
        x = ggml_reshape_4d(ctx, x, patches->ne[1], patches->ne[2],
                            patches->ne[3], kernel->ne[3]);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 1, 3, 2));
        Tensor *patch_bias = ggml_reshape_4d(
                ctx, f32(get("visual_encoder.patch_embed.proj.bias")), 1, 1,
                patch_width, 1);
        x = ggml_add(ctx, x, patch_bias);
        observe("patch_conv", x);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
        observe("patch_permute", x);
        x = norm(ggml_reshape_2d(ctx, x, patch_width, 96 * 96),
                 "visual_encoder.patch_embed.norm");
        observe("patch_embed", x);
        const int depths[4] = {2, 2, 18, 2};
        const int heads_large[4] = {6, 12, 24, 48};
        const int heads_base[4] = {4, 8, 16, 32};
        const int *heads = tag2text ? heads_base : heads_large;
        for (int stage = 0; stage < 4; ++stage) {
            int h = 96 >> stage;
            for (int block = 0; block < depths[stage]; ++block) {
                x = swin_block(x, stage, block, h, heads[stage]);
                if (block == 0)
                    observe("stage" + std::to_string(stage) + "_block0", x);
                if (block == 1)
                    observe("stage" + std::to_string(stage) + "_block1", x);
            }
            if (stage != 3) x = merge(x, stage, h);
            observe("stage" + std::to_string(stage), x);
        }
        x = norm(x, "visual_encoder.norm");
        Tensor *cls = ggml_mean(ctx, ggml_cont(ctx, ggml_transpose(ctx, x)));
        cls = ggml_cont(ctx, ggml_transpose(ctx, cls));
        x = ggml_concat(ctx, cls, x, 1);
        if (!tag2text) x = linear(x, "image_proj");
        observe("image_proj", x);
        if (model.caption) {
            image_embeddings = f32(x);
            ggml_set_output(image_embeddings);
        }

        Tensor *label_source =
                get(tag2text ? "label_embed.weight" : "label_embed");
        if (model.architecture == "ram_plus") {
            Tensor *cls_token = ggml_cont(
                    ctx, ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], 0));
            Tensor *length = ggml_sqrt(
                    ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, cls_token)));
            cls_token = ggml_div(ctx, cls_token, length);
            Tensor *similarities = ggml_mul_mat(ctx, label_source, cls_token);
            similarities = ggml_scale(ctx, similarities, 1.f / 0.07f);
            const int class_count = static_cast<int>(model.tag_names.size());
            require(class_count > 0 && label_source->ne[1] % class_count == 0,
                    "RAM++ label embedding does not match tag-list size");
            const int description_count =
                    static_cast<int>(label_source->ne[1] / class_count);
            Tensor *probabilities = ggml_soft_max(
                    ctx, ggml_reshape_2d(ctx, similarities, description_count,
                                         class_count));
            Tensor *descriptions = model.prepared_descriptions;
            probabilities = ggml_reshape_3d(ctx, probabilities,
                                            description_count, 1, class_count);
            label_source = ggml_reshape_2d(
                    ctx, ggml_mul_mat(ctx, descriptions, probabilities), 512,
                    class_count);
        }
        Tensor *labels =
                model.prepared_labels ? model.prepared_labels : label_source;
        if (!tag2text && !model.prepared_labels) {
            labels = multiply(get("wordvec_proj.weight"), f32(label_source));
            labels = ggml_relu(
                    ctx, ggml_add(ctx, labels, f32(get("wordvec_proj.bias"))));
        }
        observe("label_proj", labels);
        for (int layer = 0; layer < 2; ++layer) {
            std::string base = (tag2text ? "tag_encoder" : "tagging_head") +
                               std::string(".encoder.layer.") +
                               std::to_string(layer);
            std::string cross = base + ".crossattention";
            Tensor *q = layer == 0 && model.prepared_query
                                ? model.prepared_query
                                : linear(labels, cross + ".self.query");
            Tensor *k = linear(x, cross + ".self.key");
            Tensor *v = linear(x, cross + ".self.value");
            Tensor *attended = attention(q, k, v, 4);
            attended = linear(attended, cross + ".output.dense");
            labels = norm(ggml_add(ctx, labels, attended),
                          cross + ".output.LayerNorm", 1e-12f);
            Tensor *ff = ggml_gelu_erf(
                    ctx, linear(labels, base + ".intermediate.dense"));
            ff = linear(ff, base + ".output.dense");
            labels = norm(ggml_add(ctx, labels, ff), base + ".output.LayerNorm",
                          1e-12f);
            observe("decoder" + std::to_string(layer), labels);
        }
        if (!tag2text) return linear(labels, "fc");
        Tensor *fc_weight = ggml_reshape_2d(ctx, f32(get("fc.W")), 768, 3429);
        Tensor *fc_bias = ggml_reshape_2d(ctx, f32(get("fc.b")), 1, 3429);
        return ggml_add(ctx,
                        ggml_sum_rows(ctx, ggml_mul(ctx, labels, fc_weight)),
                        fc_bias);
    }
};

struct WordPieceTokenizer {
    std::vector<std::string> words;
    std::map<std::string, int32_t> ids;

    explicit WordPieceTokenizer(std::vector<std::string> vocabulary)
        : words(std::move(vocabulary)) {
        require(words.size() == 30522,
                "BERT base uncased vocabulary must have 30522 entries");
        for (size_t i = 0; i < words.size(); ++i)
            ids.emplace(words[i], static_cast<int32_t>(i));
        words.push_back("[DEC]");
        words.push_back("[ENC]");
    }

    explicit WordPieceTokenizer(const std::string &path)
        : WordPieceTokenizer(lines(path)) {}

    int32_t id(const std::string &word) const {
        auto it = ids.find(word);
        return it == ids.end() ? 100 : it->second;
    }

    std::vector<int32_t> encode(const std::string &text, int maximum) const {
        std::vector<std::string> basic;
        std::string current;
        auto flush = [&]() {
            if (!current.empty()) basic.push_back(std::exchange(current, ""));
        };
        for (unsigned char c : text) {
            if (std::isspace(c))
                flush();
            else if (std::ispunct(c)) {
                flush();
                basic.emplace_back(1, c);
            } else
                current += static_cast<char>(std::tolower(c));
        }
        flush();
        std::vector<int32_t> result{101};
        for (const auto &token : basic) {
            if (token.size() > 100) {
                result.push_back(100);
                continue;
            }
            std::vector<int32_t> pieces;
            bool failed = false;
            for (size_t first = 0; first < token.size();) {
                size_t last = token.size();
                bool found = false;
                while (last > first) {
                    std::string part = (first ? "##" : "") +
                                       token.substr(first, last - first);
                    if (auto it = ids.find(part); it != ids.end()) {
                        pieces.push_back(it->second);
                        first = last;
                        found = true;
                        break;
                    }
                    --last;
                }
                if (!found) {
                    failed = true;
                    break;
                }
            }
            if (failed)
                result.push_back(100);
            else
                result.insert(result.end(), pieces.begin(), pieces.end());
        }
        result.resize(std::min<size_t>(result.size(), maximum - 1));
        result.push_back(102);
        return result;
    }

    std::string decode(const std::vector<int32_t> &token_ids) const {
        std::string result;
        for (int32_t token : token_ids) {
            if (token == 102) break;
            if (token <= 102 || token >= static_cast<int32_t>(words.size()))
                continue;
            const std::string &part = words[token];
            if (part == "[DEC]" || part == "[ENC]") continue;
            if (part.rfind("##", 0) == 0) result += part.substr(2);
            // HuggingFace's BasicTokenizer keeps apostrophe as a separate
            // token with surrounding spaces (for example, "cat ' s"). Other
            // punctuation is attached to the preceding word by cleanup.
            else if (part.size() == 1 &&
                     std::ispunct(static_cast<unsigned char>(part[0])) &&
                     part[0] != '\'')
                result += part;
            else
                result += (result.empty() ? "" : " ") + part;
        }
        const std::string prompt = "a picture of ";
        if (result.rfind(prompt, 0) == 0) result.erase(0, prompt.size());
        return result;
    }
};

struct CrossCache {
    std::array<std::vector<float>, 12> keys;
    std::array<std::vector<float>, 12> values;

    CrossCache() {
        for (int layer = 0; layer < 12; ++layer) {
            keys[layer].resize(768 * 40);
            values[layer].resize(768 * 40);
        }
    }
};

struct CaptionGraph {
    Model &model;
    ggml_context *ctx;
    ggml_backend_sched_t sched;
    InferenceGraph ops;
    Tensor *tokens = nullptr;
    Tensor *image_or_tags = nullptr;
    Tensor *output = nullptr;
    ggml_cgraph *graph = nullptr;
    std::array<Tensor *, 12> cross_keys{};
    std::array<Tensor *, 12> cross_values{};
    bool produces_cross_cache = false;
    bool is_decoder = false;

    CaptionGraph(Model &m,
                 ggml_context *context,
                 ggml_backend_sched_t scheduler)
        : model(m), ctx(context), sched(scheduler), ops{m, context, false} {}

    Tensor *embeddings(const std::string &base, int length, int batch = 1) {
        tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, length * batch);
        ggml_set_input(tokens);
        Tensor *positions =
                ggml_new_tensor_1d(ctx, GGML_TYPE_I32, length * batch);
        ggml_set_input(positions);
        // Position indices are supplied after scheduler graph allocation.
        position_ids = positions;
        Tensor *word = ggml_get_rows(
                ctx, ops.get(base + ".word_embeddings.weight"), tokens);
        Tensor *position = ggml_get_rows(
                ctx, ops.get(base + ".position_embeddings.weight"), positions);
        if (batch > 1) {
            word = ggml_reshape_3d(ctx, word, word->ne[0], length, batch);
            position = ggml_reshape_3d(ctx, position, position->ne[0], length,
                                       batch);
        }
        return ops.norm(ggml_add(ctx, word, position), base + ".LayerNorm",
                        1e-12f);
    }

    Tensor *position_ids = nullptr;

    Tensor *layer(Tensor *hidden,
                  Tensor *cross,
                  const std::string &base,
                  Tensor *mask) {
        auto attend = [&](Tensor *query_input, Tensor *key_input,
                          const std::string &kind, Tensor *attention_mask) {
            const std::string path = base + "." + kind;
            Tensor *q = ops.linear(query_input, path + ".self.query");
            Tensor *k = ops.linear(key_input, path + ".self.key");
            Tensor *v = ops.linear(key_input, path + ".self.value");
            Tensor *attended =
                    ops.attention(q, k, v, 12, nullptr, attention_mask);
            attended = ops.linear(attended, path + ".output.dense");
            return ops.norm(ggml_add(ctx, query_input, attended),
                            path + ".output.LayerNorm", 1e-12f);
        };
        hidden = attend(hidden, hidden, "attention", mask);
        hidden = attend(hidden, cross, "crossattention", nullptr);
        Tensor *ff = ggml_gelu_erf(
                ctx, ops.linear(hidden, base + ".intermediate.dense"));
        ff = ops.linear(ff, base + ".output.dense");
        return ops.norm(ggml_add(ctx, hidden, ff), base + ".output.LayerNorm",
                        1e-12f);
    }

    void build_encoder(int length) {
        image_or_tags = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 145);
        ggml_set_input(image_or_tags);
        Tensor *mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, length, length);
        ggml_set_input(mask);
        attention_mask = mask;
        Tensor *hidden = embeddings("tag_encoder.embeddings", length);
        for (int i = 0; i < 12; ++i)
            hidden = layer(hidden, image_or_tags,
                           "tag_encoder.encoder.layer." + std::to_string(i),
                           mask);
        produces_cross_cache = true;
        for (int i = 0; i < 12; ++i) {
            const std::string base = "text_decoder.bert.encoder.layer." +
                                     std::to_string(i) + ".crossattention.self";
            cross_keys[i] = ops.linear(hidden, base + ".key");
            cross_values[i] = ops.linear(hidden, base + ".value");
        }
        finish(hidden);
    }

    void build_decoder_cached_cross(int length, int batch) {
        is_decoder = true;
        attention_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, length, length);
        ggml_set_input(attention_mask);
        Tensor *hidden =
                embeddings("text_decoder.bert.embeddings", length, batch);
        for (int i = 0; i < 12; ++i) {
            const std::string base =
                    "text_decoder.bert.encoder.layer." + std::to_string(i);
            const std::string self = base + ".attention";
            Tensor *q = ops.linear(hidden, self + ".self.query");
            Tensor *k = ops.linear(hidden, self + ".self.key");
            Tensor *v = ops.linear(hidden, self + ".self.value");
            Tensor *attended =
                    ops.attention(q, k, v, 12, nullptr, attention_mask);
            attended = ops.linear(attended, self + ".output.dense");
            hidden = ops.norm(ggml_add(ctx, hidden, attended),
                              self + ".output.LayerNorm", 1e-12f);
            const std::string cross = base + ".crossattention";
            cross_keys[i] =
                    batch == 1 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 768, 40)
                               : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 768, 40,
                                                    batch);
            cross_values[i] =
                    batch == 1 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 768, 40)
                               : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 768, 40,
                                                    batch);
            ggml_set_input(cross_keys[i]);
            ggml_set_input(cross_values[i]);
            // Keep cached K/V alive across repeated scheduler executions.
            ggml_set_output(cross_keys[i]);
            ggml_set_output(cross_values[i]);
            ggml_backend_sched_set_tensor_backend(sched, cross_keys[i],
                                                  model.backend);
            ggml_backend_sched_set_tensor_backend(sched, cross_values[i],
                                                  model.backend);
            q = ops.linear(hidden, cross + ".self.query");
            attended = ops.attention(q, cross_keys[i], cross_values[i], 12);
            attended = ops.linear(attended, cross + ".output.dense");
            hidden = ops.norm(ggml_add(ctx, hidden, attended),
                              cross + ".output.LayerNorm", 1e-12f);
            Tensor *ff = ggml_gelu_erf(
                    ctx, ops.linear(hidden, base + ".intermediate.dense"));
            ff = ops.linear(ff, base + ".output.dense");
            hidden = ops.norm(ggml_add(ctx, hidden, ff),
                              base + ".output.LayerNorm", 1e-12f);
        }
        const std::string base = "text_decoder.cls.predictions";
        hidden = ggml_gelu_erf(ctx,
                               ops.linear(hidden, base + ".transform.dense"));
        hidden = ops.norm(hidden, base + ".transform.LayerNorm", 1e-12f);
        hidden = ggml_mul_mat(ctx, ops.f32(ops.get(base + ".decoder.weight")),
                              hidden);
        hidden = ggml_add(ctx, hidden, ops.f32(ops.get(base + ".bias")));
        finish(hidden);
    }

    Tensor *attention_mask = nullptr;

    void finish(Tensor *result) {
        output = ops.f32(result);
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx, 20000, false);
        ggml_build_forward_expand(graph, output);
        if (produces_cross_cache)
            for (int i = 0; i < 12; ++i) {
                ggml_set_output(cross_keys[i]);
                ggml_set_output(cross_values[i]);
                ggml_build_forward_expand(graph, cross_keys[i]);
                ggml_build_forward_expand(graph, cross_values[i]);
            }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            Tensor *node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT)
                ggml_set_name(node, "ram_caption_matmul");
        }
        require(ggml_backend_sched_alloc_graph(sched, graph),
                "cannot allocate caption graph");
        if (model.profile) {
            std::map<std::string, int> fallback;
            int accelerator = 0;
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                Tensor *node = ggml_graph_node(graph, i);
                ggml_backend_t assigned =
                        ggml_backend_sched_get_tensor_backend(sched, node);
                if (assigned == model.cpu)
                    ++fallback[ggml_op_name(node->op)];
                else if (assigned == model.backend)
                    ++accelerator;
            }
            std::fprintf(stderr,
                         "caption %s nodes=%d accelerator=%d splits=%d\n",
                         is_decoder ? "decoder" : "encoder",
                         ggml_graph_n_nodes(graph), accelerator,
                         ggml_backend_sched_get_n_splits(sched));
            for (const auto &[operation, count] : fallback)
                std::fprintf(stderr, "caption CPU fallback: %s=%d\n",
                             operation.c_str(), count);
        }
    }

    void set_inputs(const std::vector<int32_t> &ids,
                    const std::vector<float> &cross,
                    const std::vector<float> &mask) {
        std::vector<int32_t> positions(ids.size());
        std::iota(positions.begin(), positions.end(), 0);
        ggml_backend_tensor_set(tokens, ids.data(), 0,
                                ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(position_ids, positions.data(), 0,
                                positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(image_or_tags, cross.data(), 0,
                                cross.size() * sizeof(float));
        ggml_backend_tensor_set(attention_mask, mask.data(), 0,
                                mask.size() * sizeof(float));
    }

    void set_decoder_inputs(const std::vector<int32_t> &ids,
                            const std::vector<float> &mask) {
        std::vector<int32_t> positions(ids.size());
        for (size_t i = 0; i < positions.size(); ++i)
            positions[i] = static_cast<int32_t>(i % attention_mask->ne[0]);
        ggml_backend_tensor_set(tokens, ids.data(), 0,
                                ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(position_ids, positions.data(), 0,
                                positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(attention_mask, mask.data(), 0,
                                mask.size() * sizeof(float));
    }

    void compute() {
        require(!aicore_cancel_requested(), "cancelled");
        require(ggml_backend_sched_graph_compute(sched, graph) ==
                        GGML_STATUS_SUCCESS,
                "caption graph compute failed");
    }

    void set_cross_cache(const CrossCache &cross) {
        for (int i = 0; i < 12; ++i) {
            const size_t key_bytes = cross.keys[i].size() * sizeof(float);
            const size_t value_bytes = cross.values[i].size() * sizeof(float);
            for (int beam = 0; beam < cross_keys[i]->ne[2]; ++beam) {
                ggml_backend_tensor_set(cross_keys[i], cross.keys[i].data(),
                                        beam * key_bytes, key_bytes);
                ggml_backend_tensor_set(cross_values[i], cross.values[i].data(),
                                        beam * value_bytes, value_bytes);
            }
        }
    }

    void read_cross_cache(CrossCache &cross) {
        for (int i = 0; i < 12; ++i) {
            ggml_backend_tensor_get(cross_keys[i], cross.keys[i].data(), 0,
                                    cross.keys[i].size() * sizeof(float));
            ggml_backend_tensor_get(cross_values[i], cross.values[i].data(), 0,
                                    cross.values[i].size() * sizeof(float));
        }
    }
};

struct CaptionEngine {
    Model &model;
    WordPieceTokenizer tokenizer;
    ggml_context *encoder_context = nullptr;
    ggml_context *decoder_context = nullptr;
    ggml_backend_sched_t encoder_sched = nullptr;
    ggml_backend_sched_t decoder_sched = nullptr;
    std::unique_ptr<CaptionGraph> encoder;
    std::unique_ptr<CaptionGraph> decoder;
    CrossCache cross_cache;
    std::vector<float> decoder_mask;
    std::vector<int32_t> prompt;
    int decoder_length = 0;
    int max_length;
    int batch_beams;
    std::string trace_dir;

    void trace_f32(const std::string &name,
                   const float *data,
                   size_t count) const {
        if (trace_dir.empty()) return;
        std::filesystem::create_directories(trace_dir);
        std::ofstream file(trace_dir + "/" + name,
                           std::ios::binary | std::ios::trunc);
        require(file.good(), "cannot write caption trace " + name);
        file.write(reinterpret_cast<const char *>(data), count * sizeof(float));
    }

    void trace_i32(const std::string &name,
                   const int32_t *data,
                   size_t count) const {
        if (trace_dir.empty()) return;
        std::filesystem::create_directories(trace_dir);
        std::ofstream file(trace_dir + "/" + name,
                           std::ios::binary | std::ios::trunc);
        require(file.good(), "cannot write caption trace " + name);
        file.write(reinterpret_cast<const char *>(data),
                   count * sizeof(int32_t));
    }

    static ggml_context *make_context() {
        ggml_init_params params{};
        params.mem_size = 128 * 1024 * 1024;
        params.no_alloc = true;
        ggml_context *context = ggml_init(params);
        require(context, "cannot allocate caption graph metadata");
        return context;
    }

    void build_decoder(int length) {
        decoder_length = length;
        decoder->build_decoder_cached_cross(length, batch_beams);
        decoder_mask.resize(length * length);
        for (int query = 0; query < length; ++query)
            for (int key = 0; key < length; ++key)
                decoder_mask[query * length + key] =
                        key <= query ? 0.f : -10000.f;
    }

    CaptionEngine(Model &m,
                  const std::vector<std::string> &vocab,
                  int maximum,
                  int beams,
                  std::string trace)
        : model(m),
          tokenizer(vocab),
          max_length(maximum),
          batch_beams(beams),
          trace_dir(std::move(trace)) {
        require(maximum >= 10 && maximum <= 512,
                "caption max length must be 10..512");
        encoder_context = make_context();
        decoder_context = make_context();
        encoder_sched = model.make_scheduler();
        decoder_sched = model.make_scheduler();
        encoder = std::make_unique<CaptionGraph>(model, encoder_context,
                                                 encoder_sched);
        decoder = std::make_unique<CaptionGraph>(model, decoder_context,
                                                 decoder_sched);
        encoder->build_encoder(40);
        build_decoder(std::min(30, maximum));
        prompt = tokenizer.encode("a picture of ", maximum);
        prompt[0] = 30522;
        prompt.pop_back();
    }

    void extend_decoder() {
        decoder.reset();
        ggml_backend_sched_free(decoder_sched);
        ggml_free(decoder_context);
        decoder_context = make_context();
        decoder_sched = model.make_scheduler();
        decoder = std::make_unique<CaptionGraph>(model, decoder_context,
                                                 decoder_sched);
        build_decoder(max_length);
        decoder->set_cross_cache(cross_cache);
    }

    ~CaptionEngine() {
        encoder.reset();
        decoder.reset();
        if (encoder_sched) ggml_backend_sched_free(encoder_sched);
        if (decoder_sched) ggml_backend_sched_free(decoder_sched);
        if (encoder_context) ggml_free(encoder_context);
        if (decoder_context) ggml_free(decoder_context);
    }

    std::string generate(const std::vector<float> &image,
                         const std::string &tags) {
        std::vector<int32_t> tag_ids = tokenizer.encode(tags, 40);
        tag_ids[0] = 30523;
        const size_t used_tags = tag_ids.size();
        tag_ids.resize(40, 0);
        std::vector<float> encoder_mask(40 * 40);
        for (int query = 0; query < 40; ++query)
            for (int key = 0; key < 40; ++key)
                encoder_mask[query * 40 + key] =
                        key < static_cast<int>(used_tags) ? 0.f : -10000.f;

        encoder->set_inputs(tag_ids, image, encoder_mask);
        encoder->compute();
        if (!trace_dir.empty()) {
            trace_i32("caption_tag_ids.i32", tag_ids.data(), tag_ids.size());
            trace_i32("caption_prompt.i32", prompt.data(), prompt.size());
            std::vector<float> hidden(ggml_nelements(encoder->output));
            ggml_backend_tensor_get(encoder->output, hidden.data(), 0,
                                    hidden.size() * sizeof(float));
            trace_f32("caption_encoder.f32", hidden.data(), hidden.size());
        }
        encoder->read_cross_cache(cross_cache);
        decoder->set_cross_cache(cross_cache);

        constexpr int beam_count = 3;
        constexpr int vocabulary = 30524;
        struct Beam {
            std::vector<int32_t> tokens;
            double score;
            int score_length;
        };
        struct Candidate {
            size_t parent;
            int32_t token;
            double score;
        };
        const auto normalized = [](const Beam &beam) {
            return beam.score / std::max(1, beam.score_length);
        };
        std::vector<Beam> beams{{prompt, 0.0, 0}};
        std::vector<Beam> finished;
        std::vector<float> logits(vocabulary);
        for (int step = static_cast<int>(prompt.size());
             step < max_length && !beams.empty(); ++step) {
            if (step > decoder_length) extend_decoder();
            std::vector<Candidate> candidates;
            if (batch_beams == beam_count) {
                std::vector<int32_t> input(batch_beams * decoder_length, 0);
                for (int index = 0; index < batch_beams; ++index) {
                    const auto &sequence =
                            beams[std::min<size_t>(index, beams.size() - 1)]
                                    .tokens;
                    std::copy(sequence.begin(), sequence.end(),
                              input.begin() + index * decoder_length);
                }
                decoder->set_decoder_inputs(input, decoder_mask);
                decoder->compute();
            }
            for (size_t beam_index = 0; beam_index < beams.size();
                 ++beam_index) {
                const auto &beam = beams[beam_index];
                if (!trace_dir.empty()) {
                    const std::string name =
                            "caption_tokens_" +
                            std::to_string(step -
                                           static_cast<int>(prompt.size())) +
                            "_beam_" + std::to_string(beam_index) + ".i32";
                    trace_i32(name, beam.tokens.data(), beam.tokens.size());
                }
                if (batch_beams == 1) {
                    std::vector<int32_t> input = beam.tokens;
                    input.resize(decoder_length, 0);
                    decoder->set_decoder_inputs(input, decoder_mask);
                    decoder->compute();
                }
                const size_t column =
                        (batch_beams == beam_count ? beam_index * decoder_length
                                                   : 0) +
                        beam.tokens.size() - 1;
                ggml_backend_tensor_get(decoder->output, logits.data(),
                                        column * vocabulary * sizeof(float),
                                        logits.size() * sizeof(float));
                if (!trace_dir.empty()) {
                    const std::string stem =
                            "caption_step_" +
                            std::to_string(step -
                                           static_cast<int>(prompt.size())) +
                            "_beam_" + std::to_string(beam_index);
                    trace_f32(stem + ".f32", logits.data(), logits.size());
                }
                const float maximum =
                        *std::max_element(logits.begin(), logits.end());
                double sum = 0.0;
                for (float value : logits)
                    sum += std::exp(double(value - maximum));
                const double logsum = double(maximum) + std::log(sum);
                std::vector<int32_t> order(vocabulary);
                std::iota(order.begin(), order.end(), 0);
                const size_t top =
                        std::min<size_t>(order.size(), beam_count * 2);
                std::partial_sort(order.begin(), order.begin() + top,
                                  order.end(), [&](int32_t a, int32_t b) {
                                      return logits[a] > logits[b];
                                  });
                for (size_t i = 0; i < top; ++i) {
                    const int32_t token = order[i];
                    if (token == 102 && step + 1 < 10) continue;
                    candidates.push_back(
                            {beam_index, token,
                             beam.score + double(logits[token]) - logsum});
                }
            }
            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate &a, const Candidate &b) {
                          return a.score > b.score;
                      });
            const std::vector<Beam> previous_beams = beams;
            beams.clear();
            const auto add_finished = [&](Beam hypothesis) {
                finished.push_back(std::move(hypothesis));
                std::sort(finished.begin(), finished.end(),
                          [&](const Beam &a, const Beam &b) {
                              return normalized(a) > normalized(b);
                          });
                if (finished.size() > beam_count) finished.resize(beam_count);
            };
            for (size_t rank = 0; rank < candidates.size(); ++rank) {
                const auto &candidate = candidates[rank];
                if (candidate.token == 102) {
                    // HuggingFace BeamSearchScorer accepts EOS only among the
                    // first `num_beams` global candidates and scores its
                    // length with the EOS token included.
                    if (rank >= beam_count) continue;
                    Beam hypothesis = previous_beams[candidate.parent];
                    hypothesis.score = candidate.score;
                    hypothesis.score_length =
                            step + 1 - static_cast<int>(prompt.size());
                    add_finished(std::move(hypothesis));
                } else if (beams.size() < beam_count) {
                    Beam next = previous_beams[candidate.parent];
                    next.tokens.push_back(candidate.token);
                    next.score = candidate.score;
                    next.score_length = static_cast<int>(next.tokens.size() -
                                                         prompt.size());
                    beams.push_back(std::move(next));
                }
                if (beams.size() == beam_count) break;
            }
            // Match Transformers' default `early_stopping=False` heuristic:
            // estimate the best possible completed score from the highest
            // scoring *running* beam at the current generated length.  Using
            // candidates.front() is not equivalent when the best candidate
            // is EOS, because that candidate is no longer a running beam.
            if (finished.size() >= beam_count && !beams.empty()) {
                const double best_possible =
                        beams.front().score /
                        std::max(1, step + 1 - static_cast<int>(prompt.size()));
                if (normalized(finished.back()) >= best_possible) break;
            }
        }
        finished.insert(finished.end(), beams.begin(), beams.end());
        require(!finished.empty(), "caption decoder returned no hypotheses");
        const auto &best =
                *std::max_element(finished.begin(), finished.end(),
                                  [&](const Beam &a, const Beam &b) {
                                      return normalized(a) < normalized(b);
                                  });
        std::string caption = tokenizer.decode(best.tokens);
        return caption;
    }
};

}  // namespace

namespace aicore {
namespace tagging {
struct Session::Impl {
    Model model;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> graph_context{
            nullptr, ggml_free};
    Tensor *input = nullptr, *output = nullptr, *embeddings = nullptr;
    ggml_cgraph *graph = nullptr;
    std::unique_ptr<CaptionEngine> caption;
    std::string resolved_device;
    std::set<int32_t> deleted;
    int threads = 1;

    explicit Impl(const aicore_tagging_options &options) {
        threads = std::max(1, options.threads);
#if defined(_OPENMP)
        // The upstream RAM runner disables OpenMP's adaptive team sizing and
        // fixes the team width before loading the graph.  Keep that process
        // contract for the AICore worker as well; otherwise libgomp may
        // resize teams between repeated images and skew both latency and
        // activation-quantized results.
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
#endif
        require(options.model_path && *options.model_path,
                "model path is required");
        const auto safe = [](const char *s) { return std::string(s ? s : ""); };
        model.load(options.model_path, options.device ? options.device : "auto",
                   options.threads, options.generate_caption != 0,
                   safe(options.tag_list_path), safe(options.threshold_path),
                   safe(options.label_embedding_path));
        resolved_device = model.backend_lease.device();
        auto lock = runtime::lock_backend_leases(
                {model.backend_lease, model.cpu_lease});
        require(!model.caption || model.architecture == "tag2text",
                "caption requires Tag2Text");
        require(!model.tag_names.empty() &&
                        model.tag_names.size() == model.tag_thresholds.size(),
                "self-contained GGUF tag list/threshold metadata required");
        model.cuda_f16_compute = model.is_cuda && !model.caption;
        // CPU Q8 matmul otherwise quantizes activations as well as weights.
        // Protect the high-resolution QKV projections where those errors
        // enter attention. Conversion is owner-scoped and cached once; the
        // GGUF weights, tag thresholds and remaining graph stay unchanged.
        model.cpu_floating_qkv =
                model.is_cpu && !model.caption && model.architecture == "ram" &&
                options.cpu_precision == AICORE_TAGGING_CPU_AUTO;
        ggml_init_params ip{};
        ip.mem_size = 128 * 1024 * 1024;
        ip.no_alloc = true;
        graph_context.reset(ggml_init(ip));
        require(bool(graph_context), "cannot allocate graph metadata");
        auto *ctx = graph_context.get();
        input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 384, 384, 3, 1);
        ggml_set_name(input, "input_image");
        ggml_set_input(input);
        InferenceGraph builder{model, ctx, false, {}, nullptr};
        builder.prepare_constants();
        output = builder.build(input);
        embeddings = builder.image_embeddings;
        ggml_set_name(output, "ram_logits");
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx, 20000, false);
        ggml_build_forward_expand(graph, output);
        if (embeddings) ggml_build_forward_expand(graph, embeddings);
        // Caption's F32 route also covers attention mul_mat nodes; tagging
        // retains F16 operands with F32 accumulation exactly as upstream.
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            Tensor *node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT)
                ggml_set_name(node, model.caption ? "ram_caption_matmul"
                                                  : "ram_tags_matmul");
        }
        require(ggml_nelements(output) == int64_t(model.tag_names.size()),
                "logit/label shape mismatch");
        require(ggml_backend_sched_alloc_graph(model.sched, graph),
                "cannot allocate tagging graph");
        if (model.caption) {
            require(!model.bert_vocab.empty(),
                    "GGUF tokenizer vocabulary is required");
            deleted.insert(model.delete_tag_indices.begin(),
                           model.delete_tag_indices.end());
            caption = std::make_unique<CaptionEngine>(
                    model, model.bert_vocab, options.max_length,
                    options.beam_batch ? options.beam_batch
                                       : (model.is_cpu ? 1 : 3),
                    std::string{});
        }
    }
    ~Impl() {
        // Schedulers hold graph nodes. Release them while metadata and leases
        // still live; shared backends are never freed by the task.
        auto lock = runtime::lock_backend_leases(
                {model.backend_lease, model.cpu_lease});
        caption.reset();
        if (model.sched) {
            ggml_backend_sched_free(model.sched);
            model.sched = nullptr;
        }
    }
};

Session::Session(const aicore_tagging_options &options)
    : impl_(std::make_unique<Impl>(options)) {}
Session::~Session() = default;
const std::vector<std::string> &Session::names() const {
    return impl_->model.tag_names;
}
const std::vector<float> &Session::thresholds() const {
    return impl_->model.tag_thresholds;
}
float Session::selectionThreshold(size_t index,
                                  const aicore_tagging_request &request) const {
    const float base =
            request.threshold < 0 ? thresholds()[index] : request.threshold;
    return base - (impl_->model.architecture == "tag2text"
                           ? request.threshold_epsilon
                           : 0.f);
}
bool Session::excluded(size_t index) const {
    return impl_->caption && impl_->deleted.count(int32_t(index));
}
const std::string &Session::device() const { return impl_->resolved_device; }

Output Session::infer(const img::Image &image,
                      const aicore_tagging_request &request) {
    using Clock = std::chrono::steady_clock;
    const auto elapsed = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto start = Clock::now();
    require(!aicore_cancel_requested(), "cancelled");
    auto pixels = preprocess(image, impl_->threads);
    const auto preprocessed = Clock::now();
    auto &s = *impl_;
    auto lock = runtime::lock_backend_leases(
            {s.model.backend_lease, s.model.cpu_lease});
    Output result;
    result.logits.resize(names().size());
    ggml_backend_tensor_set(s.input, pixels.data(), 0,
                            pixels.size() * sizeof(float));
    require(!aicore_cancel_requested(), "cancelled");
    require(ggml_backend_sched_graph_compute(s.model.sched, s.graph) ==
                    GGML_STATUS_SUCCESS,
            "tagging graph failed");
    ggml_backend_tensor_get(s.output, result.logits.data(), 0,
                            result.logits.size() * sizeof(float));
    if (s.caption) {
        std::string tags;
        for (size_t i = 0; i < result.logits.size(); ++i) {
            if (s.deleted.count(int32_t(i))) continue;
            const float threshold = selectionThreshold(i, request);
            if (1.f / (1.f + std::exp(-result.logits[i])) > threshold) {
                if (!tags.empty()) tags += " | ";
                tags += names()[i];
            }
        }
        if (request.specified_tags && *request.specified_tags &&
            std::string(request.specified_tags) != "None" &&
            std::string(request.specified_tags) != "none") {
            tags.clear();
            for (char ch : std::string(request.specified_tags))
                tags += ch == ',' ? " | " : std::string(1, ch);
        }
        std::vector<float> embeddings(ggml_nelements(s.embeddings));
        ggml_backend_tensor_get(s.embeddings, embeddings.data(), 0,
                                embeddings.size() * sizeof(float));
        result.caption = s.caption->generate(embeddings, tags);
    }
    require(!aicore_cancel_requested(), "cancelled");
    const auto end = Clock::now();
    result.timings.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    result.timings.valid_fields =
            AICORE_TIMING_PREPROCESS | AICORE_TIMING_INFERENCE;
    result.timings.preprocess_ms = elapsed(start, preprocessed);
    result.timings.inference_ms = elapsed(preprocessed, end);
    return result;
}
}  // namespace tagging
}  // namespace aicore

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "relateanything/gguf_model.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace relateanything {

namespace {

constexpr char kInputMagic[] = "RAIOv1\0";
constexpr char kOutputMagic[] = "RAOOv1\0";
constexpr const char *kGraphKind = "relation_pair_linear_v1";

template <typename T>
void read_exact(std::ifstream &stream, T *value, std::size_t count = 1) {
    const auto bytes = sizeof(T) * count;
    stream.read(reinterpret_cast<char *>(value),
                static_cast<std::streamsize>(bytes));
    if (!stream || static_cast<std::size_t>(stream.gcount()) != bytes) {
        throw std::runtime_error("truncated binary input");
    }
}

std::size_t checked_product(std::size_t a, std::size_t b, const char *label) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::runtime_error(std::string(label) + " is too large");
    }
    return a * b;
}

std::size_t metadata_size(const gguf_context *context, const char *key) {
    const int64_t id = gguf_find_key(context, key);
    if (id < 0)
        throw std::runtime_error(std::string("missing GGUF metadata: ") + key);
    switch (gguf_get_kv_type(context, id)) {
        case GGUF_TYPE_UINT32:
            return gguf_get_val_u32(context, id);
        case GGUF_TYPE_INT32: {
            const auto value = gguf_get_val_i32(context, id);
            if (value < 0)
                throw std::runtime_error(
                        std::string("negative GGUF metadata: ") + key);
            return static_cast<std::size_t>(value);
        }
        case GGUF_TYPE_UINT64:
            return static_cast<std::size_t>(gguf_get_val_u64(context, id));
        case GGUF_TYPE_INT64: {
            const auto value = gguf_get_val_i64(context, id);
            if (value < 0)
                throw std::runtime_error(
                        std::string("negative GGUF metadata: ") + key);
            return static_cast<std::size_t>(value);
        }
        default:
            throw std::runtime_error(
                    std::string("GGUF metadata is not an integer: ") + key);
    }
}

void read_tensor_data(const std::filesystem::path &path,
                      const gguf_context *gguf,
                      const std::map<std::string, ggml_tensor *> &tensors) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open GGUF tensor data: " +
                                 path.string());
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        const char *name = gguf_get_tensor_name(gguf, i);
        const auto it = tensors.find(name);
        if (it == tensors.end() || !it->second)
            throw std::runtime_error("GGUF tensor lookup failed");
        ggml_tensor *tensor = it->second;
        const std::size_t offset =
                gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, i);
        const std::size_t bytes = ggml_nbytes(tensor);
        std::vector<std::uint8_t> data(bytes);
        stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!stream)
            throw std::runtime_error("cannot seek GGUF tensor data: " +
                                     std::string(name));
        if (bytes != 0) read_exact(stream, data.data(), bytes);
        ggml_backend_tensor_set(tensor, data.data(), 0, bytes);
    }
}

void validate_shape(const ggml_tensor *tensor,
                    std::int64_t n0,
                    std::int64_t n1,
                    const char *name) {
    if (!tensor || tensor->ne[0] != n0 || tensor->ne[1] != n1 ||
        tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        throw std::runtime_error(
                std::string("unexpected shape for GGUF tensor: ") + name +
                " got [" + std::to_string(tensor ? tensor->ne[0] : -1) + "," +
                std::to_string(tensor ? tensor->ne[1] : -1) + "] expected [" +
                std::to_string(n0) + "," + std::to_string(n1) + "]");
    }
}

}  // namespace

Model::Model(const ModelOptions &options)
    : options_(options), backend_(std::make_unique<Backend>(options)) {
    const auto &path = options.model;
    require(!path.empty(), "model path is required in Options");
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &tensor_context_;
    gguf_ = gguf_init_from_file(path.string().c_str(), params);
    if (!gguf_ || !tensor_context_) {
        throw std::runtime_error("cannot load GGUF model: " + path.string());
    }

    const int64_t architecture_id =
            gguf_find_key(gguf_, "general.architecture");
    if (architecture_id >= 0 &&
        gguf_get_kv_type(gguf_, architecture_id) == GGUF_TYPE_STRING) {
        architecture_ = gguf_get_val_str(gguf_, architecture_id);
    }
    const int64_t graph_id = gguf_find_key(gguf_, "ra.graph_kind");
    require(graph_id >= 0 &&
                    gguf_get_kv_type(gguf_, graph_id) == GGUF_TYPE_STRING,
            "GGUF is missing string metadata ra.graph_kind");
    const std::string graph_kind = gguf_get_val_str(gguf_, graph_id);
    if (graph_kind == "relateanything_full_v1") {
        throw std::runtime_error(
                "official full-graph weights require FullModel and an image "
                "input "
                "(.raip or .raim); the legacy Model API accepts only adapter "
                "fixtures");
    }
    require(graph_kind == kGraphKind,
            "unsupported GGUF graph kind; expected relation_pair_linear_v1 or "
            "relateanything_full_v1");
    feature_dim_ = metadata_size(gguf_, "ra.feature_dim");
    hidden_dim_ = metadata_size(gguf_, "ra.hidden_dim");
    predicate_count_ = metadata_size(gguf_, "ra.predicate_count");
    require(feature_dim_ > 0 && hidden_dim_ > 0 && predicate_count_ > 0,
            "GGUF dimensions must be positive");

    for (int64_t i = 0; i < gguf_get_n_tensors(gguf_); ++i) {
        const char *name = gguf_get_tensor_name(gguf_, i);
        tensors_.emplace(name, ggml_get_tensor(tensor_context_, name));
    }
    const std::size_t input_dim =
            checked_product(feature_dim_, 2, "feature dimensions") + 8;
    validate_shape(
            tensor("ra.pair_proj.weight"), static_cast<std::int64_t>(input_dim),
            static_cast<std::int64_t>(hidden_dim_), "ra.pair_proj.weight");
    validate_shape(tensor("ra.pair_proj.bias"),
                   static_cast<std::int64_t>(hidden_dim_), 1,
                   "ra.pair_proj.bias");
    validate_shape(tensor("ra.predicate.weight"),
                   static_cast<std::int64_t>(hidden_dim_),
                   static_cast<std::int64_t>(predicate_count_),
                   "ra.predicate.weight");
    validate_shape(tensor("ra.predicate.bias"),
                   static_cast<std::int64_t>(predicate_count_), 1,
                   "ra.predicate.bias");
    validate_shape(tensor("ra.pair.weight"),
                   static_cast<std::int64_t>(hidden_dim_), 1, "ra.pair.weight");
    validate_shape(tensor("ra.pair.bias"), 1, 1, "ra.pair.bias");

    weight_buffer_ = ggml_backend_alloc_ctx_tensors(tensor_context_,
                                                    backend_->accelerator);
    if (!weight_buffer_)
        throw std::runtime_error("cannot allocate GGUF weights on backend");
    read_tensor_data(path, gguf_, tensors_);
}

Model::~Model() {
    if (weight_buffer_) ggml_backend_buffer_free(weight_buffer_);
    if (tensor_context_) ggml_free(tensor_context_);
    if (gguf_) gguf_free(gguf_);
}

ggml_tensor *Model::tensor(const char *name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second)
        throw std::runtime_error(std::string("missing GGUF tensor: ") + name);
    return it->second;
}

void Model::require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

PairOutput Model::infer(const PairInput &input) {
    require(input.object_count >= 2, "at least two objects are required");
    require(input.object_count <= options_.max_boxes,
            "object count exceeds --max-boxes");
    require(input.feature_dim == feature_dim_,
            "input feature dimension does not match GGUF");
    const std::size_t stride = input.feature_stride == 0 ? input.feature_dim
                                                         : input.feature_stride;
    require(stride >= input.feature_dim,
            "input feature stride is smaller than feature dimension");
    require(input.object_features.size() >= checked_product(input.object_count,
                                                            stride,
                                                            "object features"),
            "object feature buffer is too small");
    require(input.boxes_cxcywh.size() >= checked_product(input.object_count,
                                                         std::size_t(4),
                                                         "boxes"),
            "box buffer is too small");

    const std::size_t pair_count =
            input.object_count * (input.object_count - 1);
    const std::size_t input_dim = feature_dim_ * 2 + 8;
    std::vector<float> pair_data(
            checked_product(input_dim, pair_count, "pair input"));
    std::vector<std::int32_t> subject_indices;
    std::vector<std::int32_t> object_indices;
    subject_indices.reserve(pair_count);
    object_indices.reserve(pair_count);
    std::size_t pair = 0;
    for (std::size_t subject = 0; subject < input.object_count; ++subject) {
        for (std::size_t object = 0; object < input.object_count; ++object) {
            if (subject == object) continue;
            float *dst = pair_data.data() + pair * input_dim;
            const float *subject_features =
                    input.object_features.data() + subject * stride;
            const float *object_features =
                    input.object_features.data() + object * stride;
            std::copy_n(subject_features, feature_dim_, dst);
            std::copy_n(object_features, feature_dim_, dst + feature_dim_);
            std::copy_n(input.boxes_cxcywh.data() + subject * 4, 4,
                        dst + feature_dim_ * 2);
            std::copy_n(input.boxes_cxcywh.data() + object * 4, 4,
                        dst + feature_dim_ * 2 + 4);
            subject_indices.push_back(static_cast<std::int32_t>(subject));
            object_indices.push_back(static_cast<std::int32_t>(object));
            ++pair;
        }
    }

    const std::size_t graph_bytes = ggml_tensor_overhead() * 512 +
                                    ggml_graph_overhead_custom(512, false);
    ggml_init_params params{graph_bytes, nullptr, true};
    ggml_context *context = ggml_init(params);
    require(context != nullptr, "cannot create inference graph context");
    ggml_tensor *pair_input = ggml_new_tensor_2d(
            context, GGML_TYPE_F32, static_cast<int64_t>(input_dim),
            static_cast<int64_t>(pair_count));
    ggml_set_input(pair_input);
    ggml_tensor *hidden =
            ggml_mul_mat(context, tensor("ra.pair_proj.weight"), pair_input);
    ggml_set_name(hidden, "relateanything_matmul");
    hidden = ggml_add(context, hidden, tensor("ra.pair_proj.bias"));
    hidden = ggml_gelu(context, hidden);
    ggml_tensor *predicate_logits =
            ggml_mul_mat(context, tensor("ra.predicate.weight"), hidden);
    ggml_set_name(predicate_logits, "relateanything_matmul");
    ggml_tensor *predicate =
            ggml_add(context, predicate_logits, tensor("ra.predicate.bias"));
    ggml_tensor *pair_projection =
            ggml_mul_mat(context, tensor("ra.pair.weight"), hidden);
    ggml_set_name(pair_projection, "relateanything_matmul");
    ggml_tensor *pair_logits =
            ggml_add(context, pair_projection, tensor("ra.pair.bias"));
    ggml_cgraph *graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, predicate);
    ggml_build_forward_expand(graph, pair_logits);

    ggml_backend_sched_reset(backend_->scheduler);
    require(ggml_backend_sched_alloc_graph(backend_->scheduler, graph),
            "cannot allocate inference graph");
    const auto start = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(pair_input, pair_data.data(), 0,
                            ggml_nbytes(pair_input));
    require(ggml_backend_sched_graph_compute(backend_->scheduler, graph) ==
                    GGML_STATUS_SUCCESS,
            "graph execution failed");
    ggml_backend_sched_synchronize(backend_->scheduler);
    const auto end = std::chrono::steady_clock::now();

    PairOutput output;
    output.pair_count = pair_count;
    output.predicate_count = predicate_count_;
    output.subject_indices = std::move(subject_indices);
    output.object_indices = std::move(object_indices);
    output.predicate_logits.resize(predicate_count_ * pair_count);
    output.pair_logits.resize(pair_count);
    ggml_backend_tensor_get(predicate, output.predicate_logits.data(), 0,
                            ggml_nbytes(predicate));
    ggml_backend_tensor_get(pair_logits, output.pair_logits.data(), 0,
                            ggml_nbytes(pair_logits));
    output.milliseconds =
            std::chrono::duration<double, std::milli>(end - start).count();
    ggml_free(context);
    return output;
}

PairInput read_pair_input(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open input: " + path.string());
    char magic[sizeof(kInputMagic)]{};
    read_exact(stream, magic, sizeof(magic));
    if (std::memcmp(magic, kInputMagic, sizeof(magic)) != 0)
        throw std::runtime_error("invalid RAIOv1 input magic");
    std::uint32_t objects = 0;
    std::uint32_t features = 0;
    read_exact(stream, &objects);
    read_exact(stream, &features);
    if (objects == 0 || features == 0)
        throw std::runtime_error("RAIO input dimensions must be positive");
    PairInput input;
    input.object_count = objects;
    input.feature_dim = features;
    input.feature_stride = features;
    input.object_features.resize(
            checked_product(objects, features, "RAIO features"));
    input.boxes_cxcywh.resize(
            checked_product(objects, std::size_t(4), "RAIO boxes"));
    read_exact(stream, input.object_features.data(),
               input.object_features.size());
    read_exact(stream, input.boxes_cxcywh.data(), input.boxes_cxcywh.size());
    return input;
}

void write_pair_output(const std::filesystem::path &path,
                       const PairOutput &output) {
    if (output.pair_count != output.subject_indices.size() ||
        output.pair_count != output.object_indices.size() ||
        output.pair_logits.size() != output.pair_count ||
        output.predicate_logits.size() !=
                output.pair_count * output.predicate_count) {
        throw std::runtime_error("invalid pair output dimensions");
    }
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot create output: " + path.string());
    stream.write(kOutputMagic, sizeof(kOutputMagic));
    const auto pairs = static_cast<std::uint32_t>(output.pair_count);
    const auto predicates = static_cast<std::uint32_t>(output.predicate_count);
    stream.write(reinterpret_cast<const char *>(&pairs), sizeof(pairs));
    stream.write(reinterpret_cast<const char *>(&predicates),
                 sizeof(predicates));
    stream.write(reinterpret_cast<const char *>(output.subject_indices.data()),
                 static_cast<std::streamsize>(output.subject_indices.size() *
                                              sizeof(std::int32_t)));
    stream.write(reinterpret_cast<const char *>(output.object_indices.data()),
                 static_cast<std::streamsize>(output.object_indices.size() *
                                              sizeof(std::int32_t)));
    stream.write(reinterpret_cast<const char *>(output.pair_logits.data()),
                 static_cast<std::streamsize>(output.pair_logits.size() *
                                              sizeof(float)));
    stream.write(reinterpret_cast<const char *>(output.predicate_logits.data()),
                 static_cast<std::streamsize>(output.predicate_logits.size() *
                                              sizeof(float)));
    if (!stream)
        throw std::runtime_error("failed writing output: " + path.string());
}

}  // namespace relateanything

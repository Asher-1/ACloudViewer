// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QTemporaryDir>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "src/common/ggml_backend_registry.hpp"
#include "src/tasks/instantmesh/dino.cpp"

namespace {
// Removing load validation must fail this test: a valid GGUF container is not
// necessarily an executable DINO model.
bool malformed_load_check(ggml_backend_t backend, bool invalid_metadata) {
    QTemporaryDir directory;
    if (!directory.isValid()) return false;
    const std::string path =
            directory.filePath("incomplete.gguf").toStdString();
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
            ggml_init({1u << 20, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(gguf_init_empty(),
                                                             gguf_free);
    if (!ctx || !gguf) return false;
    auto* tensor = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 16, 768);
    ggml_set_name(tensor, "camera_embedder.0.weight");
    ggml_set_zero(tensor);
    gguf_add_tensor(gguf.get(), tensor);
    if (invalid_metadata)
        gguf_set_val_i32(gguf.get(), "dino.num_attention_heads", 0);
    if (!gguf_write_to_file(gguf.get(), path.c_str(), false)) return false;
    instantmesh::DinoModel model;
    std::string error;
    const bool loaded = instantmesh::dino_load(path, backend, model, &error);
    const bool rejected =
            !loaded &&
            error.find(invalid_metadata ? "hparams" : "missing tensor") !=
                    std::string::npos;
    const bool released = !model.gguf.ctx && !model.gguf.gguf &&
                          !model.gguf.buffer && model.gguf.tensors.empty() &&
                          !model.backend;
    const bool retried = !instantmesh::dino_load(path, backend, model, nullptr);
    std::printf(
            "DINO malformed %s rejected=%d released=%d retried=%d error=%s\n",
            invalid_metadata ? "metadata" : "tensors", int(rejected),
            int(released), int(retried), error.c_str());
    return rejected && released && retried;
}

bool linear_check(ggml_backend_t backend, ggml_type type) {
    constexpr int k = 96, rows = 64, columns = 32;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
            ggml_init({1u << 20, nullptr, true}), ggml_free);
    if (!ctx) return false;
    auto* weights = ggml_new_tensor_2d(ctx.get(), type, k, rows);
    auto* input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, columns);
    auto* output = instantmesh::linear(ctx.get(), input, weights, nullptr);
    auto* graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    auto alloc =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) {
        ggml_gallocr_free(alloc);
        return false;
    }
    std::vector<float> w(k * rows), x(k * columns), actual(rows * columns);
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = 0.13f * std::sin(float(i) * 0.071f) + 0.000123f;
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = 0.29f * std::cos(float(i) * 0.051f) - 0.000217f;
    if (type == GGML_TYPE_F16) {
        std::fill(w.begin(), w.end(), 0x1p-14f);
        std::fill(x.begin(), x.end(), 1.0f);
        for (int row = 0; row < rows; ++row) w[row * k] = 1.0f;
        std::vector<ggml_fp16_t> half(w.size());
        ggml_fp32_to_fp16_row(w.data(), half.data(), w.size());
        ggml_backend_tensor_set(weights, half.data(), 0,
                                half.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(weights, w.data(), 0, w.size() * sizeof(float));
    }
    ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
    const bool computed =
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (computed)
        ggml_backend_tensor_get(output, actual.data(), 0,
                                actual.size() * sizeof(float));
    ggml_gallocr_free(alloc);
    if (!computed) return false;
    double maximum = 0;
    for (int col = 0; col < columns; ++col)
        for (int row = 0; row < rows; ++row) {
            double expected = 0;
            for (int i = 0; i < k; ++i)
                expected += double(w[row * k + i]) * x[col * k + i];
            if (!std::isfinite(actual[col * rows + row])) return false;
            maximum = std::max(
                    maximum,
                    std::abs(double(actual[col * rows + row]) - expected));
        }
    std::printf("DINO linear %s max_abs=%.9g\n", ggml_type_name(type), maximum);
    return maximum < 2e-5;
}

bool check(ggml_backend_t backend, ggml_type type) {
    constexpr int channels = 64, kernel = 16 * 16 * 3, sequence = 17, batch = 2;
    instantmesh::DinoModel model;
    model.backend = backend;
    model.hp.hidden_size = channels;
    model.hp.num_hidden_layers = 1;
    model.hp.num_attention_heads = 1;
    model.hp.image_size = 64;
    model.gguf.ctx = ggml_init({1u << 20, nullptr, true});
    if (!model.gguf.ctx) return false;
    auto* ctx = model.gguf.ctx;
    const auto add = [&](const char* name, ggml_tensor* tensor) {
        model.gguf.tensors.emplace(name, tensor);
    };
    add("camera_embedder.0.weight",
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, channels));
    add("camera_embedder.2.weight",
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, channels));
    for (const auto* name :
         {"camera_embedder.0.bias", "camera_embedder.2.bias",
          "model.embeddings.patch_embeddings.projection.bias",
          "model.layernorm.weight", "model.layernorm.bias"})
        add(name, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels));
    add("model.embeddings.cls_token",
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, channels, 1, 1));
    add("model.embeddings.position_embeddings",
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, channels, sequence, 1));
    // A zero residual block preserves patches while feeding camera input into
    // the production graph.
    add("model.encoder.layer.0.adaLN_modulation.1.weight",
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, 4 * channels));
    add("model.encoder.layer.0.adaLN_modulation.1.bias",
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * channels));
    for (const auto* part :
         {"layernorm_before", "layernorm_after", "attention.attention.query",
          "attention.attention.key", "attention.attention.value",
          "attention.output.dense", "intermediate.dense", "output.dense"}) {
        const std::string prefix = std::string("model.encoder.layer.0.") + part;
        model.gguf.tensors.emplace(
                prefix + ".weight",
                std::string(part).find("layernorm") == 0
                        ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels)
                        : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels,
                                             channels));
        model.gguf.tensors.emplace(
                prefix + ".bias",
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels));
    }
    auto* projection = ggml_new_tensor_4d(ctx, type, 16, 16, 3, channels);
    add("model.embeddings.patch_embeddings.projection.weight", projection);
    model.gguf.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!model.gguf.buffer) return false;
    for (const auto& entry : model.gguf.tensors) {
        std::vector<unsigned char> zero(ggml_nbytes(entry.second), 0);
        ggml_backend_tensor_set(entry.second, zero.data(), 0, zero.size());
    }
    std::vector<float> affine(channels, 1.0f);
    ggml_backend_tensor_set(model.gguf.tensors.at("model.layernorm.weight"),
                            affine.data(), 0, affine.size() * sizeof(float));
    std::vector<float> weights(kernel * channels, 0);
    for (int channel = 0; channel < channels; ++channel) {
        weights[channel * kernel] = 1.0f;
        if (channel < channels / 2)
            std::fill(weights.begin() + channel * kernel + 1,
                      weights.begin() + (channel + 1) * kernel, 0x1p-14f);
    }
    if (type == GGML_TYPE_F32)
        for (int channel = 0; channel < channels; ++channel) {
            std::fill(weights.begin() + channel * kernel,
                      weights.begin() + (channel + 1) * kernel, 0);
            weights[channel * kernel] =
                    0.13f * std::sin(float(channel) * 0.071f) + 0.000123f;
        }
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(weights.size());
        ggml_fp32_to_fp16_row(weights.data(), half.data(), weights.size());
        ggml_backend_tensor_set(projection, half.data(), 0,
                                half.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(projection, weights.data(), 0,
                                weights.size() * sizeof(float));
    }
    std::vector<float> image(batch * 3 * 64 * 64, 1), cameras(batch * 16, 0);
    int seq = 0, hidden = 0;
    std::unique_ptr<float, decltype(&std::free)> output(
            instantmesh::dino_encode(model, image.data(), cameras.data(), batch,
                                     64, 64, &seq, &hidden),
            std::free);
    if (!output || seq != sequence || hidden != channels) return false;
    // The first half of channels is 1+767/16384, the rest 1; normalization
    // yields +/-offset/sqrt(offset^2+epsilon). F16 inputs/weights are exactly
    // representable; F32 uses weights not exactly representable in F16 and
    // double normalization.
    const double offset = 767.0 / 32768.0;
    const double normalized = offset / std::sqrt(offset * offset + 1e-12);
    double mean = 0, variance = 0;
    if (type == GGML_TYPE_F32) {
        for (int channel = 0; channel < channels; ++channel)
            mean += weights[channel * kernel];
        mean /= channels;
        for (int channel = 0; channel < channels; ++channel) {
            const double delta = weights[channel * kernel] - mean;
            variance += delta * delta;
        }
        variance /= channels;
    }
    double maximum = 0;
    for (int b = 0; b < batch; ++b)
        for (int token = 0; token < sequence; ++token)
            for (int channel = 0; channel < channels; ++channel) {
                const double value =
                        type == GGML_TYPE_F32
                                ? (weights[channel * kernel] - mean) /
                                          std::sqrt(variance + 1e-12)
                        : channel < channels / 2 ? normalized
                                                 : -normalized;
                const double expected = token == 0 ? 0 : value;
                const float actual =
                        output.get()[(b * sequence + token) * channels +
                                     channel];
                if (!std::isfinite(actual)) return false;
                maximum =
                        std::max(maximum, std::abs(double(actual) - expected));
            }
    std::printf("DINO patch %s max_abs=%.9g\n", ggml_type_name(type), maximum);
    return maximum < 2e-5;
}
}  // namespace

int main(int argc, char** argv) {
    std::string error;
    auto lease = aicore::runtime::acquire_backend_lease(
            argc > 1 ? argv[1] : "cpu", 4, &error);
    if (!lease) {
        std::fprintf(stderr, "FAIL: explicit backend unavailable: %s\n",
                     error.c_str());
        return 1;
    }
    auto lock = lease.lock();
    const bool f32 = check(lease.handle(), GGML_TYPE_F32);
    const bool f16 = check(lease.handle(), GGML_TYPE_F16);
    const bool linearF32 = linear_check(lease.handle(), GGML_TYPE_F32);
    const bool linearF16 = linear_check(lease.handle(), GGML_TYPE_F16);
    const bool missingTensors = malformed_load_check(lease.handle(), false);
    const bool invalidMetadata = malformed_load_check(lease.handle(), true);
    std::printf("actual_backend=%s\n",
                ggml_backend_dev_name(ggml_backend_get_device(lease.handle())));
    return f32 && f16 && linearF32 && linearF16 && missingTensors &&
                           invalidMetadata
                   ? 0
                   : 1;
}

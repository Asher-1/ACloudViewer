// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "ggml-alloc.h"
#include "src/common/ggml_backend_registry.hpp"
#include "src/tasks/instantmesh/vae.hpp"

namespace {
ggml_backend_t cpuBackend = nullptr;
bool traceFailed = false;
bool rejectGraphAllocation = false;
size_t graphByteBudget = 0;
bool decoderBudgetProbe = false;
size_t peakGraphBytes = 0;

// Only the fallible external allocation is injected; the production graph and
// error return stay real.
bool allocateGraph(ggml_gallocr_t allocator, ggml_cgraph* graph) {
    if (rejectGraphAllocation || !ggml_gallocr_alloc_graph(allocator, graph))
        return false;
    const size_t bytes = ggml_gallocr_get_buffer_size(allocator, 0);
    peakGraphBytes = std::max(peakGraphBytes, bytes);
    if (decoderBudgetProbe) {
        std::printf("graph_bytes=%zu accepted=%d\n", bytes,
                    !graphByteBudget || bytes <= graphByteBudget);
        std::fflush(stdout);
    }
    return !graphByteBudget || bytes <= graphByteBudget;
}
constexpr int traceNodeLimit =
        60;  // CPU references copy only a bounded prefix, never all
             // full-decoder activations.
bool compareNode(int index, ggml_tensor* gpu, ggml_tensor* cpu, void*) {
    if (gpu->type != GGML_TYPE_F32 || cpu->type != GGML_TYPE_F32) return true;
    const size_t n = ggml_nelements(gpu);
    std::vector<float> left(n), right(n);
    ggml_backend_tensor_get(gpu, left.data(), 0, n * sizeof(float));
    ggml_backend_tensor_get(cpu, right.data(), 0, n * sizeof(float));
    double error = 0, magnitude = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(left[i]) || !std::isfinite(right[i])) {
            traceFailed = true;
            return false;
        }
        error = std::max(error, std::abs(double(left[i]) - right[i]));
        magnitude = std::max(magnitude, std::abs(double(right[i])));
    }
    std::printf(
            "node=%d op=%s shape=%lld,%lld,%lld,%lld max_abs=%.9g "
            "cpu_max=%.9g\n",
            index, ggml_op_name(gpu->op), (long long)gpu->ne[0],
            (long long)gpu->ne[1], (long long)gpu->ne[2], (long long)gpu->ne[3],
            error, magnitude);
    std::fflush(stdout);
    return true;
}

ggml_status traceCompute(ggml_backend_t backend, ggml_cgraph* graph) {
    if (decoderBudgetProbe) return ggml_backend_graph_compute(backend, graph);
    ggml_init_params params{1024 * 1024, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
            ggml_init(params), ggml_free);
    auto* prefix = ggml_new_graph(context.get());
    ggml_build_forward_expand(
            prefix, ggml_graph_node(graph, std::min(traceNodeLimit,
                                                    ggml_graph_n_nodes(graph)) -
                                                   1));
    if (!ggml_backend_compare_graph_backend(backend, cpuBackend, prefix,
                                            compareNode, nullptr, nullptr, 0) ||
        traceFailed)
        return GGML_STATUS_FAILED;
    // Node-prefix diagnosis changes fusion boundaries; final output still uses
    // the original full GPU graph, not CPU accuracy proof.
    return ggml_backend_graph_compute(backend, graph);
}
}  // namespace

// Interception is confined to this test TU; production components, public ABI
// and other tasks remain unchanged.
#define ggml_gallocr_alloc_graph allocateGraph
#define ggml_backend_graph_compute traceCompute
#include "src/tasks/instantmesh/vae.cpp"
#undef ggml_backend_graph_compute
#undef ggml_gallocr_alloc_graph

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "weights-parking") {
        std::string error;
        auto device =
                aicore::runtime::acquire_backend_lease(argv[2], 4, &error);
        auto host = aicore::runtime::acquire_backend_lease("cpu", 4, &error);
        if (!device || !host) return 1;
        auto lock = aicore::runtime::lock_backend_leases({device, host});
        instantmesh::VaeModel model;
        if (!instantmesh::vae_load(argv[3], device.handle(), model, &error))
            return 77;
        std::unordered_map<std::string, std::vector<uint8_t>> expected;
        for (const auto& entry : model.gguf.tensors) {
            auto& bytes = expected[entry.first];
            bytes.resize(ggml_nbytes(entry.second));
            ggml_backend_tensor_get(entry.second, bytes.data(), 0,
                                    bytes.size());
        }
        auto original = model.gguf.buffer;
        if (model.gguf.move_weights(nullptr, &error) ||
            model.gguf.buffer != original)
            return 1;
        for (int iteration = 0; iteration < 2; ++iteration) {
            for (auto target : {host.handle(), device.handle()}) {
                if (!model.gguf.move_weights(target, &error)) return 1;
                if (ggml_backend_buffer_get_type(model.gguf.buffer) !=
                    ggml_backend_get_default_buffer_type(target))
                    return 1;
                if (model.gguf.tensors.size() != expected.size()) return 1;
                for (const auto& entry : model.gguf.tensors) {
                    std::vector<uint8_t> actual(ggml_nbytes(entry.second));
                    ggml_backend_tensor_get(entry.second, actual.data(), 0,
                                            actual.size());
                    if (actual != expected.at(entry.first)) return 1;
                }
            }
        }
        std::printf(
                "weights_parking_byte_exact=1 iterations=2 actual_backend=%s\n",
                ggml_backend_dev_name(
                        ggml_backend_get_device(device.handle())));
        return 0;
    }
    if ((argc == 7 || argc == 8) && std::string(argv[1]) == "decoder-budget") {
        std::string error;
        auto lease = aicore::runtime::acquire_backend_lease(argv[2], 4, &error);
        if (!lease) return 1;
        auto lock = lease.lock();
        instantmesh::VaeModel model;
        if (!instantmesh::vae_load(argv[3], lease.handle(), model, &error))
            return 1;
        std::vector<float> latents(4 * 120 * 80);
        std::ifstream input(argv[4], std::ios::binary);
        if (!input.read(reinterpret_cast<char*>(latents.data()),
                        latents.size() * sizeof(float)) ||
            input.peek() != EOF || std::filesystem::exists(argv[5]))
            return 1;
        for (float& value : latents) value = value / 0.75f + 0.22f;
        graphByteBudget = size_t(std::stoull(argv[6])) * 1024 * 1024;
        decoderBudgetProbe = true;
        int h = 0, w = 0;
        std::unique_ptr<float, decltype(&std::free)> output(
                instantmesh::vae_decode(model, latents.data(), 1, 120, 80, &h,
                                        &w),
                std::free);
        if (!output || h != 960 || w != 640) {
            std::fprintf(
                    stderr,
                    "decoder_budget_failed=1 peak_graph_bytes=%zu budget=%zu\n",
                    peakGraphBytes, graphByteBudget);
            return 1;
        }
        const size_t count = size_t(3) * h * w;
        for (size_t i = 0; i < count; ++i)
            if (!std::isfinite(output.get()[i])) return 1;
        if (argc == 8) {
            std::vector<float> reference(count);
            std::ifstream baseline(argv[7], std::ios::binary);
            if (!baseline.read(reinterpret_cast<char*>(reference.data()),
                               count * sizeof(float)) ||
                baseline.peek() != EOF)
                return 77;
            double maximum = 0, squared = 0;
            for (size_t i = 0; i < count; ++i) {
                if (!std::isfinite(reference[i])) return 1;
                const double difference =
                        double(output.get()[i]) - reference[i];
                maximum = std::max(maximum, std::abs(difference));
                squared += difference * difference;
            }
            std::printf(
                    "decoder_same_weight_max_abs=%.9g rmse=%.9g "
                    "independent_reference_complete=0\n",
                    maximum, std::sqrt(squared / count));
            if (!(maximum < 1e-5)) return 1;
        }
        if (std::string(argv[5]) != "-") {
            std::ofstream file(argv[5], std::ios::binary);
            file.write(reinterpret_cast<const char*>(output.get()),
                       count * sizeof(float));
            if (!file) return 1;
        }
        std::printf(
                "decoder_budget_ok=1 peak_graph_bytes=%zu actual_backend=%s\n",
                peakGraphBytes,
                ggml_backend_dev_name(ggml_backend_get_device(lease.handle())));
        return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "allocation-failure") {
        std::string error;
        auto lease = aicore::runtime::acquire_backend_lease("cpu", 4, &error);
        if (!lease) return 1;
        auto lock = lease.lock();
        instantmesh::VaeModel model;
        if (!instantmesh::vae_load(argv[2], lease.handle(), model, &error)) {
            std::fprintf(stderr,
                         "Missing or invalid VAE allocation-test model: %s\n",
                         error.c_str());
            return 77;
        }
        rejectGraphAllocation = true;
        const bool encode = std::string(argv[3]) == "encode";
        if (!encode && std::string(argv[3]) != "decode") return 1;
        std::vector<float> input(encode ? 3 * 512 * 512 : 4 * 120 * 80, 0.0f);
        int h = -1, w = -1;
        std::unique_ptr<float, decltype(&std::free)> output(
                encode ? instantmesh::vae_encode_mode(model, input.data(), 1,
                                                      512, 512, &h, &w)
                       : instantmesh::vae_decode(model, input.data(), 1, 120,
                                                 80, &h, &w),
                std::free);
        if (output || h != -1 || w != -1) return 1;
        std::printf("allocation_failure_handled=1 stage=%s\n", argv[3]);
        return 0;
    }
    if (argc != 4) {
        std::fprintf(stderr,
                     "INCOMPLETE: requires device, VAE GGUF, dec_lat.bin\n");
        return 77;
    }
    std::string error;
    auto cpu = aicore::runtime::acquire_backend_lease("cpu", 4, &error);
    auto gpu = aicore::runtime::acquire_backend_lease(argv[1], 4, &error);
    if (!cpu || !gpu) return 1;
    auto lock = aicore::runtime::lock_backend_leases({cpu, gpu});
    cpuBackend = cpu.handle();
    instantmesh::VaeModel model;
    if (!instantmesh::vae_load(argv[2], gpu.handle(), model, &error)) {
        std::fprintf(stderr, "FAIL: VAE load：%s\n", error.c_str());
        return 1;
    }
    std::vector<float> latents(4 * 80 * 120);
    std::ifstream file(argv[3], std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(latents.data()),
                   latents.size() * sizeof(float)) ||
        file.peek() != EOF)
        return 1;
    int h = 0, w = 0;
    std::unique_ptr<float, decltype(&std::free)> output(
            instantmesh::vae_decode(model, latents.data(), 1, 80, 120, &h, &w),
            std::free);
    return output && !traceFailed && h == 640 && w == 960 ? 0 : 1;
}

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Deterministic focal sparse conv unit gate (Option C, see
// src/tasks/openpcdet/DETERMINISTIC_FOCAL_DESIGN.md section 4, gate 1).
//
// Whitebox test — links the AICore_test static library, including the task-side
// CUDA extension translation unit (openpcdet_cuda_ext.cu) when built with
// AICORE_OPENPCDET_CUDA_EXT.
//
// Gates:
//   1. GPU leg: the CUDA det_sparse_conv_kernel output and the CPU
//      std::fma core (det_sparse_conv_cpu) output are BYTE-IDENTICAL
//      (memcmp == 0) over a fixed pseudo-random matrix covering
//      C_in ∈ {16, 64}, C_out ∈ {16, 32, 64}, K = 27 (focal kernel), -1 empty
//      neighbors (including a fully-empty output row), n_out = 1024.
//   2. Control: the same inputs through the stock ggml dense lowering
//      (get_rows + mul_mat + bias) differ from the deterministic core by
//      at most 2e-6 — the expected reduction-order noise scale, proving the
//      deterministic core computes the same math.
//   3. OpenMP det_sparse_conv_cpu is bit-identical to the serial
//      det_sparse_conv_rows template run (thread-scheduling independence).
//   4. det_expf stays within 2e-6 relative error of a double-precision exp
//      across the sigmoid operating range and det_sigmoid(x) + det_sigmoid(-x)
//      ≈ 1 (finiteness + saturation sanity).
//
// Exits 77 (skip) when no CUDA device is available.

#include "extensions/deterministic_sparse_conv.h"

#ifdef AICORE_OPENPCDET_CUDA_EXT
#include <cuda_runtime.h>
#endif

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "backend.h"
#include "extensions/sparse_ops.h"

namespace {

namespace det = openpcdet::det;

uint32_t s_state[4] = {0x9e3779b9u, 0x243f6a88u, 0xb7e15162u, 0xdeadbeefu};

uint32_t xorshift128(void) {
    uint32_t t = s_state[3];
    t ^= t << 11;
    t ^= t >> 8;
    s_state[3] = s_state[2];
    s_state[2] = s_state[1];
    const uint32_t s = s_state[0];
    s_state[0] = t;
    s_state[1] = s ^ (s >> 19) ^ t;
    return s_state[1];
}

float rand_unit(void) {
    return static_cast<float>(static_cast<double>(xorshift128()) /
                              4294967296.0) *
                   2.0f -
           1.0f;
}

struct failure {
    std::string message;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw failure{message};
}

// One deterministic-conv shape: fill pseudo-random inputs and compare the
// numeric paths byte for byte.
void run_shape(int in_rows,
               int c_in,
               int c_out,
               int n_out,
               int k,
               const std::string& tag) {
    std::vector<float> features(static_cast<size_t>(in_rows) * c_in);
    for (float& v : features) v = rand_unit();
    std::vector<float> weights(static_cast<size_t>(k) * c_in * c_out);
    for (float& v : weights) v = rand_unit() * 0.25f;
    std::vector<float> bias(static_cast<size_t>(c_out));
    for (float& v : bias) v = rand_unit() * 0.5f;
    // ~15% empty (-1) neighbors; the second output row is fully empty.
    std::vector<int32_t> neighbors(static_cast<size_t>(n_out) * k);
    for (int v = 0; v < n_out; ++v) {
        for (int j = 0; j < k; ++j) {
            const uint32_t roll = xorshift128() % 100;
            neighbors[static_cast<size_t>(v) * k + j] =
                    (roll < 15 || v == 1)
                            ? -1
                            : static_cast<int32_t>(xorshift128() % in_rows);
        }
    }

    // CPU std::fma core.
    std::vector<float> cpu_out(static_cast<size_t>(n_out) * c_out, 0.0f);
    det::det_sparse_conv_cpu(features.data(), c_in, neighbors.data(),
                             weights.data(), bias.data(), cpu_out.data(), n_out,
                             c_out, k);

    // Serial template instance must agree bit for bit (OpenMP cell split is
    // schedule-independent).
    std::vector<float> serial_out(static_cast<size_t>(n_out) * c_out, 0.0f);
    det::det_sparse_conv_rows(features.data(), c_in, neighbors.data(),
                              weights.data(), bias.data(), serial_out.data(),
                              n_out, c_out, k, det::det_fma);
    require(std::memcmp(cpu_out.data(), serial_out.data(),
                        cpu_out.size() * sizeof(float)) == 0,
            tag + ": OpenMP CPU core diverged from the serial template");

#ifdef AICORE_OPENPCDET_CUDA_EXT
    {
        std::vector<float> cuda_out(static_cast<size_t>(n_out) * c_out, 0.0f);
        std::string error;
        require(det::det_sparse_conv_cuda(features.data(), in_rows, c_in,
                                          neighbors.data(), weights.data(),
                                          bias.data(), cuda_out.data(), n_out,
                                          c_out, k, error),
                tag + ": CUDA deterministic conv failed: " + error);
        const int mismatch = std::memcmp(cpu_out.data(), cuda_out.data(),
                                         cpu_out.size() * sizeof(float));
        if (mismatch != 0) {
            // Byte-difference diagnostics: first divergent cell, float values
            // and ULP distance, to pin the rounding-semantic culprit.
            size_t first = 0;
            int max_ulp = 0;
            double max_abs = 0.0;
            for (size_t i = 0; i < cpu_out.size(); ++i) {
                const uint32_t a = [&] {
                    uint32_t bits;
                    std::memcpy(&bits, &cpu_out[i], 4);
                    return bits;
                }();
                const uint32_t b = [&] {
                    uint32_t bits;
                    std::memcpy(&bits, &cuda_out[i], 4);
                    return bits;
                }();
                const int ulp = a > b ? static_cast<int>(a - b)
                                      : static_cast<int>(b - a);
                const double abs = std::fabs(static_cast<double>(cpu_out[i]) -
                                             static_cast<double>(cuda_out[i]));
                if (ulp > max_ulp) max_ulp = ulp;
                if (abs > max_abs) max_abs = abs;
                if (mismatch != 0 && first == 0 && ulp != 0) first = i;
            }
            const size_t v = first / c_out, o = first % c_out;
            // Compare the CUDA result against an r-major (old-layout)
            // interpretation of the weights to identify which layout the GPU
            // actually consumed.
            double rmajor_abs = 0.0;
            {
                float acc = bias[o];
                for (int kk = 0; kk < k; ++kk) {
                    const int r = neighbors[static_cast<size_t>(v) * k + kk];
                    if (r < 0) continue;
                    const float* row =
                            features.data() + static_cast<size_t>(r) * c_in;
                    for (int c = 0; c < c_in; ++c)
                        acc = std::fma(row[c],
                                       weights[(static_cast<size_t>(kk) * c_in +
                                                c) * c_out +
                                               o],
                                       acc);
                }
                rmajor_abs = std::fabs(static_cast<double>(acc) -
                                       static_cast<double>(cuda_out[first]));
            }
            std::fprintf(stderr,
                         "diff-detail %s: first=%zu (v=%zu,o=%zu) cpu=%.9g "
                         "cuda=%.9g max_ulp=%d max_abs=%.3g rmajor_abs=%.3g\n",
                         tag.c_str(), first, v, o, cpu_out[first],
                         cuda_out[first], max_ulp, max_abs, rmajor_abs);
        }
        require(mismatch == 0,
                tag + ": CUDA kernel output is NOT byte-identical to the CPU "
                      "std::fma core (memcmp != 0)");
    }
#endif

    // Control: the stock ggml dense lowering on the CPU backend. The gate is
    // scaled by each cell's accumulated |term| sum: mul_mat reduction-order
    // noise grows with the reduction length (K*C_in terms), while layout/logic
    // bugs produce O(scale) differences (the weight-layout bug this test
    // caught showed up as an absolute diff of ~9 on a ~30 scale).
    std::vector<float> term_scale(static_cast<size_t>(n_out) * c_out, 0.0f);
    for (int v = 0; v < n_out; ++v)
        for (int o = 0; o < c_out; ++o) {
            float sum = 0.0f;
            for (int j = 0; j < k; ++j) {
                const int r = neighbors[static_cast<size_t>(v) * k + j];
                if (r < 0) continue;
                for (int c = 0; c < c_in; ++c)
                    sum += std::fabs(
                            features[static_cast<size_t>(r) * c_in + c] *
                            weights[(static_cast<size_t>(o) * k * c_in +
                                     static_cast<size_t>(j) * c_in + c)]);
            }
            term_scale[static_cast<size_t>(v) * c_out + o] = sum;
        }
    openpcdet::engine_backend backend;
    require(backend.init("cpu", 0),
            "cpu backend init failed: " + backend.error);
    ggml_init_params params{ggml_tensor_overhead() * 64 + ggml_graph_overhead(),
                            nullptr, true};
    ggml_context* ctx = ggml_init(params);
    require(ctx != nullptr, "ggml context init failed");
    ggml_tensor* features_t =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c_in, in_rows + 1);
    ggml_set_input(features_t);
    ggml_tensor* dense_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, k * n_out);
    ggml_set_input(dense_t);
    ggml_tensor* weight_t =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k * c_in, c_out);
    ggml_set_input(weight_t);
    ggml_tensor* bias_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c_out);
    ggml_set_input(bias_t);
    openpcdet::ext::sparse_indice_map map;
    map.kernel_volume = k;
    map.compact_offsets.resize(static_cast<size_t>(k) + 1);
    for (int j = 0; j <= k; ++j)
        map.compact_offsets[static_cast<size_t>(j)] = j;
    ggml_tensor* output = openpcdet::ext::build_sparse_convolution(
            ctx, features_t, dense_t, nullptr, nullptr, map, weight_t, bias_t);
    ggml_set_output(output);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    require(ggml_gallocr_alloc_graph(backend.galloc, graph),
            "ggml graph allocation failed");
    ggml_backend_tensor_set(features_t, features.data(), 0,
                            features.size() * sizeof(float));
    const std::vector<float> zeros(static_cast<size_t>(c_in), 0.0f);
    ggml_backend_tensor_set(features_t, zeros.data(),
                            static_cast<size_t>(in_rows) * c_in * sizeof(float),
                            zeros.size() * sizeof(float));
    std::vector<int32_t> dense(static_cast<size_t>(k) * n_out);
    for (size_t i = 0; i < dense.size(); ++i)
        dense[i] = neighbors[i] < 0 ? in_rows : neighbors[i];
    ggml_backend_tensor_set(dense_t, dense.data(), 0,
                            dense.size() * sizeof(int32_t));
    ggml_backend_tensor_set(weight_t, weights.data(), 0,
                            weights.size() * sizeof(float));
    ggml_backend_tensor_set(bias_t, bias.data(), 0,
                            bias.size() * sizeof(float));
    require(ggml_backend_graph_compute(backend.be, graph) ==
                    GGML_STATUS_SUCCESS,
            "ggml graph compute failed");
    std::vector<float> ggml_out(static_cast<size_t>(n_out) * c_out);
    ggml_backend_tensor_get(output, ggml_out.data(), 0,
                            ggml_out.size() * sizeof(float));
    double max_diff = 0.0;
    double max_scaled = 0.0;
    for (size_t i = 0; i < ggml_out.size(); ++i) {
        const double diff =
                static_cast<double>(std::fabs(ggml_out[i] - cpu_out[i]));
        max_diff = std::max(max_diff, diff);
        const double scale = std::max(1.0, static_cast<double>(term_scale[i]));
        max_scaled = std::max(max_scaled, diff / scale);
    }
    require(max_scaled <= 2e-6, tag + ": ggml mul_mat control differs by " +
                                        std::to_string(max_diff) +
                                        " (scaled gate 2e-6, max_scaled " +
                                        std::to_string(max_scaled) + ")");
    std::string suffix;
#ifdef AICORE_OPENPCDET_CUDA_EXT
    suffix = " cuda_memcmp=0";
#endif
    std::printf(
            "  %-26s n_out=%-5d C_in=%-3d C_out=%-3d K=%-3d "
            "ggml_max_diff=%.3g%s\n",
            tag.c_str(), n_out, c_in, c_out, k, max_diff, suffix.c_str());
    ggml_free(ctx);
}

void run_expf_gate(void) {
    double max_rel = 0.0;
    // Accuracy sweep inside the unclamped range; saturation clamps beyond it
    // are checked separately (both sides saturate long before |x| = 87).
    for (double xd = -86.5; xd <= 86.5; xd += 0.0137) {
        const float x = static_cast<float>(xd);
        const double reference = std::exp(static_cast<double>(x));
        const float got = det::det_expf(x);
        require(std::isfinite(got), "det_expf produced a non-finite value");
        const double rel =
                reference > 0.0
                        ? std::fabs((static_cast<double>(got) - reference) /
                                    reference)
                        : (got == 0.0f ? 0.0 : 1.0);
        max_rel = std::max(max_rel, rel);
    }
    require(max_rel <= 2e-6, "det_expf relative error " +
                                     std::to_string(max_rel) + " exceeds 2e-6");
    require(det::det_expf(90.0f) == 0x1.fffffep+127f,
            "det_expf overflow clamp");
    require(det::det_expf(-90.0f) == 0.0f, "det_expf underflow clamp");
    // Sigmoid pair + monotonicity sanity.
    require(std::fabs(det::det_sigmoid(0.0f) - 0.5f) < 1e-6f,
            "sigmoid(0) != 0.5");
    float previous = -1.0f;
    for (double xd = -60.0; xd <= 60.0; xd += 0.031) {
        const float value = det::det_sigmoid(static_cast<float>(xd));
        require(std::isfinite(value) && value >= previous,
                "det_sigmoid not monotonic / non-finite");
        previous = value;
    }
    require(det::det_sigmoid(-1000.0f) < 1e-30f,
            "sigmoid underflow does not saturate to ~0");
    require(det::det_sigmoid(1000.0f) == 1.0f, "sigmoid overflow not 1");
    std::printf("  %-26s max_rel=%.3g\n", "det_expf vs double exp", max_rel);
}

}  // namespace

int main() {
#ifdef AICORE_OPENPCDET_CUDA_EXT
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count <= 0) {
        std::printf(
                "SKIP: no CUDA device (%s) — GPU byte-identity leg requires "
                "one\n",
                cudaGetErrorString(count_status));
        return 77;
    }
#endif
    try {
        std::printf("deterministic focal sparse conv gates:\n");
        // K=27 is the focal 3x3x3 kernel; shapes cover the real chain
        // (C_in 16/32/64, conv_imp C_out=27 and main conv C_out 16/32/64).
        run_shape(1100, 16, 16, 1024, 27, "conv_imp_c1_shape");
        run_shape(1100, 16, 27, 1024, 27, "main_c1_shape");
        run_shape(1100, 64, 27, 1024, 27, "conv_imp_c3_shape");
        run_shape(1100, 64, 32, 1024, 27, "main_c2_shape");
        run_shape(1100, 64, 64, 1024, 27, "main_c3_shape");
        run_expf_gate();
        std::printf("ALL DETERMINISTIC SPARSE CONV GATES PASSED\n");
        return 0;
    } catch (const failure& f) {
        std::fprintf(stderr, "FAIL: %s\n", f.message.c_str());
        return 1;
    }
}

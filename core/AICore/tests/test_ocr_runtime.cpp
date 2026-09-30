// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// P1a alignment probe for the AICore `ocr` recognition path: run the
// audited rec topology over the pinned PaddlePaddle fixture and compare
// element-wise against the reference probabilities. This is the G1
// correctness anchor (CER = 0 on the fixture when argmax sequences match
// and the max |diff| stays within the f32 tolerance).

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/ggml_backend_utils.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ocr_runtime.hpp"

namespace {

// Minimal .npy reader: returns dims (C order) and raw float32 payload.
bool readNpyF32(const std::string& path,
                std::vector<int64_t>* shape,
                std::vector<float>* data) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    unsigned char header[10];
    if (fread(header, 1, 10, f) != 10) {
        fclose(f);
        return false;
    }
    if (std::memcmp(header, "\x93NUMPY", 6) != 0) {
        fclose(f);
        return false;
    }
    const unsigned short headerLen = static_cast<unsigned short>(header[8]) |
                                     static_cast<unsigned short>(header[9])
                                             << 8;
    std::string dict(headerLen, '\0');
    if (fread(dict.data(), 1, headerLen, f) != headerLen) {
        fclose(f);
        return false;
    }
    // Parse the shape tuple; the dtype must be '<f4'.
    const auto shapePos = dict.find("'shape':");
    if (shapePos == std::string::npos) {
        fclose(f);
        return false;
    }
    const auto open = dict.find('(', shapePos);
    const auto close = dict.find(')', open);
    size_t pos = open + 1;
    while (pos < close) {
        const size_t comma = dict.find(',', pos);
        const size_t end = comma == std::string::npos ? close : comma;
        const std::string token = dict.substr(pos, end - pos);
        if (!token.empty()) shape->push_back(std::atoll(token.c_str()));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    if (dict.find("'<f4'") == std::string::npos &&
        dict.find("'float32'") == std::string::npos) {
        fclose(f);
        return false;
    }
    size_t count = 1;
    for (int64_t d : *shape) count *= static_cast<size_t>(d);
    data->resize(count);
    const size_t read = fread(data->data(), sizeof(float), count, f);
    fclose(f);
    return read == count;
}

std::string argmaxSignature(const float* probs, int timeSteps, int vocab) {
    std::string signature;
    for (int t = 0; t < timeSteps; ++t) {
        const float* row = probs + static_cast<size_t>(t) * vocab;
        int best = 0;
        float bestProb = row[0];
        for (int i = 1; i < vocab; ++i)
            if (row[i] > bestProb) {
                bestProb = row[i];
                best = i;
            }
        signature += std::to_string(best) + ",";
    }
    return signature;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string fixtureDir = argc > 1 ? argv[1] : std::string();
    const std::string modelPath = argc > 2 ? argv[2] : std::string();
    // det runs the pure-CNN DBNet graph; its logits map is compared
    // element-wise without the CTC/argmax machinery. layout runs the
    // PicoDet detector and compares all 8 raw heads element-wise (the
    // head files self-identify by shape: (N,10) cls / (N,32) reg).
    const std::string task = argc > 3 ? argv[3] : "rec";
    const bool det = task == "det";
    const bool layout = task == "layout";
    const std::string stem = layout ? "layout" : (det ? "det_ch" : "rec_ch");
    const std::string backendName = argc > 4 ? argv[4] : "CPU";
    // The manifest scenarios pass their structured gate here so the probe's
    // exit code stays consistent with the enforced per-quantization
    // tolerance (F32 keeps the 1e-3 default).
    const double tolerance = argc > 7 ? std::atof(argv[7]) : 1e-3;
    if (fixtureDir.empty() || modelPath.empty()) {
        std::fprintf(stderr,
                     "usage: %s <fixture-dir> <gguf> [det] [backend]\n"
                     "  fixture-dir must contain <stem>_input.npy and\n"
                     "  <stem>_reference.npy\n",
                     argv[0]);
        return 2;
    }
    {
        // CTest skip contract: without the fetched fixtures/models the
        // probe cannot run, but the configuration is not broken.
        const std::string input = fixtureDir + "/" + stem + "_input.npy";
        if (access(input.c_str(), R_OK) != 0 ||
            access(modelPath.c_str(), R_OK) != 0) {
            std::fprintf(stderr,
                         "probe_ocr: fixtures/model not fetched, skipping\n");
            return 77;
        }
    }

    std::vector<int64_t> inputShape;
    std::vector<float> inputData;
    std::vector<int64_t> refShape;
    std::vector<float> refData;
    if (!readNpyF32(fixtureDir + "/" + stem + "_input.npy", &inputShape,
                    &inputData)) {
        std::fprintf(stderr, "failed to read %s_input.npy\n", stem.c_str());
        return 1;
    }
    std::vector<std::pair<std::vector<int64_t>, std::vector<float>>> heads;
    if (!layout && !readNpyF32(fixtureDir + "/" + stem + "_reference.npy",
                               &refShape, &refData)) {
        std::fprintf(stderr, "failed to read %s_reference.npy\n", stem.c_str());
        return 1;
    }
    if (layout) {
        // PicoDet exposes 8 heads; the paddle reference saved each as
        // layout_head<i>.npy. They self-identify by shape, so the fetch
        // order of the topology cannot mismatch the fixtures.
        for (int i = 0; i < 8; ++i) {
            std::vector<int64_t> hs;
            std::vector<float> hd;
            if (!readNpyF32(fixtureDir + "/layout_head" + std::to_string(i) +
                                    ".npy",
                            &hs, &hd)) {
                std::fprintf(stderr, "failed to read layout_head%d.npy\n", i);
                return 1;
            }
            heads.emplace_back(std::move(hs), std::move(hd));
        }
    }

    aicore::ocr::OcrWeights weights;
    std::string error;
    if (!weights.load(modelPath, &error)) {
        std::fprintf(stderr, "model load failed: %s\n", error.c_str());
        return 1;
    }
    if (!det && !layout && weights.dictionary.empty()) {
        std::fprintf(stderr, "model has no embedded dictionary\n");
        return 1;
    }

    std::fprintf(stderr, "[post-load] ops=%zu first-type=%s\n",
                 weights.program.ops.size(),
                 weights.program.ops.empty()
                         ? "?"
                         : weights.program.ops[0].type.c_str());
    // Graph-structure context (no data) and compute context (activations
    // + input). The weights live in the host-readable meta pool.
    // Depthwise layers expand to per-channel views + conv nodes; the
    // graph holds tens of thousands of tensors.
    const size_t tensorSize = ggml_tensor_overhead() * 500000;
    ggml_context* ctx = ggml_init({tensorSize * 8, nullptr,
                                   /*no_alloc=*/true});
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return 1;
    }
    ggml_tensor* input =
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, inputShape[3], inputShape[2],
                               inputShape[1], inputShape[0]);
    // User-input flag: the scheduler routes it from the CPU side and copies
    // it before the first split; the gallocr path allocates it like any
    // other leaf.
    ggml_set_input(input);

    aicore::ocr::OcrProbeOptions probe;
    if (argc > 5) probe.fetchAt = std::atoi(argv[5]);
    if (argc > 6) probe.dumpShapes = std::string(argv[6]) == "shapes";
    // Prefix bisection over the device under test: AICORE_OCR_FETCH_AT (test
    // file only, never in production sources).
    if (const char* fetchEnv = std::getenv("AICORE_OCR_FETCH_AT"))
        probe.fetchAt = std::atoi(fetchEnv);

    // Dynamic-loader build: the backend .so must be registered first.
    ggml_common::load_backends_once();
    // The manifest device tokens are lowercase; the registry names are
    // capitalized (CPU / CUDA / Vulkan / Metal).
    std::string registryName = backendName;
    if (!registryName.empty())
        registryName[0] = static_cast<char>(
                std::toupper(static_cast<unsigned char>(registryName[0])));
    ggml_backend_t backend =
            ggml_backend_init_by_name(registryName.c_str(), nullptr);
    if (!backend) {
        // Device names carry a per-device index ("CUDA0"), so an exact
        // name lookup can miss; fall back to the device class.
        const bool wantsGpu = registryName != "CPU";
        backend = ggml_backend_init_by_type(
                wantsGpu ? GGML_BACKEND_DEVICE_TYPE_GPU
                         : GGML_BACKEND_DEVICE_TYPE_CPU,
                nullptr);
    }
    if (!backend) {
        std::fprintf(stderr, "backend init failed: %s\n", registryName.c_str());
        ggml_free(ctx);
        return 1;
    }
    const bool isCpu = ggml_common::is_cpu_backend(backend);

    // Build options mirror the production session: CPU sessions route
    // 3x3 stride-1 convs through the shared Winograd custom op so the
    // manifest gates measure the winograd parity and the CPU numbers
    // reflect the production graph.
    aicore::ocr::OcrBuildOptions buildOptions;
    buildOptions.cpu_conv = isCpu;
    ggml_tensor* output = nullptr;
    std::vector<ggml_tensor*> fetchList;
    if (layout) {
        output = aicore::ocr::buildGraph(ctx, weights, input, &error, &probe,
                                         &fetchList, &buildOptions);
    } else {
        output = aicore::ocr::buildGraph(ctx, weights, input, &error, &probe,
                                         nullptr, &buildOptions);
    }
    if (!output) {
        std::fprintf(stderr, "buildGraph failed: %s\n", error.c_str());
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_t cpuBackend = backend;
    ggml_backend_sched_t sched = nullptr;
    ggml_gallocr_t alloc = nullptr;
    if (!isCpu) {
        cpuBackend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,
                                               nullptr);
        if (!cpuBackend) {
            std::fprintf(stderr, "cpu secondary backend init failed\n");
            ggml_free(ctx);
            return 1;
        }
        if (!weights.realizeHostBuffers()) {
            std::fprintf(stderr, "host buffer realization failed\n");
            ggml_free(ctx);
            return 1;
        }
        // After build: cover the folded constants emitted by buildGraph.
        weights.tagHostLeafs();
        if (!weights.realizeToDevice(backend, &error)) {
            std::fprintf(stderr, "device weight realization failed: %s\n",
                         error.c_str());
            ggml_free(ctx);
            return 1;
        }
        sched = ggml_common::new_gpu_sched({backend}, cpuBackend, 1u << 20);
        if (!sched) {
            std::fprintf(stderr, "backend scheduler init failed\n");
            ggml_free(ctx);
            return 1;
        }
    }
    ggml_cgraph* graph = ggml_new_graph_custom(
            ctx, 1u << 20, /*simple=*/false);  // depthwise expands wide
    if (layout && probe.fetchAt < 0) {
        // Every PicoDet head must live in the graph, or its tensor stays
        // unallocated and the readback aborts. Truncated builds expand
        // the single fetch node instead.
        for (ggml_tensor* t : fetchList) ggml_build_forward_expand(graph, t);
    } else {
        ggml_build_forward_expand(graph, output);
    }
    const bool allocated =
            isCpu ? (alloc = ggml_gallocr_new(
                             ggml_backend_get_default_buffer_type(backend)),
                     ggml_gallocr_alloc_graph(alloc, graph))
                  : ggml_backend_sched_alloc_graph(sched, graph);
    if (!allocated) {
        std::fprintf(stderr, "graph allocation failed\n");
        return 1;
    }
    // Upload the input, run, and read the result back. The compute pass
    // is timed over warmups + runs so the manifest can report p50/p95.
    ggml_backend_tensor_set(input, inputData.data(), 0,
                            sizeof(float) * inputData.size());
    constexpr int kWarmups = 2;
    constexpr int kRuns = 5;
    std::vector<double> timings;
    timings.reserve(kRuns);
    for (int run = -kWarmups; run < kRuns; ++run) {
        // Re-upload every run: if some graph op writes through to the
        // input buffer, a second compute must not see corrupted input.
        ggml_backend_tensor_set(input, inputData.data(), 0,
                                sizeof(float) * inputData.size());
        const auto start = std::chrono::steady_clock::now();
        if (isCpu)
            ggml_backend_graph_compute(backend, graph);
        else
            ggml_backend_sched_graph_compute(sched, graph);
        const auto stop = std::chrono::steady_clock::now();
        if (run >= 0) {
            timings.push_back(
                    std::chrono::duration<double, std::milli>(stop - start)
                            .count());
        }
    }
    std::sort(timings.begin(), timings.end());
    const double p50 = timings[timings.size() / 2];
    const double p95 = timings[static_cast<size_t>(timings.size() * 95 / 100)];
    // Graph composition (node histogram): the op mix explains where the
    // compute time goes and guards the node-count regressions of the
    // depthwise expansion.
    {
        std::map<std::string, int> hist;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            hist[ggml_op_desc(ggml_graph_node(graph, i))]++;
        }
        std::vector<std::pair<int, std::string>> top;
        for (const auto& kv : hist) top.emplace_back(kv.second, kv.first);
        std::sort(top.begin(), top.end(), std::greater<>{});
        std::fprintf(stderr, "[graph] nodes=%zu top:",
                     (size_t)ggml_graph_n_nodes(graph));
        for (size_t i = 0; i < top.size() && i < 8; ++i)
            std::fprintf(stderr, " %s=%d", top[i].second.c_str(), top[i].first);
        std::fprintf(stderr, "\n");
    }
    // The last timed run's output stays in `output`; no extra compute.
    std::vector<float> got(static_cast<size_t>(ggml_nelements(output)));
    ggml_backend_tensor_get(output, got.data(), 0, sizeof(float) * got.size());

    const int64_t timeSteps = output->ne[1];
    const int64_t vocab = output->ne[0];

    if (probe.fetchAt >= 0) {
        // Bisection mode: print the truncated activation's shape and sum
        // for comparison against the numpy simulator log.
        std::vector<float> got(ggml_nelements(output));
        ggml_backend_tensor_get(output, got.data(), 0,
                                sizeof(float) * got.size());
        double sum = 0.0;
        for (float v : got) sum += v;
        std::printf("[fetch] ne=%lld,%lld,%lld,%lld sum=%.6e\n",
                    (long long)output->ne[0], (long long)output->ne[1],
                    (long long)output->ne[2], (long long)output->ne[3], sum);
        return 0;
    }

    if (layout) {
        // Element-wise comparison of every head, matched by shape.
        float layoutMax = 0.0f;
        double layoutSum = 0.0;
        size_t layoutCount = 0;
        if (fetchList.size() != 8) {
            std::fprintf(stderr, "layout: expected 8 fetches, got %zu\n",
                         fetchList.size());
            return 1;
        }
        for (ggml_tensor* t : fetchList) {
            std::vector<float> got(ggml_nelements(t));
            ggml_backend_tensor_get(t, got.data(), 0,
                                    sizeof(float) * got.size());
            // ggml ne = {C,N,1,1} for the numpy (1,N,C) head: element
            // order matches, so the flat comparison is direct.
            std::pair<std::vector<int64_t>, std::vector<float>>* ref = nullptr;
            for (auto& h : heads) {
                if (h.first.size() == 3 && h.first[0] == 1 &&
                    h.first[1] == t->ne[1] && h.first[2] == t->ne[0]) {
                    ref = &h;
                    break;
                }
            }
            if (!ref) {
                std::fprintf(stderr,
                             "layout: no head fixture for ne=%lldx%lld\n",
                             (long long)t->ne[2], (long long)t->ne[3]);
                return 1;
            }
            if (ref->second.size() != got.size()) {
                std::fprintf(stderr,
                             "layout: head size mismatch (%zu vs %zu)\n",
                             ref->second.size(), got.size());
                return 1;
            }
            double hsum = 0.0;
            // numpy (1,N,C) is C-order over (n,c); the ggml head ne={C,N}
            // is column-major over the same logical shape, so the flat
            // indices transpose: got[c*N + n] == ref[n*C + c].
            const int64_t C = t->ne[0], N = t->ne[1];
            for (int64_t n = 0; n < N; ++n)
                for (int64_t c = 0; c < C; ++c) {
                    const float diff =
                            std::fabs(got[c * N + n] - ref->second[n * C + c]);
                    layoutMax = std::max(layoutMax, diff);
                    layoutSum += diff;
                    hsum += got[c * N + n];
                }
            layoutCount += got.size();
            std::printf("[head] ne=%lldx%lld sum=%.6e\n", (long long)t->ne[0],
                        (long long)t->ne[1], hsum);
        }
        const bool ok = layoutMax < tolerance;
        std::printf("layout probe: n=%zu maxDiff=%.6f meanDiff=%.8f ok=%s\n",
                    layoutCount, layoutMax,
                    static_cast<float>(layoutSum /
                                       static_cast<double>(layoutCount)),
                    ok ? "YES" : "NO");
        std::printf(
                "{\"task\":\"layout\",\"backend\":\"%s\",\"max_diff\":%.6f,"
                "\"mean_diff\":%.8f,\"p50_ms\":%.2f,\"p95_ms\":%.2f}\n",
                registryName.c_str(), layoutMax,
                layoutSum / static_cast<double>(layoutCount), p50, p95);
        return ok ? 0 : 1;
    }

    if (det) {
        // The DBNet map is a single-channel image: compare element-wise.
        float detMax = 0.0f;
        double detSum = 0.0;
        const size_t detCount = got.size();
        for (size_t i = 0; i < detCount; ++i) {
            const float diff = std::fabs(got[i] - refData[i]);
            detMax = std::max(detMax, diff);
            detSum += diff;
        }
        std::printf("det probe: n=%zu maxDiff=%.6f meanDiff=%.8f\n", detCount,
                    detMax,
                    static_cast<float>(detSum / static_cast<double>(detCount)));
        if (std::getenv("AICORE_OCR_CHECKPOINT")) {
            // Spatial diff distribution: edge bands point at
            // padding/stride boundary bugs; periodic bands point at
            // upsample phase bugs.
            const int64_t H = refShape[2], W = refShape[3];
            std::vector<double> rowDiff(H, 0.0), colDiff(W, 0.0);
            int64_t maxH = 0, maxW = 0;
            float maxD = 0.0f;
            for (int64_t h = 0; h < H; ++h)
                for (int64_t w = 0; w < W; ++w) {
                    const float d =
                            std::fabs(got[h * W + w] - refData[h * W + w]);
                    rowDiff[h] += d;
                    colDiff[w] += d;
                    if (d > maxD) {
                        maxD = d;
                        maxH = h;
                        maxW = w;
                    }
                }
            std::fprintf(stderr, "[ckpt] max diff at (h=%lld,w=%lld)=%.6f\n",
                         (long long)maxH, (long long)maxW, maxD);
            std::fprintf(stderr, "[ckpt] row diffs:");
            for (int64_t h = 0; h < H; ++h)
                std::fprintf(stderr, " %.1f", rowDiff[h]);
            std::fprintf(stderr, "\n[ckpt] col diffs:");
            for (int64_t w = 0; w < W; ++w)
                std::fprintf(stderr, " %.1f", colDiff[w]);
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
        }
        if (std::getenv("AICORE_OCR_SAVE_GOT"))
            std::fwrite(got.data(), sizeof(float), got.size(),
                        fopen(std::getenv("AICORE_OCR_SAVE_GOT"), "wb"));
        const bool ok = detMax < tolerance;
        std::printf("elementwise(<%g): %s\n", tolerance, ok ? "YES" : "NO");
        // Machine-readable verdict for the manifest's generic metric parser.
        std::printf(
                "{\"task\":\"det\",\"backend\":\"%s\",\"max_diff\":%.6f,"
                "\"mean_diff\":%.8f,\"p50_ms\":%.2f,\"p95_ms\":%.2f}\n",
                registryName.c_str(), detMax,
                detSum / static_cast<double>(detCount), p50, p95);
        std::printf("perf: backend=%s device=%s p50=%.2fms p95=%.2fms\n",
                    registryName.c_str(), ggml_backend_name(backend), p50, p95);
        ggml_gallocr_free(alloc);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return ok ? 0 : 1;
    }

    // The ggml output is ne={V,T,1,1}: element (t,v) sits at v + t*V.
    float maxDiff = 0.0f;
    double sumDiff = 0.0;
    const size_t count = got.size();
    for (size_t i = 0; i < count; ++i) {
        const float diff = std::fabs(got[i] - refData[i]);
        maxDiff = std::max(maxDiff, diff);
        sumDiff += diff;
    }
    const float meanDiff =
            static_cast<float>(sumDiff / static_cast<double>(count));
    const std::string gotSignature = argmaxSignature(
            got.data(), static_cast<int>(timeSteps), static_cast<int>(vocab));
    const std::string refSignature =
            argmaxSignature(refData.data(), static_cast<int>(refShape[1]),
                            static_cast<int>(refShape[2]));

    // CTC decode of the computed output.
    float meanScore = 0.0f;
    const std::string text = aicore::ocr::ctcGreedyDecode(
            got.data(), static_cast<int>(timeSteps), static_cast<int>(vocab),
            weights.dictionary, &meanScore);

    std::printf("rec probe: t=%lld vocab=%lld maxDiff=%.6f meanDiff=%.8f\n",
                static_cast<long long>(timeSteps),
                static_cast<long long>(vocab), maxDiff, meanDiff);
    std::printf("decoded text (%.3f): %s\n", meanScore, text.c_str());

    const bool signatureMatch = gotSignature == refSignature;
    const bool withinTolerance = maxDiff < tolerance;
    std::printf("argmax match: %s | tolerance(<%g): %s\n",
                signatureMatch ? "YES" : "NO", tolerance,
                withinTolerance ? "YES" : "NO");
    // Machine-readable verdict for the manifest's generic metric parser.
    std::printf(
            "{\"task\":\"rec\",\"backend\":\"%s\",\"max_diff\":%.6f,"
            "\"mean_diff\":%.8f,\"argmax_match\":%d,\"p50_ms\":%.2f,"
            "\"p95_ms\":%.2f}\n",
            registryName.c_str(), maxDiff, meanDiff, signatureMatch ? 1 : 0,
            p50, p95);
    std::printf("perf: backend=%s device=%s p50=%.2fms p95=%.2fms\n",
                registryName.c_str(), ggml_backend_name(backend), p50, p95);
    if (!signatureMatch) {
        std::fprintf(stderr, "GOT : %.120s\nREF : %.120s\n",
                     gotSignature.c_str(), refSignature.c_str());
    }

    ggml_gallocr_free(alloc);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return (signatureMatch && withinTolerance) ? 0 : 1;
}

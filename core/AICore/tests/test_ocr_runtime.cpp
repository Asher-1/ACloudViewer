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

#ifndef _WIN32
#include <unistd.h>

#include <map>
#else
#include <io.h>
// POSIX access(path, R_OK) maps onto the MSVC CRT (mode 4 = read check).
#define access _access
#define R_OK 4
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/blocked_conv.hpp"
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
    if (read != count) return false;
    // np.save of a transposed view (the PicoDet heads end in transpose2)
    // writes 'fortran_order': True, i.e. the first axis is fastest. The
    // probes compare in C order, so reorder; reading the bytes as C order
    // silently compares against the transpose.
    if (dict.find("'fortran_order': True") != std::string::npos &&
        shape->size() > 1) {
        const size_t rank = shape->size();
        std::vector<float> c(count);
        std::vector<int64_t> idx(rank, 0);
        for (size_t linear = 0; linear < count; ++linear) {
            size_t fortran = 0, stride = 1;
            for (size_t a = 0; a < rank; ++a) {
                fortran += static_cast<size_t>(idx[a]) * stride;
                stride *= static_cast<size_t>((*shape)[a]);
            }
            c[linear] = (*data)[fortran];
            for (size_t a = rank; a-- > 0;) {  // C-order odometer
                if (++idx[a] < (*shape)[a]) break;
                idx[a] = 0;
            }
        }
        *data = std::move(c);
    }
    return true;
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
    // head files self-identify by shape: (N,10) cls / (N,32) reg). The rec
    // stem follows the model's script tag (rec_ch / rec_en) so both rec
    // containers probe against their own fixture pair.
    const std::string task = argc > 3 ? argv[3] : "rec";
    const bool det = task == "det";
    const bool layout = task == "layout";
    std::string stem = layout ? "layout" : (det ? "det_ch" : "rec_ch");
    if (!det && !layout && modelPath.find("rec-en") != std::string::npos)
        stem = "rec_en";
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
    // Test-only batch tiling: replicate the N=1 fixture input across N
    // identical batch rows so the batched-attention graph can be validated
    // against its own N=1 reference and against itself (self-consistency
    // must be exact; the reference comparison quantifies matmul-order
    // noise). Never set in production sources.
    int tileN = 1;
    if (const char* tileEnv = std::getenv("AICORE_OCR_PROBE_TILE"))
        tileN = std::max(1, std::atoi(tileEnv));
    if (tileN > 1 && !det && !layout) {
        const size_t plane = inputData.size();
        std::vector<float> tiled(plane * static_cast<size_t>(tileN));
        for (int r = 0; r < tileN; ++r)
            std::copy(inputData.begin(), inputData.end(),
                      tiled.begin() + static_cast<ptrdiff_t>(r) * plane);
        inputData = std::move(tiled);
        inputShape[0] = tileN;
        std::fprintf(stderr, "[tile] rec input replicated x%d\n", tileN);
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

    aicore::ocr::OcrProbeOptions probe;
    if (argc > 5) probe.fetchAt = std::atoi(argv[5]);
    if (argc > 6) probe.dumpShapes = std::string(argv[6]) == "shapes";
    // Prefix bisection over the device under test: AICORE_OCR_FETCH_AT (test
    // file only, never in production sources).
    if (const char* fetchEnv = std::getenv("AICORE_OCR_FETCH_AT"))
        probe.fetchAt = std::atoi(fetchEnv);

    aicore::ocr::OcrWeights weights;
    std::string error;
    // A bisection indexes the SOURCE program's ops, so the load-time affine
    // folding (which removes ops) is skipped while fetchAt is set.
    if (!weights.load(modelPath, &error, /*foldAffine=*/probe.fetchAt < 0)) {
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

    // Dynamic-loader build: the backend .so must be registered first.
    ggml_common::load_backends_once();
    // The manifest device tokens are lowercase; the registry names are
    // capitalized (CPU / CUDA / Vulkan / Metal).
    std::string registryName = backendName;
    if (!registryName.empty())
        registryName[0] = static_cast<char>(
                std::toupper(static_cast<unsigned char>(registryName[0])));
    // Resolve GPU requests through the AICore registry helper: init_by_name
    // matches DEVICE names exactly ("Vulkan0"), not the UI token
    // ("vulkan"), so a plain init_by_name would return null and the
    // init_by_type fallback below would silently hand back CUDA0 — every
    // "backend=Vulkan device=CUDA0" probe row ever printed was a CUDA
    // measurement. CPU requests bypass the GPU lookup (it only enumerates
    // GPU/IGPU devices) and go straight to the CPU device class.
    ggml_backend_t backend = nullptr;
    std::string devLower = backendName;
    std::transform(devLower.begin(), devLower.end(), devLower.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    if (devLower != "cpu") {
        std::string resolved;
        backend = ggml_common::find_gpu_backend(backendName, 0, resolved);
        if (!backend) {
            // Never measure a different device than the one requested: an
            // unavailable GPU backend is a skip the runner must account for,
            // not a silent CUDA0 row labelled "Vulkan".
            std::fprintf(stderr,
                         "probe_ocr: backend %s not available, skipping\n",
                         backendName.c_str());
            ggml_free(ctx);
            return 77;
        }
        std::fprintf(stderr, "[device] requested=%s resolved=%s\n",
                     backendName.c_str(), resolved.c_str());
    } else {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,
                                            nullptr);
    }
    if (!backend) {
        std::fprintf(stderr, "backend init failed: %s\n", registryName.c_str());
        ggml_free(ctx);
        return 1;
    }
    const bool isCpu = ggml_common::is_cpu_backend(backend);

    // The probe executes exactly the graph that ships: routing (CPU gallocr
    // / CUDA-Vulkan one-device / mixed sched), conv routes and allocation
    // all come from the production owner (makeExecPlan / execBuildOptions /
    // prepareGraph). A GPU probe needs the CPU secondary the session has.
    ggml_backend_t cpuBackend = backend;
    if (!isCpu) {
        cpuBackend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,
                                               nullptr);
        if (!cpuBackend) {
            std::fprintf(stderr, "cpu secondary backend init failed\n");
            ggml_free(ctx);
            return 1;
        }
    }
    const aicore::ocr::OcrExecPlan plan = aicore::ocr::makeExecPlan(
            isCpu ? nullptr : backend, cpuBackend, weights.sched_gpu_tagging);
    aicore::ocr::OcrBuildOptions buildOptions =
            aicore::ocr::execBuildOptions(plan);
    // Blocked-island A/B: default scope mirrors production (all three
    // graphs, CPU+AVX-512 hosts); AICORE_OCR_NO_ISLAND=1 disables it for
    // the controlled baseline; AICORE_OCR_FORCE_ISLAND=1 enables it on
    // AVX2-only hosts (scalar island kernels — correctness/parity A/B only).
    // Both envs are test-file-only and never in production.
    buildOptions.blocked_island =
            isCpu && (std::getenv("AICORE_OCR_FORCE_ISLAND") != nullptr ||
                      (aicore::blk::blocked_available() &&
                       !std::getenv("AICORE_OCR_NO_ISLAND")));
    ggml_tensor* output = nullptr;
    std::vector<ggml_tensor*> fetchList;
    // Production contract: a fresh fold pool per build so the folded
    // constants can be mirrored to the device (prepareGraph rejects a build
    // without it on GPU plans). The build+prepare wall time is reported as
    // build_ms: it is the cost of every graph-cache miss in the session.
    const auto buildStart = std::chrono::steady_clock::now();
    weights.resetFoldPool();
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

    // Every PicoDet head must live in the graph, or its tensor stays
    // unallocated and the readback aborts. Truncated builds expand the
    // single fetch node instead.
    std::vector<ggml_tensor*> outputs;
    if (layout && probe.fetchAt < 0)
        outputs = fetchList;
    else
        outputs.push_back(output);
    aicore::ocr::OcrGraphExec exec;
    ggml_cgraph* graph = nullptr;
    if (!aicore::ocr::prepareGraph(weights, ctx, plan, outputs, &graph, exec,
                                   error)) {
        std::fprintf(stderr, "graph preparation failed: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "[build] build_ms=%.2f nodes=%d\n",
                 std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - buildStart)
                         .count(),
                 ggml_graph_n_nodes(graph));
    if (!det && !layout) {
        // Batch-mode diagnostic: CUDA rejects IM2COL/MUL_MAT on
        // non-contiguous activations, which fragments the sched into
        // thousands of splits. List the non-contiguous nodes so the
        // producer op can be fixed at the source.
        int nonContig = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_tensor* t = ggml_graph_node(graph, i);
            if (!ggml_is_contiguous(t) && ggml_nbytes(t) > 0) {
                if (nonContig < 12)
                    std::fprintf(stderr,
                                 "[noncontig] #%d op=%s ne=%lld,%lld,%lld,%lld "
                                 "nb=%zu,%zu,%zu\n",
                                 i, ggml_op_desc(t), (long long)t->ne[0],
                                 (long long)t->ne[1], (long long)t->ne[2],
                                 (long long)t->ne[3], (size_t)t->nb[0],
                                 (size_t)t->nb[1], (size_t)t->nb[2]);
                ++nonContig;
            }
        }
        std::fprintf(stderr, "[noncontig] total=%d / %d nodes\n", nonContig,
                     ggml_graph_n_nodes(graph));
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
        aicore::ocr::computeGraph(plan, exec, graph);
        const auto stop = std::chrono::steady_clock::now();
        if (run >= 0) {
            timings.push_back(
                    std::chrono::duration<double, std::milli>(stop - start)
                            .count());
        }
    }
    if (std::getenv("AICORE_OCR_FETCH_ALL")) {
        // Bisect helper: after the compute every intermediate tensor holds
        // valid data (gallocr). Walk all nodes and print (index, ggml op,
        // shape, sum) so the Python side can align the run against the
        // official Paddle graph op-by-op. Test-only; never built into
        // production sources.
        auto tensorSum = [](const ggml_tensor* t) {
            double s = 0.0;
            for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
                for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
                    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                        const char* row = (const char*)t->data + i3 * t->nb[3] +
                                          i2 * t->nb[2] + i1 * t->nb[1];
                        for (int64_t i0 = 0; i0 < t->ne[0]; ++i0)
                            s += *(const float*)(row + i0 * t->nb[0]);
                    }
            return s;
        };
        const int nNodes = ggml_graph_n_nodes(graph);
        const int dumpFrom =
                nNodes -
                20;  // optional tail-tensor dumps (AICORE_OCR_DUMP_TAIL)
        std::map<const void*, int> nodeIdx;
        for (int i = 0; i < nNodes; ++i)
            nodeIdx[(const void*)ggml_graph_node(graph, i)] = i;
        for (int i = 0; i < nNodes; ++i) {
            const ggml_tensor* t = ggml_graph_node(graph, i);
            if (!t || !t->data || t->type != GGML_TYPE_F32) continue;
            if (i >= dumpFrom && getenv("AICORE_OCR_DUMP_TAIL")) {
                // also dump the second operand of binary tail ops
                if (t->src[1] && t->src[1]->type == GGML_TYPE_F32 &&
                    ggml_nbytes(t->src[1]) < 1 << 16) {
                    char sp[160];
                    snprintf(sp, sizeof(sp), "/tmp/tailsrc1_%03d.f32", i);
                    FILE* sf = fopen(sp, "wb");
                    if (sf) {
                        const ggml_tensor* sv = t->src[1];
                        for (int64_t i3 = 0; i3 < sv->ne[3]; ++i3)
                            for (int64_t i2 = 0; i2 < sv->ne[2]; ++i2)
                                for (int64_t i1 = 0; i1 < sv->ne[1]; ++i1) {
                                    const char* row = (const char*)sv->data +
                                                      i3 * sv->nb[3] +
                                                      i2 * sv->nb[2] +
                                                      i1 * sv->nb[1];
                                    for (int64_t i0 = 0; i0 < sv->ne[0]; ++i0) {
                                        float v =
                                                *(const float*)(row +
                                                                i0 * sv->nb[0]);
                                        fwrite(&v, sizeof(float), 1, sf);
                                    }
                                }
                        fclose(sf);
                    }
                }
                char path[160];
                snprintf(path, sizeof(path), "/tmp/tail_%03d_%s.f32", i,
                         ggml_op_desc(t));
                FILE* df = fopen(path, "wb");
                if (df) {
                    // strided write (views may be non-contiguous)
                    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
                        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
                            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                                const char* row = (const char*)t->data +
                                                  i3 * t->nb[3] +
                                                  i2 * t->nb[2] + i1 * t->nb[1];
                                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                                    float v = *(const float*)(row +
                                                              i0 * t->nb[0]);
                                    fwrite(&v, sizeof(float), 1, df);
                                }
                            }
                    fclose(df);
                }
            }
            std::fprintf(
                    stderr,
                    "[fetchall] i=%d op=%s ne=%lld,%lld,%lld,%lld "
                    "sum=%.6e src=%d,%d data=%p s0=%p s1=%p\n",
                    i, ggml_op_desc(t), (long long)t->ne[0],
                    (long long)t->ne[1], (long long)t->ne[2],
                    (long long)t->ne[3], tensorSum(t),
                    t->src[0] ? nodeIdx.count(t->src[0]) ? nodeIdx.at(t->src[0])
                                                         : -2
                              : -1,
                    t->src[1] ? nodeIdx.count(t->src[1]) ? nodeIdx.at(t->src[1])
                                                         : -2
                              : -1,
                    (void*)t->data, t->src[0] ? t->src[0]->data : nullptr,
                    t->src[1] ? t->src[1]->data : nullptr);
        }
        std::fflush(stderr);
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
    if (const char* tmpDump = std::getenv("TMP_SCRATCH_DUMP")) {  // SCRATCH
        FILE* f = fopen(tmpDump, "wb");
        fwrite(got.data(), sizeof(float), got.size(), f);
        fclose(f);
    }

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
        // Per-batch-slab sums: identical rows must give identical slabs
        // (any spread is batch aliasing; a uniform offset is a semantic
        // divergence). The batch axis sits on ne[2] for 3-D activations
        // and ne[3] for 4-D ones.
        {
            // 4-D activations carry the batch on ne[3]; 3-D ones (the
            // flatten contract) on ne[2]. The other axis is channels.
            const int64_t b = output->ne[3] > 1 ? output->ne[3] : output->ne[2];
            if (b > 1) {
                const int64_t inner = ggml_nelements(output) / b;
                std::fprintf(stderr, "[fetch-slabs]");
                for (int64_t k = 0; k < b; ++k) {
                    double s = 0.0;
                    for (int64_t i = 0; i < inner; ++i)
                        s += got[static_cast<size_t>(k) * inner + i];
                    std::fprintf(stderr, " %lld:%.4e", (long long)k, s);
                }
                std::fprintf(stderr, "\n");
            }
        }
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
            // keeps ne[0]=C as the fastest axis, so element (n,c) sits at
            // n*C + c in both buffers and the flat comparison is direct.
            // Both sides must really be C order: the fetch is materialized
            // by buildGraph (a raw permuted view is the (C,N) transpose)
            // and readNpyF32 reorders the Fortran-order head files. The
            // two transposes once cancelled here and hid a production
            // layout decode that read every head transposed.
            const int64_t C = t->ne[0], N = t->ne[1];
            for (int64_t n = 0; n < N; ++n)
                for (int64_t c = 0; c < C; ++c) {
                    const float diff =
                            std::fabs(got[n * C + c] - ref->second[n * C + c]);
                    layoutMax = std::max(layoutMax, diff);
                    layoutSum += diff;
                    hsum += got[n * C + c];
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
        exec.free();
        ggml_backend_free(backend);
        ggml_free(ctx);
        return ok ? 0 : 1;
    }

    // The ggml output carries the batch on ne[2] (the rec graph's flatten
    // contract) with ne[3]=1, and each slab (V*T elements, v + t*V order)
    // is contiguous either way. The fixture reference is an N=1 run, so it
    // covers slab 0; when AICORE_OCR_PROBE_TILE>1 replicated the input,
    // every slab must also match its siblings EXACTLY (same graph, same
    // values, deterministic kernels).
    const int64_t outB = std::max(output->ne[2], output->ne[3]);
    const size_t slabCount =
            static_cast<size_t>(vocab) * static_cast<size_t>(timeSteps);
    float maxDiff = 0.0f;
    double sumDiff = 0.0;
    const size_t count = slabCount;
    std::vector<float> diffs(count);
    for (size_t i = 0; i < count; ++i) {
        const float diff = std::fabs(got[i] - refData[i]);
        diffs[i] = diff;
        maxDiff = std::max(maxDiff, diff);
        sumDiff += diff;
    }
    float selfDiff = 0.0f;
    if (outB > 1) {
        for (int64_t b = 1; b < outB; ++b)
            for (size_t i = 0; i < slabCount; ++i)
                selfDiff = std::max(
                        selfDiff,
                        std::fabs(got[i] -
                                  got[slabCount * static_cast<size_t>(b) + i]));
        std::printf("rec batch self-check: slabs=%lld max|slab0-slabK|=%.6f\n",
                    (long long)outB, selfDiff);
    }
    // Diff distribution: the post-softmax rec reference amplifies a few-ULP
    // pre-softmax logit difference into a large probability swing only on
    // near-tied class pairs; a percentile readout distinguishes that benign
    // tail from a systematic kernel deviation (which would lift the
    // percentiles together with the max).
    std::sort(diffs.begin(), diffs.end());
    const float dP50 = diffs[count / 2];
    const float dP99 = diffs[(count * 99) / 100];
    const float dP999 = diffs[(count * 999) / 1000];
    const size_t over1e3 =
            count - (std::lower_bound(diffs.begin(), diffs.end(), 0.001f) -
                     diffs.begin());
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
    std::printf(
            "rec diff distribution: p50=%.2e p99=%.2e p99.9=%.2e "
            ">1e-3=%zu/%zu (%.4f%%)\n",
            dP50, dP99, dP999, over1e3, count,
            100.0 * static_cast<double>(over1e3) / static_cast<double>(count));
    std::printf("decoded text (%.3f): %s\n", meanScore, text.c_str());

    const bool signatureMatch = gotSignature == refSignature;
    const bool withinTolerance = maxDiff < tolerance;
    const bool selfConsistent = outB <= 1 || selfDiff == 0.0f;
    std::printf("argmax match: %s | tolerance(<%g): %s | self: %s\n",
                signatureMatch ? "YES" : "NO", tolerance,
                withinTolerance ? "YES" : "NO", selfConsistent ? "YES" : "NO");
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

    // Slot-reuse contract (OcrGraphExec): rebuilding into the same exec
    // keeps its gallocr, grows the compute buffer only for a larger graph,
    // keeps the very same buffer for a smaller one (no fresh ReBAR mapping
    // to fault in on Vulkan), and the rebuilt graph reproduces the first
    // run bit for bit.
    bool slotReuse = true;
    if (probe.fetchAt < 0 && tileN == 1 && exec.alloc) {
        const ggml_gallocr_t firstAlloc = exec.alloc;
        const size_t firstSize = ggml_gallocr_get_buffer_size(exec.alloc, 0);
        const auto rebuild = [&](int64_t rows, std::vector<float>* out,
                                 ggml_backend_buffer_t* buffer) {
            ggml_context* c = ggml_init({tensorSize * 2, nullptr,
                                         /*no_alloc=*/true});
            ggml_tensor* in =
                    ggml_new_tensor_4d(c, GGML_TYPE_F32, inputShape[3],
                                       inputShape[2], inputShape[1], rows);
            ggml_set_input(in);
            weights.resetFoldPool();
            ggml_tensor* o = aicore::ocr::buildGraph(
                    c, weights, in, &error, nullptr, nullptr, &buildOptions);
            ggml_cgraph* g = nullptr;
            const bool built =
                    o && aicore::ocr::prepareGraph(weights, c, plan, {o}, &g,
                                                   exec, error);
            if (built) {
                std::vector<float> batch;
                for (int64_t r = 0; r < rows; ++r)
                    batch.insert(batch.end(), inputData.begin(),
                                 inputData.end());
                ggml_backend_tensor_set(in, batch.data(), 0,
                                        sizeof(float) * batch.size());
                aicore::ocr::computeGraph(plan, exec, g);
                *buffer = o->buffer;
                if (out) {
                    out->resize(static_cast<size_t>(ggml_nelements(o)));
                    ggml_backend_tensor_get(o, out->data(), 0,
                                            sizeof(float) * out->size());
                }
            }
            ggml_free(c);
            return built;
        };
        std::vector<float> again;
        ggml_backend_buffer_t largeBuffer = nullptr, smallBuffer = nullptr;
        const bool grown = rebuild(2, nullptr, &largeBuffer);
        const size_t grownSize =
                grown ? ggml_gallocr_get_buffer_size(exec.alloc, 0) : 0;
        const bool rebuilt = grown && rebuild(1, &again, &smallBuffer);
        const size_t finalSize =
                rebuilt ? ggml_gallocr_get_buffer_size(exec.alloc, 0) : 0;
        slotReuse = rebuilt && exec.alloc == firstAlloc &&
                    grownSize > firstSize && finalSize == grownSize &&
                    smallBuffer == largeBuffer && again.size() == got.size() &&
                    std::memcmp(again.data(), got.data(),
                                sizeof(float) * got.size()) == 0;
        std::printf(
                "slot reuse: %s (buffer %zu -> %zu -> %zu bytes, same "
                "allocator %d, same buffer %d)\n",
                slotReuse ? "YES" : "NO", firstSize, grownSize, finalSize,
                exec.alloc == firstAlloc ? 1 : 0,
                smallBuffer == largeBuffer ? 1 : 0);
        if (!slotReuse && !error.empty())
            std::fprintf(stderr, "slot reuse: %s\n", error.c_str());
    }

    exec.free();
    ggml_backend_free(backend);
    ggml_free(ctx);
    return (signatureMatch && withinTolerance && selfConsistent && slotReuse)
                   ? 0
                   : 1;
}

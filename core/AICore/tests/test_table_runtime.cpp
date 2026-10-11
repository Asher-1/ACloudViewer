// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Graph-level parity probe for the AICore `table` task: run the SLANet
// encoder (OCR interpreter) + ggml structure decoder over pinned Paddle
// fixtures (exact network input, Paddle per-step softmax and cell-location
// outputs, tools/export_table_fixture.py) and compare.
//
//   token_match  every fixture decodes the identical token sequence
//   max_diff     max |softmax - reference| over the decoded steps
//   loc_max_diff max |cell polygon - reference| (sigmoid domain, x488 px)
//
// Usage: test_table_runtime <fixture-dir> <gguf> [backend] [prob_tol]
//                           [loc_tol]

#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#define access _access
#define R_OK 4
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/ggml_backend_utils.hpp"
#include "ggml-backend.h"
#include "table_runtime.hpp"

namespace {

bool readNpyF32(const std::string& path,
                std::vector<int64_t>* shape,
                std::vector<float>* data) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    unsigned char header[10];
    if (fread(header, 1, 10, f) != 10 ||
        std::memcmp(header, "\x93NUMPY", 6) != 0) {
        fclose(f);
        return false;
    }
    const unsigned short headerLen = static_cast<unsigned short>(header[8]) |
                                     static_cast<unsigned short>(header[9])
                                             << 8;
    std::string dict(headerLen, '\0');
    if (fread(dict.data(), 1, headerLen, f) != headerLen ||
        (dict.find("'<f4'") == std::string::npos) ||
        // Flat C-order reads only (export_table_fixture saves contiguous
        // arrays); a Fortran-order file would compare transposed.
        dict.find("'fortran_order': True") != std::string::npos) {
        fclose(f);
        return false;
    }
    const auto open = dict.find('(', dict.find("'shape':"));
    const auto close = dict.find(')', open);
    size_t pos = open + 1;
    while (pos < close) {
        const size_t comma = dict.find(',', pos);
        const size_t end =
                comma == std::string::npos || comma > close ? close : comma;
        const std::string token = dict.substr(pos, end - pos);
        if (!token.empty() && token.find_first_not_of(' ') != std::string::npos)
            shape->push_back(std::atoll(token.c_str()));
        if (end == close) break;
        pos = end + 1;
    }
    size_t count = 1;
    for (int64_t d : *shape) count *= static_cast<size_t>(d);
    data->resize(count);
    const size_t read = fread(data->data(), sizeof(float), count, f);
    fclose(f);
    return read == count;
}

int argmaxRow(const float* row, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (row[i] > row[best]) best = i;
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <fixture-dir> <gguf> [backend] [prob_tol] "
                     "[loc_tol]\n",
                     argv[0]);
        return 2;
    }
    const std::string fixtureDir = argv[1];
    const std::string modelPath = argv[2];
    const std::string backendName = argc > 3 ? argv[3] : "cpu";
    const double probTol = argc > 4 ? std::atof(argv[4]) : 1e-3;
    const double locTol = argc > 5 ? std::atof(argv[5]) : 1e-3;
    if (access((fixtureDir + "/table_0_input.npy").c_str(), R_OK) != 0 ||
        access(modelPath.c_str(), R_OK) != 0) {
        std::fprintf(stderr,
                     "probe_table: fixtures/model not fetched, skipping\n");
        return 77;
    }

    aicore::table::TableStructureModel model;
    std::string error;
    if (!model.load(modelPath, &error)) {
        std::fprintf(stderr, "model load failed: %s\n", error.c_str());
        return 1;
    }

    ggml_common::load_backends_once();
    std::string devLower = backendName;
    std::transform(
            devLower.begin(), devLower.end(), devLower.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    ggml_backend_t gpu = nullptr;
    if (devLower != "cpu") {
        std::string resolved;
        gpu = ggml_common::find_gpu_backend(backendName, 0, resolved);
        if (!gpu) {
            std::fprintf(stderr,
                         "probe_table: backend %s not available, skipping\n",
                         backendName.c_str());
            return 77;
        }
        std::fprintf(stderr, "[device] requested=%s resolved=%s\n",
                     backendName.c_str(), resolved.c_str());
        model.setSchedGpuTagging(devLower.find("vulkan") == std::string::npos);
    }
    ggml_backend_t cpu =
            ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        std::fprintf(stderr, "cpu backend init failed\n");
        return 1;
    }

    bool tokensOk = true;
    float maxDiff = 0.0f, locMax = 0.0f;
    double sumDiff = 0.0;
    size_t countDiff = 0;
    int fixtures = 0;
    std::vector<float> lastInput;
    for (int k = 0;; ++k) {
        const std::string stem = fixtureDir + "/table_" + std::to_string(k);
        if (access((stem + "_input.npy").c_str(), R_OK) != 0) break;
        std::vector<int64_t> inShape, pShape, lShape;
        std::vector<float> input, refProbs, refLoc;
        if (!readNpyF32(stem + "_input.npy", &inShape, &input) ||
            !readNpyF32(stem + "_probs.npy", &pShape, &refProbs) ||
            !readNpyF32(stem + "_loc.npy", &lShape, &refLoc) ||
            pShape.size() != 2 || lShape.size() != 2) {
            std::fprintf(stderr, "failed to read fixture %d\n", k);
            return 1;
        }
        if (inShape.size() != 4 || inShape[2] != model.padSize() ||
            inShape[3] != model.padSize()) {
            std::fprintf(stderr, "fixture %d input is not padded to %d\n", k,
                         model.padSize());
            return 1;
        }
        aicore::table::TableStructureOutput out;
        if (!model.infer(gpu, cpu, input.data(), out, error, true)) {
            std::fprintf(stderr, "infer failed: %s\n", error.c_str());
            return 1;
        }
        const int vocab = static_cast<int>(pShape[1]);
        const int refSteps = static_cast<int>(pShape[0]);
        const int gotSteps = static_cast<int>(out.tokens.size());
        int firstMismatch = -1;
        for (int s = 0; s < std::min(refSteps, gotSteps); ++s) {
            if (argmaxRow(refProbs.data() + static_cast<size_t>(s) * vocab,
                          vocab) != out.tokens[static_cast<size_t>(s)]) {
                firstMismatch = s;
                break;
            }
        }
        const bool match = firstMismatch < 0 && refSteps == gotSteps;
        tokensOk = tokensOk && match;
        // Element-wise diffs are only meaningful while both decoders follow
        // the same token prefix.
        const int compared = firstMismatch < 0 ? std::min(refSteps, gotSteps)
                                               : firstMismatch;
        float fixtureMax = 0.0f, fixtureLoc = 0.0f;
        for (int s = 0; s < compared; ++s) {
            for (int v = 0; v < vocab; ++v) {
                const size_t i = static_cast<size_t>(s) * vocab + v;
                const float d = std::fabs(out.probs[i] - refProbs[i]);
                fixtureMax = std::max(fixtureMax, d);
                sumDiff += d;
                ++countDiff;
            }
            for (int j = 0; j < model.locReg(); ++j) {
                const size_t i = static_cast<size_t>(s) * model.locReg() + j;
                fixtureLoc =
                        std::max(fixtureLoc, std::fabs(out.loc[i] - refLoc[i]));
            }
        }
        maxDiff = std::max(maxDiff, fixtureMax);
        locMax = std::max(locMax, fixtureLoc);
        std::printf(
                "fixture %d: steps got=%d ref=%d tokens=%s max_diff=%.6f "
                "loc_max_diff=%.6f encoder=%.2fms decoder=%.2fms\n",
                k, gotSteps, refSteps, match ? "MATCH" : "MISMATCH", fixtureMax,
                fixtureLoc, out.encoderMs, out.decoderMs);
        if (!match) std::printf("  first mismatch at step %d\n", firstMismatch);
        lastInput = std::move(input);
        ++fixtures;
    }
    if (fixtures == 0) {
        std::fprintf(stderr, "no fixtures found\n");
        return 1;
    }

    // Steady-state latency on the longest fixture (the last one).
    std::vector<double> samples;
    for (int r = 0; r < 6; ++r) {
        aicore::table::TableStructureOutput out;
        const auto t0 = std::chrono::steady_clock::now();
        if (!model.infer(gpu, cpu, lastInput.data(), out, error)) return 1;
        const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
        if (r > 0) samples.push_back(ms);
    }
    std::sort(samples.begin(), samples.end());
    const double p50 = samples[samples.size() / 2];
    const double p95 = samples.back();
    const bool ok = tokensOk && maxDiff < probTol && locMax < locTol;
    std::printf(
            "table probe: fixtures=%d token_match=%s max_diff=%.6f "
            "loc_max_diff=%.6f ok=%s\n",
            fixtures, tokensOk ? "YES" : "NO", maxDiff, locMax,
            ok ? "YES" : "NO");
    std::printf(
            "{\"task\":\"table\",\"backend\":\"%s\",\"token_match\":%d,"
            "\"max_diff\":%.6f,\"mean_diff\":%.8f,\"loc_max_diff\":%.6f,"
            "\"p50_ms\":%.2f,\"p95_ms\":%.2f}\n",
            backendName.c_str(), tokensOk ? 1 : 0, maxDiff,
            countDiff ? sumDiff / static_cast<double>(countDiff) : 0.0, locMax,
            p50, p95);
    model.release();
    if (gpu) ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return ok ? 0 : 1;
}

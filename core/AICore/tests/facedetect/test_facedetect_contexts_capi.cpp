// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Two contexts with different backend-registry keys must remain valid when
// either one loads or runs. This regresses the former process-global backend,
// where loading the second context destroyed the first context's weight buffer.
// SKIP (77) without the same model/image environment used by other face tests.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "aicore/facedetect_capi.h"

namespace {

bool embed(aicore_facedetect_ctx* ctx,
           const char* image,
           std::vector<float>* output) {
    float* values = nullptr;
    int dim = 0;
    const int rc =
            aicore_facedetect_embed_path(ctx, image, 0.0f, &values, &dim);
    if (rc != 0 || values == nullptr || dim <= 0) {
        aicore_facedetect_free_buffer(values);
        return false;
    }
    output->assign(values, values + dim);
    aicore_facedetect_free_buffer(values);
    return true;
}

float cosine_distance(const std::vector<float>& a,
                      const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 1.0f;
    double dot = 0.0;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return static_cast<float>(1.0 - dot);
}

aicore_facedetect_ctx* load(const char* model, int threads) {
    aicore_facedetect_options* options = aicore_facedetect_options_new();
    if (!options) return nullptr;
    // Different thread counts intentionally select different registry entries
    // on a CPU-only host while retaining deterministic output.
    aicore_facedetect_options_set_device(options, "cpu");
    aicore_facedetect_options_set_threads(options, threads);
    aicore_facedetect_ctx* ctx = aicore_facedetect_load_opts(model, options);
    aicore_facedetect_options_free(options);
    return ctx;
}

}  // namespace

int main() {
    const char* model = std::getenv("AICORE_TEST_FACEDETECT_GGUF");
    const char* image = std::getenv("AICORE_TEST_FACEDETECT_IMAGE");
    if (!model || !image) return 77;

    aicore_facedetect_ctx* first = load(model, 1);
    aicore_facedetect_ctx* second = load(model, 2);
    if (!aicore_facedetect_is_ready(first) ||
        !aicore_facedetect_is_ready(second)) {
        std::fprintf(stderr, "failed to load independent face contexts\n");
        aicore_facedetect_free(first);
        aicore_facedetect_free(second);
        return 1;
    }

    std::vector<float> before;
    std::vector<float> middle;
    std::vector<float> after;
    const bool ok = embed(first, image, &before) &&
                    embed(second, image, &middle) &&
                    embed(first, image, &after);
    aicore_facedetect_free(second);
    aicore_facedetect_free(first);
    if (!ok) {
        std::fprintf(stderr, "interleaved face context inference failed\n");
        return 1;
    }
    const float first_distance = cosine_distance(before, after);
    const float cross_distance = cosine_distance(before, middle);
    if (first_distance > 1e-4f || cross_distance > 1e-4f) {
        std::fprintf(stderr, "context result mismatch: first=%f cross=%f\n",
                     first_distance, cross_distance);
        return 1;
    }
    std::fprintf(stderr, "test_facedetect_contexts_capi ok\n");
    return 0;
}

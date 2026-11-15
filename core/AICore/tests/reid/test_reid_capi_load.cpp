// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// ReID real-model probe: contract + accuracy/stability/timing validation
// against a ReID-encoder GGUF. Also the aicore-validate-all "reid" scenario
// executable: the default row reuses the shared yolo_models classify asset
// (legacy embed-tap flavor), and the reid-native-models rows cover the
// authoritative reid-yolo26{n,s,m,l,x} encoders converted from the official
// yolo26*-reid.onnx assets (task='reid' graphs; the graph output IS the
// embedding).
//
// Usage: test_reid_capi_load <model.gguf> <backend> [iters] [warmups]
// Env fallback: AICORE_TEST_REID_GGUF / AICORE_TEST_DEVICE
// (missing assets -> exit 77 skip).

#include <QImage>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "aicore/reid_capi.h"
#include "tests/common/test_macros.hpp"
#include "tests/common/validation_probe.hpp"

static int failures = 0;

namespace {

constexpr int kW = 640;
constexpr int kH = 480;

// Deterministic synthetic image: the C++ probe and the python baselines in
// tools/reid_validate.py see identical geometry (gradient RGB pattern).
QImage makeSyntheticImage() {
    QImage img(kW, kH, QImage::Format_RGB888);
    for (int y = 0; y < kH; ++y) {
        uchar* row = img.scanLine(y);
        for (int x = 0; x < kW; ++x) {
            row[x * 3 + 0] = (uchar)(x * 255 / kW);
            row[x * 3 + 1] = (uchar)(y * 255 / kH);
            row[x * 3 + 2] = (uchar)((x + y) * 255 / (kW + kH));
        }
    }
    return img;
}

uint64_t embedHash(const float* data, int32_t count, int32_t dim) {
    return aicore::test::fnv1a(data,
                               (size_t)count * (size_t)dim * sizeof(float));
}

}  // namespace

int main(int argc, char** argv) {
    // Args first (validate-all scenario), env fallback (standalone ctest).
    const char* gguf =
            argc > 1 ? argv[1] : std::getenv("AICORE_TEST_REID_GGUF");
    const char* device = argc > 2 ? argv[2] : std::getenv("AICORE_TEST_DEVICE");
    const int iters = argc > 3 ? std::atoi(argv[3]) : 2;
    const int warmups = argc > 4 ? std::atoi(argv[4]) : 1;
    if (!gguf || gguf[0] == '\0') {
        return 77;  // missing asset -> skip (ctest SKIP_RETURN_CODE)
    }
    if (!device || device[0] == '\0') device = "cpu";
    if (iters < 1) {
        std::fprintf(stderr, "invalid inference_runs %d\n", iters);
        return 1;
    }

    QImage img = makeSyntheticImage();
    aicore_image_view view{};
    view.data = img.constBits();
    view.width = kW;
    view.height = kH;
    view.row_stride_bytes = (size_t)img.bytesPerLine();
    view.format = AICORE_IMAGE_RGB8;
    // Fixed evaluation boxes (same geometry as tools/reid_probe.cpp).
    std::vector<float> boxes = {40,  60,  140, 220, 300, 100,
                                460, 300, 180, 200, 260, 330};
    const int32_t count = (int32_t)(boxes.size() / 4);

    // ---- load ---------------------------------------------------------------
    aicore_reid_options* opts = aicore_reid_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_reid_options_set_device(opts, device);
    aicore_reid_options_set_threads(opts, 0);
    aicore_reid_ctx* ctx = aicore_reid_load_opts(gguf, opts);
    aicore_reid_options_free(opts);
    AICORE_CHECK(ctx != nullptr);
    if (!ctx) return 1;
    if (!aicore_reid_is_ready(ctx)) {
        std::fprintf(stderr, "reid load failed: %s\n",
                     aicore_reid_last_error(ctx) ? aicore_reid_last_error(ctx)
                                                 : "?");
        failures++;
    }
    AICORE_CHECK(aicore_reid_last_error(ctx) == nullptr);
    // Header contract: embed_dim is 0 until the first successful embed call
    // validates the graph output.
    AICORE_CHECK(aicore_reid_embed_dim(ctx) == 0);

    // ---- request validation (contract errors, no crash) ---------------------
    float* out = nullptr;
    int32_t n = 0;
    int32_t dim = 0;
    AICORE_CHECK(aicore_reid_embed_image(ctx, &view, boxes.data(), count,
                                         nullptr, &n, &dim) == -1);
    AICORE_CHECK(aicore_reid_embed_image(ctx, &view, boxes.data(), -1, &out, &n,
                                         &dim) == -1);
    AICORE_CHECK(aicore_reid_embed_image(ctx, &view, nullptr, count, &out, &n,
                                         &dim) == -1);
    AICORE_CHECK(aicore_reid_embed_image(ctx, nullptr, boxes.data(), count,
                                         &out, &n, &dim) == -1);
    AICORE_CHECK(aicore_reid_last_error(ctx) != nullptr);
    AICORE_CHECK(aicore_reid_is_ready(ctx) == 1);  // errors do not kill the ctx

    // ---- empty batch --------------------------------------------------------
    out = nullptr;
    AICORE_CHECK(aicore_reid_embed_image(ctx, &view, boxes.data(), 0, &out, &n,
                                         &dim) == 0);
    AICORE_CHECK(out != nullptr && n == 0 && dim == 0);
    aicore_reid_free_buffer(out);

    // ---- warmups + repeated forwards (stability ladder) ---------------------
    uint64_t first_hash = 0;
    aicore_pipeline_timings pt{};
    for (int iter = -warmups; iter < iters; ++iter) {
        out = nullptr;
        const int rc = aicore_reid_embed_image(ctx, &view, boxes.data(), count,
                                               &out, &n, &dim);
        if (rc != 0) {
            std::fprintf(stderr, "embed failed rc=%d: %s\n", rc,
                         aicore_reid_last_error(ctx)
                                 ? aicore_reid_last_error(ctx)
                                 : "?");
            failures++;
            break;
        }
        AICORE_CHECK(n == count && dim > 0);
        AICORE_CHECK(aicore_reid_embed_dim(ctx) == dim);
        for (int32_t i = 0; i < n * dim; ++i) {
            AICORE_CHECK(std::isfinite(out[i]));
        }
        AICORE_CHECK(aicore_reid_last_pipeline_timings(ctx, &pt) == 0);
        // Honest valid_fields: preprocess/inference measured for a non-empty
        // batch; postprocess and e2e always measured.
        AICORE_CHECK((pt.valid_fields & AICORE_TIMING_PREPROCESS) != 0);
        AICORE_CHECK((pt.valid_fields & AICORE_TIMING_INFERENCE) != 0);
        AICORE_CHECK((pt.valid_fields & AICORE_TIMING_POSTPROCESS) != 0);
        AICORE_CHECK((pt.valid_fields & AICORE_TIMING_E2E) != 0);
        AICORE_CHECK(pt.e2e_ms > 0.0);
        AICORE_CHECK(pt.inference_ms > 0.0);
        AICORE_CHECK(pt.preprocess_ms > 0.0);
        const uint64_t hash = embedHash(out, n, dim);
        if (iter == 0) {
            first_hash = hash;
        } else if (iter > 0) {
            // Exact repeated-output stability across same-shape forwards.
            AICORE_CHECK(hash == first_hash);
        }
        aicore_reid_free_buffer(out);
    }

    aicore::test::printValidationResult("reid", device, first_hash, &pt);

    // ---- context destroy / recreate (cache-safety ladder) -------------------
    aicore_reid_free(ctx);
    opts = aicore_reid_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_reid_options_set_device(opts, device);
    aicore_reid_ctx* ctx2 = aicore_reid_load_opts(gguf, opts);
    aicore_reid_options_free(opts);
    AICORE_CHECK(ctx2 != nullptr && aicore_reid_is_ready(ctx2) == 1);
    if (ctx2 && aicore_reid_is_ready(ctx2)) {
        out = nullptr;
        AICORE_CHECK(aicore_reid_embed_image(ctx2, &view, boxes.data(), count,
                                             &out, &n, &dim) == 0);
        // Same-model second context must reproduce the same output.
        AICORE_CHECK(embedHash(out, n, dim) == first_hash);
        aicore_reid_free_buffer(out);
    }
    aicore_reid_free(ctx2);

    aicore_reid_shutdown();

    std::fprintf(stderr, "reid load ok: %s device=%s dim=%d\n", gguf, device,
                 dim);
    return failures == 0 ? 0 : 1;
}

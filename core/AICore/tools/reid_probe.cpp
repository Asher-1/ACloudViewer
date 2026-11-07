// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// ReID probe: deterministic synthetic image + fixed boxes -> embeddings
// and batch latency, written as JSON. Used by core/AICore/tools/
// reid_validate.py (the PyTorch / ONNX Runtime baselines run in their own
// process; numpy and libAICore must not share one process).

#include <QImage>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/reid_capi.h"

namespace {

constexpr int kW = 640, kH = 480;

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

}  // namespace

int main(int argc, char** argv) {
    const char* gguf = nullptr;
    const char* device = "cuda";
    const char* output = nullptr;
    int iters = 5;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc)
            gguf = argv[++i];
        else if (a == "--device" && i + 1 < argc)
            device = argv[++i];
        else if (a == "--output" && i + 1 < argc)
            output = argv[++i];
        else if (a == "--iters" && i + 1 < argc)
            iters = std::max(1, atoi(argv[++i]));
    }
    if (gguf == nullptr) {
        fprintf(stderr,
                "usage: reid_probe --gguf <file> [--device cuda|vulkan|cpu] "
                "[--output file.json] [--iters N]\n");
        return 2;
    }

    aicore_reid_options* opts = aicore_reid_options_new();
    aicore_reid_options_set_device(opts, device);
    aicore_reid_ctx* ctx = aicore_reid_load_opts(gguf, opts);
    if (ctx == nullptr || !aicore_reid_is_ready(ctx)) {
        fprintf(stderr, "reid load failed: %s\n",
                ctx ? aicore_reid_last_error(ctx) : "(no ctx)");
        return 3;
    }

    QImage img = makeSyntheticImage();
    aicore_image_view view;
    view.data = img.constBits();
    view.width = kW;
    view.height = kH;
    view.row_stride_bytes = (size_t)img.bytesPerLine();
    view.format = AICORE_IMAGE_RGB8;
    // Fixed evaluation boxes (same geometry as the python baselines).
    std::vector<float> boxes = {40,  60,  140, 220, 300, 100,
                                460, 300, 180, 200, 260, 330};
    const int32_t count = (int32_t)(boxes.size() / 4);

    float* out = nullptr;
    int32_t n = 0, dim = 0;
    int rc = aicore_reid_embed_image(ctx, &view, boxes.data(), count, &out, &n,
                                     &dim);
    if (rc != 0) {
        fprintf(stderr, "embed failed: %s\n", aicore_reid_last_error(ctx));
        return 4;
    }

    // Latency loop (includes crop preprocessing; the reported batch cost is
    // the end-to-end per-call time divided by the box count).
    double best = 1e18, sum = 0.0;
    for (int it = 0; it < iters; ++it) {
        float* o = nullptr;
        int32_t on = 0, od = 0;
        aicore_pipeline_timings t{};
        const clock_t c0 = clock();
        rc = aicore_reid_embed_image(ctx, &view, boxes.data(), count, &o, &on,
                                     &od);
        const double ms = 1000.0 * (clock() - c0) / CLOCKS_PER_SEC;
        if (rc != 0) {
            fprintf(stderr, "embed iter failed: %s\n",
                    aicore_reid_last_error(ctx));
            return 4;
        }
        aicore_reid_free_buffer(o);
        best = std::min(best, ms);
        sum += ms;
    }

    // Emit the JSON record: embeddings (row-major) + latency stats.
    aicore_pipeline_timings timings{};
    aicore_reid_last_pipeline_timings(ctx, &timings);
    std::string json = "{\n";
    json += "  \"gguf\": \"" + std::string(gguf) + "\",\n";
    json += "  \"device\": \"" + std::string(device) + "\",\n";
    json += "  \"count\": " + std::to_string(n) + ",\n";
    json += "  \"dim\": " + std::to_string(dim) + ",\n";
    json += "  \"batch_best_ms\": " + std::to_string(best) + ",\n";
    json += "  \"batch_mean_ms\": " + std::to_string(sum / iters) + ",\n";
    json += "  \"per_box_ms\": " + std::to_string(sum / iters / count) + ",\n";
    json += "  \"e2e_ms\": " + std::to_string(timings.e2e_ms) + ",\n";
    json += "  \"preprocess_ms\": " + std::to_string(timings.preprocess_ms) +
            ",\n";
    json += "  \"inference_ms\": " + std::to_string(timings.inference_ms) +
            ",\n";
    json += "  \"embeddings\": [";
    for (int32_t r = 0; r < n; ++r) {
        json += r ? ", [" : "[";
        for (int32_t c = 0; c < dim; ++c) {
            json += (c ? ", " : "") + std::to_string(out[(size_t)r * dim + c]);
        }
        json += "]";
    }
    json += "]\n}\n";

    if (output != nullptr) {
        FILE* f = fopen(output, "w");
        if (f == nullptr) return 5;
        fwrite(json.data(), 1, json.size(), f);
        fclose(f);
    } else {
        fwrite(json.data(), 1, json.size(), stdout);
    }
    aicore_reid_free_buffer(out);
    aicore_reid_free(ctx);
    aicore_reid_options_free(opts);
    return 0;
}

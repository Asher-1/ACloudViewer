// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Three-way SAM3 mask triage: run up to three (gguf, device) combinations on
// the same image + PVS prompt and report pairwise mask IoU. This separates
// "the quantized model deviates from f16 on every backend" (quantization
// sensitivity) from "one backend deviates while the others agree" (backend
// divergence).
//
// Usage:
//   cmp_sam3_mask_triage <image> <gguf1> <dev1> [<gguf2> <dev2>] [<gguf3> <dev3>]
//
// Output: one JSON line per run, then a JSON summary with pairwise IoU.
// Diagnostic locator only: exits 0 even when individual runs fail.

#include <QImage>
#include <QImageReader>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/sam3_capi.h"

namespace {

struct MaskResult {
    std::vector<uint8_t> mask;
    int nonzero = 0;
    float score = 0.0f;
    bool valid = false;
};

bool load_image(const char *path, std::vector<uint8_t> *rgb, int *width,
                int *height) {
    QImageReader reader(QString::fromUtf8(path));
    reader.setAutoTransform(true);
    const QImage decoded = reader.read();
    if (decoded.isNull()) return false;
    const QImage converted = decoded.convertToFormat(QImage::Format_RGB888);
    *width = converted.width();
    *height = converted.height();
    rgb->resize(static_cast<size_t>(*width) * *height * 3);
    for (int y = 0; y < *height; ++y) {
        std::memcpy(rgb->data() + static_cast<size_t>(y) * *width * 3,
                    converted.constScanLine(y), static_cast<size_t>(*width) * 3);
    }
    return true;
}

bool copy_result(aicore_sam3_seg_result *result, MaskResult *out) {
    if (!result || aicore_sam3_seg_det_count(result) <= 0) return false;
    const aicore_sam3_plane_view view = aicore_sam3_seg_mask_at(result, 0);
    if (!view.data || view.width <= 0 || view.height <= 0 ||
        view.row_stride_bytes < static_cast<size_t>(view.width))
        return false;
    out->mask.resize(static_cast<size_t>(view.width) * view.height);
    out->nonzero = 0;
    const auto *data = static_cast<const uint8_t *>(view.data);
    for (int y = 0; y < view.height; ++y) {
        const auto *row = data + static_cast<size_t>(y) * view.row_stride_bytes;
        auto *dst = out->mask.data() + static_cast<size_t>(y) * view.width;
        std::memcpy(dst, row, static_cast<size_t>(view.width));
        out->nonzero += static_cast<int>(std::count_if(
                dst, dst + view.width, [](uint8_t v) { return v != 0; }));
    }
    out->score = aicore_sam3_seg_det_score_at(result, 0);
    out->valid = std::isfinite(out->score) && out->nonzero > 0;
    return out->valid;
}

double mask_iou(const MaskResult &a, const MaskResult &b) {
    if (!a.valid || !b.valid || a.mask.size() != b.mask.size()) return 0.0;
    size_t inter = 0, uni = 0;
    for (size_t i = 0; i < a.mask.size(); ++i) {
        const bool av = a.mask[i] != 0;
        const bool bv = b.mask[i] != 0;
        inter += av && bv;
        uni += av || bv;
    }
    return uni == 0 ? 0.0 : static_cast<double>(inter) / uni;
}

bool run_one(const char *gguf, const char *device,
             const aicore_sam3_pvs_prompt &prompt,
             const std::vector<uint8_t> &rgb, int width, int height,
             MaskResult *out) {
    aicore_sam3_options *options = aicore_sam3_options_new();
    aicore_sam3_options_set_device(options, device);
    aicore_sam3_options_set_threads(options, 4);
    aicore_sam3_ctx *ctx = aicore_sam3_load_opts(gguf, options);
    aicore_sam3_options_free(options);
    if (!ctx || !aicore_sam3_is_ready(ctx)) {
        std::fprintf(stderr, "[%s] load failed: %s\n", device,
                     ctx ? aicore_sam3_last_error(ctx)
                         : aicore_sam3_last_load_error());
        aicore_sam3_free(ctx);
        return false;
    }
    if (aicore_sam3_encode_rgb(ctx, rgb.data(), width, height,
                               static_cast<size_t>(width) * 3, 1) != 0) {
        std::fprintf(stderr, "[%s] encode failed: %s\n", device,
                     aicore_sam3_last_error(ctx));
        aicore_sam3_free(ctx);
        return false;
    }
    aicore_sam3_seg_result *pvs = aicore_sam3_segment_pvs_rgb(
            ctx, &prompt, rgb.data(), width, height,
            static_cast<size_t>(width) * 3);
    const bool ok = copy_result(pvs, out);
    if (!ok) {
        std::fprintf(stderr, "[%s] PVS failed: %s\n", device,
                     aicore_sam3_last_error(ctx));
    }
    aicore_sam3_seg_result_free(pvs);
    aicore_sam3_free(ctx);
    return ok;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 6 || (argc - 2) % 2 != 0) {
        std::fprintf(stderr,
                     "usage: %s <image> <gguf1> <dev1> [<gguf2> <dev2> "
                     "[<gguf3> <dev3>]]\n",
                     argv[0]);
        return 2;
    }
    const int runs = (argc - 2) / 2;

    int width = 0, height = 0;
    std::vector<uint8_t> rgb;
    if (!load_image(argv[1], &rgb, &width, &height)) {
        std::fprintf(stderr, "could not decode image: %s\n", argv[1]);
        return 1;
    }

    // Same proportional prompt as bench_sam3_backend_acceptance (the recorded
    // successful cand1.jpg box), so results are directly comparable.
    aicore_sam3_pvs_prompt prompt{};
    prompt.box = {width * (947.0f / 1920.0f), height * (515.0f / 1280.0f),
                  width * (1084.0f / 1920.0f), height * (623.0f / 1280.0f)};
    prompt.use_box = 1;

    std::vector<MaskResult> masks(runs);
    for (int r = 0; r < runs; ++r) {
        const char *gguf = argv[2 + r * 2];
        const char *device = argv[3 + r * 2];
        const bool ok =
                run_one(gguf, device, prompt, rgb, width, height, &masks[r]);
        std::printf("{\"run\":%d,\"gguf\":\"%s\",\"device\":\"%s\",\"valid\":%s,"
                    "\"nonzero\":%d,\"score\":%.6f}\n",
                    r, gguf, device, ok ? "true" : "false", masks[r].nonzero,
                    masks[r].score);
    }
    std::printf("{\"pairwise_iou\":[");
    for (int a = 0; a < runs; ++a) {
        for (int b = a + 1; b < runs; ++b) {
            std::printf("%s{\"a\":%d,\"b\":%d,\"iou\":%.8f}",
                        (a || b) ? "," : "", a, b,
                        mask_iou(masks[a], masks[b]));
        }
    }
    std::printf("]}\n");
    return 0;
}

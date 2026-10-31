// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// End-to-end OCR benchmark: one image -> aicore_ocr_run (det + rec [+ layout])
// with warmups and p50/p95 reporting. The comparison counterpart is the
// official PaddleOCR pipeline running the same PNG on the same host
// (core/AICore/scripts/bench_paddle_official.py); the image is decoded
// through the public aicore_map_anything_load_path_rgb helper so the tool
// needs no codec of its own.
//
// usage: bench_ocr <image> <det.gguf> <rec.gguf> <device> <threads>
//                  <warmups> <iters> [layout.gguf] [--recognize-full-image]
//
// --recognize-full-image drives the public aicore_ocr_recognize ABI with the
// whole input image as one quad. It exists for official cropped-word GT sets:
// their labels exercise the published recognition model without pretending a
// detector has produced a box for an already-cropped image.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "aicore/map_anything_capi.h"
#include "aicore/ocr_capi.h"

namespace {

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t idx = static_cast<size_t>(p * (v.size() - 1));
    return v[idx];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(stderr,
                     "usage: %s <image> <det.gguf> <rec.gguf> <device> "
                     "<threads> <warmups> <iters> [layout.gguf]\n",
                     argv[0]);
        return 2;
    }
    const char* image = argv[1];
    const char* detPath = argv[2];
    const char* recPath = argv[3];
    const char* device = argv[4];
    const int threads = std::atoi(argv[5]);
    const int warmups = std::atoi(argv[6]);
    const int iters = std::atoi(argv[7]);
    const char* layoutPath = nullptr;
    bool recognizeFullImage = false;
    for (int i = 8; i < argc; ++i) {
        if (std::string(argv[i]) == "--recognize-full-image")
            recognizeFullImage = true;
        else if (!layoutPath)
            layoutPath = argv[i];
        else {
            std::fprintf(stderr, "bench_ocr: unexpected argument %s\n",
                         argv[i]);
            return 2;
        }
    }

    uint8_t* rgb = nullptr;
    int32_t w = 0, h = 0;
    if (aicore_map_anything_load_path_rgb(image, &rgb, &w, &h) != 0 || !rgb) {
        std::fprintf(stderr, "bench_ocr: failed to decode %s\n", image);
        return 1;
    }

    aicore_ocr_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = AICORE_OCR_ABI_VERSION;
    options.det_model_path = detPath;
    options.rec_model_path = recPath;
    options.layout_model_path = layoutPath;
    options.device = device;
    options.n_threads = threads;
    aicore_ocr_ctx* ctx = aicore_ocr_context_create(&options);
    if (!ctx || !aicore_ocr_context_ready(ctx)) {
        std::fprintf(stderr, "bench_ocr: context create failed: %s\n",
                     aicore_ocr_last_error());
        return 1;
    }

    aicore_image_view view{};
    view.data = rgb;
    view.width = w;
    view.height = h;
    view.format = AICORE_IMAGE_RGB8;
    view.row_stride_bytes = static_cast<size_t>(w) * 3;

    const float fullImageQuad[8] = {0.0f,
                                    0.0f,
                                    static_cast<float>(w - 1),
                                    0.0f,
                                    static_cast<float>(w - 1),
                                    static_cast<float>(h - 1),
                                    0.0f,
                                    static_cast<float>(h - 1)};
    const auto runOnce = [&] {
        return recognizeFullImage
                       ? aicore_ocr_recognize(ctx, &view, fullImageQuad, 1)
                       : aicore_ocr_run(ctx, &view);
    };
    for (int i = 0; i < warmups; ++i) {
        aicore_ocr_result* r = runOnce();
        if (r) aicore_ocr_result_release(r);
    }
    std::vector<double> e2e, det, inf, post, dbBoxes, cropResizePack, ctc,
            assembly, outputReadback, inputUpload, detGraphCompute,
            recGraphCompute, postCpuCovered, postGraphBoundary, timingGap;
    int boxes = 0;
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        aicore_ocr_result* r = runOnce();
        const auto t1 = std::chrono::steady_clock::now();
        if (!r) {
            std::fprintf(stderr, "bench_ocr: run failed: %s\n",
                         aicore_ocr_last_error());
            return 1;
        }
        const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
        e2e.push_back(ms);
        const aicore_pipeline_timings tm = aicore_ocr_timings(r);
        const aicore_ocr_post_timing_breakdown postTm =
                aicore_ocr_post_timings(r);
        det.push_back(r ? tm.preprocess_ms : 0);
        inf.push_back(tm.inference_ms);
        post.push_back(tm.postprocess_ms);
        dbBoxes.push_back(postTm.db_boxes_ms);
        cropResizePack.push_back(postTm.crop_resize_pack_ms);
        ctc.push_back(postTm.ctc_decode_ms);
        assembly.push_back(postTm.result_assembly_ms);
        outputReadback.push_back(postTm.output_readback_ms);
        inputUpload.push_back(postTm.input_upload_ms);
        detGraphCompute.push_back(postTm.det_graph_compute_ms);
        recGraphCompute.push_back(postTm.rec_graph_compute_ms);
        // CPU-owned postprocessing substages are disjoint; the postprocess_ms
        // residual covers graph boundaries (upload, readback and uninstrumented
        // host work), not a fabricated known substage.
        const double covered = postTm.db_boxes_ms + postTm.crop_resize_pack_ms +
                               postTm.ctc_decode_ms + postTm.result_assembly_ms;
        postCpuCovered.push_back(covered);
        postGraphBoundary.push_back(std::max(0.0, tm.postprocess_ms - covered));
        // Use task-internal e2e timing to exclude result release and benchmark
        // output from this conservation check.
        timingGap.push_back(tm.e2e_ms - tm.preprocess_ms - tm.inference_ms -
                            tm.postprocess_ms);
        boxes = aicore_ocr_box_count(r);
        if (i == iters - 1) {
            for (int t = 0; t < boxes; ++t) {
                const aicore_ocr_box* b = aicore_ocr_box_at(r, t);
                float cx = 0, cy = 0;
                for (int k = 0; k < 4; ++k) {
                    cx += b->quad[k][0];
                    cy += b->quad[k][1];
                }
                cx /= 4;
                cy /= 4;
                const float ew = std::hypot(b->quad[1][0] - b->quad[0][0],
                                            b->quad[1][1] - b->quad[0][1]);
                const float eh = std::hypot(b->quad[3][0] - b->quad[0][0],
                                            b->quad[3][1] - b->quad[0][1]);
                std::printf("TEXT %d %.0f %.0f %.0fx%.0f %s\n", t, cx, cy, ew,
                            eh, b->text ? b->text : "");
                std::printf("OCRBOX %d %.6f", t, b->score);
                for (int k = 0; k < 4; ++k)
                    std::printf(" %.3f,%.3f", b->quad[k][0], b->quad[k][1]);
                std::printf("\t%s\n", b->text ? b->text : "");
            }
            // Layout regions (when a layout model is loaded): class, score and
            // the axis-aligned box so backends can be diffed region-by-region
            // (sub-pixel: thin header/caption regions are ~10 px tall, where
            // integer rounding alone costs several IoU points).
            const int regions = aicore_ocr_region_count(r);
            for (int t = 0; t < regions; ++t) {
                const aicore_ocr_region* g = aicore_ocr_region_at(r, t);
                std::printf(
                        "REGION %d cls=%d order=%d score=%.3f "
                        "%.2f,%.2f %.2f,%.2f\n",
                        t, g->class_id, g->reading_order, g->score,
                        g->quad[0][0], g->quad[0][1], g->quad[2][0],
                        g->quad[2][1]);
            }
            std::fflush(stdout);
        }
        aicore_ocr_result_release(r);
    }
    std::printf(
            "{\"image\":\"%s\",\"device\":\"%s\",\"boxes\":%d,"
            "\"e2e_p50_ms\":%.2f,\"e2e_p95_ms\":%.2f,"
            "\"pre_p50_ms\":%.2f,\"inf_p50_ms\":%.2f,\"post_p50_ms\":%.2f,"
            "\"db_boxes_p50_ms\":%.2f,\"crop_resize_pack_p50_ms\":%.2f,"
            "\"ctc_decode_p50_ms\":%.2f,\"result_assembly_p50_ms\":%.2f,"
            "\"output_readback_p50_ms\":%.2f,\"input_upload_p50_ms\":%.2f,"
            "\"det_graph_compute_p50_ms\":%.2f,"
            "\"rec_graph_compute_p50_ms\":%.2f,"
            "\"post_cpu_covered_p50_ms\":%.2f,"
            "\"post_graph_boundary_p50_ms\":%.2f,"
            "\"timing_gap_p50_ms\":%.2f,\"warmups\":%d,\"iters\":%d}\n",
            image, device, boxes, percentile(e2e, 0.50), percentile(e2e, 0.95),
            percentile(det, 0.50), percentile(inf, 0.50),
            percentile(post, 0.50), percentile(dbBoxes, 0.50),
            percentile(cropResizePack, 0.50), percentile(ctc, 0.50),
            percentile(assembly, 0.50), percentile(outputReadback, 0.50),
            percentile(inputUpload, 0.50), percentile(detGraphCompute, 0.50),
            percentile(recGraphCompute, 0.50), percentile(postCpuCovered, 0.50),
            percentile(postGraphBoundary, 0.50), percentile(timingGap, 0.50),
            warmups, iters);
    aicore_ocr_context_destroy(ctx);
    aicore_map_anything_free_buffer(rgb);
    return 0;
}

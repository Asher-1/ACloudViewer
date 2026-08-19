// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Manual perf probe for the RF-DETR video path — NOT a ctest (needs a local
// GGUF; run it by hand when investigating latency reports).
//
// Decomposes one detection pass into the stages that matter for the live
// video pipeline and prints per-stage medians:
//
//   preprocess-legacy    rfdetr_preprocess(bilinear_no_antialias=false)
//                        (QImage::scaled SmoothTransformation — the path old
//                        GGUFs without rfdetr.preprocess.resize_mode take)
//   preprocess-bilinear  rfdetr_preprocess(bilinear_no_antialias=true)
//                        (hand-rolled half-pixel bilinear)
//   forward              rfdetr::rfdetr_model_forward (graph A + top-K +
//                        graph B; same graphs the plugin runs per frame)
//   select               rfdetr_select_detections
//   detect-total         all of the above as one call (via rfdetr_detect)
//
// Usage:
//   bench_rfdetr_perf <model.gguf> [device] [threads] [srcW] [srcH] [iters]
//   defaults:            -          cpu      0         1920   1080    15

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "backend.hpp"
#include "common.hpp"
#include "image_io.hpp"
#include "model_loader.hpp"
#include "postprocess.hpp"
#include "rfdetr.h"
#include "rfdetr_model.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

double median_of(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "rfdetr-base-f16.gguf";
    const char* device = argc > 2 ? argv[2] : "cpu";
    int n_threads = argc > 3 ? std::atoi(argv[3]) : 0;
    // Match rfdetr_init's resolve_n_threads semantics for the whitebox
    // sections (init_backend_ctx itself clamps <=0 to 1, which is NOT what
    // the plugin path does).
    if (n_threads <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        if (hc == 0) hc = 1;
        n_threads = (int)(hc / 2 > 0 ? hc / 2 : 1);
    }
    const int src_w = argc > 4 ? std::atoi(argv[4]) : 1920;
    const int src_h = argc > 5 ? std::atoi(argv[5]) : 1080;
    const int iters = argc > 6 ? std::max(1, std::atoi(argv[6])) : 15;

    /* --- model + backend, mirroring rfdetr_init ------------------------- */
    rfdetr_status st = RFDETR_OK;
    rfdetr::Model* m = rfdetr::model_load(model_path, &st);
    if (!m) {
        std::fprintf(stderr, "model_load failed: %s\n", rfdetr_status_str(st));
        return 1;
    }
    if (rfdetr::model_validate_tensors(*m) != RFDETR_OK) {
        std::fprintf(stderr, "model_validate_tensors failed\n");
        return 1;
    }
    rfdetr::BackendCtx bctx = rfdetr::init_backend_ctx(n_threads, device, &st);
    if (!bctx.cpu) {
        std::fprintf(stderr, "init_backend_ctx failed: %s\n",
                     rfdetr_status_str(st));
        return 1;
    }
    ggml_backend_t weight_backend = bctx.gpu ? bctx.gpu : bctx.cpu;
    if (rfdetr::model_realize_weights(*m, weight_backend) != RFDETR_OK) {
        std::fprintf(stderr, "model_realize_weights failed\n");
        return 1;
    }

    const int S = (int)m->config.image_size;
    const size_t nq = m->config.num_queries;
    const size_t nc = m->config.num_classes;
    std::fprintf(stderr,
                 "[bench] model=%s variant=%s image_size=%d queries=%zu "
                 "classes=%zu seg=%d\n",
                 model_path, m->config.variant.c_str(), S, nq, nc,
                 m->config.has_segmentation_head ? 1 : 0);
    std::fprintf(stderr,
                 "[bench] device=%s resolved=%s threads=%d src=%dx%d "
                 "iters=%d\n",
                 device, bctx.device_name.c_str(), bctx.n_threads, src_w, src_h,
                 iters);

    /* --- deterministic synthetic frame (1920x1080 default) --------------- */
    std::vector<uint8_t> rgb((size_t)src_w * src_h * 3);
    for (size_t i = 0; i < rgb.size(); ++i) {
        rgb[i] = (uint8_t)((i * 7 + (i / 31) * 13) & 0xff);
    }
    rfdetr_image* img =
            rfdetr_image_from_rgb_buffer(rgb.data(), src_w, src_h, &st);
    if (!img) {
        std::fprintf(stderr, "rfdetr_image_from_rgb_buffer failed\n");
        return 1;
    }

    const float mean[3] = {m->config.preprocess_mean[0],
                           m->config.preprocess_mean[1],
                           m->config.preprocess_mean[2]};
    const float sd[3] = {m->config.preprocess_std[0],
                         m->config.preprocess_std[1],
                         m->config.preprocess_std[2]};

    /* --- A. preprocess, both conventions -------------------------------- */
    std::vector<double> t_legacy, t_bilinear;
    float* px_legacy = nullptr;
    for (int i = 0; i < iters; ++i) {
        float* data = nullptr;
        int w = 0, h = 0;
        const auto t0 = Clock::now();
        const rfdetr_status s1 = rfdetr_preprocess(
                img, S, S, mean, sd, /*bilinear_no_antialias*/ false, &data, &w,
                &h);
        const double dt = ms_since(t0);
        if (s1 != RFDETR_OK) {
            std::fprintf(stderr, "legacy preprocess failed\n");
            return 1;
        }
        t_legacy.push_back(dt);
        if (i == 0)
            px_legacy = data; /* keep one for the forward bench */
        else
            std::free(data);
    }
    for (int i = 0; i < iters; ++i) {
        float* data = nullptr;
        int w = 0, h = 0;
        const auto t0 = Clock::now();
        const rfdetr_status s2 = rfdetr_preprocess(
                img, S, S, mean, sd, /*bilinear_no_antialias*/ true, &data, &w,
                &h);
        const double dt = ms_since(t0);
        if (s2 != RFDETR_OK) {
            std::fprintf(stderr, "bilinear preprocess failed\n");
            return 1;
        }
        t_bilinear.push_back(dt);
        std::free(data);
    }

    /* --- B. forward on the legacy-convention input ----------------------- */
    std::vector<double> t_forward;
    rfdetr::ForwardOutput fout;
    for (int i = 0; i < iters; ++i) {
        const auto t0 = Clock::now();
        fout = rfdetr::rfdetr_model_forward(*m, px_legacy, S, bctx);
        const double dt = ms_since(t0);
        if (fout.class_logits.empty() || fout.bbox_cxcywh.empty()) {
            std::fprintf(stderr, "forward failed on iter %d\n", i);
            return 1;
        }
        t_forward.push_back(dt);
    }

    /* --- C. select (postprocess) ----------------------------------------- */
    std::vector<double> t_select;
    for (int i = 0; i < iters; ++i) {
        rfdetr_detection* dets = nullptr;
        size_t n = 0;
        const auto t0 = Clock::now();
        if (!fout.masks.empty()) {
            rfdetr_select_detections_with_masks(
                    fout.class_logits.data(), fout.bbox_cxcywh.data(),
                    fout.masks.data(), fout.mask_w, fout.mask_h,
                    /*mask_threshold*/ 0.5f, nq, nc, /*threshold*/ 0.5f,
                    /*top_k*/ 300, /*class_filter*/ nullptr,
                    /*filter_len*/ 0, src_w, src_h, &dets, &n);
        } else {
            rfdetr_select_detections(fout.class_logits.data(),
                                     fout.bbox_cxcywh.data(), nq, nc,
                                     /*threshold*/ 0.5f, /*top_k*/ 300, nullptr,
                                     0, src_w, src_h, &dets, &n);
        }
        t_select.push_back(ms_since(t0));
        rfdetr_detections_free(dets, n);
    }

    /* --- D. full detect pass (preprocess + forward + select, exactly the
     *      sequence rfdetr_detect runs; the model's own resize convention) */
    std::vector<double> t_total;
    {
        const bool model_bilinear = m->config.preprocess_bilinear_no_antialias;
        for (int i = 0; i < iters; ++i) {
            rfdetr_detection* dets = nullptr;
            size_t n = 0;
            const auto t0 = Clock::now();
            float* px = nullptr;
            int pw = 0, ph = 0;
            rfdetr_preprocess(img, S, S, mean, sd, model_bilinear, &px, &pw,
                              &ph);
            rfdetr::ForwardOutput fo =
                    rfdetr::rfdetr_model_forward(*m, px, pw, bctx);
            if (!fo.masks.empty()) {
                rfdetr_select_detections_with_masks(
                        fo.class_logits.data(), fo.bbox_cxcywh.data(),
                        fo.masks.data(), fo.mask_w, fo.mask_h, 0.5f, nq, nc,
                        0.5f, 300, nullptr, 0, src_w, src_h, &dets, &n);
            } else {
                rfdetr_select_detections(
                        fo.class_logits.data(), fo.bbox_cxcywh.data(), nq, nc,
                        0.5f, 300, nullptr, 0, src_w, src_h, &dets, &n);
            }
            t_total.push_back(ms_since(t0));
            rfdetr_detections_free(dets, n);
            std::free(px);
        }
    }

    /* --- E. plugin path: rfdetr_init with the RAW requested threads/device
     *      (exactly what aicore_rfdetr_load_opts runs). Prints the resolved
     *      auto threads and backend device so silent CPU fallbacks and the
     *      auto→physical-cores mapping are visible. */
    {
        rfdetr_params p{};
        p.model_path = model_path;
        p.n_threads = n_threads;
        p.device = device;
        rfdetr_status st2 = RFDETR_OK;
        rfdetr_context* ctx = rfdetr_init(&p, &st2);
        if (!ctx) {
            std::fprintf(stderr,
                         "[bench] plugin-path: rfdetr_init failed: %s\n",
                         rfdetr_status_str(st2));
        } else {
            std::fprintf(stderr,
                         "[bench] plugin-path: resolved n_threads=%d "
                         "device=%s\n",
                         rfdetr_context_n_threads(ctx),
                         rfdetr_context_device_name(ctx));
            std::vector<double> t_plugin;
            for (int i = 0; i < iters; ++i) {
                rfdetr_detect_params dp{};
                dp.threshold = 0.5f;
                dp.top_k = 300;
                rfdetr_detection* dets = nullptr;
                size_t n = 0;
                const auto t0 = Clock::now();
                const rfdetr_status s = rfdetr_detect(ctx, img, &dp, &dets, &n);
                const double dt = ms_since(t0);
                if (s != RFDETR_OK) {
                    std::fprintf(stderr, "[bench] plugin-path detect failed\n");
                    break;
                }
                t_plugin.push_back(dt);
                rfdetr_detections_free(dets, n);
            }
            if (!t_plugin.empty()) {
                std::fprintf(stderr, "  detect-plugin-path : %8.1f\n",
                             median_of(t_plugin));
            }
            rfdetr_free(ctx);
        }
    }

    std::fprintf(stderr, "\n[bench] medians (ms, %d iters):\n", iters);
    std::fprintf(stderr, "  preprocess-legacy   : %8.1f\n",
                 median_of(t_legacy));
    std::fprintf(stderr, "  preprocess-bilinear : %8.1f\n",
                 median_of(t_bilinear));
    std::fprintf(stderr, "  forward (A+topK+B)  : %8.1f\n",
                 median_of(t_forward));
    std::fprintf(stderr, "  select              : %8.1f\n",
                 median_of(t_select));
    std::fprintf(stderr, "  detect-total        : %8.1f\n", median_of(t_total));

    std::free(px_legacy);
    rfdetr_image_free(img);
    rfdetr::free_backend_ctx(bctx);
    rfdetr::model_free(m);
    return 0;
}

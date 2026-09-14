// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Real-model GKD probe: contract validation against a published GKDT-L GGUF.
// Also the aicore-validate-all "gkd" scenario executable.
//
// Usage: test_gkd_capi_load <model.gguf> <image> <backend> [iters] [warmups]
// Env fallback: AICORE_TEST_GKD_GGUF / AICORE_TEST_GKD_IMAGE /
// AICORE_TEST_DEVICE (missing assets -> exit 77 skip).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/gkd_capi.h"
#include "tests/common/test_macros.hpp"
#include "tests/common/validation_probe.hpp"

static int failures = 0;

namespace {

std::vector<uint8_t> decodeImage(const char* path, int32_t* w, int32_t* h) {
    std::vector<uint8_t> rgb;
    uint8_t* data = nullptr;
    if (aicore_gkd_load_path_rgb(path, &data, w, h) == 0 && data != nullptr) {
        rgb.assign(data, data + (size_t)(*w) * (*h) * 3);
        aicore_gkd_free_buffer(data);
    }
    return rgb;
}

uint64_t keypointHash(const aicore_gkd_ctx* ctx) {
    uint64_t hash = 1469598103934665603ULL;
    const int n = aicore_gkd_result_keypoint_count(ctx);
    for (int i = 0; i < n; ++i) {
        const aicore_gkd_keypoint kp = aicore_gkd_result_keypoint_at(ctx, i);
        hash = aicore::test::fnv1aAppend(hash, &kp, sizeof(kp));
    }
    return hash;
}

}  // namespace

int main(int argc, char** argv) {
    // Args first (validate-all scenario), env fallback (standalone ctest).
    const char* gguf = argc > 1 ? argv[1] : std::getenv("AICORE_TEST_GKD_GGUF");
    const char* image_path =
            argc > 2 ? argv[2] : std::getenv("AICORE_TEST_GKD_IMAGE");
    const char* device = argc > 3 ? argv[3] : std::getenv("AICORE_TEST_DEVICE");
    const int iters = argc > 4 ? std::atoi(argv[4]) : 2;
    const int warmups = argc > 5 ? std::atoi(argv[5]) : 1;
    if (!gguf || gguf[0] == '\0' || !image_path || image_path[0] == '\0') {
        return 77;
    }
    if (!device || device[0] == '\0') device = "cpu";
    if (iters < 1) {
        std::fprintf(stderr, "invalid inference_runs %d\n", iters);
        return 1;
    }

    int32_t img_w = 0;
    int32_t img_h = 0;
    const std::vector<uint8_t> rgb = decodeImage(image_path, &img_w, &img_h);
    if (rgb.empty()) {
        std::fprintf(stderr, "failed to load image: %s\n", image_path);
        return 1;
    }

    // ---- load ---------------------------------------------------------------
    aicore_gkd_options* opts = aicore_gkd_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_gkd_options_set_device(opts, device);
    aicore_gkd_options_set_threads(opts, 0);
    aicore_gkd_ctx* ctx = aicore_gkd_load_opts(gguf, opts);
    aicore_gkd_options_free(opts);
    AICORE_CHECK(ctx != nullptr);
    if (!ctx) return 1;
    AICORE_CHECK(aicore_gkd_is_ready(ctx) == 1);
    AICORE_CHECK(aicore_gkd_last_error(ctx) == nullptr);
    AICORE_CHECK(aicore_gkd_context_image_size(ctx) == 384);
    AICORE_CHECK(std::strlen(aicore_gkd_context_device(ctx)) > 0);

    // ---- request validation (contract errors, no crash) --------------------
    aicore_gkd_detect_request req{};
    req.struct_size = sizeof(req);
    const char* texts[] = {"nose", "left eye"};
    {
        // no prompts -> contract rejection
        aicore_gkd_detect_request empty{};
        empty.struct_size = sizeof(empty);
        AICORE_CHECK(aicore_gkd_detect_rgb(ctx, rgb.data(), img_w, img_h,
                                           &empty) == -1);
        AICORE_CHECK(aicore_gkd_last_error(ctx) != nullptr);
        // struct-size mismatch -> contract rejection
        req.struct_size = 3;
        AICORE_CHECK(aicore_gkd_detect_rgb(ctx, rgb.data(), img_w, img_h,
                                           &req) == -1);
        req.struct_size = sizeof(req);
        // multimodal count mismatch -> contract rejection
        req.kps_texts = texts;
        req.n_kps_texts = 2;
        req.n_support_kps = 3;
        AICORE_CHECK(aicore_gkd_detect_rgb(ctx, rgb.data(), img_w, img_h,
                                           &req) == -1);
        req.n_support_kps = 0;
    }

    // ---- text-mode inference (warmups + repeated forwards) ------------------
    uint64_t first_hash = 0;
    for (int iter = -warmups; iter < iters; ++iter) {
        const int rc =
                aicore_gkd_detect_rgb(ctx, rgb.data(), img_w, img_h, &req);
        if (rc != 0) {
            std::fprintf(stderr, "detect failed rc=%d: %s\n", rc,
                         aicore_gkd_last_error(ctx) ? aicore_gkd_last_error(ctx)
                                                    : "?");
            failures++;
            break;
        }
        const int n = aicore_gkd_result_keypoint_count(ctx);
        AICORE_CHECK(n == 2);
        if (n != 2) break;
        for (int k = 0; k < n; ++k) {
            const aicore_gkd_keypoint kp =
                    aicore_gkd_result_keypoint_at(ctx, k);
            AICORE_CHECK(std::isfinite(kp.x) && std::isfinite(kp.y));
            AICORE_CHECK(std::isfinite(kp.score));
            AICORE_CHECK(kp.x_norm >= -1.05f && kp.x_norm <= 1.05f);
            AICORE_CHECK(kp.y_norm >= -1.05f && kp.y_norm <= 1.05f);
            // Recovered pixels must fall inside the image (keypoints are
            // clamped into the frame by the official recovery transform).
            AICORE_CHECK(kp.x >= 0.0f && kp.x <= (float)img_w);
            AICORE_CHECK(kp.y >= 0.0f && kp.y <= (float)img_h);
            // Semantic labels reach the consumer through the typed accessor.
            AICORE_CHECK(aicore_gkd_result_prompt_at(ctx, k) != nullptr);
        }
        float roi[4] = {0, 0, 0, 0};
        AICORE_CHECK(aicore_gkd_result_roi(ctx, roi) == 0);
        AICORE_CHECK(roi[2] >= roi[0] && roi[3] >= roi[1]);
        const uint64_t hash = keypointHash(ctx);
        if (iter == 0) {
            first_hash = hash;
        } else if (iter > 0) {
            // Exact repeated-output stability across forwards.
            AICORE_CHECK(hash == first_hash);
        }
        aicore_gkd_timings t{};
        AICORE_CHECK(aicore_gkd_last_timings(ctx, &t) == 0);
        AICORE_CHECK(t.e2e_ms > 0.0);
        aicore_pipeline_timings pt{};
        AICORE_CHECK(aicore_gkd_last_pipeline_timings(ctx, &pt) == 0);
        AICORE_CHECK((pt.valid_fields & AICORE_TIMING_E2E) != 0);
        AICORE_CHECK((pt.valid_fields & AICORE_TIMING_INFERENCE) != 0);
        AICORE_CHECK(pt.e2e_ms > 0.0);
    }

    // ---- multimodal (visual + text) path ------------------------------------
    {
        aicore_image_view view{};
        view.data = rgb.data();
        view.width = img_w;
        view.height = img_h;
        view.row_stride_bytes = (size_t)img_w * 3;
        view.format = AICORE_IMAGE_RGB8;
        // 1-shot visual prompt: two "keypoints" at 1/3 and 2/3 of the frame.
        const float support_kps[4] = {img_w / 3.0f, img_h / 3.0f,
                                      2.0f * img_w / 3.0f, 2.0f * img_h / 3.0f};
        req.support_image = &view;
        req.support_kps_xy = support_kps;
        req.support_kps_vis = nullptr;
        req.n_support_kps = 2;
        const int rc = aicore_gkd_detect_image(ctx, &view, &req);
        AICORE_CHECK(rc == 0);
        if (rc == 0) {
            AICORE_CHECK(aicore_gkd_result_keypoint_count(ctx) == 2);
            for (int k = 0; k < 2; ++k) {
                const aicore_gkd_keypoint kp =
                        aicore_gkd_result_keypoint_at(ctx, k);
                AICORE_CHECK(std::isfinite(kp.x) && std::isfinite(kp.y) &&
                             std::isfinite(kp.score));
                AICORE_CHECK(kp.x >= 0.0f && kp.x <= (float)img_w);
                AICORE_CHECK(kp.y >= 0.0f && kp.y <= (float)img_h);
            }
        }
        req.support_image = nullptr;
        req.support_kps_xy = nullptr;
        req.n_support_kps = 0;
    }

    char* json = aicore_gkd_info_json(ctx);
    AICORE_CHECK(json != nullptr && std::strstr(json, "keypoints") != nullptr);
    aicore_gkd_free_buffer(json);

    // ---- multi-ROI batch (official N_bbox top-down pipeline) ---------------
    {
        // Reference: single-ROI whole-image forward.
        AICORE_CHECK(aicore_gkd_detect_rgb(ctx, rgb.data(), img_w, img_h,
                                           &req) == 0);
        const int n_ref = aicore_gkd_result_keypoint_count(ctx);
        AICORE_CHECK(n_ref == req.n_kps_texts);
        std::vector<float> ref_norm((size_t)n_ref * 2);
        for (int j = 0; j < n_ref; ++j) {
            const aicore_gkd_keypoint kp =
                    aicore_gkd_result_keypoint_at(ctx, j);
            ref_norm[(size_t)j * 2] = kp.x_norm;
            ref_norm[(size_t)j * 2 + 1] = kp.y_norm;
        }
        // ROI 0 duplicates the whole-image box: the batched forward must
        // reproduce the single-ROI norm coordinates bit-for-bit (the B
        // dimension only batches the vision tower, it must not change the
        // math), and ROI 1 exercises a clamped interior box.
        const float boxes[8] = {
                0.0f,  0.0f,  (float)(img_w - 1),  (float)(img_h - 1),
                10.0f, 10.0f, (float)img_w / 2.0f, (float)img_h / 2.0f};
        AICORE_CHECK(aicore_gkd_detect_rgb_multi(ctx, rgb.data(), img_w, img_h,
                                                 &req, boxes, 2) == 0);
        AICORE_CHECK(aicore_gkd_result_roi_count(ctx) == 2);
        AICORE_CHECK(aicore_gkd_result_keypoint_count_at(ctx, 0) == n_ref);
        AICORE_CHECK(aicore_gkd_result_keypoint_count_at(ctx, 1) == n_ref);
        for (int j = 0; j < n_ref; ++j) {
            const aicore_gkd_keypoint batch_kp =
                    aicore_gkd_result_keypoint_at_roi(ctx, 0, j);
            AICORE_CHECK(batch_kp.x_norm == ref_norm[(size_t)j * 2] &&
                         batch_kp.y_norm == ref_norm[(size_t)j * 2 + 1]);
            const aicore_gkd_keypoint roi1_kp =
                    aicore_gkd_result_keypoint_at_roi(ctx, 1, j);
            AICORE_CHECK(std::isfinite(roi1_kp.x_norm) &&
                         std::isfinite(roi1_kp.y_norm) &&
                         std::isfinite(roi1_kp.score));
            AICORE_CHECK(roi1_kp.x >= 0.0f && roi1_kp.x <= (float)img_w);
            AICORE_CHECK(roi1_kp.y >= 0.0f && roi1_kp.y <= (float)img_h);
        }
        // Out-of-range ROI is zeroed.
        AICORE_CHECK(aicore_gkd_result_keypoint_count_at(ctx, 2) == 0);
        // NULL/empty bbox list -> contract rejection.
        AICORE_CHECK(aicore_gkd_detect_rgb_multi(ctx, rgb.data(), img_w, img_h,
                                                 &req, nullptr, 0) == -1);
    }

    const aicore_pipeline_timings* timing = nullptr;
    aicore_pipeline_timings pt{};
    if (aicore_gkd_last_pipeline_timings(ctx, &pt) == 0) timing = &pt;
    aicore::test::printValidationResult("gkd", device, first_hash, timing);

    // ---- context destroy / recreate (cache-safety ladder) -------------------
    aicore_gkd_free(ctx);
    opts = aicore_gkd_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_gkd_options_set_device(opts, device);
    aicore_gkd_ctx* ctx2 = aicore_gkd_load_opts(gguf, opts);
    aicore_gkd_options_free(opts);
    AICORE_CHECK(ctx2 != nullptr && aicore_gkd_is_ready(ctx2) == 1);
    if (ctx2 && aicore_gkd_is_ready(ctx2)) {
        req.kps_texts = texts;
        req.n_kps_texts = 2;
        AICORE_CHECK(aicore_gkd_detect_rgb(ctx2, rgb.data(), img_w, img_h,
                                           &req) == 0);
        // Same-model second context must reproduce the same output.
        AICORE_CHECK(keypointHash(ctx2) == first_hash);
    }
    aicore_gkd_free(ctx2);

    aicore_gkd_shutdown();

    std::fprintf(stderr, "gkd load ok: %s device=%s %dx%d\n", gguf, device,
                 img_w, img_h);
    return failures == 0 ? 0 : 1;
}

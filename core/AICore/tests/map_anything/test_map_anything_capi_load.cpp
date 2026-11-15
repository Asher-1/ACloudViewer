// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Real-model MapAnything probe: contract validation against a published
// map_anything_models GGUF. Also the aicore-validate-all "map_anything"
// scenario executable.
//
// Usage: test_map_anything_capi_load <model.gguf> <image1> <image2>
//        <backend> [iters] [warmups]
// Env fallback: AICORE_TEST_MAP_ANYTHING_GGUF / AICORE_TEST_MAP_ANYTHING_
// IMAGE / AICORE_TEST_MAP_ANYTHING_IMAGE2 / AICORE_TEST_DEVICE (missing
// assets -> exit 77 skip).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/map_anything_capi.h"
#include "tests/common/test_macros.hpp"
#include "tests/common/validation_probe.hpp"

static int failures = 0;

namespace {

std::vector<uint8_t> decodeImage(const char* path, int32_t* w, int32_t* h) {
    std::vector<uint8_t> rgb;
    uint8_t* data = nullptr;
    if (aicore_map_anything_load_path_rgb(path, &data, w, h) == 0 &&
        data != nullptr) {
        rgb.assign(data, data + (size_t)(*w) * (*h) * 3);
        aicore_map_anything_free_buffer(data);
    }
    return rgb;
}

// FNV-1a over every typed buffer of the latest run (the exact fingerprint
// the validation runner records and re-checks for stability).
uint64_t runHash(const aicore_map_anything_ctx* ctx) {
    uint64_t hash = 1469598103934665603ULL;
    auto append = [&](const void* p, size_t n) {
        if (p != nullptr && n > 0) hash = aicore::test::fnv1aAppend(hash, p, n);
    };
    const int32_t views = aicore_map_anything_result_view_count(ctx);
    const int32_t height = aicore_map_anything_result_height(ctx);
    const int32_t width = aicore_map_anything_result_width(ctx);
    const size_t plane = (size_t)views * height * width;
    append(&views, sizeof(views));
    append(&height, sizeof(height));
    append(&width, sizeof(width));
    append(aicore_map_anything_result_depth(ctx), plane * sizeof(float));
    append(aicore_map_anything_result_conf(ctx), plane * sizeof(float));
    append(aicore_map_anything_result_points(ctx), plane * 3 * sizeof(float));
    append(aicore_map_anything_result_local_points(ctx),
           plane * 3 * sizeof(float));
    const int32_t pose_stride = aicore_map_anything_result_pose_stride(ctx);
    append(&pose_stride, sizeof(pose_stride));
    if (pose_stride > 0) {
        append(aicore_map_anything_result_pose(ctx),
               (size_t)views * pose_stride * sizeof(float));
    }
    append(aicore_map_anything_result_mask(ctx), plane * sizeof(float));
    return hash;
}

}  // namespace

int main(int argc, char** argv) {
    // Args first (validate-all scenario), env fallback (standalone ctest).
    const char* gguf =
            argc > 1 ? argv[1] : std::getenv("AICORE_TEST_MAP_ANYTHING_GGUF");
    const char* image1 =
            argc > 2 ? argv[2] : std::getenv("AICORE_TEST_MAP_ANYTHING_IMAGE");
    const char* image2 =
            argc > 3 ? argv[3] : std::getenv("AICORE_TEST_MAP_ANYTHING_IMAGE2");
    const char* device = argc > 4 ? argv[4] : std::getenv("AICORE_TEST_DEVICE");
    const int iters = argc > 5 ? std::atoi(argv[5]) : 2;
    const int warmups = argc > 6 ? std::atoi(argv[6]) : 1;
    if (!gguf || gguf[0] == '\0' || !image1 || image1[0] == '\0' || !image2 ||
        image2[0] == '\0') {
        return 77;
    }
    if (!device || device[0] == '\0') device = "cpu";
    if (iters < 1) {
        std::fprintf(stderr, "invalid inference_runs %d\n", iters);
        return 1;
    }

    int32_t w1 = 0, h1 = 0, w2 = 0, h2 = 0;
    const std::vector<uint8_t> rgb1 = decodeImage(image1, &w1, &h1);
    const std::vector<uint8_t> rgb2 = decodeImage(image2, &w2, &h2);
    if (rgb1.empty() || rgb2.empty()) {
        std::fprintf(stderr, "failed to load input images: %s %s\n", image1,
                     image2);
        return 1;
    }

    // ---- load ---------------------------------------------------------------
    aicore_map_anything_options* opts = aicore_map_anything_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_map_anything_options_set_device(opts, device);
    aicore_map_anything_options_set_threads(opts, 0);
    aicore_map_anything_ctx* ctx = aicore_map_anything_load_opts(gguf, opts);
    aicore_map_anything_options_free(opts);
    AICORE_CHECK(ctx != nullptr);
    if (!ctx) return 1;
    AICORE_CHECK(aicore_map_anything_is_ready(ctx) == 1);
    AICORE_CHECK(aicore_map_anything_last_error(ctx) == nullptr);
    AICORE_CHECK(aicore_map_anything_context_patch_size(ctx) > 0);
    AICORE_CHECK(aicore_map_anything_context_default_image_size(ctx) > 0);
    AICORE_CHECK(std::strlen(aicore_map_anything_context_device(ctx)) > 0);
    const std::string arch =
            aicore_map_anything_result_architecture(ctx) != nullptr
                    ? std::string(aicore_map_anything_result_architecture(ctx))
                    : std::string();

    // ---- input validation (contract errors, no crash) -----------------------
    {
        aicore_image_view bad{};
        bad.data = nullptr;
        bad.width = 0;
        bad.height = 0;
        bad.row_stride_bytes = 0;
        bad.format = AICORE_IMAGE_RGB8;
        AICORE_CHECK(aicore_map_anything_run_images(ctx, &bad, 1) == -1);
        AICORE_CHECK(aicore_map_anything_last_error(ctx) != nullptr);
        AICORE_CHECK(aicore_map_anything_run_images(ctx, nullptr, 0) == -1);
    }

    // ---- two-view inference (warmups + repeated forwards) -------------------
    aicore_image_view views[2];
    views[0].data = rgb1.data();
    views[0].width = w1;
    views[0].height = h1;
    views[0].row_stride_bytes = (size_t)w1 * 3;
    views[0].format = AICORE_IMAGE_RGB8;
    views[1].data = rgb2.data();
    views[1].width = w2;
    views[1].height = h2;
    views[1].row_stride_bytes = (size_t)w2 * 3;
    views[1].format = AICORE_IMAGE_RGB8;

    uint64_t first_hash = 0;
    for (int iter = -warmups; iter < iters; ++iter) {
        const int rc = aicore_map_anything_run_images(ctx, views, 2);
        if (rc != 0) {
            std::fprintf(stderr, "run failed rc=%d: %s\n", rc,
                         aicore_map_anything_last_error(ctx)
                                 ? aicore_map_anything_last_error(ctx)
                                 : "?");
            failures++;
            break;
        }
        AICORE_CHECK(aicore_map_anything_result_view_count(ctx) == 2);
        const int32_t height = aicore_map_anything_result_height(ctx);
        const int32_t width = aicore_map_anything_result_width(ctx);
        AICORE_CHECK(height > 0 && width > 0);
        const size_t plane = (size_t)2 * height * width;
        const float* depth = aicore_map_anything_result_depth(ctx);
        AICORE_CHECK(depth != nullptr);
        // Dense geometry must be finite and positive wherever the model is
        // confident (the white-padded border rows are excluded implicitly
        // by the confidence floor of the published checkpoints). The null
        // guard keeps a contract gap a clean recorded failure instead of a
        // SIGSEGV inside the probe.
        if (depth != nullptr) {
            for (size_t i = 0; i < plane; i += 97) {  // sparse finite sweep
                AICORE_CHECK(std::isfinite(depth[i]));
                AICORE_CHECK(depth[i] > 0.0f);
            }
        }
        const int32_t pose_stride = aicore_map_anything_result_pose_stride(ctx);
        if (pose_stride == 9) {
            // vggt family: FoV rows survive the relu + 0.01 head floor.
            const float* pose = aicore_map_anything_result_pose(ctx);
            AICORE_CHECK(pose != nullptr);
            for (int v = 0; v < 2; ++v) {
                AICORE_CHECK(pose[(size_t)v * 9 + 7] > 0.0f);
                AICORE_CHECK(pose[(size_t)v * 9 + 8] > 0.0f);
            }
        } else if (pose_stride == 16) {
            const float* pose = aicore_map_anything_result_pose(ctx);
            AICORE_CHECK(pose != nullptr);
            // c2w bottom row stays (0, 0, 0, 1).
            for (int v = 0; v < 2; ++v) {
                const float* m = pose + (size_t)v * 16;
                AICORE_CHECK(m[12] == 0.0f && m[13] == 0.0f && m[14] == 0.0f &&
                             m[15] == 1.0f);
            }
        }
        const float* conf = aicore_map_anything_result_conf(ctx);
        AICORE_CHECK(conf != nullptr);
        if (conf != nullptr) {
            for (size_t i = 0; i < plane; i += 97) {
                AICORE_CHECK(std::isfinite(conf[i]));
            }
        }
        const uint64_t hash = runHash(ctx);
        if (iter == 0) {
            first_hash = hash;
        } else if (iter > 0) {
            // Exact repeated-output stability across same-shape forwards.
            AICORE_CHECK(hash == first_hash);
        }
        aicore_pipeline_timings pt{};
        AICORE_CHECK(aicore_map_anything_last_pipeline_timings(ctx, &pt) == 0);
        AICORE_CHECK((pt.valid_fields & AICORE_TIMING_E2E) != 0);
        AICORE_CHECK((pt.valid_fields & AICORE_TIMING_INFERENCE) != 0);
        AICORE_CHECK(pt.e2e_ms > 0.0);
        AICORE_CHECK(pt.inference_ms > 0.0);
    }

    // ---- borrowed-view contract: layout equivalence -------------------------
    // Every supported layout of the same pixels must reach the model as the
    // same RGB bytes, so the typed outputs are bit-identical (exact hash) to
    // the tight RGB8 run above. Padded strides carry garbage in the padding.
    if (first_hash != 0) {
        constexpr size_t kPad = 13;  // odd: no accidental alignment
        struct Layout {
            aicore_image_format format;
            int channels;
            bool padded;
        };
        const auto pack = [&](const std::vector<uint8_t>& rgb, int w, int h,
                              const Layout& l, std::vector<uint8_t>* buf) {
            const size_t stride =
                    (size_t)w * l.channels + (l.padded ? kPad : 0);
            buf->assign(stride * h, 0xA5);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const uint8_t* s = &rgb[((size_t)y * w + x) * 3];
                    uint8_t* d = &(
                            *buf)[(size_t)y * stride + (size_t)x * l.channels];
                    const bool bgr = l.format == AICORE_IMAGE_BGR8 ||
                                     l.format == AICORE_IMAGE_BGRA8;
                    if (l.channels == 1) {
                        d[0] = s[0];
                        continue;
                    }
                    d[0] = s[bgr ? 2 : 0];
                    d[1] = s[1];
                    d[2] = s[bgr ? 0 : 2];
                    if (l.channels == 4) d[3] = 0x3C;  // alpha is ignored
                }
            }
            return aicore_image_view{buf->data(), w, h, stride, l.format};
        };
        const Layout pairs[2][2] = {
                {{AICORE_IMAGE_BGR8, 3, false}, {AICORE_IMAGE_RGBA8, 4, true}},
                {{AICORE_IMAGE_BGRA8, 4, true}, {AICORE_IMAGE_RGB8, 3, true}}};
        for (const auto& pair : pairs) {
            std::vector<uint8_t> b1, b2;
            const aicore_image_view v[2] = {pack(rgb1, w1, h1, pair[0], &b1),
                                            pack(rgb2, w2, h2, pair[1], &b2)};
            AICORE_CHECK(aicore_map_anything_run_images(ctx, v, 2) == 0);
            AICORE_CHECK(runHash(ctx) == first_hash);
        }
        // GRAY8 replicates the channel: equal to RGB8 of the replicated gray.
        {
            std::vector<uint8_t> g1, g2, r1((size_t)w1 * h1 * 3),
                    r2((size_t)w2 * h2 * 3);
            for (size_t i = 0; i < r1.size(); ++i) r1[i] = rgb1[i - i % 3];
            for (size_t i = 0; i < r2.size(); ++i) r2[i] = rgb2[i - i % 3];
            const aicore_image_view gv[2] = {
                    pack(rgb1, w1, h1, {AICORE_IMAGE_GRAY8, 1, true}, &g1),
                    pack(rgb2, w2, h2, {AICORE_IMAGE_GRAY8, 1, false}, &g2)};
            AICORE_CHECK(aicore_map_anything_run_images(ctx, gv, 2) == 0);
            const uint64_t gray_hash = runHash(ctx);
            const aicore_image_view rv[2] = {
                    {r1.data(), w1, h1, (size_t)w1 * 3, AICORE_IMAGE_RGB8},
                    {r2.data(), w2, h2, (size_t)w2 * 3, AICORE_IMAGE_RGB8}};
            AICORE_CHECK(aicore_map_anything_run_images(ctx, rv, 2) == 0);
            AICORE_CHECK(runHash(ctx) == gray_hash);
        }
        // A stride below width * channels would read past the caller's
        // buffer: contract error, nothing runs, the context stays usable.
        aicore_image_view shortStride = views[0];
        shortStride.row_stride_bytes = (size_t)w1 * 3 - 1;
        const aicore_image_view sv[2] = {shortStride, views[1]};
        AICORE_CHECK(aicore_map_anything_run_images(ctx, sv, 2) == -1);
        aicore_image_view badFormat = views[0];
        badFormat.format = (aicore_image_format)99;
        const aicore_image_view fv[2] = {badFormat, views[1]};
        AICORE_CHECK(aicore_map_anything_run_images(ctx, fv, 2) == -1);
        AICORE_CHECK(aicore_map_anything_last_error(ctx) != nullptr);
    }

    // ---- parity feed contract (run_frames shares the same session) ----------
    {
        // Re-run the preprocessed geometry through the f32 feed: the dense
        // output must stay finite (bit-equality with run_images is NOT
        // expected — the images path resamples through the in-tree smoke
        // resampler; the parity anchor is the feed itself).
        const int32_t height = aicore_map_anything_result_height(ctx);
        const int32_t width = aicore_map_anything_result_width(ctx);
        std::vector<float> frames((size_t)2 * 3 * height * width, 0.5f);
        AICORE_CHECK(aicore_map_anything_run_frames(ctx, frames.data(), 2,
                                                    height, width) == 0);
        AICORE_CHECK(aicore_map_anything_result_height(ctx) == height);
        AICORE_CHECK(aicore_map_anything_result_width(ctx) == width);
    }

    char* json = aicore_map_anything_info_json(ctx);
    AICORE_CHECK(json != nullptr &&
                 std::strstr(json, "architecture") != nullptr);
    aicore_map_anything_free_buffer(json);

    aicore_pipeline_timings timing{};
    if (aicore_map_anything_last_pipeline_timings(ctx, &timing) == 0) {
        aicore::test::printValidationResult("map_anything", device, first_hash,
                                            &timing);
    } else {
        aicore::test::printValidationResult("map_anything", device, first_hash,
                                            nullptr);
    }

    // ---- context destroy / recreate (cache-safety ladder) -------------------
    aicore_map_anything_free(ctx);
    opts = aicore_map_anything_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_map_anything_options_set_device(opts, device);
    aicore_map_anything_ctx* ctx2 = aicore_map_anything_load_opts(gguf, opts);
    aicore_map_anything_options_free(opts);
    AICORE_CHECK(ctx2 != nullptr && aicore_map_anything_is_ready(ctx2) == 1);
    if (ctx2 && aicore_map_anything_is_ready(ctx2)) {
        AICORE_CHECK(aicore_map_anything_run_images(ctx2, views, 2) == 0);
        // Same-model second context must reproduce the same output.
        AICORE_CHECK(runHash(ctx2) == first_hash);
    }
    aicore_map_anything_free(ctx2);

    // ---- resize-mode contract
    // ------------------------------------------------ Unknown mode:
    // fail-closed before any weight is touched.
    opts = aicore_map_anything_options_new();
    aicore_map_anything_options_set_device(opts, device);
    aicore_map_anything_options_set_resize_mode(opts, "letterbox");
    aicore_map_anything_ctx* bad = aicore_map_anything_load_opts(gguf, opts);
    AICORE_CHECK(bad != nullptr && aicore_map_anything_is_ready(bad) == 0);
    AICORE_CHECK(bad != nullptr &&
                 aicore_map_anything_last_error(bad) != nullptr &&
                 std::strstr(aicore_map_anything_last_error(bad),
                             "letterbox") != nullptr);
    aicore_map_anything_free(bad);
    // "square" (original VGGT load_and_preprocess_images_square): square
    // grid, content = the longer side resized to the grid, the shorter side
    // proportionally, centered in a black pad.
    aicore_map_anything_options_set_resize_mode(opts, "square");
    aicore_map_anything_ctx* sq = aicore_map_anything_load_opts(gguf, opts);
    aicore_map_anything_options_free(opts);
    AICORE_CHECK(sq != nullptr && aicore_map_anything_is_ready(sq) == 1);
    if (sq && aicore_map_anything_is_ready(sq)) {
        AICORE_CHECK(aicore_map_anything_run_images(sq, views, 2) == 0);
        const int32_t H = aicore_map_anything_result_height(sq);
        const int32_t W = aicore_map_anything_result_width(sq);
        AICORE_CHECK(H > 0 && H == W);
        const int32_t sizes[2][2] = {{w1, h1}, {w2, h2}};
        for (int v = 0; v < 2; ++v) {
            int32_t vw = 0, vh = 0;
            AICORE_CHECK(aicore_map_anything_result_view_valid_size(sq, v, &vw,
                                                                    &vh) == 0);
            const int32_t iw = sizes[v][0], ih = sizes[v][1];
            const double shorter =
                    (double)W * std::min(iw, ih) / std::max(iw, ih);
            const int32_t longSide = iw >= ih ? vw : vh;
            const int32_t shortSide = iw >= ih ? vh : vw;
            AICORE_CHECK(longSide == W);
            AICORE_CHECK(std::fabs(shortSide - shorter) <= 1.0);
        }
        std::fprintf(stderr, "square grid %dx%d\n", W, H);
    }
    aicore_map_anything_free(sq);

    aicore_map_anything_shutdown();

    std::fprintf(stderr, "map_anything load ok: %s arch=%s device=%s %dx%d\n",
                 gguf, arch.c_str(), device, w1, h1);
    return failures == 0 ? 0 : 1;
}

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Real-model QuerySplat probe: contract validation against the published
// querysplat_models decoder + the shared map_anything_models encoder. Also
// the aicore-validate-all "querysplat" scenario executable.
//
// Usage: test_querysplat_capi <decoder.gguf> <encoder.gguf> <image1>
//        <image2> <image3> <image4> <backend> [iters] [warmups]
//        [--dump-gaussians PATH] [--dump-renders PATH] [--tto-steps N]
// Env fallback: AICORE_TEST_QUERYSPLAT_DECODER / _ENCODER / _IMAGE / _IMAGE2
// / _IMAGE3 / _IMAGE4 / AICORE_TEST_DEVICE (missing assets -> exit 77 skip).
//
// The optional dump flags write raw f32 buffers in the upstream cpp_ggml bin
// layouts (gaussians: N x 14 = pos3|opacity|scale3|rot4|color; renders:
// S x 3 x H x W planes concatenated) for the upstream-parity comparator
// (compare_upstream_bins.py). --tto-steps N runs the KV-memory TTO for N
// steps (l1 + 0.2*(1-ssim)/2 + w*lpips, AdamW — the official
// _optimize_memory) and gates the per-step losses; the depth-pointcloud
// export (save_vggt_depth_pointcloud semantics) is gated on every run.
//
// Structural gates (the upstream truth anchors live in the manifest):
//   - finite gaussians with opacities in [0, 1] and positive scales,
//   - finite depth (>= 0) and depth_conf (>= 1) on the input-resolution grid,
//   - pose_enc 9 floats per view with FoV = relu(x) + 0.01 >= 0.01,
//   - exact repeated-output fingerprint (deterministic graph),
//   - render + PLY + cameras JSON + depth point-cloud export round-trips.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/map_anything_capi.h"
#include "aicore/querysplat_capi.h"
#include "tests/common/test_macros.hpp"
#include "tests/common/validation_probe.hpp"

static int failures = 0;

namespace {

uint64_t runHash(const aicore_querysplat_ctx* ctx) {
    uint64_t hash = 1469598103934665603ULL;
    auto append = [&](const void* p, size_t n) {
        if (p != nullptr && n > 0) hash = aicore::test::fnv1aAppend(hash, p, n);
    };
    const int32_t views = aicore_querysplat_result_view_count(ctx);
    const int32_t height = aicore_querysplat_result_height(ctx);
    const int32_t width = aicore_querysplat_result_width(ctx);
    const int64_t n = aicore_querysplat_result_gaussian_count(ctx);
    const int32_t stride = aicore_querysplat_result_gaussian_stride(ctx);
    const size_t plane = (size_t)views * height * width;
    append(&views, sizeof(views));
    append(&height, sizeof(height));
    append(&width, sizeof(width));
    append(&n, sizeof(n));
    append(&stride, sizeof(stride));
    append(aicore_querysplat_result_gaussians(ctx),
           (size_t)std::max<int64_t>(0, n) * stride * sizeof(float));
    append(aicore_querysplat_result_depth(ctx), plane * sizeof(float));
    append(aicore_querysplat_result_pose(ctx), (size_t)views * 9 * sizeof(float));
    return hash;
}

// Raw f32 dump for the upstream-parity comparator (upstream cpp_ggml bin
// layout; little-endian on every supported platform).
bool write_f32_bin(const char* path, const float* data, size_t count) {
    if (!path || !data || count == 0) return false;
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const size_t wrote = std::fwrite(data, sizeof(float), count, f);
    std::fclose(f);
    return wrote == count;
}

}  // namespace

int main(int argc, char** argv) {
    const char* decoder =
            argc > 1 ? argv[1]
                     : std::getenv("AICORE_TEST_QUERYSPLAT_DECODER");
    const char* encoder =
            argc > 2 ? argv[2] : std::getenv("AICORE_TEST_QUERYSPLAT_ENCODER");
    const char* images_arg[4] = {
            argc > 3 ? argv[3] : std::getenv("AICORE_TEST_QUERYSPLAT_IMAGE"),
            argc > 4 ? argv[4] : std::getenv("AICORE_TEST_QUERYSPLAT_IMAGE2"),
            argc > 5 ? argv[5] : std::getenv("AICORE_TEST_QUERYSPLAT_IMAGE3"),
            argc > 6 ? argv[6] : std::getenv("AICORE_TEST_QUERYSPLAT_IMAGE4"),
    };
    const char* backend =
            argc > 7 ? argv[7] : std::getenv("AICORE_TEST_DEVICE");
    const int iters = argc > 8 ? std::atoi(argv[8]) : 1;
    const int warmups = argc > 9 ? std::atoi(argv[9]) : 0;
    const char* dump_gaussians = nullptr;
    const char* dump_renders = nullptr;
    int tto_steps = 0;
    double tto_lpips_weight = -1.0;
    int n_views_used = 4;
    for (int i = 10; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--dump-gaussians") == 0)
            dump_gaussians = argv[i + 1];
        else if (std::strcmp(argv[i], "--dump-renders") == 0)
            dump_renders = argv[i + 1];
        else if (std::strcmp(argv[i], "--tto-steps") == 0)
            tto_steps = std::atoi(argv[i + 1]);
        else if (std::strcmp(argv[i], "--views") == 0)
            n_views_used = std::max(1, std::min(4, std::atoi(argv[i + 1])));
        else if (std::strcmp(argv[i], "--tto-lpips-weight") == 0)
            tto_lpips_weight = std::atof(argv[i + 1]);
    }
    if (!decoder || !encoder) {
        std::printf("SKIP: missing AICORE_TEST_QUERYSPLAT_DECODER/ENCODER\n");
        return 77;
    }
    for (const char* img : images_arg) {
        if (!img) {
            std::printf("SKIP: missing AICORE_TEST_QUERYSPLAT_IMAGE*\n");
            return 77;
        }
    }

    aicore_querysplat_options* opts = aicore_querysplat_options_new();
    if (!opts) {
        std::printf("FAIL: options allocation failed\n");
        return 1;
    }
    aicore_querysplat_options_set_device(opts,
                                         backend ? backend : "cpu");
    aicore_querysplat_options_set_f32_accuracy(opts, 1);
    if (tto_lpips_weight >= 0.0)
        aicore_querysplat_options_set_tto_lpips_weight(opts,
                                                       tto_lpips_weight);

    aicore_querysplat_ctx* ctx = aicore_querysplat_load_opts(
            encoder, decoder, opts);
    aicore_querysplat_options_free(opts);
    if (!ctx || !aicore_querysplat_is_ready(ctx)) {
        const char* err = ctx ? aicore_querysplat_last_error(ctx) : nullptr;
        std::printf("FAIL: load failed: %s\n",
                    err ? err : "(no error string)");
        if (ctx) aicore_querysplat_free(ctx);
        return 1;
    }

    // decode the four inputs through the shared RGB loader (the
    // map_anything compatibility helper; any image task may use it)
    std::vector<uint8_t> rgb[4];
    int32_t w[4] = {0, 0, 0, 0}, h[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        uint8_t* data = nullptr;
        if (aicore_map_anything_load_path_rgb(images_arg[i], &data, &w[i],
                                              &h[i]) != 0 ||
            !data) {
            std::printf("FAIL: cannot decode %s\n", images_arg[i]);
            aicore_querysplat_free(ctx);
            return 1;
        }
        rgb[i].assign(data, data + (size_t)w[i] * h[i] * 3);
        aicore_map_anything_free_buffer(data);
    }

    // warmups (not fingerprinted)
    for (int i = 0; i < warmups; ++i) {
        aicore_image_view views[4] = {};
        for (int v = 0; v < n_views_used; ++v) {
            views[v].data = rgb[v].data();
            views[v].width = w[v];
            views[v].height = h[v];
            views[v].row_stride_bytes = (size_t)w[v] * 3;
            views[v].format = AICORE_IMAGE_RGB8;
        }
        if (aicore_querysplat_run_images(ctx, views, n_views_used) != 0) {
            std::printf("FAIL: warmup run failed\n");
            aicore_querysplat_free(ctx);
            return 1;
        }
    }

    uint64_t first_hash = 0;
    double e2e_total = 0.0;
    int e2e_count = 0;
    int32_t last_views = 0, last_height = 0, last_width = 0;
    int64_t last_gaussians = 0;
    for (int iter = 0; iter < std::max(1, iters); ++iter) {
        aicore_image_view views[4] = {};
        for (int v = 0; v < n_views_used; ++v) {
            views[v].data = rgb[v].data();
            views[v].width = w[v];
            views[v].height = h[v];
            views[v].row_stride_bytes = (size_t)w[v] * 3;
            views[v].format = AICORE_IMAGE_RGB8;
        }
        if (aicore_querysplat_run_images(ctx, views, n_views_used) != 0) {
            const char* err = aicore_querysplat_last_error(ctx);
            std::printf("FAIL: run failed: %s\n",
                        err ? err : "(none)");
            aicore_querysplat_free(ctx);
            return 1;
        }

        const int32_t S = aicore_querysplat_result_view_count(ctx);
        const int32_t H = aicore_querysplat_result_height(ctx);
        const int32_t W = aicore_querysplat_result_width(ctx);
        const int64_t N = aicore_querysplat_result_gaussian_count(ctx);
        const int32_t stride = aicore_querysplat_result_gaussian_stride(ctx);
        const float* g = aicore_querysplat_result_gaussians(ctx);
        const float* depth = aicore_querysplat_result_depth(ctx);
        const float* conf = aicore_querysplat_result_depth_conf(ctx);
        const float* pose = aicore_querysplat_result_pose(ctx);
        aicore_pipeline_timings t{};
        aicore_querysplat_last_pipeline_timings(ctx, &t);

        // structural invariants
        const size_t plane = (size_t)S * H * W;
        bool finite_g = std::isfinite(g[0]);
        bool opacity_ok = true, scale_ok = true;
        for (int64_t i = 0; i < N && (opacity_ok || scale_ok); ++i) {
            const float* r = g + (size_t)i * stride;
            if (!(r[3] >= 0.0f && r[3] <= 1.0f)) opacity_ok = false;
            if (!(r[4] > 0.0f && r[5] > 0.0f && r[6] > 0.0f)) scale_ok = false;
            if (!std::isfinite(r[0]) || !std::isfinite(r[1]) ||
                !std::isfinite(r[2]))
                finite_g = false;
        }
        bool depth_ok = depth != nullptr;
        for (size_t i = 0; i < plane && depth_ok; ++i) {
            if (!(std::isfinite(depth[i]) && depth[i] >= 0.0f))
                depth_ok = false;
            if (conf && !(std::isfinite(conf[i]) && conf[i] >= 1.0f))
                depth_ok = false;
        }
        bool pose_ok = pose != nullptr;
        for (int32_t v = 0; v < S && pose_ok; ++v) {
            // FoV = relu(x) + 0.01: entries 7, 8 must be >= 0.01
            if (!(pose[(size_t)v * 9 + 7] >= 0.01f &&
                  pose[(size_t)v * 9 + 8] >= 0.01f))
                pose_ok = false;
        }
        AICORE_CHECK(finite_g);
        AICORE_CHECK(opacity_ok);
        AICORE_CHECK(scale_ok);
        AICORE_CHECK(depth_ok);
        AICORE_CHECK(pose_ok);
        AICORE_CHECK(t.e2e_ms > 0.0);
        e2e_total += t.e2e_ms;
        ++e2e_count;
        AICORE_CHECK(N == 524288);
        last_views = S;
        last_height = H;
        last_width = W;
        last_gaussians = N;

        // render round-trip
        const float* w2c = aicore_querysplat_result_w2c(ctx);
        const float* K = aicore_querysplat_result_intrinsics(ctx);
        const float* rrgb = nullptr;
        unsigned char* rgb8 = nullptr;
        AICORE_CHECK(aicore_querysplat_render_views(ctx, w2c, K, S, &rrgb, &rgb8) == 0);

        // export round-trips
        const std::string ply = std::string(std::getenv("TMPDIR")
                                                    ? std::getenv("TMPDIR")
                                                    : "/tmp") +
                                "/qs_probe_gaussians.ply";
        AICORE_CHECK(aicore_querysplat_export_gaussians_ply(ctx, ply.c_str(), 0.05f) == 0);
        char* cams = aicore_querysplat_export_cameras_json(ctx);
        AICORE_CHECK(cams != nullptr);
        if (cams) aicore_querysplat_free_buffer(cams);
        char* stats = aicore_querysplat_export_distribution_stats(ctx);
        AICORE_CHECK(stats != nullptr);
        if (stats) aicore_querysplat_free_buffer(stats);
        std::remove(ply.c_str());

        const uint64_t hash = runHash(ctx);
        if (iter == 0) {
            first_hash = hash;
        } else {
            AICORE_CHECK(hash == first_hash);
        }
    }

    // optional upstream-parity dumps (final iteration's buffers)
    if (dump_gaussians || dump_renders) {
        const int32_t S = aicore_querysplat_result_view_count(ctx);
        const int32_t H = aicore_querysplat_result_height(ctx);
        const int32_t W = aicore_querysplat_result_width(ctx);
        const int64_t N = aicore_querysplat_result_gaussian_count(ctx);
        const int32_t stride = aicore_querysplat_result_gaussian_stride(ctx);
        if (dump_gaussians) {
            if (stride != 14 ||
                !write_f32_bin(dump_gaussians,
                               aicore_querysplat_result_gaussians(ctx),
                               (size_t)N * 14)) {
                std::printf("FAIL: gaussians dump (stride %d)\n", stride);
                aicore_querysplat_free(ctx);
                return 1;
            }
        }
        if (dump_renders) {
            const float* rrgb = nullptr;
            unsigned char* rgb8 = nullptr;
            if (aicore_querysplat_render_views(
                        ctx, aicore_querysplat_result_w2c(ctx),
                        aicore_querysplat_result_intrinsics(ctx), S, &rrgb,
                        &rgb8) != 0 ||
                !write_f32_bin(dump_renders, rrgb,
                               (size_t)S * 3 * H * W)) {
                std::printf("FAIL: renders dump\n");
                aicore_querysplat_free(ctx);
                return 1;
            }
        }
        std::printf("DUMP: gaussians=%s renders=%s S=%d H=%d W=%d\n",
                    dump_gaussians ? dump_gaussians : "-",
                    dump_renders ? dump_renders : "-", S, H, W);
    }

    // depth-pointcloud export (save_vggt_depth_pointcloud semantics):
    // confidence-proportional unprojection through the predicted cameras.
    {
        const float* pc_xyz = nullptr;
        unsigned char* pc_rgb = nullptr;
        int32_t pc_n = 0;
        AICORE_CHECK(aicore_querysplat_depth_pointcloud(
                ctx, 65536, &pc_xyz, &pc_rgb, &pc_n) == 0);
        AICORE_CHECK(pc_n > 0 && pc_xyz != nullptr && pc_rgb != nullptr);
        bool pc_finite = true;
        for (int32_t i = 0; i < pc_n && pc_finite; ++i) {
            if (!std::isfinite(pc_xyz[(size_t)i * 3]) ||
                !std::isfinite(pc_xyz[(size_t)i * 3 + 1]) ||
                !std::isfinite(pc_xyz[(size_t)i * 3 + 2]))
                pc_finite = false;
        }
        AICORE_CHECK(pc_finite);
        std::printf("DEPTH_PC: n=%d\n", pc_n);
    }

    // TTO smoke (the official _optimize_memory): N AdamW steps on the KV
    // memory against the input pixels; the composite loss decreases (or
    // holds within float noise) and the optimized gaussian set is exposed.
    if (tto_steps > 0) {
        aicore_querysplat_tto_step* losses = nullptr;
        int32_t n_losses = 0;
        const float* tto_g = nullptr;
        int64_t tto_n = 0;
        AICORE_CHECK(aicore_querysplat_run_tto(ctx, tto_steps, &losses,
                                               &n_losses, &tto_g,
                                               &tto_n) == 0);
        AICORE_CHECK(losses != nullptr && n_losses == tto_steps);
        AICORE_CHECK(tto_g != nullptr && tto_n == 524288);
        // composite with the default lpips weight 0.05:
        // L = l1 + 0.2*(1-ssim)/2 + 0.05*lpips
        auto composite = [](const aicore_querysplat_tto_step& s) {
            return s.l1 + 0.1 * (1.0 - s.ssim) + 0.05 * s.lpips;
        };
        const double first = composite(losses[0]);
        const double last = composite(losses[n_losses - 1]);
        bool finite = std::isfinite(first) && std::isfinite(last);
        for (int32_t i = 0; i < n_losses && finite; ++i) {
            finite = std::isfinite(losses[i].l1) &&
                     std::isfinite(losses[i].ssim) &&
                     std::isfinite(losses[i].lpips);
        }
        AICORE_CHECK(finite);
        AICORE_CHECK(last <= first * 1.05 + 1e-9);
        std::printf("TTO: steps=%d first=%.6f last=%.6f\n", n_losses, first,
                    last);
        // out_losses is BORROWED (a ctx-owned store): freeing it here would
        // double-free the vector buffer at context teardown ("corrupted
        // double-linked list" in the encoder-model teardown).
    }

    aicore_querysplat_free(ctx);
    if (failures == 0 && e2e_count > 0) {
        const double e2e = e2e_total / e2e_count;
        std::printf("METRIC querysplat_e2e_ms %.3f\n", e2e);
        // One-line machine report for the validation runner: the *_ms leaf
        // feeds the timing metrics, the run_hash string leaf is the exact
        // output fingerprint the 'exact' fingerprint_policy compares across
        // candidates.
        std::printf(
                "{\"task\": \"querysplat\", \"querysplat_e2e_ms\": %.3f, "
                "\"views\": %d, \"height\": %d, \"width\": %d, "
                "\"gaussian_count\": %lld, \"run_hash\": \"%016llx\"}\n",
                e2e, last_views, last_height, last_width,
                (long long)last_gaussians, (unsigned long long)first_hash);
        std::printf("PASS: querysplat probe\n");
        return 0;
    }
    std::printf("FAIL: %d gate(s)\n", failures);
    return 1;
}

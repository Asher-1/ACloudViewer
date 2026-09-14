// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Asset-driven LingBot-Map probe: loads a real GGUF (downloaded into the
// shared lingbot_models cache by the validation runner), streams a small
// synthetic frame sequence, and enforces finite outputs, per-frame timing
// honesty, and numeric stability of a reset+repeat stream. Exit 77 = the
// required asset is missing (a skip, never a pass).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/lingbot_capi.h"
#include "tests/common/test_macros.hpp"
#include "tests/common/validation_probe.hpp"

namespace {

// Deterministic synthetic [0,1] NCHW RGB frames (gradient + checker) that
// exercise the patch embedding without depending on external image assets.
std::vector<float> makeFrames(int n, int w, int h) {
    std::vector<float> frames(static_cast<size_t>(n) * 3 * w * h);
    for (int f = 0; f < n; ++f) {
        float* plane = frames.data() + static_cast<size_t>(f) * 3 * w * h;
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const float fx = static_cast<float>(x) / w;
                    const float fy = static_cast<float>(y) / h;
                    const float checker =
                            ((x / 14 + y / 14 + f + c) % 2 == 0) ? 0.25f : 0.0f;
                    plane[static_cast<size_t>(c) * w * h + y * w + x] =
                            std::fmod(0.5f * (fx + fy) + 0.2f * c + checker +
                                              0.01f * f,
                                      1.0f);
                }
            }
        }
    }
    return frames;
}

}  // namespace

// Stream state shared with the C callback. The callback runs on the
// inference thread while infer_stream is on the stack, so plain members are
// safe.
struct StreamState {
    int delivered = 0;
    uint64_t hash = 0;
    int run = 0;                     // 0 = first pass, 1 = reset+repeat
    std::vector<float> first_depth;  // frame 0 of the first pass
    double max_depth_diff = 0.0;     // repeat frame-0 depth difference
};

static int failures = 0;

int main() {
    const char* gguf = std::getenv("AICORE_TEST_LINGBOT_GGUF");
    if (!gguf || gguf[0] == '\0') return 77;
    const char* device = std::getenv("AICORE_TEST_DEVICE");
    if (!device || device[0] == '\0') device = "cpu";
    const char* skyseg = std::getenv("AICORE_TEST_LINGBOT_SKYSEG");

    aicore_lingbot_options* opts = aicore_lingbot_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_lingbot_options_set_device(opts, device);
    aicore_lingbot_options_set_threads(opts, 4);
    aicore_lingbot_options_set_stream_capacity(opts, 3);
    // Bounded KV-cache profile for constrained probes
    // (AICORE_TEST_LINGBOT_KV = "scale,window"); unset keeps the official
    // release profile (8, 64).
    if (const char* kv = std::getenv("AICORE_TEST_LINGBOT_KV")) {
        int scale = 0, window = 0;
        if (std::sscanf(kv, "%d,%d", &scale, &window) == 2 && scale > 0 &&
            window > 0) {
            aicore_lingbot_options_set_kv_profile(opts, scale, window);
        }
    }

    aicore_lingbot_ctx* ctx = aicore_lingbot_load_opts(gguf, opts);
    aicore_lingbot_options_free(opts);
    if (!ctx || aicore_lingbot_is_ready(ctx) != 1) {
        std::fprintf(stderr, "lingbot load failed: %s (path=%s)\n",
                     aicore_lingbot_last_error(ctx), gguf);
        aicore_lingbot_free(ctx);
        return 1;
    }

    char* json = aicore_lingbot_info_json(ctx);
    AICORE_CHECK(json != nullptr &&
                 std::strstr(json, "graph_version") != nullptr);
    std::fprintf(stderr, "lingbot info: %s\n", json ? json : "");
    aicore_lingbot_free_buffer(json);

    // The optional native skyseg model must load on the same backend lease.
    if (skyseg && skyseg[0]) {
        AICORE_CHECK(aicore_lingbot_skyseg_load(ctx, skyseg) == 0);
        AICORE_CHECK(aicore_lingbot_skyseg_ready(ctx) == 1);
    }

    const int W = 518;  // 37 patches
    const int H = 294;  // 21 patches (official courthouse native aspect)
    const int N = 3;
    std::vector<float> frames = makeFrames(N, W, H);

    uint64_t first_hash = 0;
    StreamState state;
    auto cb = [](void* user, const aicore_lingbot_result* r) -> int {
        auto* state = static_cast<StreamState*>(user);
        if (!r || !r->depth || !r->depth_conf || !r->c2w || !r->intrinsics) {
            return 1;  // abort
        }
        const size_t plane = static_cast<size_t>(r->width) * r->height;
        for (size_t i = 0; i < plane; ++i) {
            if (!std::isfinite(r->depth[i]) ||
                !std::isfinite(r->depth_conf[i])) {
                std::fprintf(stderr,
                             "non-finite output at frame %d index %zu: "
                             "depth=%g conf=%g\n",
                             state->delivered, i, r->depth[i],
                             r->depth_conf[i]);
                return 1;
            }
        }
        for (int i = 0; i < 16; ++i) {
            if (!std::isfinite(r->c2w[i])) {
                std::fprintf(stderr, "non-finite c2w[%d]=%g at frame %d\n", i,
                             r->c2w[i], state->delivered);
                return 1;
            }
        }
        for (int i = 0; i < 4; ++i) {
            if (!std::isfinite(r->intrinsics[i])) {
                std::fprintf(stderr, "non-finite intr[%d]=%g at frame %d\n", i,
                             r->intrinsics[i], state->delivered);
                return 1;
            }
        }
        // Frame-0 numeric stability across the reset+repeat: GPU GEMMs and
        // pooled-allocator reuse make bit-exact repeats the wrong gate, so
        // the repeat is compared with a tight numeric tolerance instead.
        if (state->delivered == 0) {
            if (state->run == 0) {
                state->first_depth.assign(r->depth, r->depth + plane);
            } else {
                double max_abs = 0.0;
                for (size_t i = 0; i < plane && i < state->first_depth.size();
                     ++i) {
                    max_abs = std::max(
                            max_abs,
                            std::fabs(static_cast<double>(r->depth[i]) -
                                      state->first_depth[i]));
                }
                state->max_depth_diff = max_abs;
            }
        }
        // Per-frame fingerprint over the depth output.
        for (size_t i = 0; i < plane; ++i) {
            state->hash =
                    state->hash * 1099511628211ULL +
                    static_cast<uint64_t>(
                            reinterpret_cast<const uint8_t*>(&r->depth[i])[0]);
            state->hash =
                    state->hash * 1099511628211ULL +
                    static_cast<uint64_t>(
                            reinterpret_cast<const uint8_t*>(&r->depth[i])[1]);
        }
        ++state->delivered;
        return 0;
    };

    if (aicore_lingbot_infer_stream(ctx, frames.data(), N, W, H, cb, &state) !=
        0) {
        std::fprintf(stderr, "infer_stream failed: %s (delivered %d)\n",
                     aicore_lingbot_last_error(ctx), state.delivered);
        aicore_lingbot_free(ctx);
        return 1;
    }
    AICORE_CHECK(state.delivered == N);
    AICORE_CHECK(aicore_lingbot_last_stream_frames(ctx) == N);
    first_hash = state.hash;

    aicore_pipeline_timings timings{};
    AICORE_CHECK(aicore_lingbot_last_pipeline_timings(ctx, &timings) == 0);
    AICORE_CHECK((timings.valid_fields & AICORE_TIMING_E2E) != 0);
    AICORE_CHECK(timings.e2e_ms > 0.0);

    // Reset + repeat: the same frames through a fresh stream must
    // reproduce frame 0 within a tight numeric tolerance on every backend
    // (CPU is bit-exact in practice; GPUs are not, hence a tolerance
    // rather than a hash-equality gate).
    AICORE_CHECK(aicore_lingbot_stream_reset(ctx) == 0);
    state.delivered = 0;
    state.hash = 0;
    state.run = 1;
    if (aicore_lingbot_infer_stream(ctx, frames.data(), N, W, H, cb, &state) !=
        0) {
        std::fprintf(stderr, "repeat infer_stream failed: %s (delivered %d)\n",
                     aicore_lingbot_last_error(ctx), state.delivered);
        aicore_lingbot_free(ctx);
        return 1;
    }
    AICORE_CHECK(state.delivered == N);
    std::fprintf(
            stderr, "repeat frame-0 max depth diff: %g (hash %llu vs %llu)\n",
            state.max_depth_diff, static_cast<unsigned long long>(state.hash),
            static_cast<unsigned long long>(first_hash));
    AICORE_CHECK(state.max_depth_diff <= 1e-3);
    std::fprintf(stderr, "lingbot stream ok: device=%s frames=%d\n", device,
                 state.delivered);

    // Sky masks must be sized to the processed frame when a sky model ran.
    if (skyseg && skyseg[0]) {
        const int needed = aicore_lingbot_last_sky_mask(ctx, nullptr, 0);
        AICORE_CHECK(needed == W * H);
        std::vector<unsigned char> mask(needed);
        AICORE_CHECK(aicore_lingbot_last_sky_mask(ctx, mask.data(), needed) ==
                     needed);
    }

    aicore_lingbot_shutdown();
    aicore_lingbot_free(ctx);

    if (failures == 0) {
        aicore::test::printValidationResult("lingbot", device, first_hash,
                                            &timings);
        std::printf("test_lingbot_capi_load: PASS\n");
        return 0;
    }
    std::printf("test_lingbot_capi_load: %d FAILURES\n", failures);
    return 1;
}

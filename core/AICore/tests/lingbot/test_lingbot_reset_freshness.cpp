// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Numeric gate for aicore_lingbot_stream_reset: a stream started after a
// reset must be byte-for-byte equivalent to the SAME stream run on a fresh
// context. The device-resident KV payload is part of the stream state —
// windowed reconstruction (reset before every window) silently attended to
// the previous window's KV tail when reset left it attached, and windows 3+
// collapsed onto identical outputs.
//
// Usage:
//   test_lingbot_reset_freshness MODEL.gguf FRAMES.bin BACKEND H W
//
// FRAMES.bin holds >= 24 tightly packed [3,H,W] float32 frames (the output
// of aicore_lingbot_preprocess_image). Exits 0 on pass, 1 on failure.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "aicore/lingbot_capi.h"

namespace {

double frameDigest(const aicore_lingbot_result* r) {
    // Cheap deterministic digest of the first streamed frame's depth: mean
    // of the first 4096 finite samples (enough to catch any KV carry-over).
    double sum = 0.0;
    int n = 0;
    const int total = r->width * r->height;
    for (int i = 0; i < total && n < 4096; ++i) {
        const float d = r->depth[i];
        if (std::isfinite(d) && d > 0.f) {
            sum += d;
            ++n;
        }
    }
    return n > 0 ? sum / n : -1.0;
}

struct RunDigest {
    double firstFrame = -2.0;
    int delivered = 0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
                     "usage: %s MODEL.gguf FRAMES.bin BACKEND H W\n",
                     argv[0]);
        return 2;
    }
    const char* modelPath = argv[1];
    const char* framesPath = argv[2];
    const std::string backend = argv[3];
    const int h = std::atoi(argv[4]);
    const int w = std::atoi(argv[5]);

    std::ifstream in(framesPath, std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "cannot open input: %s\n", framesPath);
        return 2;
    }
    const size_t bytes = static_cast<size_t>(in.tellg());
    in.seekg(0);
    std::vector<float> frames(bytes / sizeof(float));
    in.read(reinterpret_cast<char*>(frames.data()), static_cast<long>(bytes));
    const size_t frameValues = static_cast<size_t>(3) * h * w;
    if (frameValues == 0 || frames.size() < 24 * frameValues) {
        std::fprintf(stderr, "need >= 24 frames of [3,%d,%d]\n", h, w);
        return 2;
    }

    auto runStream = [&](aicore_lingbot_ctx* ctx, const float* data,
                         int n, RunDigest* out) {
        auto cb = +[](void* user, const aicore_lingbot_result* r) -> int {
            auto* d = static_cast<RunDigest*>(user);
            if (d->delivered == 0) d->firstFrame = frameDigest(r);
            ++d->delivered;
            return 0;
        };
        if (aicore_lingbot_infer_stream(ctx, data, n, w, h, cb, out) != 0) {
            std::fprintf(stderr, "infer failed: %s\n",
                         aicore_lingbot_last_error(ctx));
            return false;
        }
        return true;
    };

    auto makeOpts = [&]() {
        aicore_lingbot_options* opts = aicore_lingbot_options_new();
        aicore_lingbot_options_set_device(opts, backend.c_str());
        aicore_lingbot_options_set_threads(opts, 4);
        aicore_lingbot_options_set_stream_capacity(opts, 16);
        aicore_lingbot_options_set_kv_profile(opts, 8, 16);
        return opts;
    };

    // ---- stream A: frames [0,16) on context 1 ----
    aicore_lingbot_options* opts = makeOpts();
    aicore_lingbot_ctx* ctx1 = aicore_lingbot_load_opts(modelPath, opts);
    if (!ctx1 || aicore_lingbot_is_ready(ctx1) != 1) {
        std::fprintf(stderr, "load failed: %s\n",
                     aicore_lingbot_last_error(ctx1));
        return 1;
    }
    RunDigest runA;
    if (!runStream(ctx1, frames.data(), 16, &runA)) return 1;
    aicore_lingbot_stream_reset(ctx1);
    // ---- stream B: frames [8,24) on the SAME context after reset ----
    RunDigest runB;
    if (!runStream(ctx1, frames.data() + 8 * frameValues, 16, &runB)) return 1;
    aicore_lingbot_free(ctx1);
    aicore_lingbot_options_free(opts);

    // ---- reference: frames [8,24) on a FRESH context ----
    aicore_lingbot_options* opts2 = makeOpts();
    aicore_lingbot_ctx* ctx2 = aicore_lingbot_load_opts(modelPath, opts2);
    if (!ctx2 || aicore_lingbot_is_ready(ctx2) != 1) {
        std::fprintf(stderr, "reference load failed: %s\n",
                     aicore_lingbot_last_error(ctx2));
        return 1;
    }
    RunDigest runRef;
    if (!runStream(ctx2, frames.data() + 8 * frameValues, 16, &runRef)) return 1;
    aicore_lingbot_free(ctx2);
    aicore_lingbot_options_free(opts2);
    aicore_lingbot_shutdown();

    std::printf("digest after-reset=%.9f fresh=%.9f first-stream=%.9f\n",
                runB.firstFrame, runRef.firstFrame, runA.firstFrame);
    const double diff = std::fabs(runB.firstFrame - runRef.firstFrame);
    const double scale = std::max(1e-9, std::fabs(runRef.firstFrame));
    if (diff <= 1e-3 * scale) {
        std::printf("test_lingbot_reset_freshness: PASS (rel %.2e)\n",
                    diff / scale);
        return 0;
    }
    std::fprintf(stderr,
                 "test_lingbot_reset_freshness: FAIL (rel %.3e) — the reset "
                 "stream is NOT equivalent to a fresh context\n",
                 diff / scale);
    return 1;
}

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// AICore-side parity dump for the LingBot-Map integration. Writes the same
// per-frame LBF3 binaries as the upstream lingbot-map-cli --stream-dir
// (magic 0x4C424633 + idx32 + h + w + pose_enc[9] + depth + depth_conf +
// c2w[16] + intrinsics[4], little-endian float32), so the two engines can
// be compared element-by-element with tests/lingbot/compare_lingbot_parity.py.
//
// Usage:
//   dump_lingbot_frames MODEL.gguf FRAMES.bin BACKEND H W OUT_DIR [N_FRAMES]
//
// Prints a RESULT line with the pure-inference wall time (engine load
// excluded) for the speed A/B against the upstream CLI.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "aicore/lingbot_capi.h"

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr,
                     "usage: %s MODEL.gguf FRAMES.bin BACKEND H W OUT_DIR "
                     "[N_FRAMES] [--kv SCALE,WINDOW]\n",
                     argv[0]);
        return 2;
    }
    const char* modelPath = argv[1];
    const char* framesPath = argv[2];
    const std::string backend = argv[3];
    const int h = std::atoi(argv[4]);
    const int w = std::atoi(argv[5]);
    const std::string outDir = argv[6];
    int nFrames = argc > 7 && argv[7][0] != '-' ? std::atoi(argv[7]) : 0;
    const char* kvArg = nullptr;
    for (int i = 7; i < argc; ++i) {
        if (std::strcmp(argv[i], "--kv") == 0 && i + 1 < argc) {
            kvArg = argv[i + 1];
        }
    }

    // Read the raw float32 [N,3,H,W] frame buffer.
    std::ifstream in(framesPath, std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "cannot open input: %s\n", framesPath);
        return 2;
    }
    const size_t bytes = static_cast<size_t>(in.tellg());
    in.seekg(0);
    std::vector<float> image(bytes / sizeof(float));
    in.read(reinterpret_cast<char*>(image.data()), static_cast<long>(bytes));
    const size_t frameValues = static_cast<size_t>(3) * h * w;
    if (frameValues == 0 || image.size() % frameValues != 0) {
        std::fprintf(stderr, "input size does not match [N,3,%d,%d]\n", h, w);
        return 2;
    }
    if (nFrames <= 0) {
        nFrames = static_cast<int>(image.size() / frameValues);
    }

    aicore_lingbot_options* opts = aicore_lingbot_options_new();
    aicore_lingbot_options_set_device(opts, backend.c_str());
    aicore_lingbot_options_set_threads(opts, 4);
    aicore_lingbot_options_set_stream_capacity(opts, nFrames);
    if (kvArg) {
        int scale = 0, window = 0;
        if (std::sscanf(kvArg, "%d,%d", &scale, &window) == 2 && scale > 0 &&
            window > 0) {
            aicore_lingbot_options_set_kv_profile(opts, scale, window);
        }
    }
    aicore_lingbot_ctx* ctx = aicore_lingbot_load_opts(modelPath, opts);
    aicore_lingbot_options_free(opts);
    if (!ctx || aicore_lingbot_is_ready(ctx) != 1) {
        std::fprintf(stderr, "load failed: %s\n",
                     aicore_lingbot_last_error(ctx));
        aicore_lingbot_free(ctx);
        return 1;
    }

    // The lambda captures everything directly instead.
    struct DumpState {
        std::string outDir;
        int delivered = 0;
    } dumpState{outDir, 0};

    auto dumpCb = +[](void* user, const aicore_lingbot_result* r) -> int {
        auto* ds = static_cast<DumpState*>(user);
        char path[1024];
        std::snprintf(path, sizeof(path), "%s/frame_%04d.bin",
                      ds->outDir.c_str(), ds->delivered);
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            std::fprintf(stderr, "cannot open stream frame: %s\n", path);
            return 1;
        }
        const uint32_t magic = 0x4C424633;  // LBF3
        const uint32_t idx32 = static_cast<uint32_t>(ds->delivered);
        const uint32_t fh = static_cast<uint32_t>(r->height);
        const uint32_t fw = static_cast<uint32_t>(r->width);
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char*>(&idx32), sizeof(idx32));
        file.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
        file.write(reinterpret_cast<const char*>(&fw), sizeof(fw));
        file.write(reinterpret_cast<const char*>(r->pose_enc),
                   9 * sizeof(float));
        file.write(reinterpret_cast<const char*>(r->depth),
                   static_cast<size_t>(r->width) * r->height * sizeof(float));
        file.write(reinterpret_cast<const char*>(r->depth_conf),
                   static_cast<size_t>(r->width) * r->height * sizeof(float));
        file.write(reinterpret_cast<const char*>(r->c2w), 16 * sizeof(float));
        file.write(reinterpret_cast<const char*>(r->intrinsics),
                   4 * sizeof(float));
        ++ds->delivered;
        return 0;
    };
    const auto t0 = std::chrono::steady_clock::now();
    if (aicore_lingbot_infer_stream(ctx, image.data(), nFrames, w, h, dumpCb,
                                    &dumpState) != 0) {
        std::fprintf(stderr, "infer failed: %s (delivered %d)\n",
                     aicore_lingbot_last_error(ctx), dumpState.delivered);
        aicore_lingbot_free(ctx);
        return 1;
    }
    const double wallMs = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();

    aicore_pipeline_timings timings{};
    aicore_lingbot_last_pipeline_timings(ctx, &timings);
    std::printf(
            "RESULT engine=aicore backend=%s frames=%d wall_ms=%.1f "
            "e2e_ms=%.1f delivered=%d\n",
            backend.c_str(), nFrames, wallMs, timings.e2e_ms,
            dumpState.delivered);
    aicore_lingbot_shutdown();
    aicore_lingbot_free(ctx);
    return 0;
}

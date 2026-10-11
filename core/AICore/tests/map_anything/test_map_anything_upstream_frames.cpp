// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Candidate half of the MapAnything upstream "frames" parity gate.
//
// Writes one deterministic f32 (S,3,H,W) feed built from real images, runs it
// through aicore_map_anything_run_frames on the requested backend, and exports
// every public typed output under the upstream vggt-cli file names. The
// reference half (run_map_anything_upstream_parity.py) feeds the SAME bytes to
// the upstream cpp_ggml CLI through --bin and compares bit-for-bit, so the
// feed content only has to be realistic, not identical to any resampler.
//
// Usage: test_map_anything_upstream_frames <model.gguf> <image1> <image2>
//        <backend> <out_dir> [views=3] [iters=1] [warmups=0] [threads=0]
//        [stage_dump_dir]
// threads must equal the reference CLI --threads; stage_dump_dir only takes
// effect in MAP_ANYTHING_ENABLE_DUMP builds (diagnosis of a failing row).
// dust3r is a pair model: views is clamped to 2 there. View 3+ cycles the
// inputs, mirrored, so every view is distinct content.
// Missing inputs -> exit 77; backend fallback / any failure -> exit 1.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "aicore/map_anything_capi.h"
#include "tests/common/validation_probe.hpp"

namespace {

struct Rgb {
    std::vector<uint8_t> data;
    int32_t w = 0, h = 0;
};

Rgb decode(const char* path) {
    Rgb img;
    uint8_t* data = nullptr;
    if (aicore_map_anything_load_path_rgb(path, &data, &img.w, &img.h) == 0 &&
        data != nullptr) {
        img.data.assign(data, data + (size_t)img.w * img.h * 3);
        aicore_map_anything_free_buffer(data);
    }
    return img;
}

// Half-pixel-centred bilinear resample of one RGB8 image into a CHW f32
// plane set in [0,1]. O(H*W); any deterministic resampler works because the
// reference consumes the written bytes, not this function.
void resample_chw(const Rgb& src, bool mirror, int H, int W, float* dst) {
    const size_t plane = (size_t)H * W;
    for (int y = 0; y < H; ++y) {
        const float fy = std::clamp((y + 0.5f) * src.h / H - 0.5f, 0.0f,
                                    (float)(src.h - 1));
        const int y0 = (int)fy;
        const int y1 = std::min(y0 + 1, src.h - 1);
        const float ay = fy - y0;
        for (int x = 0; x < W; ++x) {
            const int xs = mirror ? W - 1 - x : x;
            const float fx = std::clamp((xs + 0.5f) * src.w / W - 0.5f, 0.0f,
                                        (float)(src.w - 1));
            const int x0 = (int)fx;
            const int x1 = std::min(x0 + 1, src.w - 1);
            const float ax = fx - x0;
            for (int c = 0; c < 3; ++c) {
                auto at = [&](int yy, int xx) {
                    return (float)src.data[((size_t)yy * src.w + xx) * 3 + c];
                };
                const float top = at(y0, x0) * (1 - ax) + at(y0, x1) * ax;
                const float bot = at(y1, x0) * (1 - ax) + at(y1, x1) * ax;
                dst[c * plane + (size_t)y * W + x] =
                        (top * (1 - ay) + bot * ay) / 255.0f;
            }
        }
    }
}

bool write_f32(const std::filesystem::path& path, const float* p, size_t n) {
    FILE* fp = std::fopen(path.string().c_str(), "wb");
    if (fp == nullptr) return false;
    const size_t wrote = std::fwrite(p, sizeof(float), n, fp);
    std::fclose(fp);
    return wrote == n;
}

// Resolved device must be the requested backend family: a silent fallback
// would compare the wrong upstream build and is a failure, not a pass.
bool device_matches(const std::string& want, const std::string& got) {
    std::string w = want, g = got;
    std::transform(w.begin(), w.end(), w.begin(), ::tolower);
    std::transform(g.begin(), g.end(), g.begin(), ::tolower);
    if (w == "auto" || w.empty()) return true;
    return g.rfind(w, 0) == 0;
}

uint64_t result_hash(const aicore_map_anything_ctx* ctx, size_t plane) {
    uint64_t hash = 1469598103934665603ULL;
    auto add = [&](const float* p, size_t n) {
        if (p != nullptr) hash = aicore::test::fnv1aAppend(hash, p, n * 4);
    };
    const int32_t views = aicore_map_anything_result_view_count(ctx);
    add(aicore_map_anything_result_depth(ctx), plane);
    add(aicore_map_anything_result_conf(ctx), plane);
    add(aicore_map_anything_result_points(ctx), plane * 3);
    add(aicore_map_anything_result_local_points(ctx), plane * 3);
    add(aicore_map_anything_result_mask(ctx), plane);
    add(aicore_map_anything_result_scale(ctx), 1);
    add(aicore_map_anything_result_pose(ctx),
        (size_t)views * aicore_map_anything_result_pose_stride(ctx));
    int32_t dim = 0;
    const float* text = aicore_map_anything_result_text_embedding(ctx, &dim);
    add(text, (size_t)std::max(0, dim));
    return hash;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6 || argv[1][0] == '\0' || argv[2][0] == '\0' ||
        argv[3][0] == '\0' || argv[5][0] == '\0') {
        std::fprintf(stderr,
                     "usage: %s model.gguf image1 image2 backend out_dir "
                     "[views] [iters] [warmups] [threads] [stage_dump_dir]\n",
                     argv[0]);
        return 77;
    }
    const char* gguf = argv[1];
    const char* device = argv[4][0] ? argv[4] : "cpu";
    const std::filesystem::path out_dir = argv[5];
    int views = argc > 6 ? std::atoi(argv[6]) : 3;
    const int iters = argc > 7 ? std::atoi(argv[7]) : 1;
    const int warmups = argc > 8 ? std::atoi(argv[8]) : 0;
    const int threads = argc > 9 ? std::atoi(argv[9]) : 0;
    const char* stage_dump_dir = argc > 10 ? argv[10] : "";
    if (views < 2 || iters < 1 || warmups < 0 || threads < 0) {
        std::fprintf(stderr, "invalid views/iters/warmups/threads\n");
        return 1;
    }
    const Rgb inputs[2] = {decode(argv[2]), decode(argv[3])};
    if (inputs[0].data.empty() || inputs[1].data.empty()) {
        std::fprintf(stderr, "failed to decode %s / %s\n", argv[2], argv[3]);
        return 1;
    }

    aicore_map_anything_options* opts = aicore_map_anything_options_new();
    aicore_map_anything_options_set_device(opts, device);
    aicore_map_anything_options_set_threads(opts, threads);
    if (stage_dump_dir[0] != '\0') {
        aicore_map_anything_options_set_dump_dir(opts, stage_dump_dir);
    }
    aicore_map_anything_ctx* ctx = aicore_map_anything_load_opts(gguf, opts);
    aicore_map_anything_options_free(opts);
    if (ctx == nullptr || aicore_map_anything_is_ready(ctx) != 1) {
        std::fprintf(stderr, "load failed: %s\n",
                     ctx ? aicore_map_anything_last_error(ctx) : "null ctx");
        aicore_map_anything_free(ctx);
        return 1;
    }
    const std::string resolved = aicore_map_anything_context_device(ctx);
    if (!device_matches(device, resolved)) {
        std::fprintf(stderr, "backend fallback: requested %s, resolved %s\n",
                     device, resolved.c_str());
        aicore_map_anything_free(ctx);
        return 1;
    }
    const std::string arch = aicore_map_anything_result_architecture(ctx);
    if (arch == "dust3r") views = 2;

    // Geometry: the checkpoint's nominal long side (patch-floored) and the
    // first input's aspect, patch-rounded -- the shape the images path
    // produces for landscape inputs, without depending on its resampler.
    const int patch = aicore_map_anything_context_patch_size(ctx);
    const int nominal = aicore_map_anything_context_default_image_size(ctx);
    if (patch <= 0 || nominal < patch) {
        std::fprintf(stderr, "invalid model geometry patch=%d size=%d\n", patch,
                     nominal);
        aicore_map_anything_free(ctx);
        return 1;
    }
    const int long_side = nominal / patch * patch;
    const Rgb& ref = inputs[0];
    const double aspect =
            (double)std::min(ref.w, ref.h) / std::max(ref.w, ref.h);
    const int short_side = std::max(
            patch, (int)std::lround(long_side * aspect / patch) * patch);
    const int W = ref.w >= ref.h ? long_side : short_side;
    const int H = ref.w >= ref.h ? short_side : long_side;

    const size_t plane = (size_t)H * W;
    std::vector<float> frames((size_t)views * 3 * plane);
    for (int v = 0; v < views; ++v) {
        resample_chw(inputs[v % 2], /*mirror=*/v >= 2, H, W,
                     frames.data() + (size_t)v * 3 * plane);
    }
    std::error_code ec;
    std::filesystem::create_directories(out_dir / "candidate", ec);
    if (ec ||
        !write_f32(out_dir / "frames.bin", frames.data(), frames.size())) {
        std::fprintf(stderr, "cannot write %s\n",
                     (out_dir / "frames.bin").string().c_str());
        aicore_map_anything_free(ctx);
        return 1;
    }

    uint64_t first_hash = 0;
    int failures = 0;
    for (int iter = -warmups; iter < iters; ++iter) {
        if (aicore_map_anything_run_frames(ctx, frames.data(), views, H, W) !=
            0) {
            std::fprintf(stderr, "run_frames failed: %s\n",
                         aicore_map_anything_last_error(ctx));
            aicore_map_anything_free(ctx);
            return 1;
        }
        const uint64_t hash = result_hash(ctx, (size_t)views * plane);
        if (iter == 0) first_hash = hash;
        if (iter > 0 && hash != first_hash) {
            std::fprintf(stderr, "unstable output hash at iter %d\n", iter);
            failures++;
        }
    }
    if (aicore_map_anything_result_view_count(ctx) != views ||
        aicore_map_anything_result_height(ctx) != H ||
        aicore_map_anything_result_width(ctx) != W) {
        std::fprintf(stderr, "result geometry mismatch\n");
        failures++;
    }

    // Export under the upstream vggt-cli names (see cpp_ggml src/cli.cpp).
    const size_t n = (size_t)views * plane;
    const int32_t stride = aicore_map_anything_result_pose_stride(ctx);
    int32_t text_dim = 0;
    const float* text =
            aicore_map_anything_result_text_embedding(ctx, &text_dim);
    const struct {
        const char* name;
        const float* data;
        size_t count;
    } exports[] = {
            {"pose.bin", aicore_map_anything_result_pose(ctx),
             (size_t)views * std::max(0, stride)},
            {"depth.bin", aicore_map_anything_result_depth(ctx), n},
            // vggt family exports depth_conf; pi3/mapanything/dust3r conf
            {stride == 9 ? "depth_conf.bin" : "conf.bin",
             aicore_map_anything_result_conf(ctx), n},
            {"local_points.bin", aicore_map_anything_result_local_points(ctx),
             n * 3},
            {"points.bin", aicore_map_anything_result_points(ctx), n * 3},
            {"mask.bin", aicore_map_anything_result_mask(ctx), n},
            {"scale.bin", aicore_map_anything_result_scale(ctx), 1},
            {"text_embedding.bin", text, (size_t)std::max(0, text_dim)},
    };
    for (const auto& e : exports) {
        if (e.data == nullptr || e.count == 0) continue;
        if (!write_f32(out_dir / "candidate" / e.name, e.data, e.count)) {
            std::fprintf(stderr, "cannot write %s\n", e.name);
            failures++;
        }
    }
    FILE* meta = std::fopen((out_dir / "geometry.json").string().c_str(), "w");
    if (meta == nullptr) {
        failures++;
    } else {
        std::fprintf(meta,
                     "{\"architecture\":\"%s\",\"device\":\"%s\",\"S\":%d,"
                     "\"H\":%d,\"W\":%d,\"pose_stride\":%d}\n",
                     arch.c_str(), resolved.c_str(), views, H, W, stride);
        std::fclose(meta);
    }

    aicore_pipeline_timings timing{};
    const bool has_timing =
            aicore_map_anything_last_pipeline_timings(ctx, &timing) == 0;
    aicore::test::printValidationResult("map_anything", resolved.c_str(),
                                        first_hash,
                                        has_timing ? &timing : nullptr);
    aicore_map_anything_free(ctx);
    aicore_map_anything_shutdown();
    std::fprintf(stderr,
                 "map_anything frames feed: arch=%s device=%s S=%d %dx%d\n",
                 arch.c_str(), resolved.c_str(), views, W, H);
    return failures == 0 ? 0 : 1;
}

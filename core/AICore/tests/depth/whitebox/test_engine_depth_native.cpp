// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Native-resolution e2e gate (M9-T2): load a RAW non-square PNG, run the real
// DA3 resize path (Engine::depth_native -> preprocess_real -> backbone -> DPT
// head at native res), and compare against the dumped reference forward depth
// (scripts/e2e_verify_native.py: genuine InputProcessor resize + net.head).
//
// The reference is produced at the SAME processed resolution the C++ path picks
// (e.g. 640x427 -> 504x336). PASS at a tight tolerance: resize is bit-exact and
// the forward is f32-parity, so only f32 noise (+ a possible cubic LSB)
// remains.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine.hpp"
#include "image_io.hpp"
#include "parity.hpp"

int main() {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF");
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE_NATIVE");
    if (!gguf || !base) {
        std::fprintf(stderr,
                     "[engine_depth_native] missing gguf/baseline -> SKIP\n");
        return 77;
    }

    std::string png = "dumps/native_input.png";
    if (const char* p = std::getenv("AICORE_TEST_DEPTH_IMAGE")) png = p;

    std::vector<float> ref;
    std::vector<int64_t> s;
    if (!da_parity::load_baseline(base, "native_depth", ref, s)) {
        std::fprintf(
                stderr,
                "[engine_depth_native] no native_depth in baseline -> SKIP\n");
        return 77;
    }
    uint32_t out_h = da_parity::load_kv_u32(base, "native.out_h");
    uint32_t out_w = da_parity::load_kv_u32(base, "native.out_w");

    aicore::depth::Image img;
    if (!aicore::depth::load_image_rgb(png, img)) {
        std::fprintf(stderr, "[engine_depth_native] cannot load %s -> SKIP\n",
                     png.c_str());
        return 77;
    }

    auto eng = aicore::depth::Engine::load(gguf, 0);
    if (!eng) {
        std::fprintf(stderr, "[engine_depth_native] engine load failed\n");
        return 1;
    }

    std::vector<float> depth, conf;
    int H, W;
    if (!eng->depth_native_image(img, depth, conf, H, W)) {
        std::fprintf(stderr, "[engine_depth_native] depth_native failed\n");
        return 1;
    }

    std::fprintf(
            stderr,
            "[engine_depth_native] in=%dx%d -> processed %dx%d (ref %ux%u)\n",
            img.w, img.h, W, H, out_w, out_h);
    if ((uint32_t)H != out_h || (uint32_t)W != out_w) {
        std::fprintf(
                stderr,
                "[engine_depth_native] size mismatch got %dx%d ref %ux%u\n", W,
                H, out_w, out_h);
        return 1;
    }
    // 2e-3 relative: bit-exact resize + f32-parity forward -> ~f32 noise.
    bool ok =
            da_parity::compare(depth, ref, "engine_depth_native", 2e-3f, 2e-3f);
    return ok ? 0 : 1;
}

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Fused vs unfused single-image depth parity (gpu-fuse-graph).
// Runs Engine::depth_native_image twice on the same native-resolution image:
//   unfused options -> original two-graph path (backbone graph -> host feats
//   -> head graph)
//   default options -> fused ONE-graph path (feats produced in-graph,
//   device-resident)
// and asserts the depth maps stay within the f32 re-association noise of the
// two paths. Same math, one graph: the backbone feats' vit.norm layernorm runs
// in-graph (f32 ggml_norm) and the head consumes device-resident feats, so the
// difference is accumulated order noise, not just f32-vs-f64 LSB: with DA3
// depth magnitudes ~1..10 the measured ceiling is ~1.2e-3 absolute (q4_k and
// f16 checkpoints, CPU and CUDA alike), so the gate is 5e-3 (>4x margin).
// NOTE: verified bit-identical between the pre/post EngineOptions refactor —
// the historical 1e-4 gate failed identically on both.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tasks/depth/engine.hpp"
#include "tasks/depth/image_io.hpp"

int main() {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF");
    if (!gguf) {
        std::fprintf(stderr,
                     "[fused_depth] no AICORE_TEST_DEPTH_GGUF -> SKIP\n");
        return 77;
    }
    std::string png = "dumps/native_input.png";
    if (const char* p = std::getenv("AICORE_TEST_DEPTH_IMAGE")) png = p;

    aicore::depth::Image img;
    if (!aicore::depth::load_image_rgb(png, img)) {
        std::fprintf(stderr, "[fused_depth] cannot load %s -> SKIP\n",
                     png.c_str());
        return 77;
    }
    aicore::depth::EngineOptions unfused_opts;
    unfused_opts.use_fused_graph = false;
    if (const char* dev = std::getenv("AICORE_TEST_DEVICE"))
        unfused_opts.device = dev;
    auto eng = aicore::depth::Engine::load(gguf, unfused_opts);
    if (!eng) {
        std::fprintf(stderr, "[fused_depth] engine load failed\n");
        return 1;
    }
    aicore::depth::EngineOptions fused_opts;
    fused_opts.use_fused_graph = true;
    fused_opts.device = unfused_opts.device;
    auto eng_fused = aicore::depth::Engine::load(gguf, fused_opts);
    if (!eng_fused) {
        std::fprintf(stderr, "[fused_depth] fused engine load failed\n");
        return 1;
    }

    std::vector<float> du, cu;
    int Hu, Wu;
    if (!eng->depth_native_image(img, du, cu, Hu, Wu)) {
        std::fprintf(stderr, "[fused_depth] unfused depth_native failed\n");
        return 1;
    }
    std::vector<float> df, cf;
    int Hf, Wf;
    if (!eng_fused->depth_native_image(img, df, cf, Hf, Wf)) {
        std::fprintf(stderr, "[fused_depth] fused depth_native failed\n");
        return 1;
    }
    if (Hu != Hf || Wu != Wf || du.size() != df.size()) {
        std::fprintf(stderr,
                     "[fused_depth] size mismatch unfused %dx%d (%zu) vs fused "
                     "%dx%d (%zu)\n",
                     Wu, Hu, du.size(), Wf, Hf, df.size());
        return 1;
    }
    double maxd = 0.0, maxc = 0.0;
    for (size_t i = 0; i < du.size(); ++i) {
        maxd = std::max(maxd, std::fabs((double)du[i] - (double)df[i]));
        if (i < cu.size() && i < cf.size())
            maxc = std::max(maxc, std::fabs((double)cu[i] - (double)cf[i]));
    }
    std::fprintf(stderr,
                 "[fused_depth] %dx%d fused-vs-unfused max|depth|=%.3e "
                 "max|conf|=%.3e\n",
                 Wf, Hf, maxd, maxc);
    if (maxd >= 5e-3) {
        std::fprintf(stderr, "[fused_depth] FAIL: depth diff %.3e >= 5e-3\n",
                     maxd);
        return 1;
    }
    return 0;
}

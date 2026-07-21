// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Fused vs unfused single-image depth parity (gpu-fuse-graph).
// Runs Engine::depth_native_image twice on the same native-resolution image:
//   DA_FUSED=0 -> original two-graph path (backbone graph -> host feats -> head
//   graph) DA_FUSED=1 -> fused ONE-graph path (feats produced in-graph,
//   device-resident)
// and asserts the depth maps match max|d| < 1e-4. Same math, one graph: the
// only difference is the backbone feats' vit.norm layernorm runs in-graph (f32
// ggml_norm) instead of on host (f64), so only f32 LSB noise should remain.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine.hpp"
#include "image_io.hpp"

int main() {
    const char* gguf = std::getenv("DA_TEST_GGUF");
    if (!gguf) {
        std::fprintf(stderr, "[fused_depth] no DA_TEST_GGUF -> SKIP\n");
        return 77;
    }
    std::string png = "dumps/native_input.png";
    if (const char* p = std::getenv("DA_TEST_NATIVE_PNG")) png = p;

    aicore::depth::Image img;
    if (!aicore::depth::load_image_rgb(png, img)) {
        std::fprintf(stderr, "[fused_depth] cannot load %s -> SKIP\n",
                     png.c_str());
        return 77;
    }
    auto eng = aicore::depth::Engine::load(gguf, 0);
    if (!eng) {
        std::fprintf(stderr, "[fused_depth] engine load failed\n");
        return 1;
    }

    setenv("DA_FUSED", "0", 1);
    std::vector<float> du, cu;
    int Hu, Wu;
    if (!eng->depth_native_image(img, du, cu, Hu, Wu)) {
        std::fprintf(stderr, "[fused_depth] unfused depth_native failed\n");
        return 1;
    }
    setenv("DA_FUSED", "1", 1);
    std::vector<float> df, cf;
    int Hf, Wf;
    if (!eng->depth_native_image(img, df, cf, Hf, Wf)) {
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
    if (maxd >= 1e-4) {
        std::fprintf(stderr, "[fused_depth] FAIL: depth diff %.3e >= 1e-4\n",
                     maxd);
        return 1;
    }
    return 0;
}

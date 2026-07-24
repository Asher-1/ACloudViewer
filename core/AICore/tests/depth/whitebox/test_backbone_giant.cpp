// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// M5 gate: GIANT (vitg, 40 layers, embed 1536, SwiGLU FFN) backbone parity.
// Runs the metadata-driven DinoBackbone::forward (S=1) on the giant GGUF and
// compares feat_g_{19,27,33,39} + cam_g_{19,27,33,39} against the dump.
// SLOW: 40-layer CPU forward takes tens of seconds. SKIP (77) if artifacts
// absent.
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "dino_backbone.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
int main() {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF_GIANT");
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE_GIANT");
    if (!gguf || !base) return 77;
    aicore::depth::ModelLoader ml;
    if (!ml.load(gguf)) return 1;
    std::vector<float> img;
    std::vector<int64_t> s;
    if (!da_parity::load_baseline(base, "input_image", img, s)) return 1;
    int HW = (int)(img.size() / 3);
    int H = (int)std::lround(std::sqrt((double)HW)), W = H;
    aicore::depth::Backend be;
    aicore::depth::DinoBackbone bb(ml, be);
    std::vector<std::vector<float>> feats, cams;
    if (!bb.forward(img, H, W, feats, cams)) return 1;
    const int Ls[4] = {19, 27, 33, 39};
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        std::vector<float> rf, rc;
        if (!da_parity::load_baseline(
                    base, std::string("feat_g_") + std::to_string(Ls[i]), rf,
                    s))
            return 1;
        if (!da_parity::load_baseline(
                    base, std::string("cam_g_") + std::to_string(Ls[i]), rc, s))
            return 1;
        ok &= da_parity::compare(
                feats[i], rf,
                (std::string("feat_g_") + std::to_string(Ls[i])).c_str(), 2e-3f,
                2e-3f);
        ok &= da_parity::compare(
                cams[i], rc,
                (std::string("cam_g_") + std::to_string(Ls[i])).c_str(), 2e-3f,
                2e-3f);
    }
    return ok ? 0 : 1;
}

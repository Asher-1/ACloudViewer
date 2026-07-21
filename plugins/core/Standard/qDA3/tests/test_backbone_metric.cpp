// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// M6 gate: METRIC (ViT-Large, 24 layers, embed 1024, MLP FFN, cat_token=false)
// backbone parity. Runs the metadata-driven DinoBackbone::forward (S=1) on the
// nested metric GGUF and compares feat_m_{4,11,17,23} against the nested dump.
// SLOW: 24-layer ViT-L CPU forward. SKIP (77) if artifacts absent.
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "dino_backbone.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
int main() {
    const char* gguf = std::getenv("DA_TEST_GGUF_METRIC");
    const char* base = std::getenv("DA_TEST_BASELINE_NESTED");
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
    const int Ls[4] = {4, 11, 17, 23};
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        std::vector<float> rf;
        if (!da_parity::load_baseline(
                    base, std::string("feat_m_") + std::to_string(Ls[i]), rf,
                    s))
            return 1;
        ok &= da_parity::compare(
                feats[i], rf,
                (std::string("feat_m_") + std::to_string(Ls[i])).c_str(), 2e-3f,
                2e-3f);
    }
    return ok ? 0 : 1;
}

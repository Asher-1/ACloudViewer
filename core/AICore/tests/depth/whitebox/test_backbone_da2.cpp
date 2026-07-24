// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// DA2 backbone parity: metadata-driven DinoBackbone::forward (S=1,
// cat_token=false) on the DA2 ViT-L GGUF vs get_intermediate_layers(norm=True)
// from dump_da2.py. Validates the DA2==DA3-subset feature-extraction claim.
// SKIP (77) if artifacts absent.
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "dino_backbone.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
int main() {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF_DA2");
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE_DA2");
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
                    base, std::string("feat_da2_") + std::to_string(Ls[i]), rf,
                    s))
            return 1;
        ok &= da_parity::compare(
                feats[i], rf,
                (std::string("feat_da2_") + std::to_string(Ls[i])).c_str(),
                2e-3f, 2e-3f);
    }
    return ok ? 0 : 1;
}

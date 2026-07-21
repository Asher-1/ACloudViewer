// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cmath>
#include <cstdlib>
#include <vector>

#include "dino_backbone.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
int main() {
    const char* gguf = std::getenv("DA_TEST_GGUF");
    const char* base = std::getenv("DA_TEST_BASELINE");
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
    std::vector<float> got;
    if (!bb.prepare_tokens(img, H, W, got)) return 1;
    std::vector<float> ref;
    da_parity::load_baseline(base, "pos_embed_added", ref, s);
    bool ok = da_parity::compare(got, ref, "pos_embed_added", 2e-3f, 2e-3f);
    return ok ? 0 : 1;
}

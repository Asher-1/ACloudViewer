// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// DA2 end-to-end relative-depth parity: Engine::depth_relative on the DA2 ViT-L
// GGUF vs net.forward() (dump_da2.py "depth_da2"), on the shared fixed input.
// The fixed input is already a multiple of 14 -> processed dims == input dims,
// so the C++ depth map and the reference are directly comparable. SKIP (77) if
// absent.
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "dino_backbone.hpp"
#include "dpt_head.hpp"
#include "engine.hpp"
#include "image_io.hpp"
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
    // Run backbone -> depth_relative head directly (mirrors
    // Engine::depth_relative, bypassing image IO so the input is bit-identical
    // to the dump).
    aicore::depth::Backend be;
    aicore::depth::DinoBackbone bb(ml, be);
    std::vector<std::vector<float>> feats, cams;
    if (!bb.forward(img, H, W, feats, cams)) return 1;
    aicore::depth::DptHead head(ml, be);
    std::vector<float> depth;
    if (!head.depth_relative(feats, H, W, ml.config().head_max_depth, depth))
        return 1;
    std::vector<float> ref;
    if (!da_parity::load_baseline(base, "depth_da2", ref, s)) return 1;
    return da_parity::compare(depth, ref, "depth_da2", 3e-3f, 3e-3f) ? 0 : 1;
}

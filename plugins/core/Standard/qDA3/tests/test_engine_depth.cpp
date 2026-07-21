// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// E2E fixture gate: dumped raw_image -> full Engine depth -> compare vs
// head_depth.
#include <cstdlib>
#include <vector>

#include "engine.hpp"
#include "image_io.hpp"
#include "parity.hpp"
int main() {
    const char* gguf = std::getenv("DA_TEST_GGUF");
    const char* base = std::getenv("DA_TEST_BASELINE");
    if (!gguf || !base) return 77;
    std::vector<float> raw, ref;
    std::vector<int64_t> s;
    if (!da_parity::load_baseline(base, "raw_image", raw, s)) return 77;
    if (!da_parity::load_baseline(base, "head_depth", ref, s)) return 1;
    auto eng = aicore::depth::Engine::load(gguf, 0);
    if (!eng) return 1;
    aicore::depth::Image img;
    img.w = 224;
    img.h = 224;
    img.rgb.resize(224 * 224 * 3);
    for (size_t i = 0; i < img.rgb.size(); ++i)
        img.rgb[i] = (unsigned char)(raw[i] + 0.5f);
    std::vector<float> depth, conf;
    int H, W;
    if (!eng->depth_image(img, depth, conf, H, W)) return 1;
    // full pipeline: preprocess(exact) -> backbone(~4e-4) -> head; accumulated
    // tol
    bool ok = da_parity::compare(depth, ref, "engine_depth", 5e-3f, 5e-3f);
    return ok ? 0 : 1;
}

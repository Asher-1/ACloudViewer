// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// M2 component gate: full DualDPT MAIN depth path against the dumped reference.
// Feeds the dumped feat_5/7/9/11 as the head input (isolates head from
// backbone), then gates the 4 post-resize stages, the fused map, and the final
// depth/conf.
#include <cstdlib>
#include <string>
#include <vector>

#include "backend.hpp"
#include "dpt_head.hpp"
#include "model_loader.hpp"
#include "parity.hpp"

int main() {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF");
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE");
    if (!gguf || !base) return 77;
    aicore::depth::ModelLoader ml;
    if (!ml.load(gguf)) return 1;
    aicore::depth::Backend be;

    const int H = 224, W = 224;
    const int Ls[4] = {5, 7, 9, 11};
    std::vector<std::vector<float>> feats(4);
    std::vector<int64_t> s;
    for (int i = 0; i < 4; ++i) {
        if (!da_parity::load_baseline(
                    base, std::string("feat_") + std::to_string(Ls[i]),
                    feats[i], s))
            return 1;
    }

    aicore::depth::DptHead head(ml, be);
    std::vector<float> depth, conf, fused;
    std::vector<std::vector<float>> stages;
    if (!head.depth_debug(feats, H, W, depth, conf, stages, fused)) {
        std::fprintf(stderr, "depth_debug failed\n");
        return 1;
    }

    bool ok = true;
    // ISOLATION: post-resize stages.
    for (int i = 0; i < 4; ++i) {
        std::vector<float> ref;
        if (!da_parity::load_baseline(
                    base, std::string("head_stage") + std::to_string(i), ref,
                    s))
            return 1;
        ok &= da_parity::compare(
                stages[i], ref,
                (std::string("head_stage") + std::to_string(i)).c_str(), 2e-3f,
                2e-3f);
    }
    // ISOLATION: fused (post output_conv1).
    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(base, "head_fused", ref, s)) return 1;
        ok &= da_parity::compare(fused, ref, "head_fused", 2e-3f, 2e-3f);
    }
    // MAIN gate: depth + conf.
    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(base, "head_depth", ref, s)) return 1;
        ok &= da_parity::compare(depth, ref, "head_depth", 2e-3f, 2e-3f);
    }
    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(base, "head_depth_conf", ref, s))
            return 1;
        ok &= da_parity::compare(conf, ref, "head_depth_conf", 2e-3f, 2e-3f);
    }

    return ok ? 0 : 1;
}

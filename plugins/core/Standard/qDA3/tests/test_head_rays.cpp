// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Part A gate: DualDPT AUXILIARY ray head against the dumped reference.
// Feeds the dumped feat_5/7/9/11 (same fixed 224 fixture as test_dpt_head) into
// the independent aux pyramid, then bit-exact-parity-gates ray (6ch, identity)
// + ray_conf (1ch, expp1) vs dumps/reference_rays.gguf. Uses the opt-in
// --with-aux GGUF.
#include <cstdlib>
#include <string>
#include <vector>

#include "backend.hpp"
#include "dpt_head.hpp"
#include "model_loader.hpp"
#include "parity.hpp"

int main() {
    const char* gguf = std::getenv("DA_TEST_GGUF_AUX");
    const char* base =
            std::getenv("DA_TEST_BASELINE");  // feats (reference.gguf)
    const char* rays =
            std::getenv("DA_TEST_BASELINE_RAYS");  // gold ray/ray_conf
    if (!gguf || !base || !rays) return 77;
    aicore::depth::ModelLoader ml;
    if (!ml.load(gguf)) return 1;
    aicore::depth::Backend be;

    const int H = 224, W = 224;
    const int Ls[4] = {5, 7, 9, 11};
    std::vector<std::vector<float>> feats(4);
    std::vector<int64_t> s;
    for (int i = 0; i < 4; ++i)
        if (!da_parity::load_baseline(
                    base, std::string("feat_") + std::to_string(Ls[i]),
                    feats[i], s))
            return 1;

    aicore::depth::DptHead head(ml, be);
    std::vector<float> ray, ray_conf;
    int ray_h = 0, ray_w = 0;
    if (!head.rays(feats, H, W, ray, ray_conf, ray_h, ray_w)) {
        std::fprintf(stderr, "rays() failed (aux tensors present? %s)\n", gguf);
        return 1;
    }
    std::fprintf(stderr,
                 "[rays] aux resolution H=%d W=%d ray=%zu ray_conf=%zu\n",
                 ray_h, ray_w, ray.size(), ray_conf.size());

    bool ok = true;
    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(rays, "ray", ref, s)) return 1;
        ok &= da_parity::compare(ray, ref, "ray", 2e-3f, 2e-3f);
    }
    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(rays, "ray_conf", ref, s)) return 1;
        ok &= da_parity::compare(ray_conf, ref, "ray_conf", 2e-3f, 2e-3f);
    }
    return ok ? 0 : 1;
}

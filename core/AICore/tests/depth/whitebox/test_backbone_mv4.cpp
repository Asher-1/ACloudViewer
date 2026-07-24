// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "dino_backbone.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
// M4b: multi-view (S=4) backbone WITH reference-view selection
// (saddle_balanced). Verifies (a) the internally-selected reference index b_idx
// matches the dumped refsel_b_idx, and (b) per-view feat/cam parity in ORIGINAL
// view order at 2e-3.
int main() {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF");
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE_MV4");
    if (!gguf || !base) return 77;
    aicore::depth::ModelLoader ml;
    if (!ml.load(gguf)) return 1;

    const float mean[3] = {0.485f, 0.456f, 0.406f},
                std_[3] = {0.229f, 0.224f, 0.225f};
    const int H = 224, W = 224, S = 4;
    std::vector<std::vector<float>> views(S);
    for (int v = 0; v < S; ++v) {
        std::vector<float> raw;
        std::vector<int64_t> s;  // raw_mv4_v stored HWC [224,224,3] (0..255)
        if (!da_parity::load_baseline(
                    base, std::string("raw_mv4_") + std::to_string(v), raw, s))
            return 1;
        std::vector<float> chw((size_t)3 * H * W);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < 3; ++c) {
                    float px = raw[((size_t)y * W + x) * 3 + c] / 255.0f;
                    chw[((size_t)c * H + y) * W + x] = (px - mean[c]) / std_[c];
                }
        views[v] = std::move(chw);
    }

    aicore::depth::Backend be;
    aicore::depth::DinoBackbone bb(ml, be);
    std::vector<std::vector<std::vector<float>>> feats, cams;
    int b_idx = -1;
    if (!bb.forward_mv(views, H, W, feats, cams, &b_idx)) return 1;

    bool ok = true;
    // (a) reference-view index parity.
    std::vector<float> bref;
    std::vector<int64_t> bs;
    if (!da_parity::load_baseline(base, "refsel_b_idx", bref, bs)) return 1;
    int ref_b = (int)std::lround(bref[0]);
    std::fprintf(stderr, "[refsel_b_idx] cpp=%d ref=%d -> %s\n", b_idx, ref_b,
                 (b_idx == ref_b) ? "OK" : "FAIL");
    ok &= (b_idx == ref_b);

    // (b) per-view feat/cam parity in ORIGINAL view order.
    const int Ls[4] = {5, 7, 9, 11};
    const size_t feat_view = 256 * 1536, cam_view = 1536;
    for (int i = 0; i < 4; ++i) {
        std::vector<float> rf, rc;
        std::vector<int64_t> s;
        da_parity::load_baseline(
                base, std::string("feat_mv4_") + std::to_string(Ls[i]), rf, s);
        da_parity::load_baseline(
                base, std::string("cam_mv4_") + std::to_string(Ls[i]), rc, s);
        for (int v = 0; v < S; ++v) {
            std::vector<float> rf_v(rf.begin() + (size_t)v * feat_view,
                                    rf.begin() + (size_t)(v + 1) * feat_view);
            std::vector<float> rc_v(rc.begin() + (size_t)v * cam_view,
                                    rc.begin() + (size_t)(v + 1) * cam_view);
            ok &= da_parity::compare(
                    feats[i][v], rf_v,
                    (std::string("feat_mv4_") + std::to_string(Ls[i]) + "_v" +
                     std::to_string(v))
                            .c_str(),
                    2e-3f, 2e-3f);
            ok &= da_parity::compare(
                    cams[i][v], rc_v,
                    (std::string("cam_mv4_") + std::to_string(Ls[i]) + "_v" +
                     std::to_string(v))
                            .c_str(),
                    2e-3f, 2e-3f);
        }
    }
    return ok ? 0 : 1;
}

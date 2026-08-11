// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine.hpp"
#include "image_io.hpp"
#include "parity.hpp"
// e2e multi-view gate: build 2 Images from the MV baseline raw views, run the
// full multi-view pipeline (one backbone_mv pass -> per-view depth + pose), and
// compare each view's depth/extrinsics/intrinsics against the original PyTorch
// reference.
int main() {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF");
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE_MV");
    if (!gguf || !base) return 77;
    const int H = 224, W = 224, S = 2;
    // Load reference outputs.
    std::vector<float> rdepth, rext, rintr;
    std::vector<int64_t> s;
    if (!da_parity::load_baseline(base, "depth_mv", rdepth, s)) return 77;
    if (!da_parity::load_baseline(base, "extrinsics_mv", rext, s)) return 77;
    if (!da_parity::load_baseline(base, "intrinsics_mv", rintr, s)) return 77;
    // Build 2 Images from raw_mv_0/raw_mv_1 (HWC, 0..255).
    std::vector<aicore::depth::Image> imgs(S);
    for (int v = 0; v < S; ++v) {
        std::vector<float> raw;
        std::vector<int64_t> rs;
        if (!da_parity::load_baseline(
                    base, std::string("raw_mv_") + std::to_string(v), raw, rs))
            return 77;
        imgs[v].w = W;
        imgs[v].h = H;
        imgs[v].rgb.resize((size_t)W * H * 3);
        for (size_t i = 0; i < imgs[v].rgb.size(); ++i)
            imgs[v].rgb[i] = (unsigned char)(raw[i] + 0.5f);
    }
    auto eng = aicore::depth::Engine::load(gguf, 0);
    if (!eng) return 1;
    std::vector<aicore::depth::ViewResult> views;
    int oH = 0, oW = 0;
    if (!eng->depth_pose_multi(imgs, views, oH, oW)) return 1;
    if ((int)views.size() != S || oH != H || oW != W) {
        std::fprintf(stderr, "bad shape\n");
        return 1;
    }
    const size_t dview = (size_t)H * W;
    bool ok = true;
    for (int v = 0; v < S; ++v) {
        std::vector<float> rd_v(rdepth.begin() + (size_t)v * dview,
                                rdepth.begin() + (size_t)(v + 1) * dview);
        std::vector<float> re_v(rext.begin() + (size_t)v * 12,
                                rext.begin() + (size_t)(v + 1) * 12);
        std::vector<float> ri_v(rintr.begin() + (size_t)v * 9,
                                rintr.begin() + (size_t)(v + 1) * 9);
        std::vector<float> ve(views[v].ext.begin(), views[v].ext.end());
        std::vector<float> vi(views[v].intr.begin(), views[v].intr.end());
        ok &= da_parity::compare(
                views[v].depth, rd_v,
                (std::string("mv_depth_v") + std::to_string(v)).c_str(), 5e-3f,
                5e-3f);
        ok &= da_parity::compare(
                ve, re_v, (std::string("mv_ext_v") + std::to_string(v)).c_str(),
                5e-3f, 5e-3f);
        ok &= da_parity::compare(
                vi, ri_v,
                (std::string("mv_intr_v") + std::to_string(v)).c_str(), 5e-3f,
                5e-3f);
    }
    return ok ? 0 : 1;
}

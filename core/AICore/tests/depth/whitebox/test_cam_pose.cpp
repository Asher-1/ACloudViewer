// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <array>
#include <cstdlib>
#include <vector>

#include "cam_pose.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
int main() {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF");
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE");
    if (!gguf || !base) return 77;
    aicore::depth::ModelLoader ml;
    if (!ml.load(gguf)) return 1;
    std::vector<float> ct;
    std::vector<int64_t> s;
    if (!da_parity::load_baseline(base, "cam_token_11", ct, s)) return 1;
    aicore::depth::Backend be;
    aicore::depth::CamPose cp(ml, be);
    std::array<float, 9> pe;
    std::array<float, 12> ext;
    std::array<float, 9> K;
    if (!cp.pose(ct, 224, 224, pe, ext, K)) return 1;
    std::vector<float> rpe, rext, rK;
    da_parity::load_baseline(base, "pose_enc", rpe, s);
    da_parity::load_baseline(base, "extrinsics", rext, s);
    da_parity::load_baseline(base, "intrinsics", rK, s);
    std::vector<float> vpe(pe.begin(), pe.end()), vext(ext.begin(), ext.end()),
            vK(K.begin(), K.end());
    bool ok = true;
    ok &= da_parity::compare(vpe, rpe, "pose_enc", 2e-3f, 2e-3f);
    ok &= da_parity::compare(vext, rext, "extrinsics", 2e-3f, 2e-3f);
    ok &= da_parity::compare(vK, rK, "intrinsics", 2e-3f, 2e-3f);
    return ok ? 0 : 1;
}

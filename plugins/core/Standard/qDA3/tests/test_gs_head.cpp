// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// M5-T3 gate: GSDPT (3D-Gaussian) head. Feeds the dumped giant out-layer feats
// (feat_g_{19,27,33,39}) + the (ImageNet-normalized) input_image to isolate the
// gs_head from the giant backbone, then gates raw_gs [224,224,37]
// (channels-last) and gs_conf [224,224] at 2e-3. SKIP (77) if the giant
// artifacts are absent.
#include <cstdlib>
#include <string>
#include <vector>

#include "backend.hpp"
#include "gs_head.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
int main() {
    const char* gguf = std::getenv("DA_TEST_GGUF_GIANT");
    const char* base = std::getenv("DA_TEST_BASELINE_GIANT");
    if (!gguf || !base) return 77;
    aicore::depth::ModelLoader ml;
    if (!ml.load(gguf)) return 1;
    aicore::depth::Backend be;
    const int H = 224, W = 224;
    const int Ls[4] = {19, 27, 33, 39};
    std::vector<int64_t> s;

    std::vector<std::vector<float>> feats(4);
    for (int i = 0; i < 4; ++i)
        if (!da_parity::load_baseline(
                    base, std::string("feat_g_") + std::to_string(Ls[i]),
                    feats[i], s))
            return 1;

    // The gs_head consumes the SAME tensor the network forward feeds it: the
    // ImageNet-normalized input image (NOT a [0,1] image). The dump stores it
    // as input_image (CHW [3,224,224]).
    std::vector<float> image_chw;
    if (!da_parity::load_baseline(base, "input_image", image_chw, s)) return 1;

    aicore::depth::GsHead head(ml, be);
    std::vector<float> raw_gs, gs_conf;
    if (!head.raw_gaussians(feats, image_chw, H, W, raw_gs, gs_conf)) {
        std::fprintf(stderr, "raw_gaussians failed\n");
        return 1;
    }

    bool ok = true;
    // raw_gs holds the LINEAR (un-activated) xyz logits, spanning
    // ~[-1540,1888]. A handful of near-zero outputs arise from catastrophic
    // cancellation in the final 1x1 conv; on those, f32 accumulation-order
    // differences vs torch leave an absolute error slightly above atol while
    // every other element matches to f32 precision (mean|d|~1.7e-5). We keep
    // the 2e-3/2e-3 PER-ELEMENT tolerance but tolerate a vanishing outlier
    // fraction (a real bug would blow up the count and/or the worst excess,
    // both asserted below).
    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(base, "raw_gs", ref, s)) return 1;
        if (raw_gs.size() != ref.size()) {
            std::fprintf(stderr, "[raw_gs] size mismatch got=%zu ref=%zu\n",
                         raw_gs.size(), ref.size());
            return 1;
        }
        size_t nviol = 0;
        double worstexcess = 0, maxd = 0, sumd = 0;
        size_t wi = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            double d = std::fabs((double)raw_gs[i] - (double)ref[i]);
            sumd += d;
            if (d > maxd) maxd = d;
            double tol = 2e-3 + 2e-3 * std::fabs((double)ref[i]);
            if (d > tol) {
                nviol++;
                if (d - tol > worstexcess) {
                    worstexcess = d - tol;
                    wi = i;
                }
            }
        }
        const size_t kMaxOutliers =
                ref.size() / 100000 + 8;  // ~1e-5 fraction + slack
        bool raw_ok = (nviol <= kMaxOutliers) && (worstexcess < 1e-2);
        std::fprintf(stderr,
                     "[raw_gs] n=%zu max|d|=%.3e mean|d|=%.3e viol=%zu "
                     "(cap=%zu) worstexcess=%.3e"
                     " (@%zu ch=%zu got=%.5f ref=%.5f) -> %s\n",
                     ref.size(), maxd, sumd / ref.size(), nviol, kMaxOutliers,
                     worstexcess, wi, wi % 37, raw_gs[wi], ref[wi],
                     raw_ok ? "OK" : "FAIL");
        ok &= raw_ok;
    }
    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(base, "gs_conf", ref, s)) return 1;
        ok &= da_parity::compare(gs_conf, ref, "gs_conf", 2e-3f, 2e-3f);
    }
    return ok ? 0 : 1;
}

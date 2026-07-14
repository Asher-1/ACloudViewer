// M6 gate: metric DPT single-head (depth, output_dim 1, activation "exp") + sky
// head (sky_out2a/out2b, activation "relu"), norm_type "idt" (no head.norm),
// dim_in = embed (cat_token false). Feeds the dumped feat_m_{4,11,17,23} as the
// head input (isolates head from backbone) and compares the metric depth vs
// depth_metric_head and sky vs sky_head. SKIP (77) if artifacts absent.
#include "dpt_head.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
#include <string>

int main() {
    const char* gguf = std::getenv("DA_TEST_GGUF_METRIC");
    const char* base = std::getenv("DA_TEST_BASELINE_NESTED");
    if (!gguf || !base) return 77;
    da::ModelLoader ml; if (!ml.load(gguf)) return 1;
    da::Backend be;

    const int H = 224, W = 224;
    const int Ls[4] = { 4, 11, 17, 23 };
    std::vector<std::vector<float>> feats(4);
    std::vector<int64_t> s;
    for (int i = 0; i < 4; ++i)
        if (!da_parity::load_baseline(base, std::string("feat_m_") + std::to_string(Ls[i]),
                                      feats[i], s)) return 1;

    da::DptHead head(ml, be);
    std::vector<float> depth, sky;
    if (!head.depth_sky(feats, H, W, depth, sky)) {
        std::fprintf(stderr, "depth_sky failed\n");
        return 1;
    }

    bool ok = true;
    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(base, "depth_metric_head", ref, s)) return 1;
        ok &= da_parity::compare(depth, ref, "depth_metric_head", 2e-3f, 2e-3f);
    }
    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(base, "sky_head", ref, s)) return 1;
        ok &= da_parity::compare(sky, ref, "sky_head", 2e-3f, 2e-3f);
    }
    return ok ? 0 : 1;
}

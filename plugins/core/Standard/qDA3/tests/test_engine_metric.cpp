// M6-T3 gate (SLOW, optional): full nested metric pipeline through the Engine.
// Feeds the dumped raw_image (224x224x3 uint8) through Engine::depth_metric (which
// runs the nested anyview GIANT + metric ViT-L backbones + DPT/sky heads + cam
// pose + NestedAligner) and compares depth_final / scale_factor / extrinsics_final
// vs the nested dump at 5e-3. SKIP (77) if any artifact is absent.
#include "engine.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>

int main() {
    const char* anyview = std::getenv("DA_TEST_GGUF_ANYVIEW");
    const char* metric  = std::getenv("DA_TEST_GGUF_METRIC");
    const char* base    = std::getenv("DA_TEST_BASELINE_NESTED");
    if (!anyview || !metric || !base) return 77;

    std::vector<float> raw; std::vector<int64_t> s;
    if (!da_parity::load_baseline(base, "raw_image", raw, s)) return 1;  // [224,224,3]
    const int H = 224, W = 224;
    if ((int)raw.size() != H * W * 3) { std::fprintf(stderr, "bad raw_image %zu\n", raw.size()); return 1; }

    da::Image img; img.w = W; img.h = H; img.rgb.resize((size_t)H * W * 3);
    for (size_t i = 0; i < raw.size(); ++i) {
        float v = raw[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        img.rgb[i] = (unsigned char)(v + 0.5f);
    }

    auto eng = da::Engine::load_nested(anyview, metric, 0);
    if (!eng) { std::fprintf(stderr, "load_nested failed\n"); return 1; }
    da::NestedOut out; int oH, oW;
    if (!eng->depth_metric(img, out, oH, oW)) { std::fprintf(stderr, "depth_metric failed\n"); return 1; }

    bool ok = true;
    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(base, "depth_final", ref, s)) return 1;
        ok &= da_parity::compare(out.depth, ref, "depth_final", 5e-3f, 5e-3f);
    }
    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(base, "scale_factor", ref, s)) return 1;
        std::vector<float> got = { out.scale_factor };
        ok &= da_parity::compare(got, ref, "scale_factor", 5e-3f, 5e-3f);
    }
    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(base, "extrinsics_final", ref, s)) return 1;
        std::vector<float> got(out.extrinsics.begin(), out.extrinsics.end());
        ok &= da_parity::compare(got, ref, "extrinsics_final", 5e-3f, 5e-3f);
    }
    return ok ? 0 : 1;
}

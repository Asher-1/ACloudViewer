// Standalone monocular e2e gate (DA3MONO-LARGE): load a RAW non-square PNG, run
// the mono path (Engine::depth_mono -> preprocess_real -> backbone cat_token=False
// -> DPT depth_sky head, output_dim==1 + sky), and compare depth + sky against the
// dumped reference forward (scripts/dump_mono.py: genuine InputProcessor resize +
// net.head -> {"depth","sky"}).
//
// PASS at a tight tolerance: resize is bit-exact, the forward is f32-parity, so
// only f32 noise remains. Primary criterion is depth; sky uses the same tolerance.
#include "engine.hpp"
#include "image_io.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

int main(){
    const char* gguf = std::getenv("DA_TEST_GGUF_MONO");
    const char* base = std::getenv("DA_TEST_BASELINE_MONO");
    if (!gguf || !base) { std::fprintf(stderr, "[engine_mono] missing gguf/baseline -> SKIP\n"); return 77; }

    std::string png = "dumps/mono_input.png";
    if (const char* p = std::getenv("DA_TEST_MONO_PNG")) png = p;

    std::vector<float> ref_depth, ref_sky; std::vector<int64_t> s;
    if (!da_parity::load_baseline(base, "mono_depth", ref_depth, s)) {
        std::fprintf(stderr, "[engine_mono] no mono_depth in baseline -> SKIP\n"); return 77;
    }
    if (!da_parity::load_baseline(base, "mono_sky", ref_sky, s)) {
        std::fprintf(stderr, "[engine_mono] no mono_sky in baseline -> SKIP\n"); return 77;
    }
    uint32_t out_h = da_parity::load_kv_u32(base, "mono.out_h");
    uint32_t out_w = da_parity::load_kv_u32(base, "mono.out_w");

    da::Image img;
    if (!da::load_image_rgb(png, img)) { std::fprintf(stderr, "[engine_mono] cannot load %s -> SKIP\n", png.c_str()); return 77; }

    auto eng = da::Engine::load(gguf, 0);
    if (!eng) { std::fprintf(stderr, "[engine_mono] engine load failed\n"); return 1; }
    if (!eng->is_mono()) { std::fprintf(stderr, "[engine_mono] gguf not detected as mono\n"); return 1; }

    std::vector<float> depth, sky; int H, W;
    if (!eng->depth_mono(img, depth, sky, H, W)) { std::fprintf(stderr, "[engine_mono] depth_mono failed\n"); return 1; }

    std::fprintf(stderr, "[engine_mono] in=%dx%d -> processed %dx%d (ref %ux%u)\n",
                 img.w, img.h, W, H, out_w, out_h);
    if ((uint32_t)H != out_h || (uint32_t)W != out_w) {
        std::fprintf(stderr, "[engine_mono] size mismatch got %dx%d ref %ux%u\n", W, H, out_w, out_h);
        return 1;
    }
    bool ok = da_parity::compare(depth, ref_depth, "engine_mono_depth", 2e-3f, 2e-3f);
    ok = da_parity::compare(sky, ref_sky, "engine_mono_sky", 2e-3f, 2e-3f) && ok;
    return ok ? 0 : 1;
}

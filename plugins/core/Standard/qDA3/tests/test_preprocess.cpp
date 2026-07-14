#include "preprocess.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
int main(){
    const char* gguf = std::getenv("DA_TEST_GGUF");
    const char* base = std::getenv("DA_TEST_BASELINE");
    if (!gguf || !base) return 77;
    da::ModelLoader ml; if (!ml.load(gguf)) return 1;
    std::vector<float> raw, ref; std::vector<int64_t> s;
    if (!da_parity::load_baseline(base, "raw_image", raw, s)) return 77;   // needs regenerated dump
    if (!da_parity::load_baseline(base, "input_image", ref, s)) return 1;
    // raw is (224,224,3) HWC floats 0..255
    const int H=224, W=224;
    da::Image img; img.w=W; img.h=H; img.rgb.resize((size_t)W*H*3);
    for (size_t i=0;i<img.rgb.size();++i) img.rgb[i] = (unsigned char)(raw[i] + 0.5f);
    da::Preprocessed p;
    if (!da::preprocess(img, ml.config(), p)) return 1;
    if (p.H!=H || p.W!=W) { std::fprintf(stderr, "wrong dims %dx%d\n", p.H, p.W); return 1; }
    bool ok = da_parity::compare(p.chw, ref, "preprocess", 1e-3f, 1e-3f);
    return ok ? 0 : 1;
}

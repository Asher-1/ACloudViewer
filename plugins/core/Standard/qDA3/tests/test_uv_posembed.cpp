#include "uv_posembed.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <vector>
int main(){
    const char* base = std::getenv("DA_TEST_BASELINE");
    if (!base) return 77;
    std::vector<float> ref; std::vector<int64_t> s;
    if (!da_parity::load_baseline(base, "uv_embed_64", ref, s)) return 77;
    std::vector<float> got = da::uv_pos_embed(/*pw=*/224, /*ph=*/224, /*C=*/64, /*aspect=*/1.0f, 100.f);
    bool ok = da_parity::compare(got, ref, "uv_embed_64", 1e-5f, 1e-5f);
    return ok ? 0 : 1;
}

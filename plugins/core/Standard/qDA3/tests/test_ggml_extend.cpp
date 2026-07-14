#include "ggml_extend.hpp"
#include "backend.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
int main() {
    da::Backend be; da::GraphInputPool pool;
    // layernorm of [1,2,3,4] (mean=2.5,var=1.25) with w=1,b=0 -> normalized
    std::vector<float> x = {1,2,3,4}, w = {1,1,1,1}, b = {0,0,0,0}, out;
    bool ok = be.compute([&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* tx = be.add_graph_input(ctx, pool, x.data(), 4);
        ggml_tensor* tw = be.add_graph_input(ctx, pool, w.data(), 4);
        ggml_tensor* tb = be.add_graph_input(ctx, pool, b.data(), 4);
        return da::layernorm(ctx, tx, tw, tb, 1e-6f);
    }, out);
    float sd = std::sqrt(1.25f);
    ok = ok && std::fabs(out[0] - (-1.5f/sd)) < 1e-3f && std::fabs(out[3] - (1.5f/sd)) < 1e-3f;
    std::fprintf(stderr, "layernorm -> %s\n", ok ? "OK" : "FAIL");

    // gelu_erf parity check vs exact-erf formula x*0.5*(1+erf(x/sqrt(2)))
    // guards against ggml's gelu op being the tanh approximation.
    da::GraphInputPool gpool; std::vector<float> gx = {1.0f, -1.0f, 0.5f, 2.0f}, gout;
    bool gok = be.compute([&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor* t = be.add_graph_input(ctx, gpool, gx.data(), gx.size());
        return da::gelu_erf(ctx, t);
    }, gout);
    for (size_t i = 0; i < gx.size() && gok; ++i) {
        float ref = gx[i] * 0.5f * (1.0f + std::erf(gx[i] / std::sqrt(2.0f)));
        gok = gok && std::fabs(gout[i] - ref) < 1e-3f;
    }
    std::fprintf(stderr, "gelu_erf -> %s (gelu(1)=%.4f gelu(-1)=%.4f)\n",
                 gok ? "OK" : "FAIL", gout.size() ? gout[0] : 0.f, gout.size() > 1 ? gout[1] : 0.f);

    ok = ok && gok;
    return ok ? 0 : 1;
}

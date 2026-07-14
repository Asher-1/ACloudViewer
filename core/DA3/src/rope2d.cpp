#include "rope2d.hpp"
#include <cmath>
namespace da {
RopeTables build_rope_tables(const std::vector<float>& pos_yx, int tokens, int head_dim, float freq){
    RopeTables rt; rt.head_dim = head_dim; rt.tokens = tokens;
    rt.cos.assign((size_t)head_dim*tokens, 0.f);
    rt.sin.assign((size_t)head_dim*tokens, 0.f);
    const int half = head_dim/2;     // per-axis feature size
    const int quart = half/2;        // rope pairs per axis
    for (int t=0;t<tokens;++t){
        float y = pos_yx[(size_t)t*2+0];
        float x = pos_yx[(size_t)t*2+1];
        for (int j=0;j<quart;++j){
            float invf = std::pow(freq, -2.f*(float)j/(float)half);
            float ay = y*invf, ax = x*invf;
            rt.cos[(size_t)t*head_dim + j]            = std::cos(ay);
            rt.cos[(size_t)t*head_dim + j + quart]    = std::cos(ay);
            rt.sin[(size_t)t*head_dim + j]            = std::sin(ay);
            rt.sin[(size_t)t*head_dim + j + quart]    = std::sin(ay);
            rt.cos[(size_t)t*head_dim + half + j]         = std::cos(ax);
            rt.cos[(size_t)t*head_dim + half + j + quart] = std::cos(ax);
            rt.sin[(size_t)t*head_dim + half + j]         = std::sin(ax);
            rt.sin[(size_t)t*head_dim + half + j + quart] = std::sin(ax);
        }
    }
    return rt;
}
void build_rope_inputs(ggml_context* ctx, Backend& be, GraphInputPool& pool,
                       const RopeTables& rt, ggml_tensor*& cosb, ggml_tensor*& sinb){
    const int64_t ne[3] = { rt.head_dim, 1, rt.tokens };
    cosb = be.add_graph_input_nd(ctx, pool, rt.cos.data(), ne, 3);
    sinb = be.add_graph_input_nd(ctx, pool, rt.sin.data(), ne, 3);
}
// rotate_half within each contiguous half: for half H of size head_dim/2,
// rotate_half(H) = cat(-H[quart:], H[:quart]). Combine with cos/sin.
ggml_tensor* apply_rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cosb, ggml_tensor* sinb, int head_dim){
    const int half = head_dim/2, quart = half/2;
    const int64_t heads = x->ne[1], tok = x->ne[2];
    auto part = [&](int off, int len){
        return ggml_cont(ctx, ggml_view_3d(ctx, x, len, heads, tok, x->nb[1], x->nb[2], (size_t)off * x->nb[0]));
    };
    ggml_tensor* ay = part(0, half);        // y-half
    ggml_tensor* ax = part(half, half);     // x-half
    auto rot_half = [&](ggml_tensor* h)->ggml_tensor*{
        ggml_tensor* h0 = ggml_cont(ctx, ggml_view_3d(ctx, h, quart, heads, tok, h->nb[1], h->nb[2], 0));
        ggml_tensor* h1 = ggml_cont(ctx, ggml_view_3d(ctx, h, quart, heads, tok, h->nb[1], h->nb[2], (size_t)quart*h->nb[0]));
        return ggml_concat(ctx, ggml_neg(ctx, h1), h0, 0);   // [-h1, h0]
    };
    ggml_tensor* rot = ggml_concat(ctx, rot_half(ay), rot_half(ax), 0);  // [head_dim,heads,tok]
    return ggml_add(ctx, ggml_mul(ctx, x, cosb), ggml_mul(ctx, rot, sinb));
}
}

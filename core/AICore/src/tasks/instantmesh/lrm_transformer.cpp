// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Implementation of the TriplaneTransformer as an ggml compute graph.
//
// The graph is built fresh per forward following the same pattern as dino.cpp:
// weights are referenced by name from the loaded GGUF, activations allocated
// from a gallocr, and the whole graph scheduled on the selected backend. This
// keeps the port backend-agnostic (CPU/CUDA/Vulkan share one graph).
//
// ggml note: linear(W, x) = mul_mat_exact(ctx, W, x)  (W stored as [in, out]).
#include "lrm_transformer.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "numerics.hpp"

namespace instantmesh {

namespace {

ggml_tensor *get_t(const LrmTransformerModel &m, const char *name) {
    auto it = m.gguf.tensors.find(name);
    if (it == m.gguf.tensors.end()) {
        std::fprintf(stderr, "lrm_transformer: missing tensor '%s'\n", name);
        std::abort();
    }
    return it->second;
}

// linear: y = W x + b   (W is [in, out] in ggml layout).
ggml_tensor *linear(ggml_context *ctx,
                    ggml_tensor *x,
                    ggml_tensor *W,
                    ggml_tensor *b) {
    ggml_tensor *y = mul_mat_exact(ctx, W, x);
    if (b) y = ggml_add(ctx, y, b);
    return y;
}

// LayerNorm with learned affine: norm(x) * w + b.
ggml_tensor *layer_norm(ggml_context *ctx,
                        ggml_tensor *x,
                        ggml_tensor *w,
                        ggml_tensor *b,
                        float eps) {
    ggml_tensor *n = ggml_norm(ctx, x, eps);
    n = ggml_mul(ctx, n, w);
    n = ggml_add(ctx, n, b);
    return n;
}

// in_proj packs q, k, v along the fastest (ne0) dim: qkv is [3D, seq, N].
// Slice out one of the three [D, L] projections as a view into NE0.
ggml_tensor *slice_proj(ggml_context *ctx, ggml_tensor *t, int seg) {
    const int64_t D = t->ne[0] / 3;
    return ggml_view_3d(ctx, t, D, t->ne[1], t->ne[2], t->nb[1], t->nb[2],
                        seg * D * ggml_element_size(t));
}

// Manual multi-head scaled-dot-product attention (q/k/v already projected and
// shaped [D, seq, N]). Returns the attention output [D, q_seq, N].
//   scale = 1/sqrt(head_dim), matching nn.MultiheadAttention.
ggml_tensor *multihead_attn(ggml_context *ctx,
                            ggml_tensor *q,
                            ggml_tensor *k,
                            ggml_tensor *v,
                            int hidden,
                            int n_heads,
                            int q_seq,
                            int kv_seq,
                            int batch) {
    const int head_dim = hidden / n_heads;
    // q/k/v are [hidden, seq, N] with flat = r + c*hidden (r = nh*hd + hd),
    // == torch q[seq, head*hd+hd]. Reinterpret as [hd, nh, seq] then permute to
    // [hd, seq, nh] and make contiguous (same as dino.cpp).
    auto to_head = [&](ggml_tensor *t, int seq) {
        t = ggml_reshape_4d(ctx, t, head_dim, n_heads, seq,
                            batch);            // [hd, nh, seq, B]
        t = ggml_permute(ctx, t, 0, 2, 1, 3);  // [hd, seq, nh, B]
        return ggml_cont(ctx, t);              // contiguous [hd, seq, nh, B]
    };
    q = to_head(q, q_seq);
    k = to_head(k, kv_seq);
    v = to_head(v, kv_seq);
    // ggml_flash_attn_ext (v0.18.1): q/k/v in [hd, seq, nh, B], returns
    // [hd, nh, q_seq, B] which reshapes directly to [hidden, q_seq, B].
    // Manual softmax (mul_mat + soft_max + mul_mat) is no longer needed; the
    // flash path is both faster and numerically identical on this ggml rev.
    float scale = 1.0f / std::sqrt((float)head_dim);
    ggml_tensor *out =
            ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    out = ggml_cont(ctx,
                    out);  // flash output may be non-contiguous for batch>1
    return ggml_reshape_3d(ctx, out, hidden, q_seq, batch);
}

// One BasicTransformerBlock: norm1 -> cross_attn(cond) | norm2 -> self_attn |
// norm3 -> MLP, all residual. cond_k/cond_v are the projected k/v for the
// cross-attention (dependent only on image_feats, shared across layers' query
// sides we still compute per layer since the projections differ).
ggml_tensor *basic_block(ggml_context *ctx,
                         const LrmTransformerModel &m,
                         ggml_tensor *x,
                         ggml_tensor *cond,
                         int i) {
    const LrmTransformerHparams &hp = m.hp;
    const int seq = (int)x->ne[1];
    const int cseq = (int)cond->ne[1];
    const int batch = (int)x->ne[2];
    char name[192];

    // ---- cross-attention ----
    std::snprintf(name, sizeof(name), "layers.%d.norm1.weight", i);
    ggml_tensor *n1w = get_t(m, name);
    std::snprintf(name, sizeof(name), "layers.%d.norm1.bias", i);
    ggml_tensor *n1b = get_t(m, name);
    ggml_tensor *xq = layer_norm(ctx, x, n1w, n1b, hp.layer_norm_eps);

    std::snprintf(name, sizeof(name), "layers.%d.cross_attn.q_proj_weight", i);
    ggml_tensor *cq_w = get_t(m, name);
    std::snprintf(name, sizeof(name), "layers.%d.cross_attn.k_proj_weight", i);
    ggml_tensor *ck_w = get_t(m, name);
    std::snprintf(name, sizeof(name), "layers.%d.cross_attn.v_proj_weight", i);
    ggml_tensor *cv_w = get_t(m, name);
    std::snprintf(name, sizeof(name), "layers.%d.cross_attn.out_proj.weight",
                  i);
    ggml_tensor *co_w = get_t(m, name);

    ggml_tensor *q = mul_mat_exact(ctx, cq_w, xq);    // [D, seq, N]
    ggml_tensor *k = mul_mat_exact(ctx, ck_w, cond);  // [D, cseq, N]
    ggml_tensor *v = mul_mat_exact(ctx, cv_w, cond);  // [D, cseq, N]
    ggml_tensor *ca = multihead_attn(ctx, q, k, v, hp.inner_dim, hp.num_heads,
                                     seq, cseq, batch);
    ca = mul_mat_exact(ctx, co_w, ca);  // out_proj (no bias)
    x = ggml_add(ctx, x, ca);           // residual

    // ---- self-attention ----
    std::snprintf(name, sizeof(name), "layers.%d.norm2.weight", i);
    ggml_tensor *n2w = get_t(m, name);
    std::snprintf(name, sizeof(name), "layers.%d.norm2.bias", i);
    ggml_tensor *n2b = get_t(m, name);
    ggml_tensor *xq2 = layer_norm(ctx, x, n2w, n2b, hp.layer_norm_eps);

    std::snprintf(name, sizeof(name), "layers.%d.self_attn.in_proj_weight", i);
    ggml_tensor *si_w = get_t(m, name);
    std::snprintf(name, sizeof(name), "layers.%d.self_attn.out_proj.weight", i);
    ggml_tensor *so_w = get_t(m, name);

    ggml_tensor *qkv = mul_mat_exact(ctx, si_w, xq2);  // [3D, seq, N]
    ggml_tensor *sq = ggml_cont(ctx, slice_proj(ctx, qkv, 0));
    ggml_tensor *sk = ggml_cont(ctx, slice_proj(ctx, qkv, 1));
    ggml_tensor *sv = ggml_cont(ctx, slice_proj(ctx, qkv, 2));
    ggml_tensor *sa = multihead_attn(ctx, sq, sk, sv, hp.inner_dim,
                                     hp.num_heads, seq, seq, batch);
    sa = mul_mat_exact(ctx, so_w, sa);  // out_proj (no bias)
    x = ggml_add(ctx, x, sa);           // residual

    // ---- MLP: Linear(D->4D) GELU Linear(4D->D) ----
    std::snprintf(name, sizeof(name), "layers.%d.mlp.0.weight", i);
    ggml_tensor *w1 = get_t(m, name);
    std::snprintf(name, sizeof(name), "layers.%d.mlp.0.bias", i);
    ggml_tensor *b1 = get_t(m, name);
    std::snprintf(name, sizeof(name), "layers.%d.norm3.weight", i);
    ggml_tensor *n3w = get_t(m, name);
    std::snprintf(name, sizeof(name), "layers.%d.norm3.bias", i);
    ggml_tensor *n3b = get_t(m, name);
    std::snprintf(name, sizeof(name), "layers.%d.mlp.3.weight", i);
    ggml_tensor *w2 = get_t(m, name);
    std::snprintf(name, sizeof(name), "layers.%d.mlp.3.bias", i);
    ggml_tensor *b2 = get_t(m, name);

    ggml_tensor *y = layer_norm(ctx, x, n3w, n3b, hp.layer_norm_eps);
    y = ggml_gelu_erf(ctx, linear(ctx, y, w1, b1));
    y = linear(ctx, y, w2, b2);
    return ggml_add(ctx, x, y);  // residual
}

}  // namespace

bool lrm_transformer_load(const std::string &path,
                          ggml_backend_t backend,
                          LrmTransformerModel &out,
                          std::string *error) {
    out.backend = backend;
    if (!load_gguf(path, backend, out.gguf, error)) return false;
    const gguf_context *g = out.gguf.gguf;
    out.hp.inner_dim = (int)kv_i32(g, "transformer.inner_dim", 1024);
    out.hp.num_layers = (int)kv_i32(g, "transformer.num_layers", 16);
    out.hp.num_heads = (int)kv_i32(g, "transformer.num_heads", 16);
    out.hp.cond_dim = (int)kv_i32(g, "transformer.cond_dim", 768);
    out.hp.triplane_low_res =
            (int)kv_i32(g, "transformer.triplane_low_res", 32);
    out.hp.triplane_high_res =
            (int)kv_i32(g, "transformer.triplane_high_res", 64);
    out.hp.triplane_dim = (int)kv_i32(g, "transformer.triplane_dim", 80);
    return true;
}

float *lrm_transformer_forward(const LrmTransformerModel &m,
                               const float *image_feats,
                               int N,
                               int n_cond,
                               int *out_planes,
                               int *out_dim,
                               int *out_h,
                               int *out_w) {
    const LrmTransformerHparams &hp = m.hp;
    const int low = hp.triplane_low_res;
    const int high = hp.triplane_high_res;
    const int D = hp.inner_dim;
    const int L = 3 * low * low;  // token length

    ggml_init_params iparams = {/*mem_size=*/256u << 20, /*mem_buffer=*/nullptr,
                                /*no_alloc=*/true};
    ggml_context *ctx = ggml_init(iparams);

    // inputs: cond = image features [N, L_cond, D_cond] and the token sequence
    // x = pos_embed repeated over batch. x carries no external data (it is
    // pos_embed + zero-initialized activation), so only cond is set as input.
    ggml_tensor *cond_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp.cond_dim,
                                             n_cond, N);  // [D_cond, L_cond, N]
    ggml_set_input(cond_t);

    // token sequence: pos_embed [D, L, 1] broadcast over batch.
    ggml_tensor *pos = ggml_reshape_3d(ctx, get_t(m, "pos_embed"), D, L, 1);
    ggml_tensor *x = ggml_reshape_3d(ctx, pos, D, L, N);  // [D, L, N]
    // Activations always run in F32 even when weights are quantized.
    if (x->type != GGML_TYPE_F32) {
        ggml_tensor *f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, L, N);
        x = ggml_cpy(ctx, x, f32);
    }

    // 16 transformer layers.
    for (int i = 0; i < hp.num_layers; ++i) {
        x = basic_block(ctx, m, x, cond_t, i);
    }

    // final LayerNorm.
    ggml_tensor *norm = ggml_norm(ctx, x, hp.layer_norm_eps);
    norm = ggml_add(ctx, ggml_mul(ctx, norm, get_t(m, "norm.weight")),
                    get_t(m, "norm.bias"));
    x = norm;  // [D, L, N]

    // Reorganize [D, 3*low*low, N] -> planes for the transposed conv.
    // Flat layout of x (ne0-fastest): feat + (p*low*low + yy*low + xx)*D +
    // n*L*D. We need the deconv input as [W=low, H=low, Cin=D, batch=3N] with
    // the batch dim = p*N + n. reshape to [D, low, low, 3*N] (p is the slowest
    // of the spatial triple, so ne3 = p*N + n) then permute to [W, H, D, 3N].
    ggml_tensor *planes =
            ggml_reshape_4d(ctx, x, D, low, low, 3 * N);  // [D, xx, yy, p*N+n]
    // ggml permute: new_ne[perm[i]] = a->ne[i]. To get [xx, yy, D] from [D, xx,
    // yy]:
    //   new_ne[0]=xx (a->ne1) -> perm[1]=0; new_ne[1]=yy (a->ne2) -> perm[2]=1;
    //   new_ne[2]=D  (a->ne0) -> perm[0]=2.  => permute(2, 0, 1, 3).
    planes = ggml_cont(
            ctx, ggml_permute(ctx, planes, 2, 0, 1, 3));  // [xx, yy, D, 3N]
    // NOTE: ggml_conv_transpose_2d_p0 only computes the batch-0 slice (it never
    // iterates the ne3/batch dim), so we must run it once per plane with
    // batch=1 and concat the results along the batch dim. It also requires F32
    // input, so convert the (possibly F16-quantized) planes to F32 first.
    if (planes->type != GGML_TYPE_F32) {
        ggml_tensor *f32 =
                ggml_new_tensor_4d(ctx, GGML_TYPE_F32, planes->ne[0],
                                   planes->ne[1], planes->ne[2], planes->ne[3]);
        planes = ggml_cpy(ctx, planes, f32);
    }
    ggml_tensor *dw = get_t(m, "deconv.weight");
    const int64_t nplanes = planes->ne[3];
    ggml_tensor *deconv = nullptr;
    for (int64_t p = 0; p < nplanes; p++) {
        // view of plane p as [xx, yy, D, 1]
        ggml_tensor *plane = ggml_view_4d(
                ctx, planes, planes->ne[0], planes->ne[1], planes->ne[2], 1,
                planes->nb[1], planes->nb[2], planes->nb[3],
                p * (size_t)planes->ne[0] * planes->ne[1] * planes->ne[2] *
                        ggml_element_size(planes));
        ggml_tensor *pd = ggml_conv_transpose_2d_p0(ctx, dw, plane,
                                                    2);  // [W', H', Cout, 1]
        // The single Vulkan patch chain reuses pristine transposed-convolution
        // dispatch; CPU/CUDA/Metal ignore this tag. Preserve upstream
        // staging/output rounding so the SAM3 fast path cannot alter LRM's
        // frozen compatibility gate.
        ggml_set_name(pd, "instantmesh_lrm_convt_");
        ggml_tensor *pdb = ggml_reshape_4d(ctx, get_t(m, "deconv.bias"), 1, 1,
                                           hp.triplane_dim, 1);
        pd = ggml_add(ctx, pd, pdb);
        deconv = (deconv == nullptr) ? pd : ggml_concat(ctx, deconv, pd, 3);
    }
    // deconv output [W', H', Cout, 3N] (ne0 fastest). For N==1 the flat order
    // already matches PyTorch's final [1,3, Cout, high, high] (plane = ne3,
    // Cout = ne2 fastest after spatial). Make it contiguous.
    ggml_tensor *result_t = ggml_cont(ctx, deconv);  // [W', H', Cout, 3N]

    // schedule.
    ggml_gallocr_t alloc =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result_t);
    preserve_cpu_matmul_dispatch(gf, m.backend);
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(cond_t, image_feats, 0,
                            (size_t)N * n_cond * hp.cond_dim * sizeof(float));
    ggml_backend_graph_compute(m.backend, gf);

    size_t nelem = (size_t)3 * N * hp.triplane_dim * high * high;
    float *result = (float *)malloc(nelem * sizeof(float));
    ggml_backend_tensor_get(result_t, result, 0, nelem * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    *out_planes = 3;
    *out_dim = hp.triplane_dim;
    *out_h = high;
    *out_w = high;
    return result;
}

}  // namespace instantmesh
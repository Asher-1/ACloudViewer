// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Implementation of the DINO ViT-B/16 encoder as an ggml compute graph.
//
// The graph is built fresh per forward: weights are referenced by name from the
// loaded GGUF, activations are allocated from a gallocr, and the whole graph is
// scheduled on the selected backend. This keeps the port backend-agnostic — the
// same graph runs on CPU, CUDA or Vulkan once the weights are on that backend.
//
// F32 GEMM reuses the shared STRICT_F32 hint without changing other tasks'
// arithmetic.
#include "dino.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "numerics.hpp"

namespace instantmesh {

namespace {

ggml_tensor *get_t(const DinoModel &m, const char *name) {
    auto it = m.gguf.tensors.find(name);
    if (it == m.gguf.tensors.end()) {
        std::fprintf(stderr, "dino: missing tensor '%s'\n", name);
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

// modulate(x, shift, scale) = x * (1 + scale) + shift   (adaLN, per-token).
ggml_tensor *modulate(ggml_context *ctx,
                      ggml_tensor *x,
                      ggml_tensor *shift,
                      ggml_tensor *scale) {
    // shift/scale are [hidden, B] views (non-contiguous, padded stride from the
    // 4*hidden adaLN output); make them contiguous before reshaping to
    // [hidden, 1, B] so the seq dim broadcasts.
    ggml_tensor *s = ggml_cont(ctx, scale);
    ggml_tensor *sh = ggml_cont(ctx, shift);
    s = ggml_reshape_3d(ctx, s, s->ne[0], 1, s->ne[1]);
    sh = ggml_reshape_3d(ctx, sh, sh->ne[0], 1, sh->ne[1]);
    ggml_tensor *out = ggml_add(ctx, ggml_add(ctx, ggml_mul(ctx, x, s), x), sh);
    return out;
}

// One adaLN DiT transformer layer.
ggml_tensor *dino_layer(ggml_context *ctx,
                        const DinoModel &m,
                        ggml_tensor *hidden,
                        ggml_tensor *adaln,
                        int i) {
    const DinoHparams &hp = m.hp;
    char name[192];

    // adaLN modulation: SiLU -> Linear(hidden -> 4*hidden, zero-init).
    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.adaLN_modulation.1.weight", i);
    ggml_tensor *w_ada = get_t(m, name);
    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.adaLN_modulation.1.bias", i);
    ggml_tensor *b_ada = get_t(m, name);
    ggml_tensor *mod =
            linear(ctx, ggml_silu(ctx, adaln), w_ada, b_ada);  // [4*hidden, B]
    ggml_tensor *parts[4];
    for (int p = 0; p < 4; ++p) {
        parts[p] =
                ggml_view_2d(ctx, mod, hp.hidden_size, mod->ne[1], mod->nb[1],
                             p * hp.hidden_size * ggml_element_size(mod));
    }

    // LayerNorm before + modulate -> self-attention.
    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.layernorm_before.weight", i);
    ggml_tensor *lnb_w = get_t(m, name);
    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.layernorm_before.bias", i);
    ggml_tensor *lnb_b = get_t(m, name);
    ggml_tensor *x = layer_norm(ctx, hidden, lnb_w, lnb_b, hp.layer_norm_eps);
    x = modulate(ctx, x, parts[0], parts[1]);

    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.attention.attention.query.weight", i);
    ggml_tensor *wq = get_t(m, name);
    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.attention.attention.key.weight", i);
    ggml_tensor *wk = get_t(m, name);
    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.attention.attention.value.weight", i);
    ggml_tensor *wv = get_t(m, name);
    ggml_tensor *q = mul_mat_exact(ctx, wq, x);
    ggml_tensor *k = mul_mat_exact(ctx, wk, x);
    ggml_tensor *v = mul_mat_exact(ctx, wv, x);
    // qkv_bias = True for dino-vitb16.
    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.attention.attention.query.bias", i);
    q = ggml_add(ctx, q, get_t(m, name));
    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.attention.attention.key.bias", i);
    k = ggml_add(ctx, k, get_t(m, name));
    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.attention.attention.value.bias", i);
    v = ggml_add(ctx, v, get_t(m, name));

    // Manual softmax attention. ggml v0.18.0's flash_attn_ext is buggy on
    // head_dim=64 (see tests/test_flashattn.cpp), so we build the reference
    // matmul+softmax+matmul path instead. q,k,v are [head_dim, seq, n_head, B].
    const int head_dim = hp.hidden_size / hp.num_attention_heads;
    const int seq = (int)hidden->ne[1];
    const int batch = (int)hidden->ne[2];
    // q,k,v from mul_mat are [hidden, seq, B] with flat = r + c*hidden
    // (r = nh*hd + hd) == torch q[seq, head, hd] flat. Reinterpreting as
    // [hd, nh, seq] (reshape_4d) keeps that flat, then permute to [hd, seq, nh]
    // and make contiguous so mul_mat reads the correct head-grouped values.
    auto to_head = [&](ggml_tensor *t) {
        t = ggml_reshape_4d(ctx, t, head_dim, hp.num_attention_heads, seq,
                            batch);            // [hd, nh, seq, B]
        t = ggml_permute(ctx, t, 0, 2, 1, 3);  // [hd, seq, nh, B]
        return ggml_cont(ctx, t);              // contiguous [hd, seq, nh, B]
    };
    q = to_head(q);
    k = to_head(k);
    v = to_head(v);
    // ggml_flash_attn_ext (v0.18.1): q/k/v in [hd, seq, nh, B], returns
    // [hd, nh, seq, B] directly in the layout we need. Replaces the manual
    // matmul+softmax+matmul path (flash_attn was buggy on head_dim=64 in
    // v0.18.0; fixed in v0.18.1 and verified numerically identical).
    float scale = 1.0f / std::sqrt((float)head_dim);
    ggml_tensor *attn = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f,
                                            0.0f);  // [hd, nh, seq, B]
    attn = ggml_cont(ctx,
                     attn);  // flash output may be non-contiguous for batch>1
    attn = ggml_reshape_3d(ctx, attn, hp.hidden_size, seq, batch);

    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.attention.output.dense.weight", i);
    ggml_tensor *wo = get_t(m, name);
    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.attention.output.dense.bias", i);
    ggml_tensor *bo = get_t(m, name);
    attn = linear(ctx, attn, wo, bo);
    ggml_tensor *hidden2 = ggml_add(ctx, attn, hidden);  // first residual

    // LayerNorm after + modulate + MLP (GELU).
    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.layernorm_after.weight", i);
    ggml_tensor *lna_w = get_t(m, name);
    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.layernorm_after.bias", i);
    ggml_tensor *lna_b = get_t(m, name);
    ggml_tensor *y = layer_norm(ctx, hidden2, lna_w, lna_b, hp.layer_norm_eps);
    y = modulate(ctx, y, parts[2], parts[3]);

    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.intermediate.dense.weight", i);
    ggml_tensor *wi = get_t(m, name);
    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.intermediate.dense.bias", i);
    ggml_tensor *bi = get_t(m, name);
    y = ggml_gelu(ctx, linear(ctx, y, wi, bi));

    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.output.dense.weight", i);
    ggml_tensor *wmlp = get_t(m, name);
    std::snprintf(name, sizeof(name),
                  "model.encoder.layer.%d.output.dense.bias", i);
    ggml_tensor *bmlp = get_t(m, name);
    y = linear(ctx, y, wmlp, bmlp);

    return ggml_add(ctx, y, hidden2);  // second residual
}

}  // namespace

bool dino_load(const std::string &path,
               ggml_backend_t backend,
               DinoModel &out,
               std::string *error) {
    out.backend = backend;
    if (!load_gguf(path, backend, out.gguf, error)) return false;
    const gguf_context *g = out.gguf.gguf;
    out.hp.hidden_size = (int)kv_i32(g, "dino.hidden_size", 768);
    out.hp.num_hidden_layers = (int)kv_i32(g, "dino.num_hidden_layers", 12);
    out.hp.num_attention_heads = (int)kv_i32(g, "dino.num_attention_heads", 12);
    out.hp.image_size = (int)kv_i32(g, "dino.image_size", 224);
    out.hp.patch_size = (int)kv_i32(g, "dino.patch_size", 16);
    // Reject incomplete models before graph construction; get_t/division must
    // not abort a C ABI caller.
    const auto fail = [&](const std::string &message) {
        if (error) *error = message;
        out.gguf.unload();
        out.backend = nullptr;
        return false;
    };
    const auto &hp = out.hp;
    if (hp.hidden_size <= 0 || hp.num_hidden_layers <= 0 ||
        hp.num_attention_heads <= 0 ||
        hp.hidden_size % hp.num_attention_heads != 0 || hp.image_size <= 0 ||
        hp.patch_size <= 0 || hp.image_size % hp.patch_size != 0) {
        return fail("dino: invalid hparams");
    }
    const auto required = [&](const std::string &name) {
        return out.gguf.tensors.find(name) != out.gguf.tensors.end();
    };
    for (const char *name :
         {"camera_embedder.0.weight", "camera_embedder.0.bias",
          "camera_embedder.2.weight", "camera_embedder.2.bias",
          "model.embeddings.patch_embeddings.projection.weight",
          "model.embeddings.patch_embeddings.projection.bias",
          "model.embeddings.cls_token", "model.embeddings.position_embeddings",
          "model.layernorm.weight", "model.layernorm.bias"}) {
        if (!required(name))
            return fail(std::string("dino: missing tensor '") + name + "'");
    }
    for (int i = 0; i < hp.num_hidden_layers; ++i) {
        const std::string prefix =
                "model.encoder.layer." + std::to_string(i) + ".";
        for (const char *part :
             {"adaLN_modulation.1", "layernorm_before", "layernorm_after",
              "attention.attention.query", "attention.attention.key",
              "attention.attention.value", "attention.output.dense",
              "intermediate.dense", "output.dense"}) {
            for (const char *suffix : {".weight", ".bias"}) {
                const std::string name = prefix + part + suffix;
                if (!required(name))
                    return fail("dino: missing tensor '" + name + "'");
            }
        }
    }
    return true;
}

float *dino_encode(const DinoModel &m,
                   const float *image,
                   const float *camera,
                   int B,
                   int H,
                   int W,
                   int *out_seq,
                   int *out_hidden) {
    const DinoHparams &hp = m.hp;
    const int ph = H / hp.patch_size;
    const int pw = W / hp.patch_size;
    const int n_seq = 1 + ph * pw;
    const int hidden = hp.hidden_size;

    ggml_init_params iparams = {/*mem_size=*/32u << 20, /*mem_buffer=*/nullptr,
                                /*no_alloc=*/true};
    ggml_context *ctx = ggml_init(iparams);

    // inputs (image [W,H,3,B], camera [16,B]).
    ggml_tensor *image_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, 3, B);
    ggml_tensor *cam_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, B);
    ggml_set_input(image_t);
    ggml_set_input(cam_t);

    // camera embedder: Linear(16->768) -> SiLU -> Linear(768->768).
    ggml_tensor *adaln = ggml_silu(
            ctx, linear(ctx, cam_t, get_t(m, "camera_embedder.0.weight"),
                        get_t(m, "camera_embedder.0.bias")));
    adaln = linear(ctx, adaln, get_t(m, "camera_embedder.2.weight"),
                   get_t(m, "camera_embedder.2.bias"));  // [hidden, B]

    // patch embeddings: conv2d(3->768, k=16, s=16). Kernel [768,3,16,16] ->
    // ggml [16,16,3,768].
    ggml_tensor *proj =
            get_t(m, "model.embeddings.patch_embeddings.projection.weight");
    // Patch embedding conv. For F32 weights we build a manual im2col+mul_mat
    // graph with a F32 im2col: ggml_conv_2d internally quantizes its im2col to
    // F16 (a performance tradeoff that loses ~6e-3 of precision on the first
    // layer, which we don't want for F32 parity). For quantized weights
    // (F16/Q8) Retain F16 im2col; weight quantization does not prove F16
    // accumulation meets accuracy gates. Kernel layout [16,16,3,768]
    // (KW,KH,IC,OC); image [W,H,3,N].
    ggml_tensor *conv;
    if (proj->type == GGML_TYPE_F32) {
        ggml_tensor *im2col =
                ggml_im2col(ctx, proj, image_t, 16, 16, 0, 0, 1, 1, true,
                            GGML_TYPE_F32);  // ne = {IC*KH*KW, OW, OH, N}
        // mul_mat(a, b): dst[i,j] = sum_d a[d,i]*b[d,j], a.ne0 must == b.ne0.
        ggml_tensor *im2col_2d =
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],
                                im2col->ne[3] * im2col->ne[2] *
                                        im2col->ne[1]);  // [IC*KH*KW, N*OH*OW]
        ggml_tensor *k_2d = ggml_reshape_2d(
                ctx, proj, proj->ne[0] * proj->ne[1] * proj->ne[2],
                proj->ne[3]);                                  // [IC*KH*KW, OC]
        ggml_tensor *r = mul_mat_exact(ctx, im2col_2d, k_2d);  // [N*OH*OW, OC]
        r = ggml_cont(
                ctx,
                ggml_permute(ctx,
                             ggml_reshape_4d(ctx, r, im2col->ne[1],
                                             im2col->ne[2], im2col->ne[3],
                                             proj->ne[3]),  // [OW, OH, N, OC]
                             0, 1, 3,
                             2));  // -> [OW, OH, OC, N], same as ggml_conv_2d
        conv = r;
    } else {
        // Retain official conv_2d F16 im2col/layout; explicitly accumulate in
        // F32 to preserve representable small terms.
        ggml_tensor *columns = ggml_im2col(
                ctx, proj, image_t, 16, 16, 0, 0, 1, 1, true,
                proj->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);
        ggml_tensor *product = mul_mat_exact(
                ctx,
                ggml_reshape_2d(
                        ctx, columns, columns->ne[0],
                        columns->ne[1] * columns->ne[2] * columns->ne[3]),
                ggml_reshape_2d(ctx, proj,
                                proj->ne[0] * proj->ne[1] * proj->ne[2],
                                proj->ne[3]));
        GGML_ASSERT(ggml_prec_set_acc(product, GGML_PREC_F32));
        conv = ggml_reshape_4d(ctx, product, columns->ne[1], columns->ne[2],
                               columns->ne[3], proj->ne[3]);
        conv = ggml_cont(ctx, ggml_permute(ctx, conv, 0, 1, 3, 2));
    }
    // conv is [OW, OH, OC, N] (ne0=OW, ne1=OH, ne2=OC, ne3=N). permute(1,2,0,3)
    // -> ne0=OC (from ne1), ne1=OW (from ne2), ne2=OH (from ne0) => [OC, OW,
    // OH, N]. After reshape_3d to [hidden, seq, B] this yields [seq=(ow+OH*oh),
    // hidden] with hidden fastest, row-major spatial order matching PyTorch's
    // patch_proj(image).flatten(2).transpose(1,2).
    conv = ggml_cont(ctx, ggml_permute(ctx, conv, 1, 2, 0, 3));
    ggml_tensor *patch = ggml_reshape_3d(ctx, conv, hidden, ph * pw, B);
    // conv bias: ggml_conv_2d computes only the correlation; add the
    // per-channel bias (stored [768]) broadcast over all spatial positions.
    ggml_tensor *conv_b =
            get_t(m, "model.embeddings.patch_embeddings.projection.bias");
    conv_b = ggml_reshape_3d(ctx, conv_b, hidden, 1, 1);
    patch = ggml_add(ctx, patch, conv_b);

    // prepend CLS token (torch [1,1,768] -> ggml [768,1,1], broadcast over
    // batch).
    ggml_tensor *cls = get_t(m, "model.embeddings.cls_token");
    cls = ggml_repeat(ctx, cls,
                      ggml_new_tensor_3d(ctx, cls->type, hidden, 1, B));
    ggml_tensor *hidden_states =
            ggml_concat(ctx, cls, patch, 1);  // [hidden, 1+ph*pw, B]

    // positional encoding. 224x224 -> stored [768,197,1]; broadcast over batch.
    ggml_tensor *posemb = get_t(m, "model.embeddings.position_embeddings");
    ggml_tensor *pos = ggml_reshape_3d(ctx, posemb, hidden, n_seq, 1);
    if (H != hp.image_size || W != hp.image_size) {
        std::fprintf(stderr,
                     "dino: bicubic pos-encoding interpolation for %dx%d not "
                     "yet implemented; falling back to stored (%dx%d)\n",
                     W, H, hp.image_size, hp.image_size);
    }
    hidden_states = ggml_add(ctx, hidden_states, pos);

    // 12 transformer layers.
    for (int i = 0; i < hp.num_hidden_layers; ++i) {
        hidden_states = dino_layer(ctx, m, hidden_states, adaln, i);
    }

    // output: hidden_states is [hidden(ne0), seq(ne1), batch(ne2)]
    // feature-major. Linear read is already [batch, seq, hidden] (hidden
    // fastest), so no permute is needed.
    hidden_states =
            layer_norm(ctx, hidden_states, get_t(m, "model.layernorm.weight"),
                       get_t(m, "model.layernorm.bias"), 1e-12f);
    ggml_tensor *result_t = ggml_cont(ctx, hidden_states);

    // schedule.
    ggml_gallocr_t alloc =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result_t);
    preserve_cpu_matmul_dispatch(gf, m.backend);

    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(image_t, image, 0,
                            (size_t)B * 3 * H * W * sizeof(float));
    ggml_backend_tensor_set(cam_t, camera, 0, (size_t)B * 16 * sizeof(float));
    ggml_backend_graph_compute(m.backend, gf);

    float *result = (float *)malloc((size_t)B * n_seq * hidden * sizeof(float));
    ggml_backend_tensor_get(result_t, result, 0,
                            (size_t)B * n_seq * hidden * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    *out_seq = n_seq;
    *out_hidden = hidden;
    return result;
}

}  // namespace instantmesh
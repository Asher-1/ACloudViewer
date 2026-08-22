// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/rmbg/swin_backbone.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "ggml.h"
#include "gguf.h"

namespace rmbg {

struct Q8Block {
    ggml_fp16_t d;
    int8_t qs[32];
};
static_assert(sizeof(Q8Block) == 34, "GGML Q8_0 block layout changed");

static bool decode_f32(const WeightTensor &tensor, std::vector<float> &out) {
    size_t n = 1;
    for (int64_t dim : tensor.shape) n *= (size_t)dim;
    out.resize(n);
    if (tensor.type == GGML_TYPE_F32 &&
        tensor.bytes.size() == n * sizeof(float)) {
        std::memcpy(out.data(), tensor.bytes.data(), tensor.bytes.size());
        return true;
    }
    if (tensor.type == GGML_TYPE_F16 &&
        tensor.bytes.size() == n * sizeof(ggml_fp16_t)) {
        const auto *src =
                reinterpret_cast<const ggml_fp16_t *>(tensor.bytes.data());
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(src[i]);
        return true;
    }
    if (tensor.type == GGML_TYPE_Q8_0 && n % 32 == 0 &&
        tensor.bytes.size() == n / 32 * sizeof(Q8Block)) {
        const auto *src =
                reinterpret_cast<const Q8Block *>(tensor.bytes.data());
        for (size_t block = 0; block < n / 32; ++block) {
            const float scale = ggml_fp16_to_fp32(src[block].d);
            for (size_t lane = 0; lane < 32; ++lane)
                out[block * 32 + lane] = scale * src[block].qs[lane];
        }
        return true;
    }
    out.clear();
    return false;
}

bool WeightMap::load_gguf(const char *path, std::string &err) {
    tensors.clear();
    f32_cache.clear();
    return merge_gguf(path, err);
}

bool WeightMap::merge_gguf(const char *path, std::string &err) {
    ggml_context *ctx = nullptr;
    gguf_init_params p{false, &ctx};
    gguf_context *g = gguf_init_from_file(path, p);
    if (!g) {
        err = "gguf_init_from_file failed";
        return false;
    }
    int nt = gguf_get_n_tensors(g);
    for (int i = 0; i < nt; ++i) {
        const char *name = gguf_get_tensor_name(g, i);
        ggml_tensor *t = ggml_get_tensor(ctx, name);
        WeightTensor tensor;
        tensor.type = t->type;
        for (int d = ggml_n_dims(t) - 1; d >= 0; --d)
            tensor.shape.push_back(t->ne[d]);
        tensor.bytes.resize(ggml_nbytes(t));
        std::memcpy(tensor.bytes.data(), t->data, tensor.bytes.size());
        tensors[name] = std::move(tensor);
    }
    gguf_free(g);
    ggml_free(ctx);
    return true;
}

const std::vector<float> *WeightMap::get_f32(const char *key) const {
    auto cached = f32_cache.find(key);
    if (cached != f32_cache.end()) return &cached->second;
    const WeightTensor *tensor = get_tensor(key);
    if (!tensor) return nullptr;
    std::vector<float> decoded;
    if (!decode_f32(*tensor, decoded)) return nullptr;
    return &f32_cache.emplace(key, std::move(decoded)).first->second;
}

const WeightTensor *WeightMap::get_tensor(const char *key) const {
    auto it = tensors.find(key);
    return it == tensors.end() ? nullptr : &it->second;
}

const std::vector<int64_t> *WeightMap::get_shape(const char *key) const {
    const WeightTensor *tensor = get_tensor(key);
    return tensor ? &tensor->shape : nullptr;
}

static bool require_w(const WeightMap &w,
                      const std::string &k,
                      const std::vector<float> *&out,
                      std::string &err) {
    out = w.get_f32(k.c_str());
    if (!out) {
        err = "missing weight: " + k;
        return false;
    }
    return true;
}

bool SwinBackboneForward::load_block_weights(const WeightMap &w,
                                             const std::string &prefix,
                                             int shift_size,
                                             SwinBlockWeights &out,
                                             std::string &err) {
    out.shift_size = shift_size;
    const std::vector<float> *p = nullptr;
    auto need = [&](const char *suffix) -> std::string {
        return prefix + suffix;
    };

    if (!require_w(w, need("norm1_weight"), p, err)) return false;
    out.n1w = *p;
    if (!require_w(w, need("norm1_bias"), p, err)) return false;
    out.n1b = *p;
    if (!require_w(w, need("norm2_weight"), p, err)) return false;
    out.n2w = *p;
    if (!require_w(w, need("norm2_bias"), p, err)) return false;
    out.n2b = *p;
    if (!require_w(w, need("attn_qkv_weight"), p, err)) return false;
    out.qkv_w = *p;
    if (!require_w(w, need("attn_qkv_bias"), p, err)) return false;
    out.qkv_b = *p;
    if (!require_w(w, need("attn_proj_weight"), p, err)) return false;
    out.proj_w = *p;
    if (!require_w(w, need("attn_proj_bias"), p, err)) return false;
    out.proj_b = *p;
    if (!require_w(w, need("mlp_fc1_weight"), p, err)) return false;
    out.mlp1_w = *p;
    if (!require_w(w, need("mlp_fc1_bias"), p, err)) return false;
    out.mlp1_b = *p;
    if (!require_w(w, need("mlp_fc2_weight"), p, err)) return false;
    out.mlp2_w = *p;
    if (!require_w(w, need("mlp_fc2_bias"), p, err)) return false;
    out.mlp2_b = *p;
    if (!require_w(w, need("attn_relative_position_bias_table"), p, err))
        return false;
    out.rpb_table = *p;
    out.dim = (int)out.n1w.size();
    const int rpb_rows = (2 * 12 - 1) * (2 * 12 - 1);
    out.num_heads =
            out.rpb_table.empty() ? 6 : (int)(out.rpb_table.size() / rpb_rows);

    if (require_w(w, need("attn_relative_position_index"), p, err)) {
        out.rel_pos_idx.resize(p->size());
        for (size_t i = 0; i < p->size(); ++i)
            out.rel_pos_idx[i] = (int)(*p)[i];
    } else {
        const int ws = 12;
        const int n = ws * ws;
        out.rel_pos_idx.resize((size_t)n * n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                int ph = i / ws, pw = i % ws;
                int qh = j / ws, qw = j % ws;
                int rh = qh - ph + ws - 1;
                int rw = qw - pw + ws - 1;
                out.rel_pos_idx[(size_t)i * n + j] = rh * (2 * ws - 1) + rw;
            }
        }
    }
    return true;
}

bool SwinBackboneForward::load_patch_merge_weights(const WeightMap &w,
                                                   int stage,
                                                   PatchMergeWeights &pm,
                                                   std::string &err) {
    const std::vector<float> *p = nullptr;
    const std::string prefix =
            "bb_layers_" + std::to_string(stage) + "_downsample_";
    if (!require_w(w, prefix + "norm_weight", p, err)) return false;
    pm.nw = *p;
    pm.in_dim = (int)pm.nw.size() / 4;
    if (!require_w(w, prefix + "norm_bias", p, err)) return false;
    pm.nb = *p;
    if (!require_w(w, prefix + "reduction_weight", p, err)) return false;
    pm.red_w = *p;
    return true;
}

bool SwinBackboneForward::try_load_stage_blocks(
        const WeightMap &w,
        int stage,
        int depth,
        int ws,
        std::vector<SwinBlockWeights> &out,
        std::string &err) {
    for (int i = 0; i < depth; ++i) {
        const std::string prefix = "bb_layers_" + std::to_string(stage) +
                                   "_blocks_" + std::to_string(i) + "_";
        if (!w.get_f32((prefix + "norm1_weight").c_str())) {
            return i > 0;
        }
        SwinBlockWeights bw;
        const int shift = (i % 2 == 0) ? 0 : ws / 2;
        if (!SwinBackboneForward::load_block_weights(w, prefix, shift, bw, err))
            return false;
        out.push_back(std::move(bw));
    }
    return true;
}

bool SwinBackboneForward::init(const WeightMap &w, std::string &err) {
    const std::vector<float> *p = nullptr;
    if (!require_w(w, "bb_patch_embed_proj_weight", p, err)) return false;
    pe_w_ = *p;
    if (!require_w(w, "bb_patch_embed_proj_bias", p, err)) return false;
    pe_b_ = *p;
    if (!require_w(w, "bb_patch_embed_norm_weight", p, err)) return false;
    pe_nw_ = *p;
    if (!require_w(w, "bb_patch_embed_norm_bias", p, err)) return false;
    pe_nb_ = *p;
    pe_out_c_ = cfg_.embed_dim;

    blocks_.clear();
    merges_.clear();
    const int ws = cfg_.window_size;
    static const int k_depths[] = {2, 2, 18, 2};
    for (int stage = 0; stage < 4; ++stage) {
        const std::string b0 =
                "bb_layers_" + std::to_string(stage) + "_blocks_0_norm1_weight";
        if (!w.get_f32(b0.c_str())) break;
        if (!try_load_stage_blocks(w, stage, k_depths[stage], ws, blocks_, err))
            return false;
        PatchMergeWeights pm;
        if (load_patch_merge_weights(w, stage, pm, err)) {
            merges_.push_back(std::move(pm));
        }
    }
    if (blocks_.empty()) {
        err = "no Swin blocks loaded";
        return false;
    }
    has_bb_norm_ = true;
    for (int i = 0; i < 4; ++i) {
        const std::string p = "bb_norm" + std::to_string(i) + "_";
        const std::vector<float> *wt = w.get_f32((p + "weight").c_str());
        const std::vector<float> *bs = w.get_f32((p + "bias").c_str());
        if (!wt || !bs) {
            has_bb_norm_ = false;
            break;
        }
        bb_norm_w_[i] = *wt;
        bb_norm_b_[i] = *bs;
    }
    return true;
}

float SwinBackboneForward::gelu(float x) {
    return 0.5f * x * (1.f + std::erf(x * 0.7071067811865475f));
}

void SwinBackboneForward::layer_norm(std::vector<float> &x,
                                     int rows,
                                     int cols,
                                     const std::vector<float> &gamma,
                                     const std::vector<float> &beta,
                                     float eps) {
    for (int r = 0; r < rows; ++r) {
        float *row = x.data() + (size_t)r * cols;
        double mean = 0, var = 0;
        for (int c = 0; c < cols; ++c) mean += row[c];
        mean /= cols;
        for (int c = 0; c < cols; ++c) {
            double d = row[c] - mean;
            var += d * d;
        }
        var /= cols;
        double inv = 1.0 / std::sqrt(var + eps);
        for (int c = 0; c < cols; ++c)
            row[c] = (float)((row[c] - mean) * inv * gamma[c] + beta[c]);
    }
}

void SwinBackboneForward::linear(const std::vector<float> &in,
                                 int M,
                                 int K,
                                 int N,
                                 const std::vector<float> &w,
                                 const std::vector<float> *b,
                                 std::vector<float> &out) {
    out.assign((size_t)M * N, 0.f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float s = b ? (*b)[n] : 0.f;
            for (int k = 0; k < K; ++k)
                s += in[(size_t)m * K + k] * w[(size_t)n * K + k];
            out[(size_t)m * N + n] = s;
        }
    }
}

void SwinBackboneForward::conv2d_nchw(const std::vector<float> &in,
                                      int N,
                                      int C,
                                      int H,
                                      int W,
                                      const std::vector<float> &w,
                                      int OC,
                                      int KH,
                                      int KW,
                                      int stride,
                                      const std::vector<float> *bias,
                                      std::vector<float> &out) {
    int OH = H / stride, OW = W / stride;
    out.assign((size_t)N * OC * OH * OW, 0.f);
    auto idx = [](int n, int c, int h, int w, int C, int H, int W) {
        return (((size_t)n * C + c) * H + h) * W + w;
    };
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float s = bias ? (*bias)[oc] : 0.f;
                    for (int ic = 0; ic < C; ++ic) {
                        for (int kh = 0; kh < KH; ++kh) {
                            for (int kw = 0; kw < KW; ++kw) {
                                s += in[idx(n, ic, oh * stride + kh,
                                            ow * stride + kw, C, H, W)] *
                                     w[(((size_t)oc * C + ic) * KH + kh) * KW +
                                       kw];
                            }
                        }
                    }
                    out[idx(n, oc, oh, ow, OC, OH, OW)] = s;
                }
            }
        }
    }
}

void SwinBackboneForward::spatial_roll(
        std::vector<float> &feat, int B, int H, int W, int C, int dh, int dw) {
    std::vector<float> tmp(feat.size());
    dh = ((dh % H) + H) % H;
    dw = ((dw % W) + W) % W;
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int sh = (h - dh + H) % H;
                int sw = (w - dw + W) % W;
                for (int c = 0; c < C; ++c) {
                    tmp[(((size_t)b * H + h) * W + w) * C + c] =
                            feat[(((size_t)b * H + sh) * W + sw) * C + c];
                }
            }
        }
    }
    feat = std::move(tmp);
}

void SwinBackboneForward::build_shift_attn_mask(
        int H, int W, int ws, int shift_size, std::vector<float> &mask_nW_N_N) {
    int pad_r = (ws - W % ws) % ws;
    int pad_b = (ws - H % ws) % ws;
    int Hp = H + pad_b, Wp = W + pad_r;
    const int nW = (Hp / ws) * (Wp / ws);
    const int N = ws * ws;
    mask_nW_N_N.assign((size_t)nW * N * N, 0.f);

    std::vector<int> img((size_t)Hp * Wp, 0);
    int cnt = 0;
    const int h_bounds[4] = {0, Hp - ws, Hp - shift_size, Hp};
    const int w_bounds[4] = {0, Wp - ws, Wp - shift_size, Wp};
    for (int hs = 0; hs < 3; ++hs) {
        for (int ws_ = 0; ws_ < 3; ++ws_) {
            for (int h = h_bounds[hs]; h < h_bounds[hs + 1]; ++h)
                for (int w = w_bounds[ws_]; w < w_bounds[ws_ + 1]; ++w)
                    img[(size_t)h * Wp + w] = cnt;
            ++cnt;
        }
    }

    for (int wi = 0; wi < nW; ++wi) {
        int wh = (wi / (Wp / ws)) * ws;
        int ww = (wi % (Wp / ws)) * ws;
        std::vector<int> win(N);
        for (int i = 0; i < ws; ++i)
            for (int j = 0; j < ws; ++j)
                win[(size_t)i * ws + j] = img[(size_t)(wh + i) * Wp + (ww + j)];

        float *m = mask_nW_N_N.data() + (size_t)wi * N * N;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m[i * N + j] = (win[i] == win[j]) ? 0.f : -100.f;
    }
}

void SwinBackboneForward::window_attention(std::vector<float> &x,
                                           int BnW,
                                           int N,
                                           int C,
                                           const SwinBlockWeights &bw,
                                           const std::vector<float> *shift_mask,
                                           int mask_win_idx) const {
    const int nh = bw.num_heads;
    const int hd = C / nh;
    const float scale = 1.f / std::sqrt((float)hd);
    const int M = BnW * N;

    std::vector<float> qkv;
    linear(x, M, C, 3 * C, bw.qkv_w, &bw.qkv_b, qkv);

    std::vector<float> out((size_t)M * C);
    std::vector<float> attn((size_t)N * N);

    for (int bw_i = 0; bw_i < BnW; ++bw_i) {
        const float *win_mask =
                shift_mask ? shift_mask->data() +
                                     (size_t)(mask_win_idx + bw_i) * N * N
                           : nullptr;
        for (int h = 0; h < nh; ++h) {
            float *a = attn.data();
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    float s = 0.f;
                    for (int d = 0; d < hd; ++d) {
                        const float *qi =
                                qkv.data() +
                                ((size_t)(bw_i * N + i) * 3 * C + h * hd + d);
                        const float *kj =
                                qkv.data() + ((size_t)(bw_i * N + j) * 3 * C +
                                              C + h * hd + d);
                        s += (*qi) * scale * (*kj);
                    }
                    int ridx = bw.rel_pos_idx[(size_t)i * N + j];
                    s += bw.rpb_table[(size_t)ridx * nh + h];
                    if (win_mask) s += win_mask[i * N + j];
                    a[i * N + j] = s;
                }
            }
            for (int i = 0; i < N; ++i) {
                float *row = a + i * N;
                float mx = row[0];
                for (int j = 1; j < N; ++j) mx = std::max(mx, row[j]);
                float sum = 0.f;
                for (int j = 0; j < N; ++j) {
                    row[j] = std::exp(row[j] - mx);
                    sum += row[j];
                }
                for (int j = 0; j < N; ++j) row[j] /= sum;
            }
            for (int i = 0; i < N; ++i) {
                for (int d = 0; d < hd; ++d) {
                    float s = 0.f;
                    for (int j = 0; j < N; ++j) {
                        const float *vj =
                                qkv.data() + ((size_t)(bw_i * N + j) * 3 * C +
                                              2 * C + h * hd + d);
                        s += a[i * N + j] * (*vj);
                    }
                    out[((size_t)(bw_i * N + i) * C + h * hd + d)] = s;
                }
            }
        }
    }
    x = std::move(out);
    std::vector<float> proj;
    linear(x, M, C, C, bw.proj_w, &bw.proj_b, proj);
    x = std::move(proj);
}

void SwinBackboneForward::swin_block(std::vector<float> &tokens,
                                     int B,
                                     int H,
                                     int W,
                                     const SwinBlockWeights &bw) const {
    const int C = bw.dim;
    const int ws = cfg_.window_size;
    const int L = H * W;
    std::vector<float> shortcut = tokens;

    layer_norm(tokens, B * L, C, bw.n1w, bw.n1b, cfg_.eps);

    int pad_r = (ws - W % ws) % ws;
    int pad_b = (ws - H % ws) % ws;
    int Hp = H + pad_b, Wp = W + pad_r;

    std::vector<float> feat((size_t)B * Hp * Wp * C);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int c = 0; c < C; ++c)
                    feat[(((size_t)b * Hp + h) * Wp + w) * C + c] =
                            tokens[((size_t)b * L + h * W + w) * C + c];
    }

    if (bw.shift_size > 0)
        spatial_roll(feat, B, Hp, Wp, C, -bw.shift_size, -bw.shift_size);

    const int nW = (Hp / ws) * (Wp / ws);
    const int N = ws * ws;
    std::vector<float> win((size_t)B * nW * N * C);
    for (int b = 0; b < B; ++b) {
        for (int wi = 0; wi < nW; ++wi) {
            int wh = (wi / (Wp / ws)) * ws;
            int ww = (wi % (Wp / ws)) * ws;
            for (int i = 0; i < ws; ++i) {
                for (int j = 0; j < ws; ++j) {
                    for (int c = 0; c < C; ++c) {
                        win[(((size_t)b * nW + wi) * N + i * ws + j) * C + c] =
                                feat[(((size_t)b * Hp + wh + i) * Wp + ww + j) *
                                             C +
                                     c];
                    }
                }
            }
        }
    }

    std::vector<float> shift_mask;
    const std::vector<float> *mask_ptr = nullptr;
    if (bw.shift_size > 0) {
        build_shift_attn_mask(H, W, ws, bw.shift_size, shift_mask);
        mask_ptr = &shift_mask;
    }

    for (int b = 0; b < B; ++b) {
        for (int wi = 0; wi < nW; ++wi) {
            std::vector<float> wslice((size_t)N * C);
            for (int t = 0; t < N; ++t)
                for (int c = 0; c < C; ++c)
                    wslice[(size_t)t * C + c] =
                            win[(((size_t)b * nW + wi) * N + t) * C + c];
            window_attention(wslice, 1, N, C, bw, mask_ptr, wi);
            for (int t = 0; t < N; ++t)
                for (int c = 0; c < C; ++c)
                    win[(((size_t)b * nW + wi) * N + t) * C + c] =
                            wslice[(size_t)t * C + c];
        }
    }

    for (int b = 0; b < B; ++b) {
        for (int wi = 0; wi < nW; ++wi) {
            int wh = (wi / (Wp / ws)) * ws;
            int ww = (wi % (Wp / ws)) * ws;
            for (int i = 0; i < ws; ++i) {
                for (int j = 0; j < ws; ++j) {
                    int dh = wh + i, dw = ww + j;
                    if (dh < Hp && dw < Wp) {
                        for (int c = 0; c < C; ++c) {
                            feat[(((size_t)b * Hp + dh) * Wp + dw) * C + c] =
                                    win[(((size_t)b * nW + wi) * N + i * ws +
                                         j) * C +
                                        c];
                        }
                    }
                }
            }
        }
    }

    if (bw.shift_size > 0)
        spatial_roll(feat, B, Hp, Wp, C, bw.shift_size, bw.shift_size);

    tokens.assign((size_t)B * L * C, 0.f);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int c = 0; c < C; ++c)
                    tokens[((size_t)b * L + h * W + w) * C + c] =
                            feat[(((size_t)b * Hp + h) * Wp + w) * C + c];
    }

    for (size_t i = 0; i < tokens.size(); ++i) tokens[i] += shortcut[i];

    shortcut = tokens;
    layer_norm(tokens, B * L, C, bw.n2w, bw.n2b, cfg_.eps);
    const int hidden = C * cfg_.mlp_ratio;
    std::vector<float> h1;
    linear(tokens, B * L, C, hidden, bw.mlp1_w, &bw.mlp1_b, h1);
    for (float &v : h1) v = gelu(v);
    std::vector<float> h2;
    linear(h1, B * L, hidden, C, bw.mlp2_w, &bw.mlp2_b, h2);
    for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = shortcut[i] + h2[i];
}

bool SwinBackboneForward::forward_patch_embed(
        const std::vector<float> &nchw_in,
        int H,
        int W,
        std::vector<float> &patch_embed_out,
        std::vector<float> &tokens_blc,
        std::string &err) const {
    if ((int)nchw_in.size() != 3 * H * W) {
        err = "input size mismatch";
        return false;
    }
    conv2d_nchw(nchw_in, 1, 3, H, W, pe_w_, pe_out_c_, 4, 4, 4, &pe_b_,
                patch_embed_out);
    int Wh = H / 4, Ww = W / 4;
    const int C = pe_out_c_;
    tokens_blc.assign((size_t)Wh * Ww * C, 0.f);
    for (int c = 0; c < C; ++c)
        for (int h = 0; h < Wh; ++h)
            for (int w = 0; w < Ww; ++w)
                tokens_blc[((size_t)h * Ww + w) * C + c] =
                        patch_embed_out[((size_t)c * Wh + h) * Ww + w];

    layer_norm(tokens_blc, Wh * Ww, C, pe_nw_, pe_nb_, cfg_.eps);

    for (int c = 0; c < C; ++c)
        for (int h = 0; h < Wh; ++h)
            for (int w = 0; w < Ww; ++w)
                patch_embed_out[((size_t)c * Wh + h) * Ww + w] =
                        tokens_blc[((size_t)h * Ww + w) * C + c];
    return true;
}

bool SwinBackboneForward::forward_stage0(const std::vector<float> &nchw_in,
                                         int H,
                                         int W,
                                         std::vector<float> &patch_embed_out,
                                         std::vector<float> &stage0_out,
                                         std::string &err) const {
    if ((int)nchw_in.size() != 3 * H * W) {
        err = "input size mismatch";
        return false;
    }
    if (blocks_.size() < 2) {
        err = "stage0 blocks not loaded";
        return false;
    }

    std::vector<float> flat;
    if (!forward_patch_embed(nchw_in, H, W, patch_embed_out, flat, err))
        return false;

    stage0_out = flat;
    swin_block(stage0_out, 1, H / 4, W / 4, blocks_[0]);
    swin_block(stage0_out, 1, H / 4, W / 4, blocks_[1]);
    return true;
}

bool SwinBackboneForward::forward_block(const std::vector<float> &tokens_blc,
                                        int H,
                                        int W,
                                        int block_idx,
                                        std::vector<float> &out,
                                        std::string &err) const {
    if (block_idx < 0 || block_idx >= (int)blocks_.size()) {
        err = "invalid block_idx";
        return false;
    }
    const int C = blocks_[block_idx].dim;
    if ((int)tokens_blc.size() != H * W * C) {
        err = "tokens_blc size mismatch";
        return false;
    }
    out = tokens_blc;
    swin_block(out, 1, H, W, blocks_[block_idx]);
    return true;
}

bool SwinBackboneForward::forward_patch_merge(
        const std::vector<float> &tokens_blc,
        int H,
        int W,
        int merge_idx,
        std::vector<float> &out,
        int &out_h,
        int &out_w,
        std::string &err) const {
    if (merge_idx < 0 || merge_idx >= (int)merges_.size()) {
        err = "invalid merge_idx";
        return false;
    }
    const PatchMergeWeights &pm = merges_[merge_idx];
    const int B = 1;
    const int C = pm.in_dim;
    if ((int)tokens_blc.size() != B * H * W * C) {
        err = "tokens_blc size mismatch";
        return false;
    }
    const int Hp = H + (H % 2);
    const int Wp = W + (W % 2);
    out_h = Hp / 2;
    out_w = Wp / 2;
    const int rows = out_h * out_w;
    const int K = 4 * C;
    const int OC = (int)(pm.red_w.size() / K);
    std::vector<float> merged((size_t)rows * K);
    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            const int q[4][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
            float *row = merged.data() + (size_t)(oh * out_w + ow) * K;
            for (int qd = 0; qd < 4; ++qd) {
                const int ih = oh * 2 + q[qd][0];
                const int iw = ow * 2 + q[qd][1];
                float *dst = row + (size_t)qd * C;
                if (ih < H && iw < W) {
                    const float *src =
                            tokens_blc.data() + (size_t)(ih * W + iw) * C;
                    std::memcpy(dst, src, (size_t)C * sizeof(float));
                } else {
                    std::fill(dst, dst + C, 0.f);
                }
            }
        }
    }
    layer_norm(merged, rows, K, pm.nw, pm.nb, cfg_.eps);
    linear(merged, rows, K, OC, pm.red_w, nullptr, out);
    return true;
}

bool SwinBackboneForward::forward_stage01(const std::vector<float> &nchw_in,
                                          int H,
                                          int W,
                                          std::vector<float> &patch_embed_out,
                                          std::vector<float> &stage0_out,
                                          std::vector<float> &merge_out,
                                          std::vector<float> &stage1_out,
                                          std::string &err) const {
    if (!has_stage1()) {
        err = "stage1 weights not loaded";
        return false;
    }
    if (!forward_stage0(nchw_in, H, W, patch_embed_out, stage0_out, err))
        return false;
    int H1 = 0, W1 = 0;
    const int H0 = H / 4, W0 = W / 4;
    if (!forward_patch_merge(stage0_out, H0, W0, 0, merge_out, H1, W1, err))
        return false;
    stage1_out = merge_out;
    swin_block(stage1_out, 1, H1, W1, blocks_[2]);
    swin_block(stage1_out, 1, H1, W1, blocks_[3]);
    return true;
}

bool SwinBackboneForward::forward_stage2(const std::vector<float> &nchw_in,
                                         int H,
                                         int W,
                                         std::vector<float> &patch_embed_out,
                                         std::vector<float> &stage2_out,
                                         std::string &err) const {
    if (!has_stage2()) {
        err = "stage2 weights not loaded";
        return false;
    }
    std::vector<float> s0, m0, s1;
    if (!forward_stage01(nchw_in, H, W, patch_embed_out, s0, m0, s1, err))
        return false;
    int H2 = 0, W2 = 0;
    const int H1 = H / 8, W1 = W / 8;
    if (!forward_patch_merge(s1, H1, W1, 1, stage2_out, H2, W2, err))
        return false;
    for (int i = 0; i < 18; ++i)
        swin_block(stage2_out, 1, H2, W2, blocks_[4 + i]);
    return true;
}

bool SwinBackboneForward::forward_stage3(const std::vector<float> &nchw_in,
                                         int H,
                                         int W,
                                         std::vector<float> &patch_embed_out,
                                         std::vector<float> &stage3_out,
                                         std::string &err) const {
    if (!has_stage3()) {
        err = "stage3 weights not loaded";
        return false;
    }
    std::vector<float> s2;
    if (!forward_stage2(nchw_in, H, W, patch_embed_out, s2, err)) return false;
    int H3 = 0, W3 = 0;
    const int H2 = H / 16, W2 = W / 16;
    if (!forward_patch_merge(s2, H2, W2, 2, stage3_out, H3, W3, err))
        return false;
    swin_block(stage3_out, 1, H3, W3, blocks_[22]);
    swin_block(stage3_out, 1, H3, W3, blocks_[23]);
    return true;
}

void SwinBackboneForward::tokens_to_nchw(const std::vector<float> &tokens,
                                         int H,
                                         int W,
                                         int C,
                                         std::vector<float> &nchw) {
    nchw.resize((size_t)C * H * W);
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                nchw[((size_t)c * H + h) * W + w] =
                        tokens[((size_t)h * W + w) * C + c];
            }
        }
    }
}

bool SwinBackboneForward::forward_bb_four_scales(
        const std::vector<float> &nchw_in,
        int H,
        int W,
        std::vector<float> &x1,
        std::vector<float> &x2,
        std::vector<float> &x3,
        std::vector<float> &x4,
        std::string &err) const {
    if (!has_stage3()) {
        err = "stage3 weights not loaded";
        return false;
    }
    if (!has_bb_norm_) {
        err = "bb_norm0-3 weights not loaded";
        return false;
    }
    if ((int)nchw_in.size() != 3 * H * W) {
        err = "input size mismatch";
        return false;
    }

    static const int k_depths[] = {2, 2, 18, 2};

    std::vector<float> pe, tokens;
    if (!forward_patch_embed(nchw_in, H, W, pe, tokens, err)) return false;

    int stage_h = H / 4, stage_w = W / 4;
    int block_base = 0;
    for (int stage = 0; stage < 4; ++stage) {
        for (int bi = 0; bi < k_depths[stage]; ++bi)
            swin_block(tokens, 1, stage_h, stage_w, blocks_[block_base + bi]);
        block_base += k_depths[stage];

        std::vector<float> *out = (stage == 0)   ? &x1
                                  : (stage == 1) ? &x2
                                  : (stage == 2) ? &x3
                                                 : &x4;
        std::vector<float> normed = tokens;
        layer_norm(normed, stage_h * stage_w, (int)bb_norm_w_[stage].size(),
                   bb_norm_w_[stage], bb_norm_b_[stage], cfg_.eps);
        tokens_to_nchw(normed, stage_h, stage_w, (int)bb_norm_w_[stage].size(),
                       *out);

        if (stage == 3) break;
        std::vector<float> merged;
        int nh = 0, nw = 0;
        if (!forward_patch_merge(tokens, stage_h, stage_w, stage, merged, nh,
                                 nw, err))
            return false;
        tokens = std::move(merged);
        stage_h = nh;
        stage_w = nw;
    }
    return true;
}

bool SwinBackboneForward::window_attention_only(int block_idx,
                                                std::vector<float> x_nxc,
                                                std::vector<float> &out,
                                                std::string &err) const {
    if (block_idx < 0 || block_idx >= (int)blocks_.size()) {
        err = "invalid block_idx";
        return false;
    }
    const int C = blocks_[block_idx].dim;
    const int N = cfg_.window_size * cfg_.window_size;
    if ((int)x_nxc.size() != N * C) {
        err = "window_attention_only: input size mismatch";
        return false;
    }
    window_attention(x_nxc, 1, N, C, blocks_[block_idx], nullptr);
    out = std::move(x_nxc);
    return true;
}

}  // namespace rmbg

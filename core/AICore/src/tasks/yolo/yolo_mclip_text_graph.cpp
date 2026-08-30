// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/yolo/yolo_mclip_text_graph.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include "tasks/yolo/yolo_clip_text_graph.hpp"  // shared LN/attn/MLP/L2 blocks
#include "tasks/yolo/yolo_common.hpp"           // YOLO_LOG_* sink

namespace mclip {

namespace {

#define MCLOG_ERROR(...) ::yolo::logf(AICORE_LOG_LEVEL_ERROR, "mclip: " __VA_ARGS__)
#define MCLOG_INFO(...) ::yolo::logf(AICORE_LOG_LEVEL_INFO, "mclip: " __VA_ARGS__)

ggml_tensor* find_tensor(ggml_context* ctx, const char* name) {
    ggml_tensor* t = ggml_get_tensor(ctx, name);
    if (!t) MCLOG_ERROR("tensor '%s' not found in GGUF", name);
    return t;
}

// Multi-head self-attention without the F32 cast: ggml_mul_mat consumes
// F16/Q8_0 weights natively, and GGML_OP_CAST rejects quantized types — so
// the mclip tower (unlike the CLIP tower) passes its quantized matrices
// straight through. Non-causal (DistilBERT is bidirectional).
ggml_tensor* mclip_self_attention(ggml_context* ctx, ggml_tensor* x,
                                  ggml_tensor* in_proj_w,
                                  ggml_tensor* in_proj_b,
                                  ggml_tensor* out_proj_w,
                                  ggml_tensor* out_proj_b, int n_heads,
                                  int d_head) {
    const int D = (int)x->ne[0];
    const int S = (int)x->ne[1];

    ggml_tensor* qkv = ggml_mul_mat(ctx, in_proj_w, x);
    if (in_proj_b)
        qkv = ggml_add(ctx, qkv,
                       ggml_reshape_2d(ctx, in_proj_b, in_proj_b->ne[0], 1));
    qkv = ggml_cont(ctx, qkv);

    const size_t ts = ggml_type_size(qkv->type);
    const size_t row_bytes = (size_t)3 * D * ts;
    const size_t head_bytes = (size_t)d_head * ts;
    ggml_tensor* q4 = ggml_view_4d(ctx, qkv, d_head, n_heads, S, 1,
                                   head_bytes, row_bytes, qkv->nb[3], 0);
    ggml_tensor* k4 = ggml_view_4d(ctx, qkv, d_head, n_heads, S, 1,
                                   head_bytes, row_bytes, qkv->nb[3],
                                   D * ts);
    ggml_tensor* v4 = ggml_view_4d(ctx, qkv, d_head, n_heads, S, 1,
                                   head_bytes, row_bytes, qkv->nb[3],
                                   2 * D * ts);

    ggml_tensor* kT = ggml_cont(ctx, ggml_permute(ctx, k4, 0, 2, 1, 3));
    ggml_tensor* qT = ggml_cont(ctx, ggml_permute(ctx, q4, 0, 2, 1, 3));

    const float scale = 1.0f / sqrtf((float)d_head);
    ggml_tensor* attn =
            ggml_soft_max(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, kT, qT), scale));
    ggml_tensor* vT = ggml_cont(ctx, ggml_permute(ctx, v4, 1, 2, 0, 3));
    ggml_tensor* out = ggml_mul_mat(ctx, vT, attn);
    ggml_tensor* out_merged =
            ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    ggml_tensor* out_2d = ggml_reshape_2d(ctx, out_merged, D, S);

    ggml_tensor* result = ggml_mul_mat(ctx, out_proj_w, out_2d);
    if (out_proj_b)
        result = ggml_add(ctx, result,
                          ggml_reshape_2d(ctx, out_proj_b, out_proj_b->ne[0], 1));
    return result;
}

// LayerNorm with DistilBERT's eps=1e-12 (clip_layer_norm pins 1e-5).
ggml_tensor* mclip_layer_norm(ggml_context* ctx, ggml_tensor* x,
                              ggml_tensor* weight, ggml_tensor* bias) {
    ggml_tensor* y = ggml_norm(ctx, x, 1e-12f);
    if (weight) y = ggml_mul(ctx, y, weight);
    if (bias) y = ggml_add(ctx, y, bias);
    return y;
}

// ---------------------------------------------------------------------------
// WordPiece tokenizer (BertTokenizer semantics: do_lower_case=false,
// tokenize_chinese_chars=true, strip control chars, split punctuation)
// ---------------------------------------------------------------------------

uint32_t utf8_next(const std::string& s, size_t& i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
        i += 1;
        return c;
    }
    int len = c >= 0xF0 ? 4 : (c >= 0xE0 ? 3 : (c >= 0xC0 ? 2 : 1));
    if (i + len > s.size()) {
        i += 1;
        return 0xFFFD;
    }
    uint32_t cp = c & (len == 4 ? 0x07u : (len == 3 ? 0x0Fu : 0x1Fu));
    bool ok = true;
    for (int k = 1; k < len; ++k) {
        const unsigned char cc = static_cast<unsigned char>(s[i + k]);
        if ((cc & 0xC0) != 0x80) {
            ok = false;
            break;
        }
        cp = (cp << 6) | (cc & 0x3F);
    }
    if (!ok) {
        i += 1;
        return 0xFFFD;
    }
    i += static_cast<size_t>(len);
    return cp;
}

bool is_whitespace_cp(uint32_t cp) {
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r') return true;
    // Unicode Zs (space separators).
    return cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

bool is_control_cp(uint32_t cp) {
    if (cp == '\t' || cp == '\n' || cp == '\r') return false;
    return cp < 0x20 || (cp >= 0x7F && cp <= 0x9F);
}

bool is_punctuation_cp(uint32_t cp) {
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
        (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126))
        return true;
    // Unicode categories beginning with 'P' — common ranges (exact match with
    // HF for the scripts the multilingual vocabulary covers; rare exotic
    // punctuation falls through to regular tokens, a benign difference).
    return (cp >= 0xA1 && cp <= 0xBF) || (cp >= 0x2000 && cp <= 0x206F) ||
           (cp >= 0x3000 && cp <= 0x303F) || (cp >= 0xFF01 && cp <= 0xFF0F) ||
           (cp >= 0xFF1A && cp <= 0xFF20) || (cp >= 0xFF3B && cp <= 0xFF40) ||
           (cp >= 0xFF5B && cp <= 0xFF65);
}

bool is_cjk_cp(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) ||
           (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0x2CEB0 && cp <= 0x2EBEF) ||
           (cp >= 0x30000 && cp <= 0x3134F);
}

std::vector<std::string> basic_tokenize(const std::string& text) {
    std::string spaced;
    spaced.reserve(text.size() * 2);
    for (size_t i = 0; i < text.size();) {
        const uint32_t cp = utf8_next(text, i);
        if (is_control_cp(cp)) continue;
        if (is_whitespace_cp(cp)) {
            spaced.push_back(' ');
            continue;
        }
        std::string ch;
        if (cp < 0x80) {
            ch.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            ch.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            ch.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            ch.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            ch.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            ch.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            ch.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            ch.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            ch.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            ch.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        if (is_cjk_cp(cp)) {
            spaced += " " + ch + " ";
        } else {
            spaced += ch;
        }
    }
    std::vector<std::string> words;
    std::string token;
    auto flush = [&]() {
        if (!token.empty()) {
            words.push_back(token);
            token.clear();
        }
    };
    for (size_t i = 0; i < spaced.size();) {
        const size_t begin = i;
        const uint32_t cp = utf8_next(spaced, i);
        if (is_whitespace_cp(cp)) {
            flush();
            continue;
        }
        if (is_punctuation_cp(cp)) {
            flush();
            words.push_back(spaced.substr(begin, i - begin));
            continue;
        }
        token.append(spaced, begin, i - begin);
    }
    flush();
    return words;
}

std::vector<std::string> wordpiece(TextSession* s, const std::string& word) {
    constexpr size_t kMaxInputChars = 100;
    if (word.size() > kMaxInputChars) return {s->vocab[s->unk_id]};
    std::vector<std::string> pieces;
    size_t start = 0;
    while (start < word.size()) {
        size_t end = word.size();
        std::string chosen;
        while (start < end) {
            std::string sub = word.substr(start, end - start);
            if (start > 0) sub = "##" + sub;
            auto it = s->encoder.find(sub);
            if (it != s->encoder.end()) {
                chosen = sub;
                break;
            }
            --end;
        }
        if (chosen.empty()) return {s->vocab[s->unk_id]};
        pieces.push_back(chosen);
        start = end;
    }
    return pieces;
}

// ---------------------------------------------------------------------------
// Per-string graph build + compute (true sequence length, no padding)
// ---------------------------------------------------------------------------

bool build_and_compute(TextSession* s, const std::int32_t* ids, int S,
                       float* embed) {
    if (S < 2 || S > MAX_TOKENS) {
        MCLOG_ERROR("invalid sequence length %d", S);
        return false;
    }
    // Fresh graph context at the exact sequence length. The gallocr inside
    // the backend bundle is persistent and resizes itself to the new shape.
    if (s->text_gctx) {
        ggml_free(s->text_gctx);
        s->text_gctx = nullptr;
        s->text_graph = nullptr;
        s->text_output_embed = nullptr;
    }
    const size_t mem = 16u * 1024u * 1024u;
    ggml_context* gctx = ggml_init({mem, nullptr, /*no_alloc*/ true});
    if (!gctx) {
        MCLOG_ERROR("graph context alloc failed");
        return false;
    }

    ggml_tensor* ids_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, S);
    ggml_set_input(ids_t);
    ggml_set_name(ids_t, "tokens");
    // Word embedding lookup: cast f16 -> f32, then [D, V] x [S] -> [D, S].
    // word_emb may be F16 or Q8_0: get_rows dequantizes natively.
    ggml_tensor* h = ggml_get_rows(gctx, s->word_emb, ids_t);

    // Positional embedding [MAX_POS, D] -> view [D, S] -> add.
    ggml_tensor* pe = ggml_cast(gctx, s->pos_emb, GGML_TYPE_F32);
    ggml_tensor* pe_v = ggml_view_2d(gctx, pe, EMBED_DIM, S, pe->nb[1], 0);
    h = ggml_add(gctx, h, pe_v);

    h = mclip_layer_norm(gctx, h, s->emb_ln_w, s->emb_ln_b);

    const int d_head = EMBED_DIM / N_HEADS;  // 64
    auto can = [](const ggml_tensor* a, const ggml_tensor* b) {
        return (a->ne[0] % b->ne[0] == 0) && a->ne[2] == b->ne[2] &&
               a->ne[3] == b->ne[3];
    };
    for (int i = 0; i < N_LAYERS; ++i) {
        const auto& b = s->blocks[i];
        // DistilBERT post-LN layout: attention consumes the raw stream, the
        // residual add is followed by each LayerNorm.
        ggml_tensor* att = mclip_self_attention(
                gctx, h, b.attn_in_w, b.attn_in_b, b.attn_out_w, b.attn_out_b,
                N_HEADS, d_head);
        h = mclip_layer_norm(gctx, ggml_add(gctx, h, att), b.sa_ln_w, b.sa_ln_b);
        // FFN (expanded from clip::clip_mlp_block with probe points).
        ggml_tensor* hh = ggml_mul_mat(gctx, b.ffn_in_w, h);
        hh = ggml_add(gctx, hh,
                      ggml_reshape_2d(gctx, b.ffn_in_b, b.ffn_in_b->ne[0], 1));
        hh = ggml_gelu_erf(gctx, hh);
        ggml_tensor* fo = ggml_mul_mat(gctx, b.ffn_out_w, hh);
        fo = ggml_add(gctx, fo,
                      ggml_reshape_2d(gctx, b.ffn_out_b, b.ffn_out_b->ne[0], 1));
        h = mclip_layer_norm(gctx, ggml_add(gctx, h, fo), b.out_ln_w,
                             b.out_ln_b);
    }

    // Mean pooling over the (unpadded) sequence: ggml_sum_rows sums along
    // ne0 into a [1, D] result, so transpose [D, S] -> [S, D] first to
    // collapse the token axis, then flatten back to [D].
    ggml_tensor* pooled = ggml_reshape_1d(
            gctx,
            ggml_scale(gctx,
                       ggml_sum_rows(
                               gctx, ggml_cont(gctx, ggml_transpose(gctx, h))),
                       1.0f / float(S)),
            EMBED_DIM);

    // Projection into the CLIP ViT-B/32 text space: [512, D] x [D, 1]
    // (F16/Q8_0 weights feed mul_mat natively — no cast).
    ggml_tensor* out = ggml_mul_mat(gctx, s->proj_w, pooled);
    if (s->proj_b)
        out = ggml_add(gctx, out,
                       ggml_reshape_2d(gctx, s->proj_b, OUT_DIM, 1));

    out = clip::clip_l2_norm(gctx, ggml_reshape_1d(gctx, out, OUT_DIM));
    ggml_set_output(out);
    ggml_set_name(out, "mclip_embed");

    s->text_gctx = gctx;
    s->text_graph = ggml_new_graph_custom(gctx, 4096, false);
    ggml_build_forward_expand(s->text_graph, out);
    s->text_output_embed = out;

    if (!yolo::backend_ctx_graph_alloc(s->backend, s->text_graph)) {
        MCLOG_ERROR("text graph alloc failed");
        return false;
    }
    ggml_backend_tensor_set(ids_t, ids, 0, S * sizeof(int32_t));
    if (yolo::backend_ctx_graph_compute(s->backend, s->text_graph) !=
        GGML_STATUS_SUCCESS) {
        MCLOG_ERROR("text graph compute failed");
        return false;
    }
    ggml_backend_tensor_get(s->text_output_embed, embed, 0,
                            OUT_DIM * sizeof(float));
    return true;
}

}  // namespace

// Forward declaration helper used by build_and_compute (kept tiny so the
// per-string graph builder stays in one place).
// NOTE: text_input_tokens is a per-graph tensor owned by text_gctx.

TextSession* text_create_session(const std::string& gguf_path, int threads) {
    ggml_context* weight_ctx = nullptr;
    gguf_init_params ip{};
    ip.no_alloc = false;  // map tensor data directly
    ip.ctx = &weight_ctx;

    gguf_context* g = gguf_init_from_file(gguf_path.c_str(), ip);
    if (!g) {
        MCLOG_ERROR("failed to open GGUF: %s", gguf_path.c_str());
        return nullptr;
    }

    TextSession* s = new TextSession();
    s->wctx = weight_ctx;

    int n_threads =
            threads > 0 ? threads : (int)std::thread::hardware_concurrency();
    if (n_threads <= 0) n_threads = 4;
    s->backend = yolo::init_backend_ctx(n_threads, "cpu");
    if (!s->backend.cpu) {
        text_free_session(s);
        gguf_free(g);
        return nullptr;
    }

    s->word_emb = find_tensor(s->wctx, "mclip.word_emb");
    s->pos_emb = find_tensor(s->wctx, "mclip.pos_emb");
    s->emb_ln_w = find_tensor(s->wctx, "mclip.emb_ln_w");
    s->emb_ln_b = find_tensor(s->wctx, "mclip.emb_ln_b");
    s->proj_w = find_tensor(s->wctx, "mclip.proj_w");
    s->proj_b = ggml_get_tensor(s->wctx, "mclip.proj_b");  // optional (fitted
                                                           // affine bridges)
    if (!s->word_emb || !s->pos_emb || !s->emb_ln_w || !s->emb_ln_b ||
        !s->proj_w) {
        text_free_session(s);
        gguf_free(g);
        return nullptr;
    }
    for (int i = 0; i < N_LAYERS; ++i) {
        auto& blk = s->blocks[i];
        char name[96];
        auto resolve = [&](const char* suffix) -> ggml_tensor* {
            snprintf(name, sizeof(name), "mclip.l%d.%s", i, suffix);
            return find_tensor(s->wctx, name);
        };
        blk.attn_in_w = resolve("attn_in_w");
        blk.attn_in_b = resolve("attn_in_b");
        blk.attn_out_w = resolve("attn_o_w");
        blk.attn_out_b = resolve("attn_o_b");
        blk.sa_ln_w = resolve("sa_ln_w");
        blk.sa_ln_b = resolve("sa_ln_b");
        blk.ffn_in_w = resolve("ffn_in_w");
        blk.ffn_in_b = resolve("ffn_in_b");
        blk.ffn_out_w = resolve("ffn_out_w");
        blk.ffn_out_b = resolve("ffn_out_b");
        blk.out_ln_w = resolve("out_ln_w");
        blk.out_ln_b = resolve("out_ln_b");
        if (!blk.attn_in_w || !blk.attn_in_b || !blk.attn_out_w ||
            !blk.attn_out_b || !blk.sa_ln_w || !blk.sa_ln_b || !blk.ffn_in_w ||
            !blk.ffn_in_b || !blk.ffn_out_w || !blk.ffn_out_b ||
            !blk.out_ln_w || !blk.out_ln_b) {
            MCLOG_ERROR("failed to resolve block %d", i);
            text_free_session(s);
            gguf_free(g);
            return nullptr;
        }
    }

    // WordPiece vocabulary (string array written by the converter).
    const int vid = gguf_find_key(g, "mclip.vocab");
    if (vid < 0 || gguf_get_kv_type(g, vid) != GGUF_TYPE_ARRAY) {
        MCLOG_ERROR("GGUF has no mclip.vocab array");
        text_free_session(s);
        gguf_free(g);
        return nullptr;
    }
    const size_t nv = gguf_get_arr_n(g, vid);
    s->vocab.resize(nv);
    for (size_t i = 0; i < nv; ++i) {
        s->vocab[i] = gguf_get_arr_str(g, vid, i);
        s->encoder[s->vocab[i]] = (int)i;
    }
    // Special-token ids by name (converter asserts the canonical layout, but
    // resolve dynamically so a re-ordering stays safe).
    for (const auto& [tok, id] : s->encoder) {
        if (tok == "[CLS]") s->cls_id = id;
        else if (tok == "[SEP]") s->sep_id = id;
        else if (tok == "[UNK]") s->unk_id = id;
    }

    gguf_free(g);
    YOLO_LOG_INFO("mclip: text session ready (%d threads, vocab %zu)",
                  n_threads, nv);
    return s;
}

void text_free_session(TextSession* s) {
    if (!s) return;
    yolo::free_backend_ctx(s->backend);
    if (s->wctx) ggml_free(s->wctx);
    if (s->text_gctx) ggml_free(s->text_gctx);
    delete s;
}

int text_tokenize(TextSession* s, const char* text, int32_t* tokens, int cap) {
    if (!s || !text || !tokens || cap < 2) return 0;
    std::vector<int32_t> ids;
    ids.push_back(s->cls_id);
    for (const std::string& word : basic_tokenize(text)) {
        for (const std::string& piece : wordpiece(s, word)) {
            auto it = s->encoder.find(piece);
            if (it == s->encoder.end()) {
                ids.push_back(s->unk_id);
            } else {
                ids.push_back(it->second);
            }
            if ((int)ids.size() >= cap - 1) break;
        }
        if ((int)ids.size() >= cap - 1) break;
    }
    ids.push_back(s->sep_id);
    if ((int)ids.size() > cap) {
        ids.resize(cap);
        ids.back() = s->sep_id;
    }
    for (size_t i = 0; i < ids.size(); ++i) tokens[i] = ids[i];
    return (int)ids.size();
}

bool text_encode_tokens(TextSession* s, const int32_t* tokens, int n_tokens,
                        float* embed) {
    if (!s || !tokens || !embed || n_tokens <= 0) return false;
    return build_and_compute(s, tokens, n_tokens, embed);
}

bool text_encode_string(TextSession* s, const char* text, float* embed) {
    if (!s) return false;
    int32_t tokens[MAX_TOKENS];
    const int n = text_tokenize(s, text, tokens, MAX_TOKENS);
    if (n <= 0) return false;
    return text_encode_tokens(s, tokens, n, embed);
}

}  // namespace mclip

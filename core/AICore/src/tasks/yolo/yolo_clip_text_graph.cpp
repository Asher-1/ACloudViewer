// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/yolo/yolo_clip_text_graph.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <thread>
#include <unordered_map>

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

namespace clip {

// ---------------------------------------------------------------------------
// GGUF loading helpers
// ---------------------------------------------------------------------------

static ggml_tensor* find_tensor(ggml_context* ctx, const char* name) {
    ggml_tensor* t = ggml_get_tensor(ctx, name);
    if (!t) {
        YOLO_LOG_ERROR("clip: tensor '%s' not found in GGUF", name);
    }
    return t;
}

static bool resolve_text_block(TextSession::TextBlock& blk,
                               const char* prefix,
                               ggml_context* wctx) {
    char name[128];
    auto resolve = [&](const char* suffix) -> ggml_tensor* {
        snprintf(name, sizeof(name), "text.%s.%s", prefix, suffix);
        return find_tensor(wctx, name);
    };
    blk.attn_in_w = resolve("attn.in_proj_weight");
    blk.attn_in_b = resolve("attn.in_proj_bias");
    blk.attn_out_w = resolve("attn.out_proj.weight");
    blk.attn_out_b = resolve("attn.out_proj.bias");
    blk.mlp_fc_w = resolve("mlp.c_fc.weight");
    blk.mlp_fc_b = resolve("mlp.c_fc.bias");
    blk.mlp_proj_w = resolve("mlp.c_proj.weight");
    blk.mlp_proj_b = resolve("mlp.c_proj.bias");
    blk.ln1_w = resolve("ln_1.weight");
    blk.ln1_b = resolve("ln_1.bias");
    blk.ln2_w = resolve("ln_2.weight");
    blk.ln2_b = resolve("ln_2.bias");
    return blk.attn_in_w != nullptr;
}

// ---------------------------------------------------------------------------
// Graph building helpers
// ---------------------------------------------------------------------------

ggml_tensor* clip_l2_norm(ggml_context* ctx, ggml_tensor* x) {
    ggml_tensor* sq = ggml_sqr(ctx, x);
    ggml_tensor* sum = ggml_sum_rows(ctx, sq);  // [1] (or [N, 1])
    ggml_tensor* norm = ggml_sqrt(ctx, sum);
    return ggml_div(ctx, x, norm);
}

ggml_tensor* clip_layer_norm(ggml_context* ctx,
                             ggml_tensor* x,
                             ggml_tensor* weight,
                             ggml_tensor* bias) {
    // ggml_norm: y = (x - mean) / sqrt(var + eps), eps hardcoded 1e-5f.
    ggml_tensor* y = ggml_norm(ctx, x, 1e-5f);
    if (weight) y = ggml_mul(ctx, y, weight);
    if (bias) y = ggml_add(ctx, y, bias);
    return y;
}

ggml_tensor* clip_self_attention(ggml_context* ctx,
                                 ggml_tensor* x,
                                 ggml_tensor* in_proj_w,
                                 ggml_tensor* in_proj_b,
                                 ggml_tensor* out_proj_w,
                                 ggml_tensor* out_proj_b,
                                 int n_heads,
                                 int d_head,
                                 bool causal) {
    // Cast F16 weights to F32 for computation.
    if (in_proj_w->type != GGML_TYPE_F32)
        in_proj_w = ggml_cast(ctx, in_proj_w, GGML_TYPE_F32);
    if (in_proj_b && in_proj_b->type != GGML_TYPE_F32)
        in_proj_b = ggml_cast(ctx, in_proj_b, GGML_TYPE_F32);
    if (out_proj_w->type != GGML_TYPE_F32)
        out_proj_w = ggml_cast(ctx, out_proj_w, GGML_TYPE_F32);
    if (out_proj_b && out_proj_b->type != GGML_TYPE_F32)
        out_proj_b = ggml_cast(ctx, out_proj_b, GGML_TYPE_F32);

    const int D = (int)x->ne[0];  // embed_dim
    const int S = (int)x->ne[1];  // seq_len
    const int d_h = d_head;
    const int n_h = n_heads;

    // QKV projection: mul_mat([3D, D], [D, S]) -> [3D, S].
    ggml_tensor* qkv = ggml_mul_mat(ctx, in_proj_w, x);
    if (in_proj_b)
        qkv = ggml_add(ctx, qkv,
                       ggml_reshape_2d(ctx, in_proj_b, in_proj_b->ne[0], 1));
    qkv = ggml_cont(ctx, qkv);

    // Split q/k/v as 4D views [d_h, n_h, S, 1] into the [3D, S] buffer (no
    // copy): per-position stride 3D elements, per-head stride d_h elements.
    const size_t ts = ggml_type_size(qkv->type);
    const size_t row_bytes = (size_t)3 * D * ts;
    const size_t head_bytes = (size_t)d_h * ts;
    ggml_tensor* q4 = ggml_view_4d(ctx, qkv, d_h, n_h, S, 1, head_bytes,
                                   row_bytes, qkv->nb[3], 0);
    ggml_tensor* k4 = ggml_view_4d(ctx, qkv, d_h, n_h, S, 1, head_bytes,
                                   row_bytes, qkv->nb[3], D * ts);
    ggml_tensor* v4 = ggml_view_4d(ctx, qkv, d_h, n_h, S, 1, head_bytes,
                                   row_bytes, qkv->nb[3], 2 * D * ts);

    // Permute to [d_h, S, n_h, 1] for the KQ mul_mat.
    ggml_tensor* kT = ggml_cont(ctx, ggml_permute(ctx, k4, 0, 2, 1, 3));
    ggml_tensor* qT = ggml_cont(ctx, ggml_permute(ctx, q4, 0, 2, 1, 3));

    const float scale = 1.0f / sqrtf((float)d_h);
    ggml_tensor* qs = ggml_scale(ctx, qT, scale);

    // Attention scores [S, S, n_h, 1]: ne0=key, ne1=query, ne2=head.
    ggml_tensor* attn = ggml_mul_mat(ctx, kT, qs);

    // Causal mask: query n attends keys m <= n (diag_mask_inf zeroes the
    // strictly-upper triangle), matching clip's _build_causal_attention_mask.
    if (causal) {
        attn = ggml_diag_mask_inf(ctx, attn, 0);
    }
    attn = ggml_soft_max(ctx, attn);  // softmax along keys, per query row

    // Apply attention to values: [S, d_h, n_h, 1] x [S, S, n_h, 1].
    ggml_tensor* vT = ggml_cont(ctx, ggml_permute(ctx, v4, 1, 2, 0, 3));
    ggml_tensor* out = ggml_mul_mat(ctx, vT, attn);  // [d_h, S, n_h, 1]

    // Merge heads: [d_h, S, n_h, 1] -> [D, S] contiguous.
    ggml_tensor* out_merged =
            ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    ggml_tensor* out_2d = ggml_reshape_2d(ctx, out_merged, D, S);

    ggml_tensor* result = ggml_mul_mat(ctx, out_proj_w, out_2d);
    if (out_proj_b)
        result = ggml_add(ctx, result, ggml_reshape_2d(ctx, out_proj_b, D, 1));
    return result;
}

ggml_tensor* clip_mlp_block(ggml_context* ctx,
                            ggml_tensor* x,
                            ggml_tensor* fc_w,
                            ggml_tensor* fc_b,
                            ggml_tensor* proj_w,
                            ggml_tensor* proj_b,
                            bool exact_gelu) {
    if (fc_w->type != GGML_TYPE_F32) fc_w = ggml_cast(ctx, fc_w, GGML_TYPE_F32);
    if (fc_b && fc_b->type != GGML_TYPE_F32)
        fc_b = ggml_cast(ctx, fc_b, GGML_TYPE_F32);
    if (proj_w->type != GGML_TYPE_F32)
        proj_w = ggml_cast(ctx, proj_w, GGML_TYPE_F32);
    if (proj_b && proj_b->type != GGML_TYPE_F32)
        proj_b = ggml_cast(ctx, proj_b, GGML_TYPE_F32);

    ggml_tensor* h = ggml_mul_mat(ctx, fc_w, x);  // [4*D, N]
    if (fc_b) h = ggml_add(ctx, h, ggml_reshape_2d(ctx, fc_b, fc_b->ne[0], 1));
    h = exact_gelu ? ggml_gelu_erf(ctx, h)  // exact GELU (MobileCLIP)
                   : ggml_gelu_quick(ctx,
                                     h);  // QuickGELU x*sigmoid(1.702x) (CLIP)
    h = ggml_mul_mat(ctx, proj_w, h);     // [D, N]
    if (proj_b)
        h = ggml_add(ctx, h, ggml_reshape_2d(ctx, proj_b, proj_b->ne[0], 1));
    return h;
}

// Build one transformer block (attention + MLP with residuals).
static ggml_tensor* transformer_block(ggml_context* ctx,
                                      ggml_tensor* x,
                                      ggml_tensor* ln1_w,
                                      ggml_tensor* ln1_b,
                                      ggml_tensor* attn_in_w,
                                      ggml_tensor* attn_in_b,
                                      ggml_tensor* attn_out_w,
                                      ggml_tensor* attn_out_b,
                                      ggml_tensor* ln2_w,
                                      ggml_tensor* ln2_b,
                                      ggml_tensor* mlp_fc_w,
                                      ggml_tensor* mlp_fc_b,
                                      ggml_tensor* mlp_proj_w,
                                      ggml_tensor* mlp_proj_b,
                                      int n_heads,
                                      int d_head,
                                      bool causal) {
    ggml_tensor* residual = x;
    x = clip_layer_norm(ctx, x, ln1_w, ln1_b);
    x = clip_self_attention(ctx, x, attn_in_w, attn_in_b, attn_out_w,
                            attn_out_b, n_heads, d_head, causal);
    x = ggml_add(ctx, residual, x);  // residual 1

    residual = x;
    x = clip_layer_norm(ctx, x, ln2_w, ln2_b);
    x = clip_mlp_block(ctx, x, mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b,
                       /*exact_gelu*/ false);
    x = ggml_add(ctx, residual, x);  // residual 2
    return x;
}

// ---------------------------------------------------------------------------
// BPE tokenizer (CLIP SimpleTokenizer)
// ---------------------------------------------------------------------------

// The reversible byte<->unicode mapping from GPT-2 / CLIP: printable ASCII
// and Latin-1 stay identity; the remaining 163 bytes map to chr(256 + n).
static const std::unordered_map<unsigned char, std::string>& byte_to_unicode() {
    static std::unordered_map<unsigned char, std::string> table = [] {
        std::unordered_map<unsigned char, std::string> t;
        std::vector<int> bs;
        auto add_range = [&](int lo, int hi) {
            for (int i = lo; i <= hi; i++) bs.push_back(i);
        };
        add_range('!', '~');
        add_range(0xA1, 0xAC);
        add_range(0xAE, 0xFF);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n++);
            }
        }
        for (size_t i = 0; i < bs.size(); i++) {
            t[(unsigned char)bs[i]] = std::string(1, (char)cs[i]);
        }
        return t;
    }();
    return table;
}

// Decode the next UTF-8 code point; advances `p`. Returns -1 on invalid
// input.
static int32_t utf8_decode(const unsigned char* s, size_t len, size_t* p) {
    const unsigned char c = s[*p];
    if (c < 0x80) {
        (*p)++;
        return c;
    }
    int32_t cp = 0;
    int extra = 0;
    if ((c & 0xE0) == 0xC0) {
        cp = c & 0x1F;
        extra = 1;
    } else if ((c & 0xF0) == 0xE0) {
        cp = c & 0x0F;
        extra = 2;
    } else if ((c & 0xF8) == 0xF0) {
        cp = c & 0x07;
        extra = 3;
    } else {
        (*p)++;
        return -1;
    }
    if (*p + extra >= len + 1 || *p + extra > len) {
        (*p)++;
        return -1;
    }
    for (int i = 1; i <= extra; i++) {
        const unsigned char cc = s[*p + i];
        if ((cc & 0xC0) != 0x80) {
            (*p)++;
            return -1;
        }
        cp = (cp << 6) | (cc & 0x3F);
    }
    *p += 1 + extra;
    return cp;
}

// Unicode letter / number classification covering the code ranges that occur
// in practice (Latin, Greek, Cyrillic, Hebrew, Arabic, CJK, kana, hangul).
static bool is_unicode_letter(int32_t cp) {
    if (cp < 0) return false;
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp < 0xC0) return false;
    return (cp >= 0xC0 && cp <= 0x2AF) ||     // Latin-1 supp .. Latin Extended
           (cp >= 0x370 && cp <= 0x52F) ||    // Greek + Cyrillic
           (cp >= 0x531 && cp <= 0x58F) ||    // Armenian
           (cp >= 0x590 && cp <= 0x5F4) ||    // Hebrew
           (cp >= 0x600 && cp <= 0x6FF) ||    // Arabic
           (cp >= 0x900 && cp <= 0x97F) ||    // Devanagari
           (cp >= 0x1E00 && cp <= 0x1FFF) ||  // Latin Extended Additional
           (cp >= 0x2C60 && cp <= 0x2C7F) ||  // Latin Extended-C
           (cp >= 0x3040 && cp <= 0x30FF) ||  // hiragana + katakana
           (cp >= 0x3400 && cp <= 0x9FFF) ||  // CJK
           (cp >= 0xAC00 && cp <= 0xD7AF);    // hangul
}
static bool is_unicode_number(int32_t cp) {
    return cp >= 0 &&
           ((cp >= '0' && cp <= '9') ||
            (cp >= 0x660 && cp <= 0x669) ||  // Arabic-Indic
            (cp >= 0x6F0 && cp <= 0x6F9) || (cp >= 0x96F && cp <= 0x96F) ||
            (cp >= 0xFF10 && cp <= 0xFF19));  // fullwidth
}
static bool is_unicode_space(int32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0xA0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200B) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x3000;
}

// Tokenize `text` into BPE subword tokens matching the Python
// SimpleTokenizer regex: <|startoftext|> | <|endoftext|> | 's|'t|'re|'ve|'m|
// 'll|'d | letters+ | one digit | runs of other non-space chars.
static std::vector<std::string> regex_split(const std::string& text) {
    std::vector<std::string> out;
    const unsigned char* s = (const unsigned char*)text.data();
    const size_t len = text.size();
    size_t p = 0;
    auto literal = [&](const char* lit) -> bool {
        const size_t n = strlen(lit);
        if (p + n <= len && memcmp(s + p, lit, n) == 0) {
            out.emplace_back(lit);
            p += n;
            return true;
        }
        return false;
    };
    while (p < len) {
        if (literal("<|startoftext|>") || literal("<|endoftext|>")) continue;
        if (p + 3 <= len &&
            (memcmp(s + p, "'re", 3) == 0 || memcmp(s + p, "'ve", 3) == 0 ||
             memcmp(s + p, "'ll", 3) == 0)) {
            out.emplace_back(text.substr(p, 3));
            p += 3;
            continue;
        }
        if (p + 2 <= len &&
            (memcmp(s + p, "'s", 2) == 0 || memcmp(s + p, "'t", 2) == 0 ||
             memcmp(s + p, "'m", 2) == 0 || memcmp(s + p, "'d", 2) == 0)) {
            out.emplace_back(text.substr(p, 2));
            p += 2;
            continue;
        }
        size_t q = p;
        const int32_t cp = utf8_decode(s, len, &q);
        if (cp > 0 && is_unicode_letter(cp)) {
            size_t e = q;
            while (e < len) {
                size_t q2 = e;
                int32_t c2 = utf8_decode(s, len, &q2);
                if (!(c2 > 0 && is_unicode_letter(c2))) break;
                e = q2;
            }
            out.push_back(text.substr(p, e - p));
            p = e;
        } else if (cp > 0 && is_unicode_number(cp)) {
            out.emplace_back(text.substr(p, q - p));
            p = q;
        } else if (cp > 0 && is_unicode_space(cp)) {
            p = q;
        } else {
            size_t e = q;
            while (e < len) {
                size_t q2 = e;
                int32_t c2 = utf8_decode(s, len, &q2);
                if (c2 > 0 && (is_unicode_space(c2) || is_unicode_letter(c2) ||
                               is_unicode_number(c2)))
                    break;
                e = q2;
            }
            out.push_back(text.substr(p, e - p));
            p = e;
        }
    }
    return out;
}

// whitespace_clean: collapse runs of whitespace into a single space.
static std::string whitespace_clean(std::string t) {
    std::string out;
    bool pending_space = false;
    for (unsigned char c : t) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            pending_space = true;
        } else {
            if (pending_space && !out.empty()) out += ' ';
            pending_space = false;
            out += (char)c;
        }
    }
    return out;
}

// basic_clean approximation: strip control chars (ftfy + html.unescape
// rarely fire on prompts).
static std::string basic_clean(std::string t) {
    std::string out;
    for (unsigned char c : t) {
        if (c < 0x20 && c != '\t' && c != '\n' && c != '\r') continue;
        out += (char)c;
    }
    return out;
}

// Byte-encode a token: map each UTF-8 byte through byte_to_unicode.
static std::string byte_encode(const std::string& tok) {
    const auto& table = byte_to_unicode();
    std::string out;
    for (unsigned char b : tok) out += table.at(b);
    return out;
}

// GPT-2 BPE merge of a single byte-encoded token (SimpleTokenizer.bpe).
static std::string bpe_merge(ClipBpe& bpe, const std::string& token) {
    auto it = bpe.bpe_cache.find(token);
    if (it != bpe.bpe_cache.end()) return it->second;
    // word = tuple(token[:-1]) + (token[-1] + '</w>',)
    std::vector<std::string> word;
    for (size_t i = 0; i + 1 < token.size(); i++)
        word.push_back(token.substr(i, 1));
    word.push_back(token.substr(token.size() - 1) + "</w>");
    auto pairs_of = [](const std::vector<std::string>& w) {
        std::vector<std::pair<std::string, std::string>> ps;
        for (size_t i = 0; i + 1 < w.size(); i++)
            ps.emplace_back(w[i], w[i + 1]);
        return ps;
    };
    auto ranks = [&](const std::string& a, const std::string& b) {
        for (size_t i = 0; i < bpe.merges.size(); i++) {
            if (bpe.merges[i].first == a && bpe.merges[i].second == b)
                return (int)i;
        }
        return INT_MAX;
    };
    std::vector<std::pair<std::string, std::string>> pairs = pairs_of(word);
    while (!pairs.empty()) {
        int best = INT_MAX;
        size_t bi = 0;
        for (size_t i = 0; i < pairs.size(); i++) {
            const int r = ranks(pairs[i].first, pairs[i].second);
            if (r < best) {
                best = r;
                bi = i;
            }
        }
        if (best == INT_MAX) break;
        const std::string& f = pairs[bi].first;
        const std::string& sc = pairs[bi].second;
        std::vector<std::string> nw;
        for (size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() && word[i] == f && word[i + 1] == sc) {
                nw.push_back(f + sc);
                i += 2;
            } else {
                nw.push_back(word[i]);
                i++;
            }
        }
        word.swap(nw);
        pairs = pairs_of(word);
    }
    std::string joined;
    for (size_t i = 0; i < word.size(); i++) {
        if (i) joined += ' ';
        joined += word[i];
    }
    bpe.bpe_cache[token] = joined;
    return joined;
}

bool clip_bpe_load(ClipBpe& bpe,
                   const gguf_context* g,
                   const char* vocab_key,
                   const char* merges_key,
                   int expect_vocab) {
    const int64_t vid = gguf_find_key(g, vocab_key);
    const int64_t mid = gguf_find_key(g, merges_key);
    if (vid < 0 || mid < 0) {
        YOLO_LOG_ERROR("bpe: GGUF has no %s / %s", vocab_key, merges_key);
        return false;
    }
    const size_t nv = gguf_get_arr_n(g, vid);
    const size_t nm = gguf_get_arr_n(g, mid);
    if (nv != (size_t)expect_vocab) {
        YOLO_LOG_ERROR("bpe: unexpected vocab size %zu (expected %d)", nv,
                       expect_vocab);
        return false;
    }
    bpe.vocab.resize(nv);
    for (size_t i = 0; i < nv; i++) {
        bpe.vocab[i] = gguf_get_arr_str(g, vid, i);
        bpe.encoder[bpe.vocab[i]] = (int)i;
    }
    bpe.merges.reserve(nm);
    for (size_t i = 0; i < nm; i++) {
        std::string pair = gguf_get_arr_str(g, mid, i);
        const size_t sp = pair.find(' ');
        if (sp == std::string::npos) {
            YOLO_LOG_ERROR("bpe: bad merge entry %zu: '%s'", i, pair.c_str());
            return false;
        }
        bpe.merges.emplace_back(pair.substr(0, sp), pair.substr(sp + 1));
    }
    return true;
}

int clip_bpe_tokenize(ClipBpe& bpe,
                      const char* text,
                      int ctx_len,
                      int32_t* tokens) {
    if (!text || !tokens || ctx_len <= 0) return 0;
    std::string t = text;
    std::transform(t.begin(), t.end(), t.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    t = whitespace_clean(basic_clean(t));

    std::vector<int32_t> ids;
    ids.push_back(49406);  // <|startoftext|>
    for (const std::string& tok : regex_split(t)) {
        const std::string enc = byte_encode(tok);
        const std::string merged = bpe_merge(bpe, enc);
        size_t pos = 0;
        while (pos <= merged.size()) {
            const size_t sp = merged.find(' ', pos);
            const std::string sub = sp == std::string::npos
                                            ? merged.substr(pos)
                                            : merged.substr(pos, sp - pos);
            auto entry = bpe.encoder.find(sub);
            if (entry == bpe.encoder.end()) {
                // Unknown subword: skip (CLIP <unk>-like behaviour).
                pos = sp == std::string::npos ? merged.size() + 1 : sp + 1;
                continue;
            }
            ids.push_back(entry->second);
            if (sp == std::string::npos) break;
            pos = sp + 1;
        }
    }
    ids.push_back(49407);  // <|endoftext|>
    if ((int)ids.size() > ctx_len) {
        ids.resize(ctx_len);
        ids.back() = 49407;
    }
    for (int i = 0; i < ctx_len; i++)
        tokens[i] = i < (int)ids.size() ? ids[i] : 0;
    return (int)ids.size();
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------

TextSession* text_create_session(const std::string& gguf_path, int threads) {
    ggml_context* weight_ctx = nullptr;
    gguf_init_params ip{};
    ip.no_alloc = false;  // map tensor data directly
    ip.ctx = &weight_ctx;

    gguf_context* g = gguf_init_from_file(gguf_path.c_str(), ip);
    if (!g) {
        YOLO_LOG_ERROR("clip: failed to open GGUF: %s", gguf_path.c_str());
        return nullptr;
    }

    TextSession* s = new TextSession();
    s->wctx = weight_ctx;

    // CPU-only backend bundle through the in-tree lease registry (text
    // encoding runs once per class list; GPU offload is not worth the
    // scheduler overhead).
    int n_threads =
            threads > 0 ? threads : (int)std::thread::hardware_concurrency();
    if (n_threads <= 0) n_threads = 4;
    s->backend = yolo::init_backend_ctx(n_threads, "cpu");
    if (!s->backend.cpu) {
        text_free_session(s);
        gguf_free(g);
        return nullptr;
    }

    s->text_embed_table = find_tensor(s->wctx, "text.token_embedding.weight");
    s->text_pos_embed = find_tensor(s->wctx, "text.positional_embedding");
    s->text_ln_final_w = find_tensor(s->wctx, "text.ln_final.weight");
    s->text_ln_final_b = find_tensor(s->wctx, "text.ln_final.bias");
    s->text_proj = find_tensor(s->wctx, "text.text_projection");
    if (!s->text_embed_table || !s->text_pos_embed || !s->text_ln_final_w ||
        !s->text_ln_final_b || !s->text_proj) {
        text_free_session(s);
        gguf_free(g);
        return nullptr;
    }

    for (int i = 0; i < TEXT_N_LAYERS; i++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "transformer.resblocks.%d", i);
        if (!resolve_text_block(s->text_blocks[i], prefix, s->wctx)) {
            YOLO_LOG_ERROR("clip: failed to resolve text block %d", i);
            text_free_session(s);
            gguf_free(g);
            return nullptr;
        }
    }

    if (!clip_bpe_load(s->bpe, g, "clip.vocab", "clip.merges", VOCAB_SIZE)) {
        text_free_session(s);
        gguf_free(g);
        return nullptr;
    }

    // Build the text encoder graph. Weights stay on host memory (encoding
    // runs once per class list, not per frame).
    {
        const size_t mem = 16u * 1024u * 1024u;  // graph intermediates
        ggml_context* gctx = ggml_init({mem, nullptr, /*no_alloc*/ true});
        if (!gctx) {
            text_free_session(s);
            gguf_free(g);
            return nullptr;
        }

        // Input: token ids [TEXT_CTX].
        s->text_input_tokens =
                ggml_new_tensor_1d(gctx, GGML_TYPE_I32, TEXT_CTX);
        ggml_set_input(s->text_input_tokens);
        ggml_set_name(s->text_input_tokens, "tokens");

        // Token embedding lookup: [VOCAB, D] x [CTX] -> [D, CTX].
        ggml_tensor* embed_f32 =
                ggml_cast(gctx, s->text_embed_table, GGML_TYPE_F32);
        ggml_tensor* h = ggml_get_rows(gctx, embed_f32, s->text_input_tokens);

        // Positional embedding: GGUF stores torch [CTX, D] transposed, so
        // the ggml layout is already [D, CTX] — direct add.
        ggml_tensor* pos_f32 =
                ggml_cast(gctx, s->text_pos_embed, GGML_TYPE_F32);
        h = ggml_add(gctx, h, pos_f32);

        const int d_head = EMBED_DIM / TEXT_N_HEADS;  // 64
        for (int i = 0; i < TEXT_N_LAYERS; i++) {
            auto& b = s->text_blocks[i];
            h = transformer_block(gctx, h, b.ln1_w, b.ln1_b, b.attn_in_w,
                                  b.attn_in_b, b.attn_out_w, b.attn_out_b,
                                  b.ln2_w, b.ln2_b, b.mlp_fc_w, b.mlp_fc_b,
                                  b.mlp_proj_w, b.mlp_proj_b, TEXT_N_HEADS,
                                  d_head, /*causal*/ true);
        }

        h = clip_layer_norm(gctx, h, s->text_ln_final_w, s->text_ln_final_b);

        // EOT gather at the runtime position (x[arange(batch),
        // text.argmax(-1)]); positions after EOT are zero-padded, so the
        // first EOT is the argmax.
        s->text_eot_idx = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
        ggml_set_input(s->text_eot_idx);
        ggml_set_name(s->text_eot_idx, "eot_idx");
        h = ggml_get_rows(gctx, h, s->text_eot_idx);  // [D, 1]

        // Text projection: x @ P. GGUF stores torch P [D, D] row-major, so
        // ggml A[i0,i1] = P[i1,i0]; mul_mat needs A = P, hence the
        // permute+cont flip.
        ggml_tensor* proj_f32 = ggml_cast(gctx, s->text_proj, GGML_TYPE_F32);
        ggml_tensor* proj_T =
                ggml_cont(gctx, ggml_permute(gctx, proj_f32, 1, 0, 2, 3));
        h = ggml_mul_mat(gctx, proj_T, ggml_reshape_2d(gctx, h, EMBED_DIM, 1));

        h = clip_l2_norm(gctx, ggml_reshape_1d(gctx, h, EMBED_DIM));
        h = ggml_reshape_1d(gctx, h, EMBED_DIM);

        ggml_set_output(h);
        ggml_set_name(h, "text_embed");

        s->text_gctx = gctx;
        s->text_graph = ggml_new_graph_custom(gctx, 2048, false);
        ggml_build_forward_expand(s->text_graph, h);
        s->text_output_embed = h;

        if (!yolo::backend_ctx_graph_alloc(s->backend, s->text_graph)) {
            YOLO_LOG_ERROR("clip: text graph alloc failed");
            text_free_session(s);
            gguf_free(g);
            return nullptr;
        }
    }

    gguf_free(g);
    YOLO_LOG_INFO("clip: text session ready (%d threads)", n_threads);
    return s;
}

void text_free_session(TextSession* s) {
    if (!s) return;
    yolo::free_backend_ctx(s->backend);
    if (s->wctx) ggml_free(s->wctx);
    if (s->text_gctx) ggml_free(s->text_gctx);
    delete s;
}

bool text_encode_tokens(TextSession* s, const int32_t* tokens, float* embed) {
    if (!s || !s->text_graph || !tokens || !embed) return false;

    ggml_backend_tensor_set(s->text_input_tokens, tokens, 0,
                            TEXT_CTX * sizeof(int32_t));

    // EOT position: first 49407 in the sequence.
    int32_t eot = TEXT_CTX - 1;
    for (int i = 0; i < TEXT_CTX; i++) {
        if (tokens[i] == VOCAB_SIZE - 1) {
            eot = i;
            break;
        }
    }
    ggml_backend_tensor_set(s->text_eot_idx, &eot, 0, sizeof(int32_t));

    if (yolo::backend_ctx_graph_compute(s->backend, s->text_graph) !=
        GGML_STATUS_SUCCESS) {
        YOLO_LOG_ERROR("clip: text graph compute failed");
        return false;
    }
    ggml_backend_tensor_get(s->text_output_embed, embed, 0,
                            EMBED_DIM * sizeof(float));
    return true;
}

int text_tokenize(TextSession* s, const char* text, int32_t* tokens) {
    if (!s) return 0;
    return clip_bpe_tokenize(s->bpe, text, TEXT_CTX, tokens);
}

bool text_encode_string(TextSession* s, const char* text, float* embed) {
    if (!s) return false;
    int32_t tokens[TEXT_CTX];
    text_tokenize(s, text, tokens);
    return text_encode_tokens(s, tokens, embed);
}

}  // namespace clip
